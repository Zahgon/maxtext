# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of the kvcache."""

from typing import Any

import jax
import jax.numpy as jnp

from flax import linen as nn
from flax import nnx

from aqt.jax.v2 import config as aqt_config
from aqt.jax.v2.aqt_tensor import QTensor as KVTensor
from aqt.jax.v2.flax import aqt_flax

from maxtext.layers import nnx_wrappers
from maxtext.layers.initializers import variable_to_logically_partitioned

from maxtext.common.common_types import Array, AxisNames, AxisIdxes, Config, CACHE_BATCH_PREFILL, DType, MODEL_MODE_PREFILL, MODEL_MODE_TRAIN, MODEL_MODE_AUTOREGRESSIVE, CACHE_HEADS_NONE, DECODING_ACTIVE_SEQUENCE_INDICATOR
from maxtext.common.common_types import CACHE_BATCH, CACHE_SEQUENCE, CACHE_HEADS, CACHE_KV, CACHE_SCALE_BATCH, CACHE_SCALE_SEQUENCE, CACHE_SCALE_HEADS, CACHE_SCALE_KV


MAX_INT8 = 127.5
MAX_INT4 = 7.5
E4M3_MAX = jnp.finfo(jnp.float8_e4m3fn).max.astype(jnp.float32)




def transpose_tuple(items: tuple[Any, ...], axis_order: AxisIdxes) -> tuple[Any, ...]:
  return tuple((items[i] for i in axis_order))


class KVQuant:
  """Class to configure quantization for KV cache."""

  axis_cfg = ""
  dtype = None

  def __init__(self, config: Config):
    assert config.quantize_kvcache
    self.axis_cfg = config.kv_quant_axis
    self.dtype = self._get_dtype(config.kv_quant_dtype)

  def _get_dtype(self, dtype_cfg: str):
    if dtype_cfg == "int4":
      return jnp.int4
    if dtype_cfg == "int8":
      return jnp.int8
    if dtype_cfg == "fp8":
      return jnp.float8_e4m3fn
    raise ValueError(f"Invalid kv_quant_dtype: {dtype_cfg}")

  def _get_max_axis(self, axis_names: AxisNames):
    if self.axis_cfg == "dkv":
      return axis_names.index(CACHE_KV)
    if self.axis_cfg == "heads_and_dkv":
      return (axis_names.index(CACHE_HEADS), axis_names.index(CACHE_KV))
    raise ValueError(f"Invalid KV quant axis cfg: {self.axis_cfg}")

  def quantize(self, kv: Array, axis_names: AxisNames):
    """Quantize key/values stored in kvcache."""
    assert self.axis_cfg, "KV quant axis cannot be None"
    max_axis = self._get_max_axis(axis_names)
    scale = jnp.max(jnp.abs(kv), axis=max_axis, keepdims=True)
    if self.dtype == jnp.int8:
      value = jnp.int8(jnp.rint(kv * (MAX_INT8 / scale)))
      return value, scale
    if self.dtype == jnp.int4:
      value = jnp.int4(jnp.rint(kv * (MAX_INT4 / scale)))
      return value, scale
    if self.dtype == jnp.float8_e4m3fn:
      value = jnp.float8_e4m3fn(kv * (E4M3_MAX / scale))
      return value, scale
    raise ValueError(f"Invalid KV quant dtype:{self.dtype}.")

  def einsum_fn_with_rhs_qtensor(
      self,
      rhs_dequant_mode=None,
      rhs_calibration_mode=None,
      lhs_dequant_mode=None,
      lhs_calibration_mode=None,
  ):
    """einsum function where QTensor is the right-hand-side"""
    # Assumes kv is already quantized.
    einsum = jnp.einsum
    if self.dtype != jnp.float8_e4m3fn:
      num_bits = 4 if self.dtype == jnp.int4 else 8
      kv_cfg = aqt_config.dot_general_make(
          lhs_bits=None,
          rhs_bits=num_bits,
          bwd_bits=None,
          use_fwd_quant=False,
      )
    else:
      kv_cfg = aqt_config.config_fwd_fp8()

    if rhs_dequant_mode:
      aqt_config.set_fwd_dequant_mode(kv_cfg, rhs_dequant_mode=rhs_dequant_mode)
    if rhs_calibration_mode:
      aqt_config.set_fwd_calibration_mode(
          kv_cfg,
          rhs_calibration_mode=rhs_calibration_mode,
      )
    if lhs_dequant_mode:
      aqt_config.set_fwd_dequant_mode(kv_cfg, lhs_dequant_mode=lhs_dequant_mode)
    if lhs_calibration_mode:
      aqt_config.set_fwd_calibration_mode(
          kv_cfg,
          lhs_calibration_mode=lhs_calibration_mode,
      )
    einsum = aqt_flax.AqtEinsum(
        rhs_quant_mode=aqt_flax.QuantMode.TRAIN,
        lhs_freeze_mode=aqt_flax.FreezerMode.NONE,
        rhs_freeze_mode=aqt_flax.FreezerMode.NONE,
        cfg=kv_cfg,
    )
    return einsum

  def einsum_fn_with_rhs_qtensor_and_dequant(self):
    """Get einstein summation for different dequant modes."""
    if self.dtype == jnp.float8_e4m3fn:
      return self.einsum_fn_with_rhs_qtensor(
          lhs_dequant_mode=aqt_config.DequantMode.THIS_INPUT,
          lhs_calibration_mode=aqt_config.CalibrationMode.REMAINING_AXIS,
          rhs_dequant_mode=aqt_config.DequantMode.OTHER_INPUT,
          rhs_calibration_mode=aqt_config.CalibrationMode.REMAINING_AXIS,
      )
    else:
      return self.einsum_fn_with_rhs_qtensor(
          rhs_dequant_mode=aqt_config.DequantMode.OTHER_INPUT,
          rhs_calibration_mode=aqt_config.CalibrationMode.REMAINING_AXIS,
      )


def kv_cache_as_linen(
    *,
    max_prefill_length: int,
    max_target_length: int,
    batch: int,
    key_seq_len: int,
    value_seq_len: int,
    key_heads: int,
    value_heads: int,
    key_head_size: int,
    value_head_size: int,
    dtype: DType,
    kv_quant: None | KVQuant = None,
    prefill_cache_logical_axis_names: AxisNames = (CACHE_BATCH_PREFILL, CACHE_SEQUENCE, CACHE_HEADS, CACHE_KV),
    cache_logical_axis_names: AxisNames = (CACHE_BATCH, CACHE_SEQUENCE, CACHE_HEADS, CACHE_KV),
    cache_scale_logical_axis_names: AxisNames = (
        CACHE_SCALE_BATCH,
        CACHE_SCALE_SEQUENCE,
        CACHE_SCALE_HEADS,
        CACHE_SCALE_KV,
    ),
    prefill_cache_axis_order: AxisIdxes = (1, 2, 0, 3),
    ar_cache_axis_order: AxisIdxes = (1, 2, 0, 3),
    key_axis_order: AxisIdxes = (2, 0, 1, 3),
    use_chunked_prefill: bool = False,
    model_mode: str = MODEL_MODE_PREFILL,
    name: str | None = None,
):
  """Initializes the KVCache module and returns it as a Linen module.

  Args:
    max_prefill_length: The maximum prefill length.
    max_target_length: The maximum target length.
    batch: The batch size.
    key_seq_len: The key sequence length.
    value_seq_len: The value sequence length.
    key_heads: The number of key heads.
    value_heads: The number of value heads.
    key_head_size: The key head size.
    value_head_size: The value head size.
    dtype: The data type.
    kv_quant: The KVQuant configuration.
    prefill_cache_logical_axis_names: The logical axis names for the prefill cache.
    cache_logical_axis_names: The logical axis names for the cache.
    cache_scale_logical_axis_names: The logical axis names for the cache scale.
    prefill_cache_axis_order: The axis order for the prefill cache.
    ar_cache_axis_order: The axis order for the autoregressive cache.
    key_axis_order: The axis order for the key.
    use_chunked_prefill: Whether to use chunked prefill.
    model_mode: The model mode.
    name: The name of the Linen module.

  Returns:
    A Linen module that wraps the NNX `KVCache` module.
  """
  pass


class BaseCache(nnx.Module):
  """Abstract base class for Caches."""


class KVCache(BaseCache):
  """Implementation of the KVCache."""

  def __init__(
      self,
      max_prefill_length: int,
      max_target_length: int,
      # TODO(bvandermoon): Can we get batch, key_seq_len, value_seq_len, key_heads,
      # and value_heads from key/value after migrating Attention to NNX?
      batch: int,
      key_seq_len: int,
      value_seq_len: int,
      key_heads: int,
      value_heads: int,
      key_head_size: int,
      value_head_size: int,
      dtype: DType,
      kv_quant: None | KVQuant = None,
      prefill_cache_logical_axis_names: AxisNames = (CACHE_BATCH_PREFILL, CACHE_SEQUENCE, CACHE_HEADS, CACHE_KV),
      cache_logical_axis_names: AxisNames = (CACHE_BATCH, CACHE_SEQUENCE, CACHE_HEADS, CACHE_KV),
      cache_scale_logical_axis_names: AxisNames = (
          CACHE_SCALE_BATCH,
          CACHE_SCALE_SEQUENCE,
          CACHE_SCALE_HEADS,
          CACHE_SCALE_KV,
      ),
      prefill_cache_axis_order: AxisIdxes = (1, 2, 0, 3),
      ar_cache_axis_order: AxisIdxes = (1, 2, 0, 3),
      key_axis_order: AxisIdxes = (2, 0, 1, 3),
      use_chunked_prefill: bool = False,
      model_mode: str = MODEL_MODE_PREFILL,
      *,
      # Not used in KVCache but passed in by nnx_wrappers.to_linen.
      # TODO: Remove when bridge no longer needed
      rngs: nnx.Rngs = None,
  ):
    """Initializes the KVCache module.

    Args:
      max_prefill_length: The maximum prefill length.
      max_target_length: The maximum target length.
      batch: The batch size.
      key_seq_len: The key sequence length.
      value_seq_len: The value sequence length.
      key_heads: The number of key heads.
      value_heads: The number of value heads.
      key_head_size: The key head size.
      value_head_size: The value head size.
      dtype: The data type.
      kv_quant: The KVQuant configuration.
      prefill_cache_logical_axis_names: The logical axis names for the prefill cache.
      cache_logical_axis_names: The logical axis names for the cache.
      cache_scale_logical_axis_names: The logical axis names for the cache scale.
      prefill_cache_axis_order: The axis order for the prefill cache.
      ar_cache_axis_order: The axis order for the autoregressive cache.
      key_axis_order: The axis order for the key.
      model_mode: The model mode.
      use_chunked_prefill: Whether to use chunked prefill.
      rngs: The random number generators for initialization.
    """
    super().__init__()
    self.max_prefill_length = max_prefill_length
    self.max_target_length = max_target_length
    self.batch = batch
    self.key_seq_len = key_seq_len
    self.value_seq_len = value_seq_len
    self.key_heads = key_heads
    self.value_heads = value_heads
    self.key_head_size = key_head_size
    self.value_head_size = value_head_size
    self.dtype = dtype
    self.kv_quant = kv_quant
    self.prefill_cache_logical_axis_names = prefill_cache_logical_axis_names
    self.cache_logical_axis_names = cache_logical_axis_names
    self.cache_scale_logical_axis_names = cache_scale_logical_axis_names
    self.prefill_cache_axis_order = prefill_cache_axis_order
    self.ar_cache_axis_order = ar_cache_axis_order
    self.key_axis_order = key_axis_order
    self.model_mode = model_mode
    self.use_chunked_prefill = use_chunked_prefill

    if model_mode in (MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE):
      self._initialize_prefill_caches(model_mode)
      self._initialize_ar_cache_vars(model_mode)





  def _get_cached_kv_dtype(self):
    return self.kv_quant.dtype if self.kv_quant else self.dtype

  def _get_cache_scale_logical_shape(self, heads, cache_length):
    assert self.kv_quant
    if self.kv_quant.axis_cfg == "dkv":
      return (self.batch, cache_length, heads, 1)
    if self.kv_quant.axis_cfg == "heads_and_dkv":
      return (self.batch, cache_length, 1, 1)
    raise ValueError(f"Invalid config for kv_quant_axis:{self.kv_quant.axis_cfg}")

  def _initialize_prefill_caches(self, model_mode):
    """Get a shaped abstraction of the state"""

    cache_length = self.max_prefill_length
    dtype = self._get_cached_kv_dtype()

    if model_mode == MODEL_MODE_PREFILL:
      cache_logical_axis_names = self.prefill_cache_logical_axis_names
    else:
      cache_logical_axis_names = self.cache_logical_axis_names
    cache_axis_names = transpose_tuple(cache_logical_axis_names, self.prefill_cache_axis_order)

    cache_logical_shape = (self.batch, cache_length, self.key_heads, self.key_head_size)
    cache_shape_key = transpose_tuple(cache_logical_shape, self.prefill_cache_axis_order)

    cache_logical_shape = (self.batch, cache_length, self.value_heads, self.value_head_size)
    cache_shape_value = transpose_tuple(cache_logical_shape, self.prefill_cache_axis_order)

    self.cached_prefill_key = nnx.Cache(
        jnp.zeros(cache_shape_key, dtype=dtype),
        out_sharding=cache_axis_names,
    )
    self.cached_prefill_value = nnx.Cache(
        jnp.zeros(cache_shape_value, dtype=dtype),
        out_sharding=cache_axis_names,
    )

    if model_mode == MODEL_MODE_PREFILL:
      segment_id_axis_names = (CACHE_BATCH_PREFILL, CACHE_SEQUENCE)
    else:
      segment_id_axis_names = (CACHE_BATCH, CACHE_SEQUENCE)

    self.cache_prefill_segment_id = nnx.Cache(
        jnp.zeros((cache_logical_shape[0], cache_length), dtype=jnp.int32),
        out_sharding=segment_id_axis_names,
    )

    if self.kv_quant:
      cache_scale_axis_names = transpose_tuple(self.cache_scale_logical_axis_names, self.prefill_cache_axis_order)

      cache_scale_logical_shape = self._get_cache_scale_logical_shape(self.key_heads, cache_length)
      cache_key_scale_shape = transpose_tuple(cache_scale_logical_shape, self.prefill_cache_axis_order)

      cache_scale_logical_shape = self._get_cache_scale_logical_shape(self.value_heads, cache_length)
      cache_value_scale_shape = transpose_tuple(cache_scale_logical_shape, self.prefill_cache_axis_order)

      self.cached_prefill_key_scale = nnx.Cache(
          jnp.zeros(cache_key_scale_shape, dtype=jnp.bfloat16),
          out_sharding=cache_scale_axis_names,
      )
      self.cached_prefill_value_scale = nnx.Cache(
          jnp.zeros(cache_value_scale_shape, dtype=jnp.bfloat16),
          out_sharding=cache_scale_axis_names,
      )
    else:
      self.cached_prefill_key_scale = None
      self.cached_prefill_value_scale = None


  def _initialize_ar_cache_vars(self, model_mode):
    """get ar cache vars"""

    dtype = self._get_cached_kv_dtype()
    if self.max_target_length <= self.max_prefill_length:
      raise ValueError(
          f"max_target_length: {self.max_target_length} should be greater than max_prefill_length:"
          f" {self.max_prefill_length}!"
      )
    cache_length = self.max_target_length - self.max_prefill_length

    if model_mode == MODEL_MODE_PREFILL:
      cache_logical_axis_names = self.prefill_cache_logical_axis_names
    else:
      cache_logical_axis_names = self.cache_logical_axis_names
    cache_axis_names = transpose_tuple(cache_logical_axis_names, self.ar_cache_axis_order)

    cache_logical_shape = (self.batch, cache_length, self.key_heads, self.key_head_size)
    cache_shape_key = transpose_tuple(cache_logical_shape, self.ar_cache_axis_order)

    cache_logical_shape = (self.batch, cache_length, self.value_heads, self.value_head_size)
    cache_shape_value = transpose_tuple(cache_logical_shape, self.ar_cache_axis_order)

    # TODO(b/339703100): investigate the issue why with_logical_partitioning doesn't enforce sharding
    self.cached_ar_key = nnx.Cache(
        jnp.zeros(cache_shape_key, dtype=dtype),
        out_sharding=cache_axis_names,
    )
    self.cached_ar_key[...] = nn.with_logical_constraint(
        self.cached_ar_key[...],
        cache_axis_names,
    )

    self.cached_ar_value = nnx.Cache(
        jnp.zeros(cache_shape_value, dtype=dtype),
        out_sharding=cache_axis_names,
    )
    self.cached_ar_value[...] = nn.with_logical_constraint(
        self.cached_ar_value[...],
        cache_axis_names,
    )

    if model_mode == MODEL_MODE_PREFILL:
      segment_id_axis_names = (CACHE_BATCH_PREFILL, CACHE_SEQUENCE)
    else:
      segment_id_axis_names = (CACHE_BATCH, CACHE_SEQUENCE)
    self.cache_ar_segment_id = nnx.Cache(
        jnp.zeros((cache_logical_shape[0], cache_length), dtype=jnp.int32),
        out_sharding=segment_id_axis_names,
    )

    self.cached_ar_lengths = nnx.Cache(
        jnp.zeros((cache_logical_shape[0],), dtype=jnp.int32),
        out_sharding=(CACHE_BATCH,),
    )

    if self.kv_quant:
      cache_scale_axis_names = transpose_tuple(self.cache_scale_logical_axis_names, self.ar_cache_axis_order)

      cache_scale_logical_shape = self._get_cache_scale_logical_shape(self.key_heads, cache_length)
      cache_key_scale_shape = transpose_tuple(cache_scale_logical_shape, self.ar_cache_axis_order)

      cache_scale_logical_shape = self._get_cache_scale_logical_shape(self.value_heads, cache_length)
      cache_value_scale_shape = transpose_tuple(cache_scale_logical_shape, self.ar_cache_axis_order)

      self.cached_ar_key_scale = nnx.Cache(
          jnp.zeros(cache_key_scale_shape, dtype=jnp.bfloat16),
          out_sharding=cache_scale_axis_names,
      )
      self.cached_ar_value_scale = nnx.Cache(
          jnp.zeros(cache_value_scale_shape, dtype=jnp.bfloat16),
          out_sharding=cache_scale_axis_names,
      )
    else:
      self.cached_ar_key_scale = None
      self.cached_ar_value_scale = None

    self.cache_ar_index = nnx.Cache(
        jnp.zeros((1,), dtype=jnp.int32),
        out_sharding=(),
    )


  def kv_cache_chunked_prefill(
      self, key: Array, value: Array, decoder_segment_ids: Array, previous_chunk: None | Array = None
  ):
    """Update the current kv cache into previous chunk and return needed length.

    The previous chunk kv cache should be in the model's param.

    Prefill cache need to be max prefill length to prevent different shape of kv cache.
    Different shape of kv cache in previous chunk could produce different compiled graph.

    Args:
      key: in shape [b, s, n, d].
      value: in shape [b, s, n, d].
      decoder_segment_ids: [b, s] -- marking segment ids for tokens
      previous_chunk:
        In shape [b, s]. The tokens without padding in previous chunk.
        Use to preserve the previous kv cache.

    Returns:
      key, value, decoder_segment_id.
    """
    pass

  def kv_cache_prefill(
      self,
      key: Array,
      value: Array,
      decoder_segment_ids: Array,
  ):
    """In prefill mode, we zero out the existing cache, run the computation and
    prepare the cache as necessary.

    Args:
      key: in shape [b, s, n, d].
      value: in shape [b, s, n, d].
      decoder_segment_ids: [b, s] -- marking segment ids for tokens

    Returns:
      key, value, decoder_segment_id.

    """
    pass

  def update_ar_key_value(
      self,
      one_token_key: Array,
      one_token_value: Array,
      key_caches: tuple[nnx.Cache, nnx.Cache | None],
      value_caches: tuple[nnx.Cache, nnx.Cache | None],
      one_hot_indices: Array,
      lengths: Array,
      use_ragged_attention: bool,
  ) -> None:
    """Adds a single token's results to the ar kv cache

    Args:
        one_token_key (Array): Key of one token to add to the cache
        one_token_value (Array): Value of one token to add to the cache
        cached_ar_key (tuple[nnx.Cache, nnx.Cache|None],): Cached keys to add new token key to, possibly with scale
        cached_ar_value (tuple[nnx.Cache, nnx.Cache|None],: Cached values to add new token value to, possible with scale
        one_hot_indices (Array): Location of the new token within the cache

    Returns:
        tuple[Array, Array]: Updated caches for key and value with new token info added
    """
    pass

  def get_cached_values(self, cache_vars, target_dtype, cache_axis_order) -> jax.Array | KVTensor:
    """get cached values"""
    pass

  def kv_cache_autoregressive(
      self,
      key: Array,
      value: Array,
      use_ragged_attention: bool = False,
  ):
    """In autoregressive mode, we update the cache for this entry and
       then return the full cache.

    Args:
      key: in shape [b, 1, n, d].
      value: in shape [b, 1, n, d].
      decoder_segment_ids: [b, 1] -- marking segment ids for tokens

    Returns:
      tuple of (key, value, segment_id) for both prefill and ar cache,
    Raises:
      ValueError: when key/value shape is not [batch, 1, num_heads, heads_dim].
    """
    pass

  def __call__(
      self,
      key: Array,
      value: Array,
      decoder_segment_ids: Array,
      model_mode: str,
      use_ragged_attention: bool = False,
      previous_chunk: Any = None,
  ) -> tuple:
    """KV cache takes the current state and updates the state accordingly.

    The key and value have dimension [b, s, n_kv, d],
    but we cache them with a reshape as defined in *_axis_order config as a TPU
    fusion optimization. This also enables the "scatter via one-hot
    broadcast" trick, which means we do a one-hot broadcast instead of a
    scatter/gather operations, resulting in a 3-4x speedup in practice.

    Args:
      key: in shape [b, s, n_kv, d].
      value: in shape [b, s, n_kv, d].
      model_mode: model mode controlling model.

    Returns:
      two tuples of (k, v, decoder_segments) -- either can be Nones

    """
    if model_mode == MODEL_MODE_PREFILL:
      if self.use_chunked_prefill:
        return self.kv_cache_chunked_prefill(key, value, decoder_segment_ids, previous_chunk), None
      else:
        return self.kv_cache_prefill(key, value, decoder_segment_ids), None
    elif model_mode == MODEL_MODE_AUTOREGRESSIVE:
      return self.kv_cache_autoregressive(key, value, use_ragged_attention)
    else:
      raise ValueError(f"Model Mode isn't supported! {model_mode=}")


class GatedDeltaNetCache(BaseCache):
  """Cache for Linear Attention (Gated Delta Net).

  Stores the fixed-size recurrent state and the sliding window state for convolution.
  """

  def __init__(
      self,
      batch: int,
      num_heads: int,
      k_head_dim: int,
      v_head_dim: int,
      conv_kernel_size: int,
      conv_dim: int,
      dtype: DType,
      cache_batch_axis_name: str = CACHE_BATCH,
      cache_heads_axis_name: str = CACHE_HEADS,
  ):
    super().__init__()
    self.batch = batch
    self.dtype = dtype

    # 1. Recurrent State (S) for the Delta Rule
    # Shape: [Batch, Heads, K_Dim, V_Dim]
    # We maintain the running state matrix.
    self.recurrent_state = nnx.Cache(
        jnp.zeros((int(batch), num_heads, k_head_dim, v_head_dim), dtype=dtype),
        # Sharding: Batch, Heads, None (K), None (V)
        out_sharding=(cache_batch_axis_name, cache_heads_axis_name, None, None),
    )

    # 2. Convolution State for the 1D Conv
    # Shape: [Batch, Kernel_Size - 1, Conv_Dim]
    # We store the last (K-1) inputs to perform the sliding window conv during decoding.
    self.conv_state = nnx.Cache(
        jnp.zeros((int(batch), conv_kernel_size - 1, conv_dim), dtype=dtype),
        # Sharding: Batch, None (Time), None (Dim)
        out_sharding=(cache_batch_axis_name, None, None),
    )

  def __call__(self):
    """Returns the cache variables for the layer to use."""
    return self


def gated_delta_net_cache_as_linen(
    *,
    batch: int,
    num_heads: int,
    head_dim: int,
    conv_kernel_size: int,
    conv_dim: int,
    dtype: DType,
    name: str | None = None,
):
  """Initializes the GatedDeltaNetCache and returns it as a Linen module."""
  pass


def mla_kv_cache_as_linen(
    *,
    max_prefill_length: int,
    max_target_length: int,
    batch: int,
    key_seq_len: int,
    value_seq_len: int,
    key_head_size: int,
    value_head_size: int,
    dtype: DType,
    key_heads: int = 1,
    value_heads: int = 1,
    kv_quant: None | KVQuant = None,
    prefill_cache_axis_order: AxisIdxes = (1, 2, 0, 3),
    ar_cache_axis_order: AxisIdxes = (1, 2, 0, 3),
    use_chunked_prefill: bool = False,
    model_mode: str = MODEL_MODE_PREFILL,
    name: str | None = None,
):
  """Initializes the MlaKVCache module and returns it as a Linen module.

  Args:
    max_prefill_length: The maximum prefill length.
    max_target_length: The maximum target length.
    batch: The batch size.
    key_seq_len: The key sequence length.
    value_seq_len: The value sequence length.
    key_head_size: The key head size.
    value_head_size: The value head size.
    dtype: The data type.
    key_heads: The number of key heads.
    value_heads: The number of value heads.
    kv_quant: The KVQuant configuration.
    prefill_cache_axis_order: The axis order for the prefill cache.
    ar_cache_axis_order: The axis order for the autoregressive cache.
    use_chunked_prefill: Whether to use chunked prefill.
    model_mode: The model mode.
    name: The name of the Linen module.

  Returns:
    A Linen module that wraps the NNX `MlaKVCache` module.
  """
  pass


class MlaKVCache(KVCache):
  """Implementation of the KVCache for MLA."""

  def __init__(
      self,
      max_prefill_length: int,
      max_target_length: int,
      # TODO(bvandermoon): Can we get batch, key_seq_len, value_seq_len,
      # key_head_size, value_head_size, key_heads, and value_heads from
      # key/value after migrating Attention to NNX?
      batch: int,
      key_seq_len: int,
      value_seq_len: int,
      key_head_size: int,
      value_head_size: int,
      dtype: DType,
      key_heads: int = 1,
      value_heads: int = 1,
      kv_quant: None | KVQuant = None,
      prefill_cache_logical_axis_names: AxisNames = (CACHE_BATCH_PREFILL, CACHE_SEQUENCE, CACHE_HEADS_NONE, CACHE_KV),
      cache_logical_axis_names: AxisNames = (CACHE_BATCH, CACHE_SEQUENCE, CACHE_HEADS_NONE, CACHE_KV),
      cache_scale_logical_axis_names: AxisNames = (
          CACHE_SCALE_BATCH,
          CACHE_SCALE_SEQUENCE,
          CACHE_SCALE_HEADS,
          CACHE_SCALE_KV,
      ),
      prefill_cache_axis_order: AxisIdxes = (1, 2, 0, 3),
      ar_cache_axis_order: AxisIdxes = (1, 2, 0, 3),
      key_axis_order: AxisIdxes = (2, 0, 1, 3),
      use_chunked_prefill: bool = False,
      model_mode: str = MODEL_MODE_PREFILL,
      *,
      # Not used in MlaKVCache but passed in by nnx_wrappers.to_linen.
      # TODO: Remove when bridge no longer needed
      rngs: nnx.Rngs = None,
  ):
    """Initializes the MlaKVCache module.

    Args:
      max_prefill_length: The maximum prefill length.
      max_target_length: The maximum target length.
      batch: The batch size.
      key_seq_len: The key sequence length.
      value_seq_len: The value sequence length.
      key_head_size: The key head size.
      value_head_size: The value head size.
      dtype: The data type.
      key_heads: The number of key heads.
      value_heads: The number of value heads.
      kv_quant: The KVQuant configuration.
      prefill_cache_logical_axis_names: The logical axis names for the prefill
        cache.
      cache_logical_axis_names: The logical axis names for the cache.
      cache_scale_logical_axis_names: The logical axis names for the cache
        scale.
      prefill_cache_axis_order: The axis order for the prefill cache.
      ar_cache_axis_order: The axis order for the autoregressive cache.
      key_axis_order: The axis order for the key.
      use_chunked_prefill: Whether to use chunked prefill.
      model_mode: The model mode.
      rngs: The random number generators for initialization.
    """
    super().__init__(
        max_prefill_length=max_prefill_length,
        max_target_length=max_target_length,
        batch=batch,
        key_seq_len=key_seq_len,
        value_seq_len=value_seq_len,
        key_heads=key_heads,
        value_heads=value_heads,
        key_head_size=key_head_size,
        value_head_size=value_head_size,
        dtype=dtype,
        kv_quant=kv_quant,
        prefill_cache_logical_axis_names=prefill_cache_logical_axis_names,
        cache_logical_axis_names=cache_logical_axis_names,
        cache_scale_logical_axis_names=cache_scale_logical_axis_names,
        prefill_cache_axis_order=prefill_cache_axis_order,
        ar_cache_axis_order=ar_cache_axis_order,
        key_axis_order=key_axis_order,
        use_chunked_prefill=use_chunked_prefill,
        model_mode=model_mode,
        rngs=rngs,
    )



  def __call__(
      self,
      key_latent: Array,
      key_rope: Array,
      decoder_segment_ids: Array,
      model_mode: str,
      use_ragged_attention: bool = False,
      previous_chunk: Any = None,
  ) -> tuple[
      None | tuple[Array, Array, Array],
      None | tuple[Array, Array, Array, Array],
  ]:
    assert model_mode != MODEL_MODE_TRAIN, "incorrectly updating kvcache in train mode."
    assert self.kv_quant is None, "kvcache quantization not supported with mla."
    key_latent = self.key_latent_add_head_dim(key_latent)
    prefill_cache, ar_cache = super().__call__(key_latent, key_rope, decoder_segment_ids, model_mode)
    if prefill_cache:
      key_latent, key_rope, decoder_segments_ids = prefill_cache
      prefill_cache = (
          self.key_latent_remove_head_dim(key_latent),
          key_rope,
          decoder_segments_ids,
      )
    if ar_cache:
      key_latent, key_rope, decoder_segments_ids, lengths = ar_cache
      ar_cache = (
          self.key_latent_remove_head_dim(key_latent),
          key_rope,
          decoder_segments_ids,
          lengths,
      )
    return prefill_cache, ar_cache
