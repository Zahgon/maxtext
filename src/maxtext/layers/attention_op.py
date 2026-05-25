#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  pytype: disable=module-attr
"""Attentions Ops Layers."""

import dataclasses
import functools
from functools import partial
import math
from typing import Any, Callable, Optional, Tuple

from flax import linen as nn
from flax import nnx
from flax.linen import partitioning
import jax
from jax import lax
from jax.ad_checkpoint import checkpoint_name
from jax.experimental import pallas as pl
from jax.experimental.pallas.ops.gpu import attention as gpu_pallas_attention
from jax.experimental.pallas.ops.gpu import decode_attention as gpu_pallas_decode_attention
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.common.common_types import (
    Array,
    AttentionType,
    AxisIdxes,
    AxisNames,
    BATCH_ATTN,
    CACHE_BATCH,
    CACHE_BATCH_PREFILL,
    CACHE_HEADS,
    CACHE_KV,
    CACHE_SCALE_BATCH,
    CACHE_SCALE_HEADS,
    CACHE_SCALE_KV,
    CACHE_SCALE_SEQUENCE,
    CACHE_SEQUENCE,
    Config,
    DECODE_BATCH,
    DECODE_LENGTH,
    DECODING_ACTIVE_SEQUENCE_INDICATOR,
    DEFAULT_MASK_VALUE,
    DType,
    D_KV,
    HEAD,
    KV_LENGTH,
    LENGTH,
    MODEL_MODE_AUTOREGRESSIVE,
    MODEL_MODE_PREFILL,
    MODEL_MODE_TRAIN,
    PREFILL_LENGTH,
    Q_LENGTH,
)
from maxtext.inference import page_manager
from maxtext.inference.kvcache import KVQuant, KVTensor
from maxtext.kernels.attention import jax_flash_attention
from maxtext.kernels.attention.ragged_attention import ragged_gqa
from maxtext.kernels.attention.ragged_attention import ragged_mha
from maxtext.layers import nnx_wrappers
from maxtext.layers.initializers import variable_to_logically_partitioned
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.utils import max_utils
from maxtext.utils.sharding import logical_to_mesh_axes, maybe_shard_with_pspec
import numpy as np
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as tokamax_splash_kernel
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as tokamax_splash_mask
# pylint: disable=line-too-long, g-doc-args, g-doc-return-or-yield, bad-continuation, g-inconsistent-quotes
# pytype: disable=attribute-error

# Used to pass in splash attention block sizes from config.
global_block_q = 0
global_block_kv = 0
global_block_kv_compute = 0
global_block_q_dkv = 0
global_block_kv_dkv = 0
global_block_kv_dkv_compute = 0
global_block_q_dq = 0
global_block_kv_dq = 0
global_use_fused_bwd_kernel = False
global_q_layout = ""
global_k_layout = ""
global_v_layout = ""

dynamic_vector_slice_in_dim = jax.vmap(lax.dynamic_slice_in_dim, in_axes=(None, 0, None, None))


def validate_compute_axis_order(s: AxisIdxes) -> None:
  valid_compute_axis_order = ((0, 1, 2, 3), (0, 2, 1, 3))
  if s not in valid_compute_axis_order:  # currently supported compute_axis_order
    raise ValueError(
        "Invalid compute_axis_order was passed. Valid options ",
        valid_compute_axis_order,
    )


def apply_mask_to_logits(logits: Array, mask: Array):
  """Applies a floating-point mask to a set of logits.

  The mask is represented as a tensor with some dtype where 0 represents true
  and values
  below a large negative number (here set to
  get_large_negative_number(logits.dtype) / 2) represent false. Applying the
  mask
  leaves the logits alone in the true case and replaces them by
  get_large_negative_number(logits.dtype) in the false case. Previously, this
  was
  done by adding the logits to the mask; however, this leads to a bad fusion
  decision in the compiler that saves the values in memory rather than
  just the predicate. This implementation avoids that problem.

  from
  https://github.com/google/praxis/blob/4712a6b9ee13e224b86e235ff55f7c6bab9fbab3/praxis/py_utils.py#L706

  Args:
    logits: A JTensor of logit values.
    mask: A JTensor of mask values with the encoding described in the function
      documentation.

  Returns:
    Masked logits.
  """
  pass


def validate_gpu_flash_attention(sinks: Array | None, record_max_logits: bool) -> None:
  """Helper function to check for unsupported features with flash attention on GPU."""
  pass


# TODO(agagik): change splash_attention_mask._ComputableMask to be non protected
class ChunkedCausalMask(splash_attention_mask._ComputableMask):  # pylint: disable=protected-access
  """Lazy chunked causal mask.

  Attention is causal within each chunk (0, K), (K, 2K), (2K, 3K), ... tokens
  attend to each other but not across chunks.
  Llama4 models use interleaved chunk attention along with global attention.

  This mask class inherits from splash_attention_mask._ComputableMask and is
  designed to be used with Splash Attention.
  It allows the mask logic to be computed on-the-fly or fused into the attention
  kernel, avoiding the memory cost of
  materializing the full (sequence_length, sequence_length) boolean mask array,
  which can be prohibitive for long sequences.
  """

  #: The size of each attention chunk.
  chunk_size: int

  def __init__(
      self,
      shape: tuple[int, int],
      chunk_size: int,
      shard_count: int = 1,
  ):
    if chunk_size <= 0:
      raise ValueError("chunk_size must be positive")
    self.chunk_size = chunk_size

    # Define the mask function for chunk attention
    def chunked_causal_mask_function(q_ids, kv_ids):
      """Computes the mask logic for the given slice indices."""
      pass

    # Initialize the parent ComputableMask with this function
    super().__init__(
        shape=shape,
        mask_function=chunked_causal_mask_function,
        shard_count=shard_count,
    )

  # Implement equality and hashing based on relevant attributes
  def __eq__(self, other: object):
    if not isinstance(other, type(self)):
      return NotImplemented
    # Compare shape, chunk_size, and the underlying q_sequence array
    return (
        self.shape == other.shape
        and self.chunk_size == other.chunk_size
        and np.array_equal(self.q_sequence, other.q_sequence)
    )

  def __hash__(self):
    return hash(
        (
            type(self),
            self.shape,
            self.chunk_size,
            self.q_sequence.tobytes() if self.q_sequence is not None else None,
        )
    )


def _generate_chunk_attention_mask(mask_shape: tuple[int, int], chunk_size: int, q_offset: int = 0) -> jax.Array:
  """Generates an explicit boolean mask for chunked causal attention.

  This function computes the full boolean mask array where True indicates
  attention is allowed based on chunked causal rules (tokens attend only
  within the same chunk, and causally within that chunk).

  Args:
    mask_shape: The desired shape of the mask (q_seq_len, kv_seq_len).
    chunk_size: The size of the attention chunks.

  Returns:
    A boolean mask of shape `mask_shape` where True indicates attention is
    allowed according to chunked causal rules, and False otherwise.

  Raises:
    ValueError: If chunk_window_size is None or not positive.
  """

  row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0) + q_offset
  col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
  if chunk_size <= 0:
    raise ValueError("chunk_size must be positive")

  # chunk mask calculation
  same_chunk = (row_ids // chunk_size) == (col_ids // chunk_size)
  chunk_mask = same_chunk & (row_ids >= col_ids)
  return chunk_mask


def _make_block_mask_indices(bidirectional_mask):
  """Creates block mask identifying segments based on a bidirectional mask.

  Args:
    bidirectional_mask: boolean mask, e.g. [011110011010].

  Returns:
    block mask for segments, e.g. [011110022030].
  """
  # Left pad 0.
  padded_mask = jnp.pad(bidirectional_mask, [(0, 0), (1, 0)], constant_values=0)
  boundary = padded_mask[..., 1:] > padded_mask[..., :-1]
  numbered_boundary = jnp.cumsum(boundary, axis=-1)
  return bidirectional_mask * numbered_boundary


def _make_bidirectional_block_mask(bidirectional_mask):
  """Creates bidirectional block mask from bidirectional_mask, where True corresponds to image tokens.

  bidirectional_mask shape: [B, L] bidirectional_block_mask shape: [B, L, L]
  Examples: bidirectional_mask = [[0, 1, 1, 1, 0, 0]] bidirectional_block_mask =
  [[

      [False, False, False, False, False, False],
      [False,  True,  True,  True, False, False],
      [False,  True,  True,  True, False, False],
      [False,  True,  True,  True, False, False],
      [False, False, False, False, False, False],
      [False, False, False, False, False, False],
  ]]
  """
  q_block_indices = _make_block_mask_indices(bidirectional_mask)
  kv_block_indices = q_block_indices
  bidirectional_block_mask = (kv_block_indices[:, None, :] == q_block_indices[..., None]) & (
      q_block_indices[..., None] > 0
  )
  return bidirectional_block_mask


def attention_op_as_linen(
    *,
    config: Config,
    mesh: Mesh,
    attention_kernel: str,
    max_target_length: int,
    num_query_heads: int,
    num_kv_heads: int,
    float32_qk_product: bool = False,
    max_prefill_predict_length: int = -1,
    float32_logits: bool = False,
    flash_axis_names_q: AxisNames = (BATCH_ATTN, HEAD, LENGTH, D_KV),
    flash_axis_names_kv: AxisNames = (BATCH_ATTN, HEAD, KV_LENGTH, D_KV),
    flash_axis_names_splash_kernel: AxisNames = (HEAD, LENGTH),
    prefill_cache_logical_axis_names: AxisNames = (
        CACHE_BATCH_PREFILL,
        CACHE_SEQUENCE,
        CACHE_HEADS,
        CACHE_KV,
    ),
    cache_logical_axis_names: AxisNames = (
        CACHE_BATCH,
        CACHE_SEQUENCE,
        CACHE_HEADS,
        CACHE_KV,
    ),
    cache_scale_logical_axis_names: AxisNames = (
        CACHE_SCALE_BATCH,
        CACHE_SCALE_SEQUENCE,
        CACHE_SCALE_HEADS,
        CACHE_SCALE_KV,
    ),
    ragged_qkv_axis_names: AxisNames = (
        CACHE_BATCH,
        CACHE_HEADS,
        CACHE_SEQUENCE,
        CACHE_KV,
    ),
    ragged_lengths_names: AxisNames = (CACHE_BATCH,),
    compute_axis_order: AxisIdxes = (0, 1, 2, 3),
    key_axis_order: AxisIdxes = (2, 0, 1, 3),
    reshape_q: bool = False,
    dropout_rate: float = 0.0,
    dtype: DType = jnp.float32,
    quant: Optional[Quant] = None,
    kv_quant: Optional[KVQuant] = None,
    attention_type: AttentionType = AttentionType.GLOBAL,  # Default to global attention
    attn_logits_soft_cap: float | None = None,
    sliding_window_size: int | None = None,
    chunk_attn_window_size: int | None = None,
    use_ragged_attention: bool = False,
    ragged_block_size: int = 256,
):
  """A factory function to create an AttentionOp as a Linen module.

  This function serves as a bridge to use the NNX-based `AttentionOp` within a
  Linen model.
  """
  pass


class AttentionOp(nnx.Module):
  """Attention operation"""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      attention_kernel: str,
      max_target_length: int,
      num_query_heads: int,
      num_kv_heads: int,
      float32_qk_product: bool = False,
      max_prefill_predict_length: int = -1,
      float32_logits: bool = False,
      flash_axis_names_q: AxisNames = (BATCH_ATTN, HEAD, LENGTH, D_KV),
      flash_axis_names_kv: AxisNames = (BATCH_ATTN, HEAD, KV_LENGTH, D_KV),
      flash_axis_names_splash_kernel: AxisNames = (HEAD, LENGTH),
      prefill_cache_logical_axis_names: AxisNames = (
          CACHE_BATCH_PREFILL,
          CACHE_SEQUENCE,
          CACHE_HEADS,
          CACHE_KV,
      ),
      cache_logical_axis_names: AxisNames = (
          CACHE_BATCH,
          CACHE_SEQUENCE,
          CACHE_HEADS,
          CACHE_KV,
      ),
      cache_scale_logical_axis_names: AxisNames = (
          CACHE_SCALE_BATCH,
          CACHE_SCALE_SEQUENCE,
          CACHE_SCALE_HEADS,
          CACHE_SCALE_KV,
      ),
      ragged_qkv_axis_names: AxisNames = (
          CACHE_BATCH,
          CACHE_HEADS,
          CACHE_SEQUENCE,
          CACHE_KV,
      ),
      ragged_lengths_names: AxisNames = (CACHE_BATCH,),
      compute_axis_order: AxisIdxes = (0, 1, 2, 3),
      key_axis_order: AxisIdxes = (2, 0, 1, 3),
      reshape_q: bool = False,
      dropout_rate: float = 0.0,
      dtype: DType = jnp.float32,
      quant: Optional[Quant] = None,
      kv_quant: Optional[KVQuant] = None,
      attention_type: AttentionType = AttentionType.GLOBAL,  # Default to global attention
      attn_logits_soft_cap: float | None = None,
      sliding_window_size: int | None = None,
      chunk_attn_window_size: int | None = None,
      use_ragged_attention: bool = False,
      ragged_block_size: int = 256,
      rngs: nnx.Rngs | None = None,
  ):
    """Initializes the AttentionOp module.

    Args:
      config: The configuration for the model.
      mesh: The device mesh.
      attention_kernel: The attention kernel to use.
      max_target_length: The maximum target length.
      num_query_heads: The number of query heads.
      num_kv_heads: The number of key/value heads.
      float32_qk_product: Whether to compute qk_product in float32.
      max_prefill_predict_length: The maximum prefill predict length.
      float32_logits: Whether to compute logits in float32.
      flash_axis_names_kv: The logical axis names for the KV cache in flash
        attention.
      flash_axis_names_q: The logical axis names for the query in flash
        attention.
      flash_axis_names_splash_kernel: The logical axis names for the splash
        attention kernel.
      prefill_cache_logical_axis_names: The logical axis names for the prefill
        cache.
      cache_logical_axis_names: The logical axis names for the cache.
      cache_scale_logical_axis_names: The logical axis names for the cache
        scale.
      ragged_qkv_axis_names: The logical axis names for ragged QKV tensors.
      ragged_lengths_names: The logical axis names for ragged lengths.
      compute_axis_order: The order of axes for computation.
      key_axis_order: The order of axes for the key. ... and other configuration
        parameters.
      rngs: The random number generators for initialization, passed by the
        nnx.to_linen wrapper.
    """
    self.config = config
    self.mesh = mesh
    self.attention_kernel = attention_kernel
    self.max_target_length = max_target_length
    self.num_query_heads = num_query_heads
    self.num_kv_heads = num_kv_heads
    self.float32_qk_product = float32_qk_product
    self.max_prefill_predict_length = max_prefill_predict_length
    self.float32_logits = float32_logits
    self.flash_axis_names_q = flash_axis_names_q
    self.flash_axis_names_kv = flash_axis_names_kv
    self.flash_axis_names_splash_kernel = flash_axis_names_splash_kernel
    self.prefill_cache_logical_axis_names = prefill_cache_logical_axis_names
    self.cache_logical_axis_names = cache_logical_axis_names
    self.cache_scale_logical_axis_names = cache_scale_logical_axis_names
    self.ragged_qkv_axis_names = ragged_qkv_axis_names
    self.ragged_lengths_names = ragged_lengths_names
    self.compute_axis_order = compute_axis_order
    self.key_axis_order = key_axis_order
    self.reshape_q = reshape_q
    self.dropout_rate = dropout_rate
    self.dtype = dtype
    self.quant = quant
    self.kv_quant = kv_quant
    self.attention_type = attention_type
    self.attn_logits_soft_cap = attn_logits_soft_cap
    self.sliding_window_size = sliding_window_size
    self.chunk_attn_window_size = chunk_attn_window_size
    self.use_ragged_attention = use_ragged_attention
    self.ragged_block_size = ragged_block_size
    self.rngs = rngs

    def maybe_create_nnx(einsum, *args):
      if isinstance(einsum, nn.Module):
        return nnx_wrappers.ToNNX(einsum, rngs=rngs).lazy_init(*args)
      return einsum

    # qk_product
    if self.kv_quant:
      # Dummy inputs for lazy initialization
      b = 1
      t_prefill = self.max_prefill_predict_length
      t_ar = 1  # Autoregressive mode has a query length of 1
      n = self.num_query_heads
      n_kv = self.num_kv_heads
      d = self.config.head_dim
      g = n // n_kv
      s_prefill = self.max_prefill_predict_length
      s_ar = self.max_target_length

      # Dummy query/key/value shapes as before...
      dummy_query_prefill = jnp.zeros((b, t_prefill, n_kv, g, d), dtype=self.dtype)
      dummy_key_prefill = jnp.zeros((b, s_prefill, n_kv, d), dtype=self.dtype)
      dummy_query_ar = jnp.zeros((b, t_ar, n_kv, g, d), dtype=self.dtype)
      dummy_key_ar = jnp.zeros((b, s_ar, n_kv, d), dtype=self.dtype)

      dummy_attn_weights_prefill = jnp.zeros((b, n_kv, g, t_prefill, s_prefill), dtype=jnp.float32)
      dummy_value_prefill = jnp.zeros((b, s_prefill, n_kv, d), dtype=self.dtype)
      dummy_attn_weights_ar = jnp.zeros((b, n_kv, g, t_ar, s_ar), dtype=jnp.float32)
      dummy_value_ar = jnp.zeros((b, s_ar, n_kv, d), dtype=self.dtype)

      # Prefill AqtEinsum instances
      self.AqtEinsum_0 = maybe_create_nnx(
          self.kv_quant.einsum_fn_with_rhs_qtensor(),
          "btkgd,bskd->bkgts",
          dummy_query_prefill,
          dummy_key_prefill,
      )
      self.AqtEinsum_1 = maybe_create_nnx(
          self.kv_quant.einsum_fn_with_rhs_qtensor_and_dequant(),
          "bkgts,bskd->btkgd",
          dummy_attn_weights_prefill,
          dummy_value_prefill,
      )
      # Autoregressive AqtEinsum instances
      self.AqtEinsum_2 = maybe_create_nnx(
          self.kv_quant.einsum_fn_with_rhs_qtensor(),
          "btkgd,bskd->bkgts",
          dummy_query_ar,
          dummy_key_ar,
      )
      self.AqtEinsum_3 = maybe_create_nnx(
          self.kv_quant.einsum_fn_with_rhs_qtensor_and_dequant(),
          "bkgts,bskd->btkgd",
          dummy_attn_weights_ar,
          dummy_value_ar,
      )
    else:
      self.AqtEinsum_0 = jnp.einsum
      self.AqtEinsum_1 = jnp.einsum
      self.AqtEinsum_2 = jnp.einsum
      self.AqtEinsum_3 = jnp.einsum

  def _logical_to_mesh_axes(self, logical_name):
    logical_rules = None if self.config.using_pipeline_parallelism else self.config.logical_axis_rules
    return logical_to_mesh_axes(logical_name, mesh=self.mesh, rules=logical_rules)

  def check_attention_inputs(self, query: Array, key: Array | KVTensor, value: Array | KVTensor) -> None:
    """Check attention inputs."""
    pass

  def _maybe_shard_with_pspec(self, inputs, pspec: jax.sharding.PartitionSpec | None):
    return maybe_shard_with_pspec(
        inputs,
        pspec,
        mesh=self.mesh,
        shard_mode=self.config.shard_mode,
        debug_sharding=self.config.debug_sharding,
        extra_stack_level=1,
    )

  def generate_attention_mask(
      self,
      query,
      key,
      decoder_segment_ids: Array | None,
      model_mode: str,
      previous_chunk: Any = None,
      bidirectional_mask: Any = None,
  ) -> Array | None:
    """Generates a combined attention mask for Transformer models.

    This function constructs an attention mask by potentially combining
    several types of masks based on the input parameters and model
    configuration. The generated mask dictates which query-key pairs are
    allowed to attend to each other.

    The masking logic can enforce:
    1.  **Sequence Separation:** Using `decoder_segment_ids`, attention is
      confined within distinct sequences in a batch. This is crucial when
      multiple unrelated sequences are packed together.
    2.  **Causality:** Preventing attention to future positions. This is
      standard for autoregressive decoding. For chunked prefill, as
      described in the SARATHI paper [2], causality is adjusted based
      on `previous_chunk` information.
    3.  **Specialized Attention Patterns:** Depending on `self.attention_type`,
      it can apply:
      * Local Sliding Window Attention: Restricts attention to a
          fixed-size window around each query position.
      * Chunk Attention: Divides sequences into chunks and applies
          masking at the chunk level.
    4.  **Bidirectional Attention for Sub-sequences:** If `bidirectional_mask`
      is provided (e.g., for image tokens in a multimodal model),
      those parts of the sequence can attend bidirectionally, and this
      mask is OR-ed with other generated masks.

    The overall approach and specific masking techniques are influenced by
    efficient attention mechanisms like those found in the Pallas MHA
    Flash Attention reference [1].

    Args:
      query: The query tensor, typically of shape `[batch_size,
        q_sequence_length, num_heads, head_dim]`. Used primarily for deriving
        sequence length.
      key: The key tensor, typically of shape `[batch_size, kv_sequence_length,
        num_heads, head_dim]`. Used primarily for deriving sequence length.
      decoder_segment_ids: Optional `Array` of shape `[batch_size,
        q_sequence_length]`. Identifies distinct sequences within the batch.
        Attention is restricted to elements within the same segment ID. In
        autoregressive mode, specific values (e.g.,
        `common_types.DECODING_ACTIVE_SEQUENCE_INDICATOR`) can mark the
        currently active sequence for decoding.
      model_mode: A string (e.g., `common_types.MODEL_MODE_AUTOREGRESSIVE`,
        `MODEL_MODE_PREFILL`) indicating the operational mode. This
        significantly influences mask generation, particularly how causality and
        segment separation are handled.
      previous_chunk: Optional. Information about previously processed key/value
        chunks, often a tensor representing the previous keys/values. Used to
        correctly offset causal masks in chunked attention or streaming
        scenarios. Its shape might be `[batch_size, prev_kv_sequence_length,
        ...]`.
      bidirectional_mask: Optional `Array` of shape `[batch_size,
        kv_sequence_length]`. If provided, this boolean mask indicates tokens
        (e.g., image tokens) that are allowed to attend bidirectionally. The
        resulting block-wise bidirectional mask is combined with other masks
        using a logical OR.

    Returns:
      An `Array` representing the attention mask, with shape
       `[batch_size, 1, 1, q_sequence_length, kv_sequence_length]`.
      It is broadcastable to the shape
       `[batch_size, num_kv_heads, group_size=n_q // n_kv, q_sequence_length,
       kv_sequence_length]`.
      Positions with `0.0` allow attention, while positions with
       `DEFAULT_MASK_VALUE` (a large negative number) prevent it.
      Returns `None` if no masking is determined to be necessary based on
       the inputs and configuration.

    References:
      [1] JAX Pallas MHA Flash Attention:
          https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/flash_attention.py
      [2] SARATHI: Efficient LLM Inference by Piggybacking Decodes with
          Chunked Prefills - ArXiv:2308.16369 (https://arxiv.org/abs/2308.16369)
    """
    pass

  def calculate_moba_gate_logic(self, q_item, k_item, q_pos_item):
    """Computes the block-level MoBA gating intermediates for one batch item.

    Args:
      q_item: Query tensor shaped `[q_len, n_q_heads, head_dim]`.
      k_item: Key tensor shaped `[kv_len, n_kv_heads, head_dim]`.
      q_pos_item: Absolute query positions shaped `[q_len]`, used to derive the
        chunk index for each query. For example, during prefill after 128 tokens
        have been processed `q_pos_item` is `jnp.arange(128, 128 + q_len)`,
        while in autoregressive decode with a single query token it is
        `jnp.array([kv_len - 1])`.

    Returns:
      `need_attend`, a boolean mask of shape `[n_kv_heads, g, q_len, num_block]`
      indicating which key blocks each query should attend to. The additional
      values in the returned tuple are debug intermediates used for logging and
      diagnostics when inspecting the gating behaviour.
    """
    pass

  def generate_moba_mask_single_item(self, q_item, k_item, q_positions):
    """Generates the token-level MoBA additive mask for a single batch item."""
    pass

  def _generate_moba_mask(self, query: Array, key: Array, q_positions: Array) -> Array:
    """Builds the token-level MoBA additive mask for the whole batch.

    Args:
      query: Query tensor shaped `[batch, q_len, n_q_heads, head_dim]`.
      key: Key tensor shaped `[batch, kv_len, n_kv_heads, head_dim]`.
      q_positions: Absolute query positions shaped `[q_len]`, shared across the
        batch, identifying the starting offset of each query token. For example,
        in prefill after 128 tokens we pass `jnp.arange(128, 128 + q_len)`,
        while in autoregressive decode with a single new token the vector is
        `[kv_len - 1]` for each batch element.

    Returns:
      Additive attention mask with shape
      `[batch, n_kv_heads, n_q_heads // n_kv_heads, q_len, kv_len]` containing
      `0.` for permitted positions and `-inf` for masked ones.
    """
    pass

  def apply_attention(
      self,
      query: Array,
      key: Array | KVTensor,
      value: Array | KVTensor,
      decoder_segment_ids: Array | None,
      segment_positions: Array | None,
      lengths: Array | None,
      model_mode: str,
      use_ragged_attention: bool = False,
      previous_chunk: Any = None,
      bidirectional_mask: Any = None,
      sinks: Array | None = None,
      indexer_mask: Array | None = None,
      record_max_logits: bool = False,
      *,
      qk_product_einsum: Callable[..., Array],
      wv_product_einsum: Callable[..., Array],
  ):
    """Apply attention"""
    pass

  def gpu_ragged_attention(
      self,
      q: Array,
      k: Array | KVTensor,
      v: Array | KVTensor,
      lengths: Array,
      block_size: int,
  ):
    """gpu ragged attention"""
    pass

  def tpu_ragged_attention(
      self,
      query: Array,
      key: Array | KVTensor,
      value: Array | KVTensor,
      lengths: Array,
      block_size: int,
  ) -> tuple[Array, Array, Array]:
    """Ragged Attention."""
    pass

  def tpu_flash_attention(
      self,
      query: Array,
      key: Array,
      value: Array,
      decoder_segment_ids: Array | None,
      attn_logits_soft_cap: float | None = None,
      sinks: Array | None = None,
      indexer_mask: Array | None = None,
      record_max_logits: bool = False,
  ) -> tuple[Array, Array]:
    """TPU Flash Attention."""

    cp_size = self.config.context_parallel_size
    load_balanced_context_parallel = self.config.context_parallel_load_balance

    # Transpose to ('batch', 'heads', 'length', 'kv')
    query = jnp.transpose(query, axes=(0, 2, 1, 3))
    key = jnp.transpose(key, axes=(0, 2, 1, 3))
    value = jnp.transpose(value, axes=(0, 2, 1, 3))
    segment_axis_names_q = None
    segment_axis_names_kv = None
    sink_axis_names = self._logical_to_mesh_axes((HEAD,))
    if decoder_segment_ids is not None:
      segment_axis_names_q = self._logical_to_mesh_axes((BATCH_ATTN, Q_LENGTH))
      segment_axis_names_kv = self._logical_to_mesh_axes((BATCH_ATTN, KV_LENGTH))

    axis_names_splash_kernel = self._logical_to_mesh_axes(self.flash_axis_names_splash_kernel)
    axis_names_q = self._logical_to_mesh_axes(self.flash_axis_names_q)
    axis_names_kv = self._logical_to_mesh_axes(self.flash_axis_names_kv)
    indexer_mask_axis_names = self._logical_to_mesh_axes((BATCH_ATTN, Q_LENGTH, KV_LENGTH))

    global global_block_q, global_block_kv, global_block_kv_compute, global_block_q_dkv, global_block_kv_dkv
    global global_block_kv_dkv_compute, global_block_q_dq, global_block_kv_dq, global_use_fused_bwd_kernel
    global global_q_layout, global_k_layout, global_v_layout
    global_block_q = self.config.sa_block_q
    global_block_kv = self.config.sa_block_kv
    global_block_kv_compute = self.config.sa_block_kv_compute
    global_block_q_dkv = self.config.sa_block_q_dkv
    global_block_kv_dkv = self.config.sa_block_kv_dkv
    global_block_kv_dkv_compute = self.config.sa_block_kv_dkv_compute
    global_block_q_dq = self.config.sa_block_q_dq
    global_block_kv_dq = self.config.sa_block_kv_dq
    global_use_fused_bwd_kernel = self.config.sa_use_fused_bwd_kernel
    global_q_layout = self.config.sa_q_layout
    global_k_layout = self.config.sa_k_layout
    global_v_layout = self.config.sa_v_layout

    devices_in_data_fsdp = self.mesh.shape.get("data", 1) * self.mesh.shape.get("fsdp", 1)
    assert (query.shape[0] / devices_in_data_fsdp).is_integer(), (
        "Batch dimension should be shardable among the devices in data and fsdp"
        " axis"
        f" got {query.shape[0]=}/{devices_in_data_fsdp=}"
    )

    # create_splash_attention config
    def create_sa_config(config, query, key, attn_logits_soft_cap):
      if config.use_tokamax_splash:
        sa_config = tokamax_splash_kernel.SplashConfig(
            block_q=min(global_block_q, query.shape[2]),
            block_kv=min(global_block_kv, key.shape[2]),
            block_kv_compute=min(global_block_kv_compute, key.shape[2]),
            block_q_dkv=min(global_block_q_dkv, query.shape[2]),
            block_kv_dkv=min(global_block_kv_dkv, key.shape[2]),
            block_kv_dkv_compute=min(global_block_kv_dkv_compute, query.shape[2]),
            use_fused_bwd_kernel=True,  # tokamax only supports fused bwd kernel
            q_layout=tokamax_splash_kernel.QKVLayout[global_q_layout],
            k_layout=tokamax_splash_kernel.QKVLayout[global_k_layout],
            v_layout=tokamax_splash_kernel.QKVLayout[global_v_layout],
            attn_logits_soft_cap=attn_logits_soft_cap,
            residual_checkpoint_name="context",
            fwd_cost_estimate=pl.CostEstimate(
                flops=config.cost_estimate_flops_fwd,
                transcendentals=0,
                bytes_accessed=0,
            )
            if config.cost_estimate_flops_fwd >= 0
            else None,
            bwd_cost_estimate=pl.CostEstimate(
                flops=config.cost_estimate_flops_bwd,
                transcendentals=0,
                bytes_accessed=0,
            )
            if config.cost_estimate_flops_bwd >= 0
            else None,
            dq_reduction_steps=config.dq_reduction_steps if config.dq_reduction_steps > 0 else None,
            use_experimental_scheduler=config.use_splash_scheduler,
        )
      else:
        sa_config = splash_attention_kernel.BlockSizes(
            block_q=min(global_block_q, query.shape[2]),
            block_kv=min(global_block_kv, key.shape[2]),
            block_kv_compute=min(global_block_kv_compute, key.shape[2]),
            block_q_dkv=min(global_block_q_dkv, query.shape[2]),
            block_kv_dkv=min(global_block_kv_dkv, key.shape[2]),
            block_kv_dkv_compute=min(global_block_kv_dkv_compute, query.shape[2]),
            block_q_dq=None if global_use_fused_bwd_kernel else min(global_block_q_dq, query.shape[2]),
            block_kv_dq=None if global_use_fused_bwd_kernel else min(global_block_kv_dq, query.shape[2]),
            use_fused_bwd_kernel=global_use_fused_bwd_kernel,
            q_layout=splash_attention_kernel.QKVLayout[global_q_layout],
            k_layout=splash_attention_kernel.QKVLayout[global_k_layout],
            v_layout=splash_attention_kernel.QKVLayout[global_v_layout],
        )
      return sa_config

    sa_config = create_sa_config(self.config, query, key, attn_logits_soft_cap)
    mask_shape = (query.shape[2], key.shape[2])  # (q_seq_len, kv_seq_len)
    mask_module = tokamax_splash_mask if self.config.use_tokamax_splash else splash_attention_mask
    if self.attention_type == AttentionType.FULL:
      mask = mask_module.FullMask(mask_shape)
    else:
      mask = mask_module.CausalMask(shape=mask_shape)

    # Create LoadBalancedCausalMask if cp and load_balancing
    if cp_size > 1 and load_balanced_context_parallel:
      mask = LoadBalancedCausalMask(shape=mask_shape, cp_size=cp_size)

    # TODO: figure out local_sliding attention + load_balancing, default is global
    # Apply local masking if local sliding attention is enabled.
    if self.attention_type == AttentionType.LOCAL_SLIDING:
      if self.sliding_window_size is None:
        raise ValueError("Sliding_window_size must be set if Local Sliding attention type")
      mask &= mask_module.LocalMask(
          shape=(query.shape[2], key.shape[2]),
          window_size=(self.sliding_window_size, self.sliding_window_size),
          offset=0,
      )
    elif self.attention_type == AttentionType.CHUNK:
      if self.chunk_attn_window_size is None:
        raise ValueError("chunk_attn_window_size must be set for chunk attention type")

      mask &= ChunkedCausalMask(
          shape=(query.shape[2], key.shape[2]),
          chunk_size=self.chunk_attn_window_size,
      )

    max_logit_value = None
    if self.config.use_tokamax_splash:
      # Create mask
      single_head_mask = mask  # tokamax now just uses a single mask and assumes broadcast to all heads
      if self.config.use_max_logit_estimate > 0:
        sa_config = dataclasses.replace(sa_config, max_logit_const=self.config.use_max_logit_estimate)

      # Create the splash attention kernel object separately, jit it for performance
      @partial(
          jax.jit,
          static_argnames=[
              "single_head_mask",
          ],
      )
      def wrap_splash_kernel(single_head_mask):
        splash_kernel = tokamax_splash_kernel.make_splash_mha(
            mask=single_head_mask,
            config=sa_config,
            q_seq_shards=cp_size,  # axis for sequence sharding,
        )
        return splash_kernel

      splash_kernel = wrap_splash_kernel(single_head_mask)
      segment_axis_names_splash_kernel = self._logical_to_mesh_axes((Q_LENGTH,))
      splash_kernel = self._maybe_shard_with_pspec(splash_kernel, segment_axis_names_splash_kernel)
    elif self.config.use_jax_splash:
      if self.config.use_max_logit_estimate > 0:
        sa_config = dataclasses.replace(sa_config, max_logit_const=self.config.use_max_logit_estimate)
      segment_axis_names_splash_kernel = nn.logical_to_mesh_axes((Q_LENGTH,))
    else:
      # Create multi-head mask
      multi_head_mask = splash_attention_mask.MultiHeadMask(masks=(mask,) * query.shape[1])

      # Create the splash attention kernel object separately, jit it for performance
      @partial(
          jax.jit,
          static_argnames=[
              "multi_head_mask",
              "shard_head_size",
          ],
      )
      def wrap_splash_kernel(multi_head_mask, shard_head_size=1):
        splash_kernel = splash_attention_kernel.make_splash_mha(
            mask=multi_head_mask,
            head_shards=shard_head_size,  # the size of the axis if sharding over heads
            q_seq_shards=cp_size,  # axis for sequence sharding
            block_sizes=sa_config,
            attn_logits_soft_cap=attn_logits_soft_cap,
            residual_checkpoint_name="context",
        )
        return splash_kernel

      head_physical_axes = logical_to_mesh_axes((HEAD,), self.mesh)[0]
      head_physical_axes = (head_physical_axes,) if isinstance(head_physical_axes, str) else (head_physical_axes or ())
      shard_head_size = math.prod(self.mesh.shape.get(ax, 1) for ax in head_physical_axes)
      splash_kernel = wrap_splash_kernel(multi_head_mask, shard_head_size)
      named_sharding = jax.sharding.NamedSharding(self.mesh, axis_names_splash_kernel)
      segment_axis_names_splash_kernel = splash_kernel.manual_sharding_spec(named_sharding)
      splash_kernel = jax.tree.map(
          lambda arr, spec: None if arr is None else self._maybe_shard_with_pspec(arr, spec),
          splash_kernel,
          segment_axis_names_splash_kernel,
          is_leaf=lambda x: x is None,
      )

    # Now call the function wrap_flash_attention which does the actual computation.
    # The splash kernel is passed as a parameter to the function. Since we have the shard map
    # decorating the wrap_flash_attention function, the data will be correctly sharded
    # meaning q will be sharded over sequence aka context length but K and V will be duplicated
    # The shardings are specified in the in_specs and out_specs of the shard_map decorator:
    # 'segment_axis_names_q' maps to ['activation_q_length', ['context']] meaning that q is sharded over the context axis
    #  'segment_axis_names_kv' maps to ['activation_kv_length', []] meaning that K and V are not sharded
    # splash_kernel is sharded over (HEAD, LENGTH)

    if record_max_logits:
      # max_logits will share similar sharding as query but last dim is unrelated to model
      # Using None for the last dimension of max_logits sharding
      if isinstance(axis_names_q, jax.sharding.PartitionSpec):
        # max_logits is Rank 3 (batch, heads, seq), so drop the last dimension (d_kv)
        max_logits_spec = jax.sharding.PartitionSpec(*axis_names_q[:-1])
      else:
        # Fallback if axis_names_q is not a PartitionSpec (unlikely in this context)
        max_logits_spec = axis_names_q[:-1]
      out_specs = (axis_names_q, max_logits_spec)
    else:
      # Safely pad the specs with None to match the (attention_output, None) return type
      out_specs = (axis_names_q, None)

    @functools.partial(
        jax.shard_map,
        mesh=self.mesh,
        in_specs=(
            axis_names_q,
            axis_names_kv,
            axis_names_kv,
            segment_axis_names_q,
            segment_axis_names_kv,
            None,  # no sharding for config
            segment_axis_names_splash_kernel,
            None,  # no sharding for cp_size
            None,  # no sharding for load_balanced_context_parallel
            sink_axis_names,  # sharding align with query heads
            indexer_mask_axis_names,
        ),
        out_specs=out_specs,
        check_vma=False,
    )
    def wrap_flash_attention(
        query,
        key,
        value,
        decoder_segment_ids_q,
        decoder_segment_ids_kv,
        sa_config,
        splash_kernel,
        cp_size,
        load_balanced_context_parallel,
        sinks,
        indexer_mask,
    ):
      # If load_balanced_context_parallel is enabled, reorder the key and value tensors
      # to ensure that they are contiguous in memory.
      # This is necessary for the splash attention kernel to work correctly because it expects
      # the K and V to be contiguous. Note that K and V are not sharded over the sequence aka context axis
      # This was we get the unsharded unpermuted key and value tensors
      if cp_size > 1 and load_balanced_context_parallel:
        key = max_utils.reorder_sequence(tensor=key, cp_size=cp_size, seq_dim=2, to_contiguous=True)
        value = max_utils.reorder_sequence(tensor=value, cp_size=cp_size, seq_dim=2, to_contiguous=True)
        decoder_segment_ids_unpermuted = max_utils.reorder_sequence(
            tensor=decoder_segment_ids_kv,
            cp_size=cp_size,
            seq_dim=1,
            to_contiguous=True,
        )

      if decoder_segment_ids_q is not None:
        if cp_size > 1 and load_balanced_context_parallel:
          decoder_segment_ids_tuple = splash_attention_kernel.SegmentIds(
              decoder_segment_ids_q, decoder_segment_ids_unpermuted
          )
        else:
          # if cp=1, decoder_segment_ids_q is the same as decoder_segment_ids_kv
          decoder_segment_ids_tuple = splash_attention_kernel.SegmentIds(decoder_segment_ids_q, decoder_segment_ids_kv)
      else:
        decoder_segment_ids_tuple = None

      if self.config.use_tokamax_splash:
        if self.config.use_indexer and indexer_mask is not None:
          # Construct the splash kernel call with dynamic mask

          # Iterate over batch dimension for (query, key, value, segment, sinks, mask)
          attn_fn = jax.vmap(dynamic_mask_splash_kernel, (0, 0, 0, 0, None, 0))
          indexer_mask = jnp.isclose(indexer_mask, 0.0)

          if record_max_logits:
            attention_output, max_logits = attn_fn(query, key, value, decoder_segment_ids_tuple, sinks, indexer_mask)
            return attention_output, max_logits
          else:
            attention_output, _ = attn_fn(query, key, value, decoder_segment_ids_tuple, sinks, indexer_mask)
            return attention_output, None
        else:
          kernel = partial(splash_kernel, max_logit_value=max_logit_value)

          if record_max_logits:


            attention_output, max_logits = jax.vmap(kernel_fn, in_axes=(0, 0, 0, 0, None))(
                query, key, value, decoder_segment_ids_tuple, sinks
            )
            return attention_output, max_logits
          else:
            attention_output = jax.vmap(lambda q, k, v, d, s: kernel(q, k, v, d, sinks=s), in_axes=(0, 0, 0, 0, None))(
                query, key, value, decoder_segment_ids_tuple, sinks
            )
            return attention_output, None
      elif self.config.use_jax_splash:
        materialized_mask = jnp.asarray(mask[:, :])
        attention_output = jax_flash_attention.flash_attention_block_masked(
            query,
            key,
            value,
            decoder_segment_ids_tuple,
            block_kv=self.config.sa_block_kv,
            block_q=self.config.sa_block_q,
            mask=materialized_mask,
            mask_value=DEFAULT_MASK_VALUE,
        )
        if record_max_logits:
          # The native JAX splash attention implementation does not currently expose the softmax statistics
          # (e.g., max_logits) required for QK-Clip. Use tokamax splash attention if max logit recording is needed.
          raise NotImplementedError("record_max_logits not supported for jax_splash")
      else:
        attention_output = jax.vmap(splash_kernel, in_axes=(0, 0, 0, 0, None))(
            query, key, value, decoder_segment_ids_tuple, sinks
        )
        if record_max_logits:
          raise NotImplementedError("record_max_logits not supported for legacy splash")

      return attention_output, None

    query = self._maybe_shard_with_pspec(query, axis_names_q)
    key = self._maybe_shard_with_pspec(key, axis_names_kv)
    value = self._maybe_shard_with_pspec(value, axis_names_kv)
    decoder_segment_ids_q = self._maybe_shard_with_pspec(decoder_segment_ids, segment_axis_names_q)
    decoder_segment_ids_kv = self._maybe_shard_with_pspec(decoder_segment_ids, segment_axis_names_kv)
    sinks = self._maybe_shard_with_pspec(sinks, sink_axis_names)
    indexer_mask = self._maybe_shard_with_pspec(indexer_mask, indexer_mask_axis_names)

    ret = wrap_flash_attention(
        query,
        key,
        value,
        decoder_segment_ids_q,
        decoder_segment_ids_kv,
        sa_config,
        None if self.config.use_jax_splash else splash_kernel,
        cp_size,
        load_balanced_context_parallel,
        sinks,
        indexer_mask,
    )

    x, max_logits = ret
    x = jnp.transpose(x, axes=(0, 2, 1, 3))

    if record_max_logits:
      # Max over sequence length (dim 2 of max_logits)
      # max_logits from kernel is (batch, heads, q_len)
      # output needs to be (batch, heads)
      # Note: q_len is sharded. We first reduce locally.
      max_logits_local = jnp.max(max_logits, axis=2)

      return x, max_logits_local

    return x, None

  def cudnn_flash_attention(
      self,
      query: Array,
      key: Array,
      value: Array,
      decoder_segment_ids: Array | None,
      segment_positions: Array | None,
      model_mode: str = MODEL_MODE_TRAIN,
  ) -> Array:
    """CUDNN Flash Attention with Transformer Engine.
    1. Stable API, supports MHA, GQA, SWA, Packing and Context Parallelism
    2. Context Parallelism currently only supports causal masking
    3. Only Ring attention has packing support with striped load balancing
      (context_parallel_strategy="ring" and context_parallel_load_balance=true)
    4. Breaks with TE 2.12 and 2.13 (known bug); works with TE stable release <=2.11 or >=2.14.
    """
    pass

  def cudnn_jax_flash_attention(
      self,
      query: Array,
      key: Array,
      value: Array,
      decoder_segment_ids: Array | None,
      model_mode: str = MODEL_MODE_TRAIN,
  ) -> tuple[Array, Array]:
    """CUDNN Flash Attention with JAX SDPA API."""
    pass

  def compute_local_attention(
      self,
      attn_weights: Array,
      value: Array | KVTensor,
      q_seq_len: int,
      model_mode: str,
      wv_product_einsum: Callable[..., Array],
      sinks: Array | None = None,
  ) -> tuple[Array, Array, Array]:
    """Computes the attention of a local subset of the kv cache.

    Local attention results will need to be combined with any other local
    attentions and normalized Based on
    https://github.com/google-research/google-research/blob/master/scaling_transformer_inference_efficiency/attention.py

    Args:
        attn_weights (Array): Product of query and key
        value (Array): Current value
        aqt_rng (PRNGKey | None): Optional rng

    Returns:
        (local_out, local_max,): where
          local_out is local unnormalized output
          local_max is the local max of exponentials
          local_sum is the sum of exponentials for this chunk, divided by
          exp(local_max).
    """
    pass


  def apply_attention_dot(
      self,
      query: Array,
      key: Array | KVTensor,
      value: Array | KVTensor,
      decoder_segment_ids: Array | None,
      model_mode: str = MODEL_MODE_TRAIN,
      previous_chunk: Any = None,
      bidirectional_mask: Any = None,
      sinks: Array | None = None,
      indexer_mask: Array | None = None,
      record_max_logits: bool = False,
      *,
      qk_product_einsum: Callable[..., Array],
      wv_product_einsum: Callable[..., Array],
  ):
    """Apply Attention."""
    pass

  def qk_product(
      self,
      query: Array,
      key: Array | KVTensor,
      q_seq_len: int,
      model_mode: str,
      einsum: Callable[..., Array],
  ) -> Array:
    """Query-Key product.

    Args:
      query: Query projection, in shape of [b, t, n, d]
      key: Key projection in shape of [b, s, n_kv, d]

    Returns:
      results in shape [b, n_kv, n // n_kv, t, s].

    Annotations:
      b: batch size
      t: query length
      s: key / value length
      d: head / kv dimension
      n: number of query heads
      n_kv: number of kv heads, sometimes annotated as k
      n // n_kv: number of group for query, sometimes annotated with g
    """
    pass

  def wv_product(
      self,
      attn_weights: Array,
      value: Array | KVTensor,
      model_mode: str,
      einsum: Callable[..., Array],
  ) -> Array:
    """weighted value product.

    Args:
      attn_weights: Computed results of qk_einsum, in shape [b, n_kv, n // n_kv,
        t, s]
      value: Value projection, in shape of [b, s, n_kv, d]

    Returns:
      result in shape [b, t, n, d]

    Annotations:
      b: batch size
      t: query length
      s: key / value length
      d: head / kv dimension
      n: number of query heads
      n_kv: number of kv heads, sometimes annotated as k
      n // n_kv: number of group for query, sometimes annotated with g
    """
    pass


  def normalize_cudnn_attention(self, local_outs, local_stats):
    """Normalize across two cuDNN attentions

    Args:
        local_outs (list): List of outputs entries for each cudnn attention in
          shape [b, t, n, d].
        local_stats (list): List of logsumexp entries for each cudnn attention
          in shape [b, n, t].

    Returns:
        Array: Combined attention that has been normalized in shape [b, t, n,
        d].
    """
    pass

  def normalize_attention(self, local_outs, local_maxes, local_sums):
    """Normalize across multiple localized attentions

    Args:
        local_outs (list): List of unnormalized outputs entries for each local
          attention
        local_maxes (list): List of max exponentials entries for each local
          attention
        local_sums (list): List of exponential sum entries for each local
          attention

    Returns:
        Array: Combined attention that has been normalized
    """
    pass

  def __call__(
      self,
      query,
      key,
      value,
      decoder_segment_ids,
      inputs_positions,
      model_mode,
      cached_values=None,
      previous_chunk=None,
      bidirectional_mask=None,
      sinks=None,
      indexer_mask: Optional[Array] = None,
      slot: Optional[int] = None,
      page_state: Optional[page_manager.PageState] = None,
      record_max_logits: bool = False,
  ):
    if cached_values is None:
      prefill_kv_cache, ar_kv_cache = None, None
    else:
      prefill_kv_cache, ar_kv_cache = cached_values[0], cached_values[1]
    if model_mode != MODEL_MODE_TRAIN:
      assert prefill_kv_cache
      key, value, decoder_segment_ids = prefill_kv_cache

    indexer_mask_prefill = None
    indexer_mask_ar = None
    if indexer_mask is not None:
      prefill_len = key.shape[1]
      indexer_mask_prefill = indexer_mask[:, :, :prefill_len]
      if ar_kv_cache is not None:
        indexer_mask_ar = indexer_mask[:, :, prefill_len:]

    prefill_unnormalized_output, prefill_exponentials_max, prefill_exponentials_sum = self.apply_attention(
        query=query,
        key=key,
        value=value,
        decoder_segment_ids=decoder_segment_ids,
        segment_positions=inputs_positions,
        lengths=None,
        model_mode=model_mode,
        use_ragged_attention=self.use_ragged_attention,
        previous_chunk=previous_chunk,
        bidirectional_mask=bidirectional_mask,
        sinks=sinks,
        indexer_mask=indexer_mask_prefill,
        record_max_logits=record_max_logits,
        qk_product_einsum=self.AqtEinsum_0,
        wv_product_einsum=self.AqtEinsum_1,
    )

    # Return the "prefill" cache if it actually the combined prefill+ar kv cache
    if ar_kv_cache is None:
      if prefill_exponentials_sum is not None:
        return prefill_unnormalized_output / prefill_exponentials_sum
      return prefill_unnormalized_output

    key, value, decoder_segment_ids, lengths = ar_kv_cache

    ar_unnormalized_output, ar_exponentials_max, ar_exponentials_sum = self.apply_attention(
        query=query,
        key=key,
        value=value,
        decoder_segment_ids=decoder_segment_ids,
        segment_positions=inputs_positions,
        lengths=lengths,
        model_mode=model_mode,
        use_ragged_attention=self.use_ragged_attention,
        bidirectional_mask=bidirectional_mask,
        indexer_mask=indexer_mask_ar,
        qk_product_einsum=self.AqtEinsum_2,
        wv_product_einsum=self.AqtEinsum_3,
    )

    if ar_unnormalized_output is not None:
      unnormalized_outputs = [
          prefill_unnormalized_output,
          ar_unnormalized_output,
      ]
      exponentials_maxes = [prefill_exponentials_max, ar_exponentials_max]
      exponentials_sums = [prefill_exponentials_sum, ar_exponentials_sum]
      if prefill_exponentials_max is not None and prefill_exponentials_sum is None:
        prefill_stat = prefill_exponentials_max
        ar_stat = ar_exponentials_max
        stats = [prefill_stat, ar_stat]
        return self.normalize_cudnn_attention(unnormalized_outputs, stats)
      else:
        return self.normalize_attention(unnormalized_outputs, exponentials_maxes, exponentials_sums)
    else:
      return prefill_unnormalized_output / prefill_exponentials_sum


# pylint: disable=protected-access
class LoadBalancedCausalMask(splash_attention_mask._ComputableMask):
  """Lazy causal mask, prevents the model from attending to future tokens.

  Attributes:
    offset: Offset of q start wrt kv. A positive offset shifts the bottom
      triangle upward, a negative one shifts it downward. A negative offset
      makes the first 'offset' rows of the attention matrix all 0s which leads
      to undefined softmax.
  """

  offset: int
  shape: tuple[int, int]
  cp_size: int

  def __init__(
      self,
      shape: tuple[int, int],
      offset: int = 0,
      shard_count: int = 1,
      cp_size: int = 4,
  ):
    self.offset = offset


    arr = np.arange(shape[0])
    # we reorder the mask to be load balanced following the same approach as
    # used to reorder the input tokens
    out = max_utils.reorder_mask_load_balancing(arr[None, :, None, None], cp_size, 1)
    q_sequence = out[0, :, 0, 0]

    mask_function = causal_mask_function

    super().__init__(
        shape=shape,
        mask_function=mask_function,
        shard_count=shard_count,
    )
    self.q_sequence = q_sequence

  def __eq__(self, other: object):
    if not isinstance(other, type(self)):
      return NotImplemented

    return self.shape == other.shape and self.offset == other.offset and np.array_equal(self.q_sequence, other.q_sequence)

  def __hash__(self):
    return hash(
        (
            type(self),
            self.shape,
            self.offset,
            self.q_sequence.tobytes() if self.q_sequence is not None else None,
        )
    )
