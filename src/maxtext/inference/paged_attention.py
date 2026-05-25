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

"""Paged Attention Op

WARNING: THIS FILE IS A WORK IN PROGRESS.
"""

import functools

from flax import linen as nn
from flax import nnx
import jax
from jax.experimental.pallas.ops.tpu.paged_attention import paged_attention_kernel
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from maxtext.common.common_types import Array, AxisNames, BATCH, DType, D_KV, HEAD, LENGTH, MODEL_MODE_AUTOREGRESSIVE, MODEL_MODE_PREFILL
from maxtext.inference import page_manager
from maxtext.inference import paged_attention_kernel_v2
from maxtext.layers.initializers import variable_to_logically_partitioned
from maxtext.utils.sharding import logical_to_mesh_axes

_use_kernel_v2 = False


def paged_attention_op_as_linen(
    *,
    mesh: Mesh,
    num_pages: int,
    tokens_per_page: int,
    max_pages_per_slot: int,
    max_pages_per_prefill: int,
    pages_per_compute_block: int,
    num_kv_heads: int,
    kv_head_dim_size: int,
    dtype: DType = jnp.float32,
    attn_logits_soft_cap: float | None = None,
    query_axis_names: AxisNames = (BATCH, LENGTH, HEAD, D_KV),
    kv_pages_axis_names: AxisNames = (
        "paged_kv_heads",
        "num_pages",
        "tokens_per_page",
        "paged_kv_head_dim_size",
    ),
):
  """A factory function to create a PagedAttentionOp as a Linen module.

  This function serves as a bridge to use the NNX-based `PagedAttentionOp`
  within a Linen model. It wraps the `PagedAttentionOp` module using
  `nnx.bridge.to_linen`, making it compatible with the Linen API. This is
  useful for gradual migration of a codebase from Linen to NNX.

  Args:
    mesh: The device mesh for sharding.
    num_pages: The total number of pages in the KV cache.
    tokens_per_page: The number of tokens each page can hold.
    max_pages_per_slot: The maximum number of pages a single sequence can use.
    max_pages_per_prefill: The maximum number of pages for a prefill sequence.
    pages_per_compute_block: The number of pages processed in one kernel block.
    num_kv_heads: The number of key/value heads.
    kv_head_dim_size: The dimension of each key/value head.
    dtype: The data type for computations.
    attn_logits_soft_cap: The soft cap for attention logits.
    query_axis_names: The logical axis names for the query tensor.
    kv_pages_axis_names: The logical axis names for the KV cache pages.

  Returns:
    A Linen module that wraps the NNX `PagedAttentionOp` module.
  """
  pass


class PagedAttentionOp(nnx.Module):
  """An NNX module for paged attention.

  This module implements the paged attention mechanism, which is an efficient
  method for handling attention in autoregressive models with long sequences.
  It divides the KV cache into fixed-size "pages" to manage memory dynamically.
  """

  def __init__(
      self,
      mesh: Mesh,
      num_pages: int,
      tokens_per_page: int,
      max_pages_per_slot: int,
      max_pages_per_prefill: int,
      pages_per_compute_block: int,
      num_kv_heads: int,
      kv_head_dim_size: int,
      dtype: DType = jnp.float32,
      attn_logits_soft_cap: float | None = None,
      query_axis_names: AxisNames = (BATCH, LENGTH, HEAD, D_KV),
      kv_pages_axis_names: AxisNames = (
          "paged_kv_heads",
          "num_pages",
          "tokens_per_page",
          "paged_kv_head_dim_size",
      ),
      *,
      # Not used in Embed but passed in by nnx.bridge.to_linen.
      # TODO: Remove when bridge no longer needed
      rngs: nnx.Rngs,
  ):
    """Initializes the PagedAttentionOp module.

    Args:
      mesh: The device mesh for sharding.
      num_pages: The total number of pages in the KV cache.
      tokens_per_page: The number of tokens each page can hold.
      max_pages_per_slot: The maximum number of pages a single sequence can use.
      max_pages_per_prefill: The maximum number of pages for a prefill sequence.
      pages_per_compute_block: The number of pages processed in one kernel block.
      num_kv_heads: The number of key/value heads.
      kv_head_dim_size: The dimension of each key/value head.
      dtype: The data type for computations.
      attn_logits_soft_cap: The soft cap for attention logits.
      query_axis_names: The logical axis names for the query tensor.
      kv_pages_axis_names: The logical axis names for the KV cache pages.
      rngs: The random number generators for initialization (required by NNX).
    """

    self.mesh = mesh
    self.num_pages = num_pages
    self.tokens_per_page = tokens_per_page
    self.max_pages_per_slot = max_pages_per_slot
    self.max_pages_per_prefill = max_pages_per_prefill
    self.pages_per_compute_block = pages_per_compute_block
    self.num_kv_heads = num_kv_heads
    self.kv_head_dim_size = kv_head_dim_size
    self.dtype = dtype
    self.attn_logits_soft_cap = attn_logits_soft_cap
    self.query_axis_names = query_axis_names
    self.kv_pages_axis_names = kv_pages_axis_names

    self.kv_pages_shape = (
        self.num_kv_heads,
        self.num_pages,
        self.tokens_per_page,
        self.kv_head_dim_size,
    )

    self.key_pages = nnx.Cache(
        jnp.zeros(self.kv_pages_shape, dtype=self.dtype),
        out_sharding=self.kv_pages_axis_names,
    )
    self.value_pages = nnx.Cache(
        jnp.zeros(self.kv_pages_shape, dtype=self.dtype),
        out_sharding=self.kv_pages_axis_names,
    )

  def _maybe_materialize_cache(self, cache: nnx.Cache) -> nnx.Cache:
    """Materializes the cache if it's currently a ShapeDtypeStruct."""
    pass

  def get_kv_pages(self):
    """Retrieves the key and value page caches.

    This method ensures the KV cache pages are materialized (if they are abstract
    ShapeDtypeStructs, a temporary state during Linen bridge initialization) and
    applies the necessary sharding constraints.

    Returns:
      A tuple containing the key pages and value pages caches (`nnx.Cache`).
    """
    pass

  def pad_qkv(self, *qkv):
    """Pad input to kv_head_dim_size"""
    pass

  def paged_dot_product_attention_with_max_and_sum(self, query, key, value):
    """paged dot product attention with max & sum"""
    pass

  # TODO(rupliu): add sharding when SPMD is fully supported
  def paged_attention_v2_prefill(
      self,
      query: Array,
      key_pages_cache: nnx.Cache,
      value_pages_cache: nnx.Cache,
      page_state: page_manager.PageState,
  ) -> Array:
    """Apply ragged input Paged Attention in prefill only. The assumption
    is the batch_size is only 1
    """
    pass

  # TODO(rupliu): add sharding when SPMD is fully supported
  def paged_attention_v2_decode(
      self,
      query: Array,
      key_pages_cache: nnx.Cache,
      value_pages_cache: nnx.Cache,
      page_state: page_manager.PageState,
  ) -> Array:
    """Apply ragged input Paged Attention in decode only."""
    pass

  # v1 kernel has around 20% performance gain than v2 kernel in decode only task
  def paged_attention_v1_decode(
      self,
      query: Array,
      key_pages_cache: nnx.Cache,
      value_pages_cache: nnx.Cache,
      page_state: page_manager.PageState,
  ) -> Array:
    """Apply Paged Attention v1 in decode only."""
    pass

  def __call__(
      self,
      query: Array,
      key: Array,
      value: Array,
      decoder_segment_ids: Array,
      model_mode: str,
      previous_chunk=None,
      slot: None | int = None,
      page_state: None | page_manager.PageState = None,
  ):
    """Applies the paged attention mechanism.

    This is the main entry point for the module. It takes query, key, and value
    tensors and performs paged attention based on the current model mode
    (prefill or autoregressive).

    Args:
      query: The query tensor.
      key: The key tensor for the current step.
      value: The value tensor for the current step.
      decoder_segment_ids: Segment IDs for the decoder, used for masking.
      model_mode: The current operational mode, either 'prefill' or
        'autoregressive'.
      previous_chunk: Information about previously processed chunks, used for
        chunked prefill.
      slot: The batch slot index for the current request.
      page_state: The current state of the page manager.

    Returns:
        A tuple (output, exponentials_max, exponentials_sum) containing:
        - The attention output tensor.
        - The max of the exponentials (for prefill mode with dot-product attention).
        - The sum of the exponentials (for prefill mode with dot-product attention).
        The latter two are None for autoregressive mode, as this is handled
        internally by the paged attention kernel.
    """

    key_pages_cache, value_pages_cache = self.get_kv_pages()
    query, key, value = self.pad_qkv(query, key, value)

    # update kv pages and call page attention kernel
    if model_mode == MODEL_MODE_PREFILL:
      self.update_prefill_step_pages(key_pages_cache, value_pages_cache, key, value, slot, page_state)
      if _use_kernel_v2:
        return (
            self.paged_attention_v2_prefill(query, key_pages_cache, value_pages_cache, page_state),
            None,
            None,
        )
      return self.paged_dot_product_attention_with_max_and_sum(query, key, value)
    elif model_mode == MODEL_MODE_AUTOREGRESSIVE and page_state is not None:
      self.update_decode_step_pages(key_pages_cache, value_pages_cache, key, value, page_state)
      if _use_kernel_v2:
        return (
            self.paged_attention_v2_decode(query, key_pages_cache, value_pages_cache, page_state),
            None,
            None,
        )
      return (
          self.paged_attention_v1_decode(query, key_pages_cache, value_pages_cache, page_state),
          None,
          None,
      )
    else:
      raise NotImplementedError(model_mode)

  def update_prefill_step_pages(
      self,
      key_pages_cache: nnx.Cache,  # [num_kv_heads, num_pages, tokens_per_page, head_dim]
      value_pages_cache: nnx.Cache,
      key: Array,
      value: Array,
      slot: int,
      page_state: page_manager.PageState,
  ) -> None:
    """Update pages for prefill step."""
    pass

  def update_decode_step_pages(self, key_pages_cache, value_pages_cache, key, value, page_state):
    """Update decode-step pages"""
    pass
