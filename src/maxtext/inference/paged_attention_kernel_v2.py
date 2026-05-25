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

# TODO(rupliu): This is a copied version of
# https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/ragged_paged_attention.py
# Clean up this file when Maxtext migrate to latext Jax version

"""TPU-Friendly Ragged Paged Attention kernel.

This kernel offers a highly optimized implementation of ragged paged attention,
specifically designed for TPU and compatible with a wide range of model
specifications. It supports mixed prefill and decoding, enhancing throughput
during inference.
"""

import functools

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


class MultiPageAsyncCopyDescriptor:
  """Descriptor for async copy of multiple K/V pages from HBM."""

  def __init__(
      self,
      pages_hbm_ref,  # [total_num_pages, page_size, num_kv_heads_per_blk, head_dim]
      vmem_buf,  # [num_kv_pages_per_blk, page_size, num_kv_heads_per_blk, head_dim]
      sem,
      page_indices_ref,  # i32[max_num_seqs, pages_per_seq]
      offset,  # [seq_idx, kv_pages_start]
  ):
    self._vmem_buf = vmem_buf
    seq_id, kv_pages_start = offset
    self._async_copies = [
        pltpu.make_async_copy(
            pages_hbm_ref.at[page_indices_ref[seq_id, kv_pages_start + i]],
            vmem_buf.at[i],
            sem,
        )
        for i in range(vmem_buf.shape[0])
    ]

  def start(self):
    """Starts the async copies."""
    for async_copy in self._async_copies:
      async_copy.start()



def ref_ragged_paged_attention(
    queries: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    k_pages: jax.Array,  # [total_num_pages, page_size, num_kv_heads, head_dim]
    v_pages: jax.Array,  # [total_num_pages, page_size, num_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1],
    *,
    sm_scale: float = 1.0,
    mask_value: float = DEFAULT_MASK_VALUE,
):
  """Ref ragged paged attention."""
  pass


# Expect to run these checks during runtime.
def validate_inputs_on_runtime(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    k_pages: jax.Array,  # [total_num_pages, page_size, num_kv_heads, head_dim]
    v_pages: jax.Array,  # [total_num_pages, page_size, num_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs,  # i32[1]
):
  """validate inputs on runtime"""
  pass


# Expect to run these checks during compile time.
def check_inputs_shapes(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    k_pages: jax.Array,  # [total_num_pages, page_size, num_kv_heads, head_dim]
    v_pages: jax.Array,  # [total_num_pages, page_size, num_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs,  # i32[1]
):
  """check shapes of inputs"""
  pass


def ragged_paged_attention_kernel(
    # Prefetch
    kv_lens_ref,  # [max_num_seqs]
    page_indices_ref,  # [max_num_seqs, pages_per_seq]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    seq_buf_idx_ref,
    # TODO(jevinjiang): if OOM in SMEM, consider pack to other scalar refs.
    num_seqs_ref,
    # Input
    q_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
    k_pages_hbm_ref,  # [total_num_pages, page_size, num_kv_heads, head_dim]
    v_pages_hbm_ref,  # [total_num_pages, page_size, num_kv_heads, head_dim]
    # Output
    o_ref,  # [num_q_per_blk, num_q_heads_per_blk, head_dim]
    # Scratch
    k_bufs,  # [2, num_kv_pages_per_blk, page_size, num_kv_heads_per_blk, head_dim]
    v_bufs,  # [2, num_kv_pages_per_blk, page_size, num_kv_heads_per_blk, head_dim]
    sems,  # [2, 2]
    l_ref,  # [num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128]
    m_ref,  # [num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128]
    *,
    sm_scale: float,
    mask_value: float,
):
  """ragged paged-attention kernel"""
  pass






def get_min_heads_per_blk(num_q_heads, num_kv_heads, q_dtype, kv_dtype):
  """get min heads per block"""
  pass


@functools.partial(
    jax.jit,
    static_argnames=[
        "sm_scale",
        "mask_value",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
    ],
)
def ragged_paged_attention(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    # TODO(jevinjiang): create a write_to_kv_cache kernel!
    k_pages: jax.Array,  # [total_num_pages, page_size, num_kv_heads, head_dim]
    v_pages: jax.Array,  # [total_num_pages, page_size, num_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    *,
    sm_scale: float = 1.0,
    mask_value: float = DEFAULT_MASK_VALUE,
    num_kv_pages_per_block: int = 16,
    num_queries_per_block: int = 128,
    vmem_limit_bytes: int | None = None,
):
  """Ragged paged attention that supports mixed prefill and decode.

  Args:
    q: concatenated all sequences' queries.
    k_pages: paged K cache. Normally in HBM.
    v_pages: paged V cache. Normally in HBM.
    kv_lens: padded kv lengths. Only the first num_seqs values are valid.
    page_indices: the first index indicates which page to use in the kv cache
      for each sequence. Only the first num_seqs values are valid.
    cu_q_lens: the cumulative sum of the effective query lengths. Similar to
      kv_lens, only the first num_seqs+1 values are valid.
    num_seqs: the dynamic number of sequences.
    sm_scale: the softmax scale which will be applied to the Q@K^T.
    mask_value: mask value for causal mask.
    num_kv_pages_per_block: number of kv pages to be processed in one flash
      attention block in the pallas kernel.
    num_queries_per_block: number of kv pages to be processed in one flash
      attention block in the pallas kernel.
    vmem_limit_bytes: the vmem limit for the pallas kernel.

  Returns:
    The output of the attention.
  """
  pass
