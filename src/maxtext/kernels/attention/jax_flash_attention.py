#  Copyright 2026 Google LLC
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
"""JAX implementation without using Pallas for Flash Attention."""

from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
from maxtext.kernels.attention import splash_attention_kernel

SegmentIds = splash_attention_kernel.SegmentIds


# This function computes masked flash attention using a block-sparse approach.
# This implementation keeps the full batch and number of heads dimensions
# throughout the attention computation while iterating through blocks of the
# key/value sequence and, within each, iterates through blocks of the query
# sequence. The `mask_blocked` is used to skip computations for blocks where all
# attention scores are masked out, improving efficiency for sparse masks.
def flash_attention_block_masked(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    segment_ids: SegmentIds | None,
    block_kv: int,
    block_q: int,
    mask: jnp.ndarray,
    mask_value: float,
    cap: Optional[float] = None,
    save_residuals: bool = False,
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]]:
  """Computes masked flash attention using block-sparse masking.

  Args:
    q: Query tensor with shape (batch_size, num_kv_heads,
      num_q_heads_per_kv_head, q_seq_len, head_dim).
    k: Key tensor with shape (batch_size, num_kv_heads, kv_seq_len, head_dim).
    v: Value tensor with shape (batch_size, num_kv_heads, kv_seq_len,
      v_head_dim).
    segment_ids: SegmentIds are a mechanism to ensure that there is no
      cross-attention between segments (fraction of a sequence) that have been
      concatenated together into a sequence. Each array is a list of ids
      (integers). Only tokens with the same id are allowed to attend to each
      other. It stores the segment ids of the query and key/value sequences.
    block_kv: Block size for the key/value sequence dimension.
    block_q: Block size for the query sequence dimension.
    mask: The full attention mask with shape of (q_seq_len, kv_seq_len). This
      mask will be used for all batches.
    mask_value: The value to use for masked-out attention scores.
    cap: Optional cap for attention logits. This helps to prevent extremely
      large logits: capped_logits = jnp.tanh(logits / attn_logits_soft_cap) *
      attn_logits_soft_cap
    save_residuals: Whether to save residuals. If True, returns a tuple of
      (output, dict=(logsumexp, max_logits)). Both `logsumexp` and `max_logits`
      are of shape (batch_size, num_kv_heads, num_q_heads // num_kv_heads,
      q_seq_len).

  Returns:
    If save_residuals is True, returns a tuple containing:
      - The output of the attention computation.
      - A dict of (logsumexp, max_logits)
    Otherwise, returns the output of the attention computation.
  """
  batch_size, num_q_heads, q_seq_len, qk_head_dim_size = q.shape
  _, num_kv_heads, kv_seq_len, _ = k.shape
  v_head_dim_size = v.shape[-1]
  data_type = q.dtype
  q_groups = num_q_heads // num_kv_heads
  q = q.reshape(
      (
          batch_size,
          num_kv_heads,
          q_groups,
          q_seq_len,
          qk_head_dim_size,
      )
  )

  # Calculate the number of key/value and query blocks.
  num_kv_blocks = kv_seq_len // block_kv
  num_q_blocks = q_seq_len // block_q

  # Before applying the segment mask, we need to broadcast the mask in batch
  # dimension since we have same logic for all batches.
  mask_full = jnp.broadcast_to(mask[None, :, :], (batch_size, q_seq_len, kv_seq_len))

  if segment_ids is not None:
    segment_ids_q = segment_ids.q[:, :, None]
    segment_ids_kv = segment_ids.kv[:, None, :]
    mask_full = jnp.logical_and(mask_full, segment_ids_q == segment_ids_kv)
  mask_blocked = jax.jit(mask_blocker, static_argnums=[1, 2])(mask_full, block_q, block_kv)

  # Initialize `l` (logsumexp) and `m` (max_logits) for the online softmax.
  # `l` is initialized to 0 since no blocks have been processed yet and the sum
  # is 0.
  l = jnp.zeros((batch_size, num_kv_heads, q_groups, q_seq_len), dtype=data_type)
  # `m` is initialized to the mask_value so that the first block's maximum logit
  # correctly becomes the running maximum.
  m = jnp.full(
      (batch_size, num_kv_heads, q_groups, q_seq_len),
      mask_value,
      dtype=data_type,
  )

  output = jnp.zeros(
      (
          batch_size,
          num_kv_heads,
          q_groups,
          q_seq_len,
          v_head_dim_size,
      ),
      dtype=data_type,
  )

  # Outer loop over the key/value blocks.

  output, l, m = jax.lax.fori_loop(0, num_kv_blocks, outer_loop_body, (output, l, m), unroll=True)

  # Reshape the output to drop the size one dimension at index 2,
  # which corresponds to `num_q_heads // num_kv_heads` when
  # num_q_heads == num_kv_heads.
  output = output.squeeze(axis=2)
  if not save_residuals:
    # To avoid remat of the output, we can use context=hbm remat policy as in
    # maxtext/configs/types.py
    return output

  l = l.squeeze(axis=2)
  m = m.squeeze(axis=2)
  stats = {"logsumexp": m + jnp.log(l), "max_logits": m}
  stats = jax.tree.map(jax.lax.stop_gradient, stats)
  return output, stats


def mask_blocker(mask: jnp.ndarray, block_q: int, block_kv: int) -> jnp.ndarray:
  """Creates a blocked mask from a full mask.

  Args:
    mask: The attention mask with shape of (batch_size, q_seq_len, kv_seq_len).
    block_q: Block size for the query sequence dimension.
    block_kv: Block size for the key/value sequence dimension.

  Returns:
    A blocked mask where each element indicates the number of non-zero
    elements in the corresponding block of the original mask.
  """
  pass
