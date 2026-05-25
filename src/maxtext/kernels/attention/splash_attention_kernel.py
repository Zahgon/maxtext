# pylint: skip-file
from __future__ import annotations

# Copyright 2023–2026 Google LLC
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


# Implementation of Sparse Flash Attention, a.k.a. "Splash" attention.

from collections.abc import Callable, Mapping
import dataclasses
import enum
import functools
from types import UnionType
from typing import Any, Literal, NamedTuple, overload

import jax
from jax import ad_checkpoint
from jax import lax
from jax import tree_util
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as mask_lib
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask_info as mask_info_lib
import jax.numpy as jnp
import numpy as np

partial = functools.partial
DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)
NUM_LANES = 128
NUM_SUBLANES = 8
# We predefine some useful dimension numbers for dot_general
NN_DIM_NUMBERS = (((1,), (0,)), ((), ()))  # standard matmul
NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))  # RHS transposed

# mypy: ignore-errors


class SegmentIds(NamedTuple):
  """SegmentIds for Q and KV sequences.

  SegmentIds are a mechanism to ensure that there is no cross-attention between
  segments (fraction of a sequence) that have been concatenated together into a
  sequence. Each array is a list of ids (integers). Only tokens with the same
  id are allowed to attend to each other.

  The static mask (e.g. causal) is "and-ed" with the segment id mask to form
  the actual attention mask. It is important that the latter does not have any
  all-zero rows (along dimension kv). Otherwise it would result in a invalid
  softmax (the denominator would be 0).
  This condition holds for causal self-attention because in this case segment
  ids form a block diagonal matrix so at least one element in each row is set.
  It is easy to break this condition with non-self-attention configurations.
  """

  q: jax.Array  # [q_seq_len]
  kv: jax.Array  # [kv_seq_len]


# Return type of SplashAttention function that implements the custom vjp rule.
# `jax.Array` no residuals
# `tuple[jax.Array, tuple[jax.Array,]]` # residuals,
SplashCustomReturnType = jax.Array | tuple[jax.Array, tuple[jax.Array,]]

SplashResidualsType = tuple[
    jax.Array,  # q
    jax.Array,  # k
    jax.Array,  # v
    None | SegmentIds,  # segment_ids
    jax.Array,  # out
    jax.Array,  # logsumexp
    None | mask_info_lib.MaskInfo,  # dq_mask_info
    None | mask_info_lib.MaskInfo,  # dkv_mask_info
]

MaskFunctionType = Callable[..., jax.Array]


def get_kernel_name(
    block_metadata: Mapping[str, Any],
    is_mqa: bool,
    save_residuals: bool,
    is_segmented: bool,
    phase: str,
) -> str:
  """Returns a unique name for all SplashAttention kernel variants."""
  pass


# Reference attention implementations


@overload
def _attention_reference(
    mask: jax.Array,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    save_residuals: Literal[False],
    mask_value: float,
    custom_type: str,
    attn_logits_soft_cap: float | None,
) -> jax.Array:
  """Reference attention implementation."""
  ...


@overload
def _attention_reference(
    mask: jax.Array,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    save_residuals: Literal[True],
    mask_value: float,
    custom_type: str,
    attn_logits_soft_cap: float | None,
) -> tuple[jax.Array, tuple[jax.Array]]:
  ...


def _attention_reference(
    mask: jax.Array,  # [q_seq_len, kv_seq_len]
    q: jax.Array,  # [q_seq_len, head_dim]
    k: jax.Array,  # [kv_seq_len, head_dim]
    v: jax.Array,  # [kv_seq_len, head_dim]
    segment_ids: SegmentIds | None,
    mask_value: float,
    save_residuals: bool,
    custom_type: str,
    attn_logits_soft_cap: float | None,
):
  """Reference attention implementation."""
  pass


def _attention_reference_default(
    mask: jax.Array,  # [q_seq_len, kv_seq_len]
    q: jax.Array,  # [q_seq_len, head_dim]
    k: jax.Array,  # [kv_seq_len, head_dim]
    v: jax.Array,  # [kv_seq_len, head_dim]
    segment_ids: SegmentIds | None,
    mask_value: float,
    save_residuals: bool,
    custom_type: str,
    attn_logits_soft_cap: float | None,
):
  """Reference attention default implementation."""
  pass


def attention_reference(
    mask: jax.Array,  # [q_seq_len, kv_seq_len]
    q: jax.Array,  # [q_seq_len, head_dim]
    k: jax.Array,  # [kv_seq_len, head_dim]
    v: jax.Array,  # [kv_seq_len, head_dim]
    segment_ids: SegmentIds | None,
    *,
    mask_value: float = DEFAULT_MASK_VALUE,
    save_residuals: bool = False,
    custom_type: str = "flash",
    attn_logits_soft_cap: float | None = None,
) -> SplashCustomReturnType:
  """Reference attention implementation."""
  pass


def _attention_reference_custom_fwd(
    mask: jax.Array,  # [q_seq_len, kv_seq_len]
    q: jax.Array,  # [q_seq_len, head_dim]
    k: jax.Array,  # [kv_seq_len, head_dim]
    v: jax.Array,  # [kv_seq_len, head_dim]
    segment_ids: SegmentIds | None,
    mask_value: float,
    save_residuals: bool,
    custom_type: str,
    attn_logits_soft_cap: float | None,
):
  """Reference attention custom forward implementation."""
  pass


def _attention_reference_custom_bwd(
    mask_value: float,
    save_residuals: bool,
    custom_type: str,
    attn_logits_soft_cap: float | None,
    res,
    do: jax.Array,
) -> tuple[None, jax.Array, jax.Array, jax.Array, None]:
  """Reference attention custom backward implementation."""
  pass


_attention_reference_custom = jax.custom_vjp(_attention_reference, nondiff_argnums=(5, 6, 7, 8))
_attention_reference_custom.defvjp(_attention_reference_custom_fwd, _attention_reference_custom_bwd)


def attention_reference_custom(
    mask: jax.Array,  # [q_seq_len, kv_seq_len]
    q: jax.Array,  # [q_seq_len, head_dim]
    k: jax.Array,  # [kv_seq_len, head_dim]
    v: jax.Array,  # [kv_seq_len, head_dim]
    segment_ids: SegmentIds | None,
    *,
    mask_value: float = DEFAULT_MASK_VALUE,
    save_residuals: bool = False,
    custom_type: str = "flash",
    attn_logits_soft_cap: float | None = None,
):
  """Reference attention custom implementation."""
  pass


def make_attention_reference(
    mask: mask_lib.Mask | np.ndarray,
    is_mqa: bool,
    backward_impl: str = "vanilla",
    **params: Any,
) -> Callable:
  """Returns a function that computes reference attention."""
  pass


make_masked_mha_reference = partial(make_attention_reference, is_mqa=False)
make_masked_mqa_reference = partial(make_attention_reference, is_mqa=True)


# Splash attention implementation


# We use an IntEnum to make it JSON serializable as regen metadata.
class QKVLayout(enum.IntEnum):
  HEAD_DIM_MINOR = enum.auto()  # [..., seq_len, head_dim]
  SEQ_MINOR = enum.auto()  # [..., head_dim, seq_len]




@dataclasses.dataclass(frozen=True, slots=True)
class BlockSizes:
  """Tile sizes parameterizing SplashAttention kernels.

  Those parameters have negligible effect on numerics, but affect performance
  greatly.

  Note that changing the layouts only influences the physical layout that the
  kernel will enforce. The logical interface to splash attention always takes
  the head dimension as the minormost one.
  """

  block_q: int
  block_kv: int
  block_kv_compute: int | None = None

  block_q_dkv: int | None = None
  block_kv_dkv: int | None = None
  block_kv_dkv_compute: int | None = None

  block_q_dq: int | None = None
  block_kv_dq: int | None = None

  use_fused_bwd_kernel: bool = False

  q_layout: QKVLayout = QKVLayout.HEAD_DIM_MINOR
  k_layout: QKVLayout = QKVLayout.HEAD_DIM_MINOR
  v_layout: QKVLayout = QKVLayout.HEAD_DIM_MINOR

  def __post_init__(self):
    if self.block_kv_compute is None:
      object.__setattr__(self, "block_kv_compute", self.block_kv)
    if self.block_kv_dkv_compute is None:
      object.__setattr__(self, "block_kv_dkv_compute", self.block_kv_dkv)
    if self.use_fused_bwd_kernel:
      if self.block_q_dq is not None or self.block_kv_dq is not None:
        raise ValueError("Block sizes for dq kernel are not needed with a fused kernel.")




def _next_nonzero(
    h,
    i,
    j,
    data_next_ref,
    block_mask_ref,
    m_next_ref,
    next_i=False,
):
  """Returns the next nonzero index and the mask for the current index."""
  pass


def _apply_mask_and_soft_cap(
    qk: jax.Array,
    mask_value: float,
    should_not_mask,
    mask_ref,
    q_sequence_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    *,
    attn_logits_soft_cap: float,
    k_slice: pl.Slice,
    k_offset: int | jax.Array,
    bq: int,
    k_in_lanes=True,
    mask_function=None,
) -> jax.Array | tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Applies the mask and soft cap to the logits."""
  assert mask_ref is None or q_sequence_ref is None
  assert (q_sequence_ref is None) == (mask_function is None)

  masks = []
  if mask_ref is not None:
    if k_in_lanes:
      mask = mask_ref[:, k_slice]
    else:
      mask = mask_ref[k_slice, :]

    masks.append(jnp.bitwise_or(mask, jnp.broadcast_to(should_not_mask, mask.shape)))
  if mask_function is not None:
    # Compute the mask using the given q_sequence indices.
    # KV indices are computed on the fly. This works because we only support Q
    # sequence sharding. If we wanted to compute Q indices too, then we would
    # need to keep into account the current shard along Q sequence.

    if k_in_lanes:
      assert q_sequence_ref.shape == (bq, NUM_LANES)

      k_sequence = k_offset + jax.lax.broadcasted_iota(jnp.int32, (bq, k_slice.size), 1)

      repeats, rem = divmod(k_slice.size, NUM_LANES)
      assert rem == 0
      q_sequence = jnp.tile(q_sequence_ref[...], (1, repeats))  # [bq, k_slice.size]
    else:
      assert q_sequence_ref.shape == (NUM_SUBLANES, bq)

      k_sequence = k_offset + jax.lax.broadcasted_iota(jnp.int32, (k_slice.size, bq), 0)
      q_sequence = q_sequence_ref[:1, :]  # [1, bq]
      q_sequence = jnp.broadcast_to(q_sequence, (k_slice.size, bq))

    assert q_sequence.shape == k_sequence.shape
    computed_mask = mask_function(q_sequence, k_sequence)  # pytype: disable=wrong-arg-count
    if computed_mask.dtype != jnp.dtype(jnp.bool_):
      raise ValueError("Mask function must return a boolean-valued array, but got:" f" {computed_mask.dtype}")
    masks.append(computed_mask)

  if q_segment_ids_ref is not None:
    if k_in_lanes:
      kv_ids = kv_segment_ids_ref[:1, k_slice]  # [1, k_slice]
      repeats, rem = divmod(kv_ids.shape[1], NUM_LANES)
      if rem:
        raise NotImplementedError(f"block_kv must be a multiple of {NUM_LANES}")
      q_ids = jnp.tile(q_segment_ids_ref[:], (1, repeats))  # [bq, bkv]
    else:
      assert bq == q_segment_ids_ref.shape[-1]
      repeats, rem = divmod(bq, NUM_LANES)
      if rem:
        raise NotImplementedError(f"block_q must be a multiple of {NUM_LANES}")
      kv_ids = jnp.tile(kv_segment_ids_ref[k_slice, :], (1, repeats))  # [k_slice, bq]
      q_ids = q_segment_ids_ref[:1, :]  # [1, bq]
    masks.append(q_ids == kv_ids)

  def cap_logits(logits):
    if attn_logits_soft_cap is not None:
      logits = jnp.tanh(qk / attn_logits_soft_cap)
      return logits * attn_logits_soft_cap
    else:
      return logits

  if masks:
    mask = functools.reduce(jnp.logical_and, masks)
    qk = cap_logits(qk)
    qk = jnp.where(mask, qk, mask_value)
  else:
    qk = cap_logits(qk)
  return qk


def flash_attention_kernel(
    # Prefetched inputs
    data_next_ref,
    block_mask_ref,
    mask_next_ref,
    # Inputs
    q_ref,
    k_ref,
    v_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    mask_ref,
    q_sequence_ref,
    # Outputs
    m_scratch_ref,
    l_scratch_ref,
    o_scratch_ref,
    o_ref,
    logsumexp_ref=None,
    *,
    mask_value: float,
    grid_width: int,
    bq: int,
    bkv: int,
    bkv_compute: int,
    head_dim_v: int,
    q_layout: QKVLayout,
    k_layout: QKVLayout,
    v_layout: QKVLayout,
    attn_logits_soft_cap: float | None,
    mask_function: MaskFunctionType | None,
):
  """Flash attention kernel."""
  pass


@overload
def _splash_attention_forward(
    fwd_mask_info: mask_info_lib.MaskInfo,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    mask_value: float,
    is_mqa: bool,
    block_sizes: BlockSizes,
    residual_checkpoint_name: str | None,
    mask_function: MaskFunctionType | None,
    save_residuals: Literal[False] = False,
    attn_logits_soft_cap: float | None = None,
) -> jax.Array:
  ...


@overload
def _splash_attention_forward(
    fwd_mask_info: mask_info_lib.MaskInfo,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    mask_value: float,
    is_mqa: bool,
    block_sizes: BlockSizes,
    residual_checkpoint_name: str | None,
    mask_function: MaskFunctionType | None,
    save_residuals: Literal[True],
    attn_logits_soft_cap: float | None = None,
) -> SplashCustomReturnType:
  ...




def _splash_attention_forward(
    fwd_mask_info: mask_info_lib.MaskInfo,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    mask_value: float,
    is_mqa: bool,
    block_sizes: BlockSizes,
    residual_checkpoint_name: str | None,
    save_residuals: bool,
    mask_function: MaskFunctionType | None,
    attn_logits_soft_cap: float | None = None,
    interpret: bool = False,
) -> SplashCustomReturnType:
  """Forward pass for splash attention."""
  pass


@partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9, 10, 11, 12, 13, 14))
def _splash_attention_custom(
    fwd_mask_info: mask_info_lib.MaskInfo,
    dq_mask_info: mask_info_lib.MaskInfo | None,
    dkv_mask_info: mask_info_lib.MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    save_residuals: bool,
    mask_value: float,
    is_mqa: bool,
    block_sizes: BlockSizes,
    residual_checkpoint_name: str | None,
    mask_function: MaskFunctionType | None,
    attn_logits_soft_cap: float | None = None,
    interpret: bool = False,
) -> SplashCustomReturnType:
  """Custom splash attention kernel."""
  pass


def _splash_attention_fwd(
    fwd_mask_info: mask_info_lib.MaskInfo,
    dq_mask_info: mask_info_lib.MaskInfo | None,
    dkv_mask_info: mask_info_lib.MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    save_residuals: bool,
    mask_value: float,
    is_mqa: bool,
    block_sizes: BlockSizes,
    residual_checkpoint_name: str | None,
    mask_function: MaskFunctionType | None,
    attn_logits_soft_cap: float | None = None,
    interpret: bool = False,
) -> tuple[
    tuple[jax.Array],
    SplashResidualsType,
]:
  """Forward pass for splash attention."""
  pass


def _flash_attention_dq_kernel(
    # Prefetched inputs
    data_next_ref,
    block_mask_ref,
    mask_next_ref,
    # Inputs
    q_ref,
    k_ref,
    v_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    logsumexp_ref,
    do_ref,
    di_ref,
    mask_ref,
    q_sequence_ref,
    # Outputs
    dq_scratch_ref,
    dq_ref,
    *,
    mask_value: float,
    grid_width: int,
    bq: int,
    bkv: int,
    attn_logits_soft_cap: float | None = None,
    q_layout: QKVLayout,
    k_layout: QKVLayout,
    v_layout: QKVLayout,
    mask_function: MaskFunctionType | None,
):
  """Backprop kernel for the DQ part of flash attention."""
  pass


def _splash_attention_bwd_dq(
    q,
    k,
    v,
    segment_ids,
    logsumexp,
    do,
    di,
    *,
    bq: int,
    bkv: int,
    is_mqa: bool,
    mask_info: mask_info_lib.MaskInfo,
    mask_value: float,
    attn_logits_soft_cap: float | None,
    q_layout: QKVLayout,
    k_layout: QKVLayout,
    v_layout: QKVLayout,
    mask_function: MaskFunctionType | None,
    interpret: bool,
):
  """Backward pass for the DQ part of splash attention."""
  pass


def _flash_attention_dkv_kernel(
    # Prefetched inputs
    data_next_ref,
    block_mask_ref,
    mask_next_ref,
    # Inputs
    q_ref,
    k_ref,
    v_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    logsumexp_ref,
    do_ref,
    di_ref,
    mask_ref,
    q_sequence_ref,
    # Outputs
    dq_scratch_ref,
    dk_scratch_ref,
    dv_scratch_ref,
    dq_ref,
    dk_ref,
    dv_ref,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    mask_value: float,
    grid_width: int,
    bq: int,
    bkv_compute: int,
    is_mqa: bool,
    attn_logits_soft_cap: float | None,
    q_layout: QKVLayout,
    k_layout: QKVLayout,
    v_layout: QKVLayout,
    bkv: int,
    mask_function: MaskFunctionType | None,
):
  """Backward pass for the DKV part of splash attention."""
  pass


def _splash_attention_bwd_dkv(
    q,
    k,
    v,
    segment_ids,
    logsumexp,
    do,
    di,
    *,
    bq: int,
    bkv: int,
    bkv_compute: int,
    is_mqa: bool,
    mask_info: mask_info_lib.MaskInfo,
    mask_value: float,
    attn_logits_soft_cap: float | None,
    use_fused_bwd_kernel: bool,
    q_layout: QKVLayout,
    k_layout: QKVLayout,
    v_layout: QKVLayout,
    mask_function: MaskFunctionType | None,
    interpret: bool,
):
  """Backward pass for the DKV part of splash attention."""
  pass


def _splash_attention_bwd(
    save_residuals: bool,
    mask_value: float,
    is_mqa: bool,
    block_sizes: BlockSizes,
    residual_checkpoint_name: str | None,
    mask_function: MaskFunctionType | None,
    attn_logits_soft_cap: float | None,
    interpret: bool,
    res: SplashResidualsType,
    do: jax.Array,
) -> tuple[
    mask_info_lib.MaskInfo | None,  # fwd_mask_info
    mask_info_lib.MaskInfo | None,  # dq_mask_info
    mask_info_lib.MaskInfo | None,  # dvk_mask_info
    jax.Array,  # q
    jax.Array,  # k
    jax.Array,  # v
    SegmentIds | None,  # segmend_ids
]:
  """Backward pass for splash attention."""
  pass


_splash_attention_custom.defvjp(_splash_attention_fwd, _splash_attention_bwd)


@partial(
    jax.jit,
    static_argnames=[
        "is_mqa",
        "block_sizes",
        "save_residuals",
        "mask_value",
        "attn_logits_soft_cap",
        "residual_checkpoint_name",
        "mask_function",
        "interpret",
    ],
)
def _splash_attention(
    fwd_mask_info: mask_info_lib.MaskInfo,
    dq_mask_info: mask_info_lib.MaskInfo | None,
    dkv_mask_info: mask_info_lib.MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None = None,
    *,
    is_mqa: bool,
    block_sizes: BlockSizes | None,
    save_residuals: bool,
    mask_value: float,
    attn_logits_soft_cap: float | None,
    residual_checkpoint_name: str | None,
    mask_function: MaskFunctionType | None,
    interpret: bool,
) -> SplashCustomReturnType:
  """For dynamic masks, `partial_mask_blocks` has shape (head_count, q_blocks, kv_blocks, block_q, block_kv).

  This shape allows sharding across both head count and query sequence
  dimensions.

  Note: The leading dimensions (head_count, q_blocks, kv_blocks) must be
  collapsed into a single dimension before being passed to the kernel.
  """
  pass


@partial(
    jax.jit,
    static_argnames=[
        "is_mqa",
        "block_sizes",
        "save_residuals",
        "mask_value",
        "attn_logits_soft_cap",
        "residual_checkpoint_name",
        "mask_function",
        "interpret",
    ],
)
def _splash_attention_manual_fwd(
    fwd_mask_info: mask_info_lib.MaskInfo,
    dq_mask_info: mask_info_lib.MaskInfo | None,
    dkv_mask_info: mask_info_lib.MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None = None,
    sinks: jax.Array | None = None,
    *,
    is_mqa: bool,
    block_sizes: BlockSizes | None,
    save_residuals: bool,
    mask_value: float,
    attn_logits_soft_cap: float | None,
    residual_checkpoint_name: str | None,
    mask_function: MaskFunctionType | None,
    interpret: bool,
) -> SplashCustomReturnType:
  """Returns both the attention output and logsumexp.

  This is useful when manually controlling remat in the backward pass, as both
  can be returned as residuals from the forward pass."""
  pass


def _splash_attention_manual_bwd(
    fwd_mask_info: mask_info_lib.MaskInfo,
    dq_mask_info: mask_info_lib.MaskInfo | None,
    dkv_mask_info: mask_info_lib.MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    out: jax.Array,
    logsumexp: jax.Array,
    do: jax.Array,
    segment_ids: SegmentIds | None = None,
    sinks: jax.Array | None = None,
    *,
    is_mqa: bool,
    block_sizes: BlockSizes | None,
    save_residuals: bool,
    mask_value: float,
    attn_logits_soft_cap: float | None,
    residual_checkpoint_name: str | None,
    mask_function: MaskFunctionType | None,
    interpret: bool,
):
  """Transpose of _splash_attention_manual_fwd that uses attention output and logsumexp."""
  pass


@jax.tree_util.register_pytree_node_class
class SplashAttentionKernel:
  """Defines a SplashAttention kernel object."""

  def __init__(
      self,
      fwd_mask_info: mask_info_lib.MaskInfo,
      dq_mask_info: mask_info_lib.MaskInfo | None,
      dkv_mask_info: mask_info_lib.MaskInfo | None,
      **kwargs,
  ):
    self.kwargs = kwargs
    self.fwd_mask_info = fwd_mask_info
    self.dq_mask_info = dq_mask_info
    self.dkv_mask_info = dkv_mask_info

  def __call__(self, *args, **kwargs) -> SplashCustomReturnType:
    return _splash_attention(
        self.fwd_mask_info,
        self.dq_mask_info,
        self.dkv_mask_info,
        *args,
        **kwargs,
        **self.kwargs,
    )



  def manual_sharding_spec(self, sharding: jax.sharding.NamedSharding):
    """Returns a value that can be used as a shard_map partition spec for the kernel."""
    if self.fwd_mask_info.data_next is not None:
      block_mask_shape = self.fwd_mask_info.data_next.shape
      try:
        shard_shape = sharding.shard_shape(block_mask_shape)
      except ValueError as exc:
        raise ValueError("The sharding must divide the mask blocks evenly between devices") from exc
      if block_mask_shape[-1] != shard_shape[-1]:
        raise ValueError("Sharding the kv sequence dimension is not supported")
    spec = sharding.spec
    assert len(spec) == 2
    replicated = jax.sharding.PartitionSpec()
    partial_mask_blocks_spec = spec if self.fwd_mask_info.is_dynamic_mask else replicated
    # Shard q_sequence over the sequence dimension only.
    q_sequence_spec = jax.sharding.PartitionSpec(spec[1])
    mask_info_specs = mask_info_lib.MaskInfo(  # pytype: disable=wrong-arg-types
        data_next=spec if self.fwd_mask_info.data_next is not None else None,
        mask_next=spec if self.fwd_mask_info.mask_next is not None else None,
        block_mask=spec if self.fwd_mask_info.block_mask is not None else None,
        partial_mask_blocks=partial_mask_blocks_spec if self.fwd_mask_info.partial_mask_blocks is not None else None,
        q_sequence=q_sequence_spec if self.fwd_mask_info.q_sequence is not None else None,
    )
    return SplashAttentionKernel(
        mask_info_specs,
        mask_info_specs if self.dq_mask_info is not None else None,
        mask_info_specs if self.dkv_mask_info is not None else None,
        **self.kwargs,
    )

  def tree_flatten(self):
    return (
        (self.fwd_mask_info, self.dq_mask_info, self.dkv_mask_info),
        self.kwargs,
    )

  @classmethod
  def tree_unflatten(cls, kwargs, values):
    fwd_mask_info, dq_mask_info, dkv_mask_info = values
    # NamedTuples are not preserved during pytree serialization.
    dq_mask_info = mask_info_lib.MaskInfo(*dq_mask_info) if dq_mask_info is not None else None
    dkv_mask_info = mask_info_lib.MaskInfo(*dkv_mask_info) if dkv_mask_info is not None else None
    return SplashAttentionKernel(
        mask_info_lib.MaskInfo(*fwd_mask_info),
        dq_mask_info,
        dkv_mask_info,
        **kwargs,
    )


def _make_splash_attention(
    mask: np.ndarray | jax.Array | mask_lib.MultiHeadMask,
    *,
    block_sizes: BlockSizes | None = None,
    is_mqa: bool,
    save_residuals: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    attn_logits_soft_cap: float | None = None,
    downcast_smem_data: bool = True,
    head_shards: int,
    q_seq_shards: int,
    residual_checkpoint_name: str | None = None,
    interpret: bool = False,
):
  """Creates a SplashAttentionKernel."""
  pass


make_splash_mha = partial(_make_splash_attention, is_mqa=False)
make_splash_mqa = partial(_make_splash_attention, is_mqa=True)

make_splash_mha_single_device = partial(make_splash_mha, is_mqa=False, head_shards=1, q_seq_shards=1)

make_splash_mqa_single_device = partial(make_splash_mha, is_mqa=True, head_shards=1, q_seq_shards=1)
