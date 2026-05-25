# Copyright 2026 Google LLC
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

"""Ragged gather reduce kernel implementation from tpu-inference."""
# Source from experimental/users/kyuyeunk/vllm/kernels/sparse_core/ragged_gather_reduce.py

import functools
import math
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp
from packaging.version import Version

# JAX <= 0.10.0 used `out_shape`/`scratch_shapes` kwargs for `pl.kernel`; later
# versions renamed them to `out_type`/`scratch_types`.
if Version(jax.__version__) <= Version("0.10.0"):
  _OUT_KW = "out_shape"
  _SCRATCH_KW = "scratch_shapes"
else:
  _OUT_KW = "out_type"
  _SCRATCH_KW = "scratch_types"


# ceil up to the nearest multiple of b.
def _align_to(a, b):
  return ((a + b - 1) // b) * b


def _fallback_implementation(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
) -> jax.Array:
  """Fallback to JAX implementation."""
  out = x[indices] * topk_weights[:, None].astype(jnp.float32)
  out = jnp.where(valid_rows_mask[:, None], out, 0)
  out = out.reshape(-1, reduce_group_size, out.shape[-1])
  out = jnp.sum(out, axis=1).astype(x.dtype)
  return out


def main_kernel(
    # Inputs.
    num_rows_per_row_partition_ref: jax.Ref,
    in_hbm_ref: jax.Ref,
    src_indices_hbm_ref: jax.Ref,
    dst_indices_hbm_ref: jax.Ref,
    topk_weights_hbm_ref: jax.Ref,
    # Outputs.
    out_hbm_ref: jax.Ref,
    # Scratch.
    num_rows_per_row_partition_vmem_ref: jax.Ref,
    out_vmem_ref: jax.Ref,
    prev_iter_last_row_vmem_ref: jax.Ref,
    src_indices_vmem_ref: jax.Ref,
    dst_indices_vmem_ref: jax.Ref,
    topk_weights_vmem_ref: jax.Ref,
    sem_ref: jax.Ref,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
    num_row_partitions: int,
    num_column_partitions: int,
):
  """Main Pallas kernel for ragged gather and reduction on SparseCore."""
  pass


def _preprocess(
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
    num_row_partitions: int,
    num_simd_lanes: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Preprocesses indices for ragged gather reduce."""
  assert indices.ndim == 1, "Ragged scatter only supports 1d indices."

  row_partition_size = indices.shape[0] // num_row_partitions
  valid_rows_mask = valid_rows_mask.reshape(num_row_partitions, -1)

  # Move all the valid source rows to the beginning of each row partition.
  sorted_by_validity = jnp.argsort(~valid_rows_mask, descending=False, stable=True, axis=-1)
  sorted_by_validity += (
      jnp.broadcast_to(
          jnp.arange(num_row_partitions)[:, None],
          (num_row_partitions, row_partition_size),
      )
      * row_partition_size
  )
  sorted_by_validity = sorted_by_validity.reshape(-1)

  src_indices = indices[sorted_by_validity]
  # `reduce_group_size` source rows are mapped (and reduced) to the same output
  # row.
  dst_indices = sorted_by_validity // reduce_group_size
  topk_weights = topk_weights[sorted_by_validity]
  topk_weights = topk_weights.astype(jnp.float32)

  num_src_rows_per_row_partition = jnp.sum(valid_rows_mask, axis=-1)
  assert num_row_partitions <= num_simd_lanes
  num_src_rows_per_row_partition = jnp.pad(
      num_src_rows_per_row_partition.astype(jnp.int32),
      (0, num_simd_lanes - num_row_partitions),
  )
  # If there is no valid source row in a reduce group, we set the mask to
  # False, so that the output for that group is set to zero.
  mask = jnp.any(valid_rows_mask.reshape(-1, reduce_group_size), axis=-1)

  return (
      src_indices,
      dst_indices,
      topk_weights,
      num_src_rows_per_row_partition,
      mask,
  )


@functools.partial(jax.jit, static_argnames=("reduce_group_size",))
def ragged_gather_reduce(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
) -> jax.Array:
  """Gathers `x` according to `indices`, applies weights and masks, and reduces.

  This function performs a gathered lookup from `x` using `indices`, scales the
  obtained rows by `topk_weights`, masks out any rows where `valid_rows_mask` is
  False, and then groups every `reduce_group_size` rows together and reduces
  them via summation.

  The typical use case of this kernel is unpermute + local-reduction in the
  MOE after GMM. Compared to maxtext.src.maxtext.kernels.gather_reduce_sc,
  this kernel provides better performance if large sparsity exists in
  `valid_rows_mask`. For example, expert_parallelism =8, 16 etc.

  Args:
    x: A 2D JAX array of input features with shape `(input_size, hidden_size)`.
    indices: A 1D JAX array of indices to gather with shape `(input_size,)`.
    topk_weights: A 1D JAX array of weights to scale the gathered rows with
      shape `(input_size,)`.
    valid_rows_mask: A 1D boolean JAX array indicating which gathered rows are
      valid, with shape `(input_size,)`.
    reduce_group_size: An integer representing the number of consecutive rows to
      reduce (sum) together.

  Returns:
    A 2D JAX array of reduced data with shape
    `(input_size // reduce_group_size, hidden_size)`.
  """

  assert x.ndim == 2, "ragged_gather_reduce only supports 2d inputs."
  assert indices.ndim == 1, "ragged_gather_reduce only supports 1d indices."
  assert topk_weights.ndim == 1, "ragged_gather_reduce only supports 1d topk_weights."
  assert valid_rows_mask.ndim == 1, "ragged_gather_reduce only supports 1d valid_rows_mask."

  sc_info = pltpu.get_tpu_info().sparse_core
  if sc_info is None:
    return _fallback_implementation(x, indices, topk_weights, valid_rows_mask, reduce_group_size)

  # Heuristic threshold on whether to fallback for small inputs.
  dtype = x.dtype
  dtype_bytes = jax.dtypes.itemsize_bits(dtype) // 8
  if jnp.size(x) * dtype_bytes * 2 < pltpu.get_tpu_info().vmem_capacity_bytes * 0.6:
    # For small {input + output}, it's likely that both can be put in TC VMEM,
    # so it's likely faster to run TC-based implementation on it than going
    # through SC, without data movement to/from HBM.
    return _fallback_implementation(x, indices, topk_weights, valid_rows_mask, reduce_group_size)

  hidden_size = x.shape[-1]
  input_size = indices.size
  num_simd_lanes = sc_info.num_lanes
  num_cores = sc_info.num_cores * sc_info.num_subcores

  # This kernel partitions the output's columns into `num_column_partitions` and
  # partition the output's rows into `num_row_partitions` and run each
  # {row_partition} x {column_partition} combination on a separate SC subcore
  # for parallelism. With such work partitioning, we guarantee that there won't
  # be write collision (from different subcores) to the any output row X column.
  #
  # Each column partition should be multiple of 128 (number of lanes) due to
  # DMA requirements. Unless requiring padding on the column dimension, larger
  # column partitions (thus smaller row partitions given fixed num_cores) is
  # more preferable because large row partition may lead to imbalanced load
  # (valid_rows_mask may have more rows in some partitions than others).
  # Most LLM's hidden size is multiple of 1024, `num_column_partitions=8` should
  # work well in practice without requiring padding on the column size.
  num_column_partitions = 8
  assert num_cores % num_column_partitions == 0
  num_rows_partitions = num_cores // num_column_partitions

  aligned_hidden_size = _align_to(hidden_size, 128 * num_column_partitions)
  col_size = aligned_hidden_size // num_column_partitions
  row_tile_size = num_simd_lanes
  padded_input_size = _align_to(
      input_size,
      math.lcm(num_rows_partitions * row_tile_size, reduce_group_size),
  )
  pad_input_size = padded_input_size - input_size

  x = jnp.pad(
      x,
      ((0, pad_input_size), (0, aligned_hidden_size - hidden_size)),
      constant_values=0,
  )
  indices = jnp.pad(indices, (0, pad_input_size), constant_values=0)
  topk_weights = jnp.pad(topk_weights, (0, pad_input_size), constant_values=0)
  valid_rows_mask = jnp.pad(valid_rows_mask, (0, pad_input_size), constant_values=False)

  (
      src_indices,
      dst_indices,
      topk_weights,
      num_src_rows_per_row_partition,
      mask,
  ) = _preprocess(
      indices,
      topk_weights,
      valid_rows_mask,
      reduce_group_size,
      num_rows_partitions,
      num_simd_lanes,
  )

  vector_mesh = plsc.VectorSubcoreMesh(
      num_cores=sc_info.num_cores,
      num_subcores=sc_info.num_subcores,
      core_axis_name="core",
      subcore_axis_name="subcore",
  )
  # Each output row from `main_kernel` will be of type float32, and then casted
  # to the input dtype when doing the filter operation.
  out = pl.kernel(  # pytype: disable=wrong-keyword-args
      functools.partial(
          main_kernel,
          core_axis_name=vector_mesh.core_axis_name,
          subcore_axis_name=vector_mesh.subcore_axis_name,
          num_row_partitions=num_rows_partitions,
          num_column_partitions=num_column_partitions,
      ),
      compiler_params=pltpu.CompilerParams(
          use_tc_tiling_on_sc=True,
          disable_bounds_checks=True,
      ),
      mesh=vector_mesh,
      name="sc_ragged_gather_reduce",
      **{
          _OUT_KW: jax.ShapeDtypeStruct(
              (padded_input_size // reduce_group_size, aligned_hidden_size),
              jnp.float32,
          ),
          _SCRATCH_KW: dict(  # pylint: disable=use-dict-literal
              num_rows_per_row_partition_vmem_ref=pltpu.VMEM((num_simd_lanes,), jnp.int32),
              out_vmem_ref=pltpu.VMEM((num_simd_lanes, col_size), jnp.uint32),
              prev_iter_last_row_vmem_ref=pltpu.VMEM((1, col_size), jnp.uint32),
              src_indices_vmem_ref=pltpu.VMEM((num_simd_lanes,), jnp.int32),
              dst_indices_vmem_ref=pltpu.VMEM((num_simd_lanes,), jnp.int32),
              topk_weights_vmem_ref=pltpu.VMEM((num_simd_lanes,), jnp.float32),
              sem_ref=pltpu.SemaphoreType.DMA((2,)),
          ),
      },
  )(num_src_rows_per_row_partition, x, src_indices, dst_indices, topk_weights)

  # If there is no valid source row in a reduce group, set that group's output
  # to zero.
  return jnp.where(
      mask[:, None],
      out.astype(x.dtype),
      jnp.zeros_like(out, dtype=x.dtype),
  )[: (input_size // reduce_group_size), :hidden_size]
