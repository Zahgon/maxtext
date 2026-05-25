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

"""Ragged gather kernel implementation from tpu-inference."""
# Source from https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/kernels/sparse_core/ragged_gather.py

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
from packaging.version import Version


# JAX <= 0.10.0 used `out_shape`/`scratch_shapes` kwargs for `pl.kernel`; later
# versions renamed them to `out_type`/`scratch_types`.
if Version(jax.__version__) <= Version("0.10.0"):
  _OUT_KW = "out_shape"
  _SCRATCH_KW = "scratch_shapes"
else:
  _OUT_KW = "out_type"
  _SCRATCH_KW = "scratch_types"


def main_kernel(
    # Inputs.
    start_ref: jax.Ref,
    end_ref: jax.Ref,
    in_hbm_ref: jax.Ref,
    indices_hbm_ref: jax.Ref,
    # Outputs.
    out_hbm_ref: jax.Ref,
    # Scratch.
    start_vmem_ref: jax.Ref,
    end_vmem_ref: jax.Ref,
    out_vmem_ref: jax.Ref,
    indices_vmem_ref: jax.Ref,
    sem_ref: jax.Ref,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
):
  """Core ragged gather operation"""
  pass


def calculate_col_size(hidden_size: int) -> int:
  """Calculate col size for ragged gather kernel."""
  tpu_info = pltpu.get_tpu_info()
  sc_info = tpu_info.sparse_core
  assert sc_info is not None
  num_lanes = tpu_info.num_lanes
  num_simd_lanes = sc_info.num_lanes

  match tpu_info.chip_version:
    case 6:
      target_bytes = (256 * 1024) * 0.9
    case 7:
      target_bytes = (512 * 1024) * 0.9
    case _:
      target_bytes = (128 * 1024) * 0.9

  base_bytes = num_simd_lanes * hidden_size * (32 // 8)
  num_cols = 1

  while pl.cdiv(base_bytes, num_cols * num_lanes) * num_lanes > target_bytes:
    num_cols += 1
  return pl.cdiv(hidden_size, (num_cols * num_lanes)) * num_lanes


@jax.jit
def ragged_gather(x: jax.Array, indices: jax.Array, start: jax.Array, end: jax.Array) -> jax.Array:
  """Perform gather on indices within dynamic array start and end."""

  assert x.ndim == 2, "Ragged gather only supports 2d inputs."
  assert indices.ndim == 1, "Ragged gather only supports 1d indices."

  if jnp.isscalar(start):
    start = start[None]
  if jnp.isscalar(end):
    end = end[None]

  dtype = x.dtype

  sc_info = pltpu.get_tpu_info().sparse_core
  if sc_info is None:
    # Sparse core is not available. Fallback to regular gather.
    return x[indices]

  hidden_size = x.shape[-1]
  out_size = indices.size

  num_simd_lanes = sc_info.num_lanes
  num_cores = sc_info.num_cores * sc_info.num_subcores
  block_size = num_simd_lanes * num_cores
  col_size = calculate_col_size(hidden_size)

  # Pad to align to the block size.
  out_pad_size = pl.cdiv(out_size, block_size) * block_size - out_size
  indices = jnp.pad(indices, ((0, out_pad_size)))

  aligned_hidden_size = pl.cdiv(hidden_size, col_size) * col_size

  vector_mesh = plsc.VectorSubcoreMesh(
      num_cores=sc_info.num_cores,
      num_subcores=sc_info.num_subcores,
      core_axis_name="core",
      subcore_axis_name="subcore",
  )
  return pl.kernel(  # pytype: disable=wrong-keyword-args
      functools.partial(
          main_kernel,
          core_axis_name=vector_mesh.core_axis_name,
          subcore_axis_name=vector_mesh.subcore_axis_name,
      ),
      compiler_params=pltpu.CompilerParams(
          use_tc_tiling_on_sc=True,
          disable_bounds_checks=True,
      ),
      mesh=vector_mesh,
      name="sc_ragged_gather",
      **{
          _OUT_KW: jax.ShapeDtypeStruct((out_size + out_pad_size, aligned_hidden_size), dtype),
          _SCRATCH_KW: [
              pltpu.VMEM((num_simd_lanes,), jnp.int32),
              pltpu.VMEM((num_simd_lanes,), jnp.int32),
              pltpu.VMEM((num_simd_lanes, col_size), jnp.uint32),
              pltpu.VMEM((num_simd_lanes,), jnp.int32),
              pltpu.SemaphoreType.DMA((2,)),
          ],
      },
  )(start, end, x, indices)[:out_size, :hidden_size]
