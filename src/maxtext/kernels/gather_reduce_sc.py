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

"""SparseCore gather-reduce kernel implementation.

This module contains a kernel implementation for performing a gather-reduce
operation on TPU SparseCore. It groups rows of an operand based on provided
indices, sums them up, and scatters the results.
"""

import array
import functools
from typing import Any

import jax
from jax import core
from jax.experimental import mosaic
from jax.experimental.mosaic.dialects import tpu
import jax.experimental.pallas.tpu as pltpu
from jax.interpreters import mlir
import jax.numpy as jnp
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import func
from jaxlib.mlir.dialects import memref
from jaxlib.mlir.dialects import scf
from jaxlib.mlir.dialects import vector


class VectorTypeHelper:
  """Helper to create VectorType with a specific element type."""

  def __init__(self, element_type_fn):
    self.element_type_fn = element_type_fn

  def __getitem__(self, shape):
    if isinstance(shape, int):
      shape = [shape]
    return ir.VectorType.get(shape, self.element_type_fn())


_I32 = VectorTypeHelper(functools.partial(ir.IntegerType.get_signless, 32))
_F32 = VectorTypeHelper(ir.F32Type.get)
_BF16 = VectorTypeHelper(ir.BF16Type.get)


@jax.jit(
    static_argnames=[
        "reduce_group_size",
        "single_sc",
        "col_chunk_size",
        "loop_unroll_factor_1",
        "loop_unroll_factor_2",
        "loop_unroll_factor_3",
        "loop_parallel_access_1",
        "loop_parallel_access_2",
        "loop_parallel_access_3",
        "topk_wgt_zero_nan",
    ],
)
def sc_gather_reduce(
    op: jax.Array,
    idx: jax.Array,
    topk_weights: jax.Array | None = None,
    *,
    reduce_group_size: int,
    single_sc: bool = False,
    col_chunk_size: int = int(3.5 * 1024),
    row_chunk_size: int = 16,  # writing back 2 rows given reduce size of 8
    loop_unroll_factor_1: int = 2,
    loop_unroll_factor_2: int = 2,
    loop_unroll_factor_3: int = 8,
    loop_parallel_access_1: bool = True,
    loop_parallel_access_2: bool = False,
    loop_parallel_access_3: bool = False,
    topk_wgt_zero_nan: bool = False,
) -> jax.Array:
  """Performs a gather-reduce operation on SparseCore.

  This kernel groups rows of the operand `op` based on `idx`, sums them up,
  and scatters the results. The gather and add operations are performed in fp32,
  and the results are written back in bf16.

  Equivalent jax numpy code:
  ```
    gathered = op[idx, :]
    if topk_wgt_local is not None:
      flat_weights = topk_wgt_local.flatten()
      gathered = gathered * flat_weights[:, None].astype(acc_dtype)
    gathered = jnp.reshape(gathered, (-1, reduce_group_size, op.shape[1]))
    output = jnp.sum(gathered.astype(acc_dtype), axis=1).astype(jnp.bfloat16)
    ```

  Args:
    op: The operand matrix in fp32 [B, K] to reduce.
    idx: The indices in int32[M,] guiding the reduction and scatter.
    topk_weights: Optional weights to apply to the gathered operands.
    reduce_group_size: The size of the groups to reduce.
    single_sc: Whether to use a single SparseCore.
    col_chunk_size: The size of column chunks to process.
    row_chunk_size: The size of row chunks for internal processing.
    loop_unroll_factor_1: Unroll factor for the main loop over column chunks.
    loop_unroll_factor_2: Unroll factor for the loop over row chunks in offset
      calculation.
    loop_unroll_factor_3: Unroll factor for the inner loop within offset
      calculation.
    loop_parallel_access_1: Enables parallel access for the main column chunk
      loop.
    loop_parallel_access_2: Enables parallel access for the row chunk loop in
      offset calculation.
    loop_parallel_access_3: Enables parallel access for the inner loop within
      offset calculation.
    topk_wgt_zero_nan: If true, treat zero topk_weights as indicators of NaN
      during multiplication, resulting in zero output.

  Returns:
    The result of operation, bf16 matrix [M/reduce_group_size, K].
  """
  pass
