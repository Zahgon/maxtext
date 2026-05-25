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

"""SparseCore gather-reduce kernel implementation using Pallas.

This module contains a Pallas kernel implementation for performing a
gather-reduce operation on TPU SparseCore. It groups rows of an operand
based on provided indices, sums them up, and scatters the results.
"""

import functools

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp


def sc_gather_reduce(
    op: jax.Array,
    idx: jax.Array,
    topk_weights: jax.Array | None = None,
    *,
    reduce_group_size: int,
    single_sc: bool = False,
    col_chunk_size: int = int(3.5 * 1024),
    row_chunk_size: int = 512,
    topk_wgt_zero_nan: bool = False,
) -> jax.Array:
  """Performs a gather-reduce operation on SparseCore.

  This kernel groups rows of the operand ``op`` based on ``idx``, sums them
  up, and scatters the results. The gather and add operations are performed
  in fp32, and the results are written back in bf16.

  Equivalent JAX code::

    gathered = op[idx, :]
    if topk_weights is not None:
      flat_weights = topk_weights.flatten()
      gathered = gathered * flat_weights[:, None].astype(jnp.float32)
    gathered = jnp.reshape(gathered, (-1, reduce_group_size, op.shape[1]))
    output = jnp.sum(gathered.astype(jnp.float32), axis=1).astype(jnp.bfloat16)

  Args:
    op: The operand matrix [B, K] in f32 or bf16 to gather from and reduce.
    idx: The indices [M,] in int32 guiding the gather.
    topk_weights: Optional weights [M // 128, 128] in bf16 to apply to the
      gathered rows before reduction.
    reduce_group_size: The number of gathered rows to sum per output row.
    single_sc: Whether to use a single SparseCore.
    col_chunk_size: The size of column chunks to process.
    row_chunk_size: The size of row chunks for internal processing. Must be ``2
      * reduce_group_size``.
    topk_wgt_zero_nan: If True, treat zero ``topk_weights`` as indicators of NaN
      during multiplication, resulting in zero output.

  Returns:
    The reduced result as a bf16 matrix [M / reduce_group_size, K].
  """
  pass
