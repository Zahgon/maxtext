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

"""Grouped matrix multiplication operations with custom VJPs."""

# pylint: disable=too-many-positional-arguments

import dataclasses
import functools
from typing import List, Literal, Tuple
import jax
import jax.numpy as jnp
from maxtext.kernels.megablox import backend
from maxtext.layers import quantizations
import qwix
import qwix.pallas as qpl
import tokamax


DRHS_RAGGED_DOT_DIM_NUMS = jax.lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(([0], [0]), ([], [])),
    lhs_ragged_dimensions=[0],
    rhs_group_dimensions=[],
)


def gmm(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    preferred_element_type: jnp.dtype = jnp.float32,
    tiling: tuple[int, int, int, int, int, int, int, int, int] = (
        128,
        128,
        128,
        128,
        128,
        128,
        128,
        128,
        128,
    ),
    group_offset: jnp.ndarray | None = None,
    existing_out: jnp.ndarray | None = None,
    transpose_rhs: bool = False,
    interpret: bool = False,
    lhs_quantize_dtype: Literal[jnp.int4, jnp.int8] | None = None,
    rhs_quantize_dtype: Literal[jnp.int4, jnp.int8] | None = None,
    use_qwix_quantization: bool = False,
    use_tokamax_backend: bool = False,
    weight_gather_axes: List[Tuple[str, int]] | None = None,
    # TODO(amandaliang): get rid of the qwix_rule in favor of Qwix's interception feature
    qwix_rule: qwix.QtRule | None = None,
    use_manual_quantization: bool = False,
):
  """Grouped matrix multiplication operation."""
  quantization_rule = None
  if use_qwix_quantization:
    # get_current_rule has to be called outside of the _gmm_fwd function.
    quantization_rule = qwix_rule if qwix_rule else qpl.get_current_rule("gmm")
    if quantization_rule and not isinstance(quantization_rule, qwix.QtRule):
      raise ValueError("Expect a QtRule for quantized training.")
  else:
    # Handcraft a rule that matches the AQT's behavior.
    if lhs_quantize_dtype or rhs_quantize_dtype:
      quantization_rule = qwix.QtRule(
          weight_qtype=rhs_quantize_dtype,
          weight_calibration_method="absmax",
          act_qtype=lhs_quantize_dtype,
          act_calibration_method="absmax",
      )

  gmm_fwd_bwd = lambda *args: _gmm_fwd(*args)[0]  # pylint: disable=C3001
  gmm_fwd_bwd = jax.custom_vjp(gmm_fwd_bwd, nondiff_argnums=(3, 4, 7, 8, 9, 10, 11, 12))
  gmm_fwd_bwd.defvjp(_gmm_fwd, functools.partial(_gmm_bwd, lhs.dtype, rhs.dtype))
  return gmm_fwd_bwd(
      lhs,
      rhs,
      group_sizes,
      preferred_element_type,
      tiling,
      group_offset,
      existing_out,
      transpose_rhs,
      interpret,
      quantization_rule,
      use_tokamax_backend,
      weight_gather_axes,
      use_manual_quantization,
  )


def _gmm_fwd(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    preferred_element_type: jnp.dtype = jnp.float32,
    tiling: tuple[int, int, int, int, int, int, int, int, int] = (
        128,
        128,
        128,
        128,
        128,
        128,
        128,
        128,
        128,
    ),
    group_offset: jnp.ndarray | None = None,
    existing_out: jnp.ndarray | None = None,
    transpose_rhs: bool = False,
    interpret: bool = False,
    quantization_rule: qwix.QtRule | None = None,
    use_tokamax_backend: bool = False,
    weight_gather_axes: List[Tuple[str, int]] | None = None,
    use_manual_quantization: bool = False,
) -> tuple[
    jnp.ndarray,
    tuple[
        jnp.ndarray | qpl.QArray,
        jnp.ndarray | qpl.QArray,
        jnp.ndarray,
        jnp.ndarray | None,
    ],
]:
  """Forward function for GMM VJP."""
  if quantization_rule:
    if quantization_rule.act_qtype and not isinstance(lhs, qpl.QArray):
      lhs = qpl.quantize(
          lhs,
          quantization_rule.act_qtype,
          channelwise_axes=[] if quantization_rule.disable_channelwise_axes else [0],
          calibration_method=quantization_rule.act_calibration_method,
      )
    if quantization_rule.weight_qtype and not isinstance(rhs, qpl.QArray):
      if not use_manual_quantization:
        rhs = qpl.quantize(
            rhs,
            quantization_rule.weight_qtype,
            # If only considering the fwd pass, we could also enable channelwise
            # axes for the group axis, i.e., [0, 1 or 2]. However, this makes the
            # bwd pass unable to reuse the scale easily.
            channelwise_axes=([] if quantization_rule.disable_channelwise_axes else ([1] if transpose_rhs else [2])),
            calibration_method=quantization_rule.weight_calibration_method,
        )
      else:
        rhs = quantizations.manual_quantize(
            rhs,
            quantization_rule.weight_calibration_method,
            quantization_rule.weight_qtype,
        )
      # QAG is only supported for following conditions
  if use_tokamax_backend:
    if quantization_rule and quantization_rule.bwd_qtype:
      if quantization_rule.weight_calibration_method.startswith("fixed") and isinstance(rhs, qpl.QArray):
        if weight_gather_axes:
          for axis_name, axis_idx in weight_gather_axes:
            rhs_qvalue = jax.lax.all_gather(rhs.qvalue, axis_name, axis=axis_idx, tiled=True)
            rhs = dataclasses.replace(rhs, qvalue=rhs_qvalue)
    # Handle transpose_rhs manually as ragged_dot assumes (G, K, N)
    if transpose_rhs:
      rhs = rhs.swapaxes(1, 2)

    if use_manual_quantization:
      out = tokamax.ragged_dot(
          lhs=lhs,
          rhs=rhs,
          group_sizes=group_sizes,
          precision=jax.lax.Precision.DEFAULT,
          preferred_element_type=preferred_element_type,
          group_offset=group_offset,
          implementation="mosaic",
          manual_axis_type=jax.sharding.ManualAxisType(varying=frozenset(["data", "fsdp", "expert"])),
      )
    else:
      out = tokamax.ragged_dot(
          lhs=lhs,
          rhs=rhs,
          group_sizes=group_sizes,
          precision=jax.lax.Precision.DEFAULT,
          preferred_element_type=preferred_element_type,
          group_offset=group_offset,
          implementation="mosaic",
      )
  else:
    out = backend.gmm(
        lhs,
        rhs,
        group_sizes,
        preferred_element_type,
        tiling[:3],
        group_offset,
        existing_out,
        transpose_rhs=transpose_rhs,
        interpret=interpret,
    )
  return out, (lhs, rhs, group_sizes, group_offset)


def _gmm_bwd(
    lhs_dtype: jax.typing.DTypeLike,
    rhs_dtype: jax.typing.DTypeLike,
    preferred_element_type: jnp.dtype,
    tiling: tuple[int, int, int, int, int, int, int, int, int],
    transpose_rhs: bool,
    interpret: bool,
    quantization_rule: qwix.QtRule | None,
    use_tokamax_backend: bool,
    weight_gather_axes: List[Tuple[str, int]] | None,
    use_manual_quantization: bool,
    residual: tuple[
        jnp.ndarray | qpl.QArray,
        jnp.ndarray | qpl.QArray,
        jnp.ndarray,
        jnp.ndarray | None,
    ],
    grad: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, None, None, jnp.ndarray]:
  """Backward function for throughput GMM VJP."""
  pass
