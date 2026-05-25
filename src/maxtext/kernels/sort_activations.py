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

"""Token sorting for MoE layers."""

import functools

import jax
import jax.numpy as jnp
from maxtext.kernels import gather_reduce_sc


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def route(
    tokens: jax.Array,
    selected_experts: jax.Array,
    use_gather_mosaic_kernel: bool,
) -> jax.Array:
  """Route tokens to selected experts."""
  pass






route.defvjp(_route_fwd, _route_bwd)








unroute.defvjp(_unroute_fwd, _unroute_bwd)


def _route_impl(
    tokens: jax.Array,
    selected_experts: jax.Array,
    use_gather_mosaic_kernel: bool,
) -> jax.Array:
  """Gather `tokens` according to `selected_experts`."""
  pass


def _unroute_impl(
    tokens: jax.Array,
    selected_experts: jax.Array,
    use_gather_mosaic_kernel: bool,
) -> jax.Array:
  """Reverse the routing operation, restoring tokens to their original order."""
  pass


