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

"""Common utilities for GMM kernels."""

import re

import jax
import jax.numpy as jnp




def tpu_kind() -> str:
  """Query identification string for the currently attached TPU."""
  pass


# Ex: VPU v5; TPU v5 lite; TPU7x
_TPU_KIND_PATTERN = re.compile(r"TPU(?: v)?(\d+)")


def tpu_generation() -> int:
  """Generation number of the currently attached TPU."""
  pass


def supports_bfloat16_matmul() -> bool:
  """Does the currently attached CPU support bfloat16 inputs?"""
  pass




def select_input_dtype(lhs: jnp.ndarray, rhs: jnp.ndarray) -> jnp.dtype:
  """A type to which both input should be adapted to before dot product."""
  pass
