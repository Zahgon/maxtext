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

"""
Utility functions to support the HF checkpoint conversion and verification process in test_hf.py.
"""

import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils

import torch.nn.functional as F
import torch

from tabulate import tabulate


def convert_jax_weight_to_torch(weight: "jax.Array", dtype: None | str = None) -> torch.Tensor:
  expected_dtype = str(weight.dtype) if dtype is None else dtype
  expected_shape = weight.shape
  weight = multihost_utils.process_allgather(weight)
  weight = np.array(weight, dtype="float32")
  torch_dtype = getattr(torch, expected_dtype)
  torch_array = torch.from_numpy(weight).to(torch_dtype).reshape(expected_shape)
  return torch_array


def check_arrays_match(arrayA, arrayB, atol=0.01, rtol=1e-5):
  """
  Compare two sets of arrays for equality within the specified absolute and relative tolerances.

  This function handles both PyTorch tensors and JAX arrays, automatically
  converting between the two if necessary. If the arrays don't match within
  the specified tolerance, it prints detailed information about the mismatches.

  Args:
      arrayA (torch.Tensor | jax.Array): First set of arrays to compare
      arrayB (torch.Tensor | jax.Array): Second set of arrays to compare
      atol (float, optional): Absolute tolerance for comparison. Defaults to 0.01.
      rtol (float, optional): Relative tolerance for comparison. Defaults to 1e-5.

  Returns:
      bool: True if the arrays match within the specified tolerances, False otherwise.
  """
  pass


def check_predicted_tokens_match(logits_a, logits_b, tolerance=0.1):
  """Compares the top predicted tokens from each set of logits and ensures their
  disagreement rate doesn't exceed the tolerance threshold. Raises an AssertionError
  if the disagreement is too high.

  Args:
      logits_a (jax.Array | torch.Tensor | np.ndarray): First set of model output logits
      logits_b (jax.Array | torch.Tensor | np.ndarray): Second set of model output logits to compare against logits_a
      tolerance (float, optional): Maximum allowed fraction of token prediction disagreements,
          must be between 0.0 and 1.0. Defaults to 0.05 (5%).

  Examples:
      >>> logits1 = get_model_output(input1)
      >>> logits2 = get_model_output(input2)
      >>> check_predicted_tokens_match(logits1, logits2, tolerance=0.03)  # Allows 3% disagreement
  """
  pass


def get_logits_comparison_metrics(logitsA, logitsB):
  """
  Calculate various comparison metrics between two sets of logits.

  This function computes several metrics to compare the similarity and differences
  between two sets of logits, including KL divergence, absolute differences,
  and agreement in top-k predictions.

  Args:
      logitsA (jax.Array | torch.Tensor | np.ndarray): First set of logits to compare
      logitsB (jax.Array | torch.Tensor | np.ndarray): Second set of logits to compare

  Returns:
      dict: A dictionary containing the following metrics:
          - max_kl_div: Maximum KL divergence between probability distributions
          - abs_diff: Maximum absolute difference between probabilities
          - disagreement_top5: Proportion of positions where top-5 predictions differ
          - disagreement_top1: Proportion of positions where top-1 predictions differ

  Notes:
      The function also prints a formatted table of the metrics using tabulate.
  """
  pass
