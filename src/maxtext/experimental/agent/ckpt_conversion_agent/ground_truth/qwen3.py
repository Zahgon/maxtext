# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen3 ckpt conversation agent ground truth hook functions."""

import numpy as np


def QWEN3_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, scan_layers=False, saving_to_hf=False):
  """Creates parameter transformation functions for Qwen3.

  This function provides a dictionary of transformation functions (hooks) for
  converting Qwen3 model parameters between MaxText and Hugging Face formats.
  It handles embedding padding and kernel reshaping.

  Args:
    config (dict): Model configuration dictionary, including
      'num_hidden_layers' and optionally 'num_experts'.
    scan_layers (bool, optional): Whether the model uses scanned layers.
      Defaults to False.
    saving_to_hf (bool, optional): The direction of conversion. True for
      MaxText to Hugging Face, False for the reverse. Defaults to False.

  Returns:
    dict: A dictionary mapping MaxText parameter names to their corresponding
      transformation functions.
  """
  pass
