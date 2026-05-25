"""
 Copyright 2025 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

"""Example code for agent, only gemma2 mappings"""

import numpy as np


def GEMMA2_MAXTEXT_TO_HF_PARAM_MAPPING(config, scan_layers=False):
  """Returns mapping between MaxText and HuggingFace Gemma2 weight paths.

  Args:
      config (dict): Model configuration dictionary containing at least
        'num_hidden_layers'.
      scan_layers (bool, optional): Whether the MaxText model uses layer
        scanning optimization. When True, decoder layers are stacked into a
        single tensor. Defaults to False.

  Returns:
      dict: A mapping where keys are MaxText parameter paths and values are
        either single strings (HF parameter path) for unscanned parameters or
        lists of strings (HF parameter paths) for stacked layers when
        `scan_layers=True`.

  Notes:
      - MaxText uses a paired layer approach where two HF decoder layers are
        treated as one MaxText decoder layer.
      - MaxText layer `i` corresponds to HF layers `2i` and `2i+1`.
      - Local components map to even-numbered HF decoder layers (0, 2, 4...).
      - Global components map to odd-numbered HF decoder layers (1, 3, 5...).
  """
  pass


def GEMMA2_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, scan_layers=False, saving_to_hf=False):
  """Creates parameter transformation functions for Gemma2 conversion.

  This function generates a mapping of transformation functions that handle the
  necessary conversions between MaxText and HuggingFace parameter formats for
  Gemma2, including operations like padding, reshaping, and scaling.

  Args:
      config (dict): Model configuration dictionary that must contain:
          - num_hidden_layers (int): Number of layers in the model.
          - head_dim (int): Dimension of attention heads.
          - hidden_size (int): Model's hidden dimension size.
      scan_layers (bool, optional): Controls the output format for layer
        parameters. True for batched, False for individual. Defaults to False.
      saving_to_hf (bool, optional): Determines the direction of transformation.
        True for MaxText to HuggingFace, False for the reverse. Defaults to
        False.

  Returns:
      dict: A mapping from MaxText parameter names to transformation functions.
        The value can be a single function or a list of functions to be
        applied sequentially.
  """
  pass
