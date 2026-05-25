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

"""Lllama3 ckpt conversation agent ground truth hook functions."""

import jax

import numpy as np


def LLAMA31_MAXTEXT_TO_HF_PARAM_HOOK_FN(config, scan_layers=False, saving_to_hf=False):
  """Creates parameter transformation functions for converting between MaxText and
  HuggingFace formats.
  This function generates a mapping of transformation functions that handle the necessary
  conversions between MaxText and HuggingFace parameter formats, including operations like
  reshaping.
  """
  pass
