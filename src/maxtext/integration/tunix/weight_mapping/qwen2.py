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

"""Defines the weight mapping from MaxText's Qwen2 model to a vLLM-compatible format.

This module provides the `QWEN2_VLLM_MAPPING` dataclass, which contains all the
necessary configurations to convert MaxText's Qwen2 model weights into a
format that can be loaded by HuggingFace's vLLM. This includes:
- A direct mapping of parameter names.
- Sharding specifications for distributed environments.
"""

from dataclasses import dataclass


@dataclass
class QWEN2_VLLM_MAPPING:
  """Mapping MaxText Qwen2 weights to vLLM's Qwen2 weights."""

  @staticmethod
  def to_hf_hook_fns():
    """Returns a dictionary of hook functions to be applied to MaxText weights.

    Returns:
      An empty dictionary, as no hook functions are needed for this mapping.
    """
    pass

  @staticmethod
  def to_hf_transpose_keys():
    """Returns a list of keys for weights that need to be transposed.

    Returns:
      An empty dictionary, as no keys require transposition for this mapping.
    """
    pass

  @staticmethod
  def lora_to_hf_mappings():
    """Provides the mapping for LoRA (Low-Rank Adaptation) weights.

    Returns:
      None, as LoRA mappings are not defined for this model.
    """
    pass

  @staticmethod
  def to_hf_mapping():
    """Mapping from MaxText model to HuggingFace vLLM model.

    Currently, the param mapping conforms to the Tunix API, which combines the
    param name & sharding in one dictionary.
    This is subject to change in the future where we can decouple the two.
    """
    pass
