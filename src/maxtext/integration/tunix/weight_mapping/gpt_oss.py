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

"""Mapping MaxText GPT-OSS (MoE) weights to vLLM/tpu-inference keys."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class GPT_OSS_VLLM_MAPPING:
  """
  Mapping definition from MaxText GPT-OSS (Scanned/Interleaved) to vLLM JAX NNX.
  Supports:
  - Modulo Interleaving (e.g., Block 0 -> Layers 0, 2, 4...)
  """

  @staticmethod
  def lora_to_hf_mappings():
    """Provides the mapping for LoRA (Low-Rank Adaptation) weights.
    Returns:
        None, as LoRA mappings are not defined for this model.
    """
    pass

  @staticmethod
  def to_hf_hook_fns():
    """Returns hook functions to fuse interleaved weights."""
    pass

  @staticmethod
  def to_hf_transpose_keys():
    """Returns keys that need to be transposed."""
    pass

  @staticmethod
  def to_hf_mapping(
      layer_cycle_interval: int = 2, total_num_layers: int = 36, interleave_style: str = "modulo"
  ) -> Dict[str, Tuple[str, Tuple[Optional[str], ...]]]:
    """Returns the weight mapping for the model.
    Args:
        layer_cycle_interval: The interval at which layers are cycled.
        total_num_layers: The total number of layers in the model.
        interleave_style: The style of interleaving used for the layers.
    Returns:
        A dictionary mapping MaxText parameter names to vLLM parameter names.
    """
    pass
