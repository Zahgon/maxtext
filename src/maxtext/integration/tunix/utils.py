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

"""Utils for Tunix integration."""

import re

import maxtext.integration.tunix.weight_mapping as weight_mapping  # pylint: disable=consider-using-from-import
from maxtext.checkpoint_conversion.utils.param_mapping import PARAM_MAPPING
from maxtext.checkpoint_conversion.utils.param_mapping import VLLM_HOOK_FNS

STANDALONE_VLLM_WEIGHT_MAPPING = weight_mapping.StandaloneVllmWeightMapping()

# This static map provides the architectural knowledge (sharding) that is
# not present in the original HF mapping.
# Keys are the "generalized" MaxText names (e.g., base.decoder.layers...).
_SHARDING_KNOWLEDGE_MAP = {
    # Non-layer parameters
    "base.token_embedder.embedding": ("model", None),
    "base.decoder.decoder_norm.scale": (None,),
    "base.decoder.logits_dense.kernel": (None, "model"),
    # --- Attention (generic for scanned/unscanned) ---
    "base.decoder.layers.pre_self_attention_layer_norm.scale": (None, "layer"),
    "base.decoder.layers.self_attention.query.kernel": (
        None,
        "layer",
        "model",
        None,
    ),
    "base.decoder.layers.self_attention.key.kernel": (
        None,
        "layer",
        "model",
        None,
    ),
    "base.decoder.layers.self_attention.value.kernel": (
        None,
        "layer",
        "model",
        None,
    ),
    "base.decoder.layers.self_attention.query_norm.scale": (None, "layer"),
    "base.decoder.layers.self_attention.key_norm.scale": (None, "layer"),
    "base.decoder.layers.self_attention.out.kernel": (
        "model",
        "layer",
        None,
        None,
    ),
    "base.decoder.layers.post_self_attention_layer_norm.scale": (None, "layer"),
    # --- Dense MLP (generic for scanned/unscanned) ---
    "base.decoder.layers.mlp.wi_0.kernel": (None, "layer", "model"),
    "base.decoder.layers.mlp.wi_1.kernel": (None, "layer", "model"),
    "base.decoder.layers.mlp.wo.kernel": ("model", "layer", None),
    # --- MoE (generic for scanned/unscanned) ---
    "base.decoder.layers.moe_block.gate.kernel": (None, "layer", "model"),
    "base.decoder.layers.moe_block.wi_0": ("expert", "layer", None, "model"),
    "base.decoder.layers.moe_block.wi_1": ("expert", "layer", None, "model"),
    "base.decoder.layers.moe_block.wo": ("expert", "layer", "model", None),
    # --- Deepseek Attention ---
    "base.decoder.layers.self_attention.wq_a.kernel": (
        None,
        "layer",
        "model",
        None,
    ),
    "base.decoder.layers.self_attention.wq_b.kernel": (
        None,
        "layer",
        "model",
        None,
    ),
    "base.decoder.layers.self_attention.q_norm.scale": (None, "layer"),
    "base.decoder.layers.self_attention.wkv_a.kernel": (
        None,
        "layer",
        "model",
        None,
    ),
    "base.decoder.layers.self_attention.wkv_b.kernel": (
        None,
        "layer",
        "model",
        None,
    ),
    "base.decoder.layers.self_attention.kv_norm.scale": (None, "layer"),
    # --- Deepseek MoE ---
    "base.decoder.layers.moe_block.shared_experts.wi_0.kernel": (
        None,
        "layer",
        "model",
    ),
    "base.decoder.layers.moe_block.shared_experts.wi_1.kernel": (
        None,
        "layer",
        "model",
    ),
    "base.decoder.layers.moe_block.shared_experts.wo.kernel": (
        "model",
        "layer",
        None,
    ),
    "base.decoder.layers.moe_block.gate.bias": (None, "layer", "model"),
}


class VllmWeightMapping:
  """Mapping MaxText model weights to vLLM's model weights."""

  def __init__(self, model_name, config=None, use_standalone_mappings=False):
    self.model_name = model_name
    self.config = config
    self.use_standalone_mappings = use_standalone_mappings
    self._sharding_knowledge_map = _SHARDING_KNOWLEDGE_MAP

  def to_hf_mapping(self):
    """Returns a mapping from MaxText parameter names to HuggingFace parameter names."""
    pass


  def to_hf_hook_fns(self):
    """Returns a mapping from MaxText parameter names to transformation functions."""
    pass


  def _generalize_maxtext_key(self, maxtext_key):
    """Generalizes the MaxText key to a common vLLM format."""
    pass

  def _generalize_hf_value(self, hf_value):
    """Extracts and generalizes the Hugging Face name from the hf_value."""
    pass

  def _correct_hf_wildcard_name(self, wildcard_name):
    """Corrects the generated Hugging Face wildcard name."""
    pass

  def convert_hf_map_to_sharding_map(self, hf_mapping):
    """Converts a MaxText-to-HF name map into a generic MaxText-to-vLLM sharding map.

    Args:
      hf_mapping (dict): The output from QWEN3_MAXTEXT_TO_HF_PARAM_MAPPING.
        - Keys are MaxText param names (e.g., "params-decoder-layers...").
        - Values are HF param names (str) or lists of names (list).

    Returns:
      dict: A mapping from generalized MaxText names (e.g.,
        "base.decoder.layers.mlp.wi_0.kernel") to a tuple containing:
        (str: generalized HF/vLLM name, tuple: sharding specification).
    """
    pass
