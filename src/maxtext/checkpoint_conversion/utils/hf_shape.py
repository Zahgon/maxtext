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

"""Hugging Face shape checkpoint conversion utils."""


def GEMMA3_HF_WEIGHTS_TO_SHAPE(config):
  """Generates a shape mapping for Hugging Face Gemma3 parameters.

  This function computes the expected shapes for all parameters in a Hugging
  Face Gemma3 model, including both the text and vision components. The shapes
  are derived from the provided model configuration.

  Args:
    config (dict): The Hugging Face model configuration dictionary. It must
      contain 'text_config' and 'vision_config' sub-dictionaries with all
      necessary architectural details (e.g., hidden_size, num_layers).

  Returns:
    dict: A dictionary where keys are Hugging Face parameter names (e.g.,
    'model.language_model.embed_tokens.weight') and values are lists of
    integers representing the tensor's shape.
  """
  pass


def GEMMA4_HF_WEIGHTS_TO_SHAPE(config):
  """Generates shape mapping for Hugging Face Gemma4 parameters.

  Handles both multimodal (with vision tower) and text-only variants, as well
  as MoE (26B) and dense (31B) text configurations. Shapes are per-layer aware:
  local (sliding) attention layers use head_dim, while global (full) attention
  layers use global_head_dim and num_global_key_value_heads.

  Args:
    config (dict): The Hugging Face model configuration dictionary. Must contain
      'text_config' with architectural details. May contain 'vision_config' for
      multimodal models.

  Returns:
    dict: A dictionary mapping Hugging Face parameter names to their shapes.
  """
  pass


def GEMMA4_SMALL_HF_WEIGHTS_TO_SHAPE(config):
  """Generates HF parameter shapes for Gemma 4 small (E2B / E4B).

  Differs from GEMMA4_HF_WEIGHTS_TO_SHAPE in that it:
    * derives global-vs-sliding from the per-model ``layer_types`` list
      (E2B has period-5, E4B has period-6),
    * emits the Per-Layer-Embedding parameters when ``hidden_size_per_layer_input`` > 0,
    * omits k_proj/v_proj/k_norm/v_norm shapes on KV-shared layers, and
    * doubles ``intermediate_size`` on shared layers when ``use_double_wide_mlp``
      is set (E2B).
  """
  pass


def GEMMA2_HF_WEIGHTS_TO_SHAPE(config):
  """Returns mapping between HuggingFace weights path and weights shape.

  Args:
      config (dict): Model configuration dictionary, defined in `model_configs.py`

  Returns:
      dict: A mapping where:
          - Keys are HuggingFace model parameter paths
          - Values are parameter shape as a list
  """
  pass


def DEEPSEEK_HF_WEIGHTS_TO_SHAPE(config):
  """Returns mapping between HuggingFace weights path and their shape derived from HF config.

  Args:
      config (dict): HF configuration dictionary
        e.g., https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/config.json

  Returns:
      dict: A mapping where:
          - Keys are HuggingFace model parameter paths
          - Values are parameter shape as a list

  To check expected mapping:
    from transformers import AutoModelForCausalLM
    model_name = "deepseek-ai/DeepSeek-V2-Lite"
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")
    for name, val in model.named_parameters():
      print(name, val.shape)
  """
  pass


def QWEN3_NEXT_HF_WEIGHTS_TO_SHAPE(config):
  """Returns mapping between HuggingFace Qwen3-Next weights path and their shape."""
  pass


def GPT_OSS_HF_WEIGHTS_TO_SHAPE(config):
  """Returns mapping between HuggingFace GptOss weights path and their shape."""
  pass


def QWEN_HF_WEIGHTS_TO_SHAPE(config):
  """Returns mapping between HuggingFace Qwen weights path and the HuggingFace weights shape.

  Args:
      config (dict): HF configuration dictionary (from Qwen3TextConfig.to_dict())
          e.g., https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json

  Returns:
      dict: A mapping where:
          - Keys are HuggingFace model parameter paths
          - Values are parameter shape as a list

  To check expected mapping:
    from transformers import AutoModelForCausalLM
    model_name = "Qwen/Qwen3-0.6B"
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")
    for name, val in model.named_parameters():
      print(name, val.shape)
  """
  pass


def LLAMA31_HF_WEIGHTS_TO_SHAPE(config):
  """Returns mapping between HuggingFace weights path and weights shape.

  Args:
      config (dict): Model configuration dictionary, defined in `model_configs.py`

  Returns:
      dict: A mapping where:
          - Keys are HuggingFace model parameter paths
          - Values are parameter shape as a List
  """
  pass


def MIXTRAL_HF_WEIGHTS_TO_SHAPE(config):
  """
  Returns a mapping of Hugging Face parameter names to their tensor shapes.

  Args:
      config (dict): The model configuration dictionary.

  Returns:
      A dictionary mapping Hugging Face parameter paths to their tensor shapes.
  """
  pass


# {maxtext model name: {hf weight name: hf shape}}
HF_SHAPE = {
    "gemma2-2b": GEMMA2_HF_WEIGHTS_TO_SHAPE,
    "gemma2-9b": GEMMA2_HF_WEIGHTS_TO_SHAPE,
    "gemma2-27b": GEMMA2_HF_WEIGHTS_TO_SHAPE,
    "gemma3-4b": GEMMA3_HF_WEIGHTS_TO_SHAPE,
    "gemma3-12b": GEMMA3_HF_WEIGHTS_TO_SHAPE,
    "gemma3-27b": GEMMA3_HF_WEIGHTS_TO_SHAPE,
    "gemma4-26b": GEMMA4_HF_WEIGHTS_TO_SHAPE,
    "gemma4-31b": GEMMA4_HF_WEIGHTS_TO_SHAPE,
    "gemma4-e2b": GEMMA4_SMALL_HF_WEIGHTS_TO_SHAPE,
    "gemma4-e4b": GEMMA4_SMALL_HF_WEIGHTS_TO_SHAPE,
    "qwen2.5-1.5b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "qwen2.5-7b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "qwen2.5-14b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "qwen3-0.6b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "qwen3-4b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "qwen3-4b-thinking-2507": QWEN_HF_WEIGHTS_TO_SHAPE,
    "qwen3-8b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "qwen3-14b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "qwen3-32b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "llama3.1-8b": LLAMA31_HF_WEIGHTS_TO_SHAPE,
    "llama3.1-8b-Instruct": LLAMA31_HF_WEIGHTS_TO_SHAPE,
    "llama3.1-70b": LLAMA31_HF_WEIGHTS_TO_SHAPE,
    "llama3.1-405b": LLAMA31_HF_WEIGHTS_TO_SHAPE,
    "qwen3-30b-a3b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "qwen3-235b-a22b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "qwen3-480b-a35b": QWEN_HF_WEIGHTS_TO_SHAPE,
    "deepseek2-16b": DEEPSEEK_HF_WEIGHTS_TO_SHAPE,
    "deepseek3-671b": DEEPSEEK_HF_WEIGHTS_TO_SHAPE,
    "deepseek3.2-671b": DEEPSEEK_HF_WEIGHTS_TO_SHAPE,
    "gpt-oss-20b": GPT_OSS_HF_WEIGHTS_TO_SHAPE,
    "gpt-oss-120b": GPT_OSS_HF_WEIGHTS_TO_SHAPE,
    "mixtral-8x7b": MIXTRAL_HF_WEIGHTS_TO_SHAPE,
    "mixtral-8x22b": MIXTRAL_HF_WEIGHTS_TO_SHAPE,
}
