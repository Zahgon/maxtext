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

"""Qwen3 MaxText to vLLM weight converters."""

import functools
import gc
import logging

import jax
import jax.numpy as jnp
from jaxtyping import PyTree

from maxtext.integration.vllm.torchax_converter.base import BaseMaxTextToVLLMConverter


class Qwen3MaxTextToVLLMConverter(BaseMaxTextToVLLMConverter):
  """Qwen3-specific MaxText to vLLM converter."""

  def _convert_global(self, params):
    logging.info("_convert_global: embed_tokens...")
    self._to_embed_tokens(params)
    logging.info("_convert_global: final_norm...")
    self._to_final_norm(params)
    logging.info("_convert_global: lm_head...")
    self._to_lm_head(params)
    logging.info("_convert_global: done")

  def _convert_attn(self, params):
    logging.info("_convert_attn: pre_self_attention_layer_norm...")
    pre_ln = params["base"]["decoder"]["layers"]["pre_self_attention_layer_norm"]["scale"]
    convert_pre_ln = self._transpose_unstack(pre_ln)
    assert len(convert_pre_ln) == self.num_layers, f"Expected {self.num_layers} layers, got {len(convert_pre_ln)}"
    for i, layer in enumerate(convert_pre_ln):
      self.vllm_state[f"vllm_model.model.layers.{i}.input_layernorm.weight"] = layer
    del convert_pre_ln

    logging.info("_convert_attn: post_self_attention_layer_norm...")
    post_ln = params["base"]["decoder"]["layers"]["post_self_attention_layer_norm"]["scale"]
    converted_post_ln = self._transpose_unstack(post_ln)
    assert len(converted_post_ln) == self.num_layers, f"Expected {self.num_layers} layers, got {len(converted_post_ln)}"
    for i, layer in enumerate(converted_post_ln):
      self.vllm_state[f"vllm_model.model.layers.{i}.post_attention_layernorm.weight"] = layer
    del post_ln, converted_post_ln

    logging.info("_convert_attn: self_attention (qkv/o/norms)...")
    attn = params["base"]["decoder"]["layers"]["self_attention"]
    self_attn = self._to_attn(attn)
    for key, layers in self_attn.items():
      self.vllm_state.update({f"vllm_model.model.layers.{i}.{key}": layer for i, layer in enumerate(layers)})
    del attn, self_attn
    logging.info("_convert_attn: done")
    gc.collect()

  def _convert_moe(self, params):
    logging.info("_convert_moe: extracting moe_block...")
    moe = params["base"]["decoder"]["layers"]["moe_block"].to_pure_dict()
    prefix = "vllm_model.model.layers"

    logging.info("_convert_moe: gate weights...")
    self.vllm_state.update(
        {f"{prefix}.{i}.mlp.gate.weight": weight for i, weight in enumerate(self._to_mlp_gate(moe["gate"]["kernel"]))}
    )
    del moe["gate"]
    gc.collect()

    logging.info("_convert_moe: expert down (w2) weights...")
    self.vllm_state.update(
        {f"{prefix}.{i}.mlp.experts.w2_weight": weight for i, weight in enumerate(self._to_mlp_expert_down(moe["wo"]))}
    )
    del moe["wo"]
    gc.collect()

    logging.info("_convert_moe: expert gate+up (w13) weights (fuse_all jit+vmap)...")
    self._to_mlp_expert_gate_up(
        moe["wi_0"],
        moe["wi_1"],
        prefix,
        "mlp.experts.w13_weight",
    )
    del moe["wi_0"], moe["wi_1"], moe
    logging.info("_convert_moe: done")
    gc.collect()

  def _to_final_norm(self, params):
    self.vllm_state["vllm_model.model.norm.weight"] = jnp.array(params["base"]["decoder"]["decoder_norm"]["scale"])

  def _to_embed_tokens(self, params):
    self.vllm_state["vllm_model.model.embed_tokens.weight"] = jnp.array(params["base"]["token_embedder"]["embedding"])

  def _to_lm_head(self, params):
    self.vllm_state["vllm_model.lm_head.weight"] = self._transpose_2d(params["base"]["decoder"]["logits_dense"]["kernel"])

  def _to_attn(self, attn: PyTree) -> dict[str, jax.Array]:
    """Convert MaxText attention parameters into per-layer vLLM weights."""
    tp = min(self.vllm_tp, self.config.base_num_kv_heads)
    compute = self._make_attn_compute(tp)
    return compute(attn)

  @staticmethod
  @functools.lru_cache(maxsize=8)
  def _make_attn_compute(tp: int):
    """Build the cached JIT that packs QKV and output projections for a TP size."""


    return _compute

  def _to_mlp_gate(self, param):
    param = self._transpose_gate(param)
    return self._unstack_layer(param)

  def _to_mlp_expert_down(self, param):
    param = self._transpose_expert_down(param)
    param = jnp.transpose(param, (0, 1, 3, 2))
    return self._unstack_layer(param)

  def _to_mlp_expert_gate_up(self, wi_0, wi_1, layer_key_prefix, layer_key_suffix):
    """Fuse MoE gate and up projections into the vLLM `w13` layout."""
    fuse_all = self._make_fuse_all(self.vllm_tp)

    logging.info("_to_mlp_expert_gate_up: dispatching _fuse_all (single JIT+vmap)...")
    fused = fuse_all(wi_0, wi_1)
    logging.info(
        "_to_mlp_expert_gate_up: _fuse_all complete, shape=%s, unstacking layers...",
        fused.shape,
    )
    del wi_0, wi_1
    gc.collect()

    for i, layer_i in enumerate(jnp.unstack(fused, axis=0)):
      layer_i = jnp.transpose(layer_i, (0, 2, 1))
      self.vllm_state[f"{layer_key_prefix}.{i}.{layer_key_suffix}"] = layer_i
      if i % 8 == 7:
        gc.collect()
    del fused
    gc.collect()

  @staticmethod
  @functools.lru_cache(maxsize=8)
  def _make_fuse_all(tp: int):
    """Build the cached JIT that fuses all expert gate and up weights."""


    return _fuse_all

  @staticmethod
  @jax.jit
  def _unstack_layer(param):
    """Split a stacked layer tensor into a tuple of per-layer tensors."""
    return jnp.unstack(param, axis=0)

  @staticmethod
  @jax.jit
  def _transpose_unstack(x):
    return jnp.unstack(jnp.transpose(x, (1, 0)))

  @staticmethod
  @jax.jit
  def _transpose_2d(x):
    return jnp.transpose(x, (1, 0))

  @staticmethod
  @jax.jit
  def _transpose_gate(param):
    return jnp.transpose(param, (1, 2, 0))

  @staticmethod
  @jax.jit
  def _transpose_expert_down(param):
    return jnp.transpose(param, (1, 0, 3, 2))
