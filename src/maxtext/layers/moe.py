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


"""MoE related Layers."""

import enum
import functools
import math
import random
from typing import Iterable, Optional, Tuple, Union

from aqt.jax.v2 import aqt_tensor as aqt
from flax import nnx
import jax
from jax import ad_checkpoint as adc
from jax.experimental import xla_metadata
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from maxtext.common import common_types as ctypes
from maxtext.common.common_types import ShardMode
from maxtext.kernels import megablox as mblx
from maxtext.layers import attentions, linears, nnx_wrappers, quantizations
from maxtext.layers.initializers import NdInitializer, default_bias_init, nd_dense_init, variable_to_logically_partitioned
from maxtext.kernels.ragged.ragged_sort import a2a_ragged_sort
from maxtext.kernels.ragged.ragged_sort import a2a_ragged_unsort
from maxtext.kernels.ragged.ragged_sort import ring_ragged_sort
from maxtext.kernels.ragged.ragged_sort import ring_ragged_unsort
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils.sharding import create_sharding, maybe_shard_with_logical, maybe_shard_with_pspec
from maxtext.utils.sharding import logical_to_mesh_axes
import numpy as np
import qwix
from qwix.contrib.sparsity import sparsity_module
import qwix.pallas as qpl
import tokamax

set_xla_metadata = xla_metadata.set_xla_metadata


DISPATCH = "dispatch"
COMBINE = "combine"


def _sort_activations(
    inputs: jax.Array,
    sort_indices: jax.Array,
    use_custom_vjp: bool,
) -> jax.Array:
  """Sort activations by `sort_indices`.

  If `use_custom_vjp=True`, then we use a custom backward pass that
  reverses the sort order. Specifically, this unsort operation is simply a sort
  with `jnp.argsort(sort_indices)` as the sort indices. This is only needed in
  the case where the compiler generates a less efficient backward pass op.

  Note that `use_custom_vjp=True` assumes that `sort_indices` is a permutation
  of `jnp.arange(inputs.shape[0])`.

  Args:
    inputs: `(tokens, ...)`-shaped array of input activations to sort.
    sort_indices: `(tokens,)`-shaped array containing the sort order.
    use_custom_vjp: Whether to use the explicit backward pass.

  Returns:
    `(tokens, ...)`-shaped array of input activations sorted by `sort_indices`.
  """
  assert inputs.shape[0] == sort_indices.shape[0]

  with jax.named_scope("sort_activations"):
    if use_custom_vjp:
      return _sort_activations_custom(inputs, sort_indices)
    return inputs[sort_indices, ...]


@jax.custom_vjp
def _sort_activations_custom(inputs: jax.Array, sort_indices: jax.Array) -> jax.Array:
  """Sort functions with custom vjp."""
  return inputs[sort_indices, ...]


def _sort_activations_custom_fwd(inputs: jax.Array, sort_indices: jax.Array) -> tuple[jax.Array, jax.Array]:
  """Forward pass of the custom vjp for `_sort_activations()`."""
  pass


def _sort_activations_custom_bwd(residuals: jax.Array, grads: jax.Array) -> tuple[jax.Array, None]:
  """Backward pass of the custom vjp for `_sort_activations()`."""
  pass


_sort_activations_custom.defvjp(_sort_activations_custom_fwd, _sort_activations_custom_bwd)


def get_batchsplit_init_kernel_axes():
  return (
      ("embed_moe", None, "expert_only"),
      ("embed_moe", "expert_only", None),
  )


def random_routing(rng_key, gate_logits, num_experts_per_tok):
  """Performs random routing of tokens to experts.

  Args:
    rng_key: A JAX PRNGKey for randomness.
    gate_logits: A JAX array of shape (batch_size, sequence_length, num_experts)
      representing the logits for each expert.
    num_experts_per_tok: The number of experts to select for each token.

  Returns:
    A tuple containing:
      - top_k_indices: JAX array of shape (batch_size, sequence_length,
      num_experts_per_tok)
                       representing the indices of the selected experts for each
                       token.
      - top_k_weights: JAX array of shape (batch_size, sequence_length,
      num_experts_per_tok)
                       representing the weights for the selected experts.
  """
  bs, seq_len, num_experts = gate_logits.shape
  selected_num = bs * seq_len * num_experts_per_tok
  # Directly generate random integers in the range [0, num_experts)
  top_k_indices = jax.random.randint(
      rng_key,
      shape=(selected_num,),
      minval=0,
      maxval=num_experts,
      dtype=jnp.int32,
  )
  top_k_indices = top_k_indices.reshape(bs, seq_len, num_experts_per_tok)
  top_k_weights = jnp.take_along_axis(gate_logits, top_k_indices, axis=-1)
  return top_k_weights, top_k_indices


def calculate_load_balance_updates(top_k_indices, num_experts, rate):
  """
  Computes a bias adjustment update based on expert load.
  Used in DeepSeek V3: https://arxiv.org/html/2412.19437v1.
  Implementation reference: https://arxiv.org/pdf/2408.15664.

  Args:
      top_k_indices: Shape (batch, sequence, top_k).
      num_experts: Total number of experts.
      rate: The update rate.

  Returns:
      update: The value to add to the expert bias. Shape (num_experts,).
  """
  flat_indices = top_k_indices.ravel()
  expert_counts = jnp.bincount(flat_indices, length=num_experts)

  total_tokens = flat_indices.size
  average_load = total_tokens / num_experts
  direction = jnp.sign(average_load - expert_counts)
  output = direction * rate
  return output


class GateLogit(nnx.Module):
  """A layer used to compute gate logits, allowing to return the pre bias values for DeepSeek routing."""

  def __init__(
      self,
      in_features_shape: Union[Iterable[int], int],
      out_features_shape: Union[Iterable[int], int],
      model_name: str,
      mesh: Mesh,
      rngs: nnx.Rngs,
      axis: Union[Iterable[int], int] = -1,
      weight_dtype: ctypes.DType = jnp.float32,
      dtype: ctypes.DType = jnp.float32,
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
      kernel_axes: Tuple[Optional[str], ...] = (),
      use_bias: bool = False,
      score_func: str = "",
      quant: Optional[quantizations.AqtQuantization] = None,
      shard_mode: ShardMode = ShardMode.AUTO,
      matmul_precision: str = "default",
  ):
    """Initializes the GateLogit module.

    Attributes:
      in_features_shape: The shape of the input features.
      out_features_shape: The shape of the output features, typically the number of experts.
      model_name: The name of the model.
      rngs: An `nnx.Rngs` object used for initializing parameters.
      axis: The axis or axes over transformation is applied.
      weight_dtype: The data type of the kernel weights.
      dtype: The data type for the computation.
      kernel_init: The initializer function for the kernel weight matrix.
      kernel_axes: A tuple of logical axis names for partitioning the kernel.
      use_bias: Whether to add learnable bias in gate logit scores. When enabled,
        this bias aids expert load balancing (like in DeepSeek V3), and is not
        part of the loss calculation.
      score_func: Scoring function for output normalization before applying bias.
      quant: The quantization configuration. If None, no quantization is applied.
      matmul_precision: The precision level for the matrix multiplication.
    """
    self.in_features_shape = linears.canonicalize_tuple(in_features_shape)
    self.out_features_shape = linears.canonicalize_tuple(out_features_shape)
    self.model_name = model_name
    self.mesh = mesh
    self.axis = linears.canonicalize_tuple(axis)
    self.weight_dtype = weight_dtype
    self.dtype = dtype
    self.kernel_init = kernel_init
    self.kernel_axes = kernel_axes
    self.use_bias = use_bias
    self.score_func = score_func
    self.quant = quant
    self.shard_mode = shard_mode
    self.matmul_precision = matmul_precision

    # Parameter initialization
    kernel_shape = self.in_features_shape + self.out_features_shape
    kernel_in_axis = np.arange(len(self.axis))
    kernel_out_axis = np.arange(len(self.axis), len(self.axis) + len(self.out_features_shape))

    if not quantizations.in_serve_mode(self.quant):
      self.kernel = nnx.Param(
          self.kernel_init(
              rngs.params(),
              kernel_shape,
              self.weight_dtype,
              kernel_in_axis,
              kernel_out_axis,
          ),
          out_sharding=self.kernel_axes,
      )

    if self.use_bias:
      bias_axes = self.kernel_axes[-len(self.out_features_shape) :]
      bias_shape = kernel_shape[-len(self.out_features_shape) :]
      self.bias = nnx.Param(
          default_bias_init(rngs.params(), bias_shape, self.weight_dtype),
          out_sharding=bias_axes,
      )
    else:
      self.bias = None

    if quant:
      dot_general_cls = quant.dot_general_cls(mesh_axes=kernel_axes)
      dot_general_linen = dot_general_cls()
      quant_dot_general = nnx_wrappers.ToNNX(dot_general_linen, rngs=rngs)
      self._quant_dot_general_name = f"{type(dot_general_linen).__name__}_0"
      setattr(self, self._quant_dot_general_name, quant_dot_general)
      dummy_inputs = jnp.zeros((1, *self.in_features_shape), dtype=self.dtype)
      self(dummy_inputs, _initializing=True)
    else:
      self._quant_dot_general_name = None


  def __call__(self, inputs: jax.Array, _initializing: bool = False) -> Tuple[jax.Array, Optional[jax.Array]]:
    inputs = jnp.asarray(inputs, self.dtype)
    norm_axis = linears.normalize_axes(self.axis, inputs.ndim)

    if quantizations.in_serve_mode(self.quant):
      kernel_shape = self.in_features_shape + self.out_features_shape
      kernel = jnp.zeros(kernel_shape, dtype=self.dtype)
    else:
      kernel = self.kernel[...]
    kernel = jnp.asarray(kernel, self.dtype)

    contract_ind = tuple(range(0, len(norm_axis)))
    output_sharding = (
        create_sharding(self.mesh, ("activation_batch", "activation_length", None))
        if self.shard_mode == ShardMode.EXPLICIT
        else None
    )
    output = linears._compute_dot_general_nnx(
        inputs,
        kernel,
        norm_axis,
        contract_ind,
        self.matmul_precision,
        self.quant_dot_general,
        _initializing,
        out_sharding=output_sharding,
    )
    pre_bias_logits = None

    if self.score_func:
      output = linears._convert_to_activation_function(self.score_func)(output)
      if self.model_name.startswith("deepseek3"):
        pre_bias_logits = output

    if self.use_bias:
      bias = jnp.asarray(self.bias[...], self.dtype)
      output += bias
    return output, pre_bias_logits


class RoutedMoE(nnx.Module):
  """Implements a routed MoE block."""

  def __init__(
      self,
      config: ctypes.Config,
      num_experts: int,
      num_experts_per_tok: int,
      mesh: jax.sharding.Mesh,
      kernel_init: attentions.NdInitializer,
      kernel_axes: Tuple[Optional[str], ...],
      rngs: nnx.Rngs,
      intermediate_dim: int = 2048,
      weight_dtype: ctypes.DType = jnp.float32,
      dtype: ctypes.DType = jnp.float32,
      quant: Optional[quantizations.AqtQuantization] = None,
  ):
    """Initializes the RoutedMoE module.

    Attributes:
      config: The main config setting.
      num_experts: Number of experts.
      num_experts_per_tok: Number of experts for each token.
      mesh: Mesh, device mesh.
      kernel_init: The initializer function for the kernel weight matrix.
      kernel_axes: A tuple of logical axis names for partitioning the kernel.
      rngs: An `nnx.Rngs` object used for initializing parameters.
      intermediate_dim: Intermediate dimension of MoE.
      weight_dtype: The data type of the kernel weights.
      dtype: The data type for the computation.
      quant: The quantization configuration. If None, no quantization is applied.
    """
    self.config = config
    self.num_experts = num_experts
    self.num_experts_per_tok = num_experts_per_tok
    self.mesh = mesh
    self.kernel_init = kernel_init
    self.kernel_axes = kernel_axes
    self.intermediate_dim = intermediate_dim
    self.weight_dtype = weight_dtype
    self.dtype = dtype
    self.quant = quant
    self.rngs = rngs

    self.moe_expert_input_dim = (
        self.config.emb_dim if self.config.moe_expert_input_dim <= 0 else self.config.moe_expert_input_dim
    )

    if self.config.shard_exp_on_fsdp:
      # special sharding for dsv3
      self.wi_kernel_axes = ("embed_moe", None, "mlp_moe")
      self.wo_kernel_axes = ("embed_moe", "mlp_moe", None)
    elif self.config.use_batch_split_schedule:
      self.wi_kernel_axes, self.wo_kernel_axes = get_batchsplit_init_kernel_axes()
    else:
      self.wi_kernel_axes = ("exp", "embed_moe", "mlp_moe")
      self.wo_kernel_axes = ("exp", "mlp_moe", "embed_moe")

    if self.config.attention == "vllm_rpa":
      # vLLM uses 'model' as the tensor parallelism axis name
      self._tensor_parallelism_name = ("model", "attn_dp")
    else:
      self._tensor_parallelism_name = "tensor"

    if self.config.attention == "vllm_rpa" and self.config.enable_dp_attention:
      self._expert_parallelism_name = "attn_dp_expert"
    elif self.config.custom_mesh_and_rule == ctypes.CustomRule.CP_AS_EP:
      # when custom mesh and rule is cp-as-ep, context axis is same with expert in MoE component
      self._expert_parallelism_name = ("context", "expert")
    else:
      self._expert_parallelism_name = "expert"

    self.gate = GateLogit(
        in_features_shape=self.moe_expert_input_dim,
        out_features_shape=self.num_experts,
        mesh=self.mesh,
        model_name=self.config.model_name,
        dtype=jnp.float32 if self.config.float32_gate_logits else self.dtype,
        weight_dtype=self.weight_dtype,
        quant=self.quant,
        kernel_init=self.kernel_init,
        kernel_axes=self.kernel_axes,
        use_bias=self.config.routed_bias,
        # tpu-inference applies the score function in the fused_moe_gmm kernel,
        # so we don't apply it here to avoid redundant computation.
        # See https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/layers/common/fused_moe_gmm.py#L58.
        score_func="" if self.config.attention == "vllm_rpa" else self.config.routed_score_func,
        matmul_precision=self.config.matmul_precision,
        shard_mode=config.shard_mode,
        rngs=self.rngs,
    )
    rule = qpl.get_current_rule("gmm")
    sparsity_rule = None
    if rule is not None:
      if not isinstance(rule, qwix.QtRule):
        raise ValueError("Expect a QtRule for quantized training.")
      if rule.additional_qt_config and "sparsity_rule" in rule.additional_qt_config:
        q_s_rule = rule.additional_qt_config["sparsity_rule"]
        if q_s_rule and q_s_rule.weight_sparsity_n and q_s_rule.weight_sparsity_m:
          sparsity_rule = q_s_rule

    if sparsity_rule is not None:
      self.wi_0_sparsity_module = sparsity_module.SparsityModule(
          shape=(self.num_experts, self.config.emb_dim, self.intermediate_dim),
          sharding_axes=self.wi_kernel_axes,
          sparsity_rule=sparsity_rule,
      )
      self.wi_1_sparsity_module = sparsity_module.SparsityModule(
          shape=(self.num_experts, self.config.emb_dim, self.intermediate_dim),
          sharding_axes=self.wi_kernel_axes,
          sparsity_rule=sparsity_rule,
      )
      self.wo_sparsity_module = sparsity_module.SparsityModule(
          shape=(self.num_experts, self.intermediate_dim, self.config.emb_dim),
          sharding_axes=self.wo_kernel_axes,
          sparsity_rule=sparsity_rule,
      )
    else:
      self.wi_0_sparsity_module = None
      self.wi_1_sparsity_module = None
      self.wo_sparsity_module = None

    # pylint: disable=protected-access
    self.activation_fn = linears._convert_to_activation_function(self.config.mlp_activations[0])

    kernel_in_axis = np.arange(1)
    kernel_out_axis = np.arange(1, 2)

    if quantizations.in_serve_mode(self.quant):
      # During aqt convert state we delete kernel weight from params to save
      # memory. Instead they are retrieved from the tensors stored in the 'aqt'
      # collection.
      self.wi_0 = jnp.zeros((num_experts, self.moe_expert_input_dim, intermediate_dim))
      self.wi_1 = jnp.zeros((num_experts, self.moe_expert_input_dim, intermediate_dim))
      self.wo = jnp.zeros((num_experts, intermediate_dim, self.moe_expert_input_dim))
    elif self.config.prefuse_moe_weights and self.config.attention == "vllm_rpa":
      # Pad model dimension in Fused MoE weight kernels for GMM_v2 execution.
      moe_intermediate_dim = (
          self.config.padded_base_moe_mlp_dim
          if self.config.padded_base_moe_mlp_dim is not None
          else self.intermediate_dim
      )
      self.wi = nnx.Param(
          self.kernel_init(
              self.rngs.params(),
              (num_experts, self.moe_expert_input_dim, moe_intermediate_dim * 2),
              weight_dtype,
              kernel_in_axis,
              kernel_out_axis,
          ),
          out_sharding=self.wi_kernel_axes,
      )
      self.wo = nnx.Param(
          self.kernel_init(
              self.rngs.params(),
              (self.num_experts, self.intermediate_dim, self.moe_expert_input_dim),
              self.weight_dtype,
              kernel_in_axis,
              kernel_out_axis,
          ),
          out_sharding=self.wo_kernel_axes,
      )
    else:
      # Pad model dimension in Unfused MoE weight kernels for GMM_v2 execution.
      moe_intermediate_dim = (
          self.config.padded_base_moe_mlp_dim
          if self.config.padded_base_moe_mlp_dim is not None
          else self.intermediate_dim
      )
      self.wi_0 = nnx.Param(
          self.kernel_init(
              self.rngs.params(),
              (num_experts, self.moe_expert_input_dim, moe_intermediate_dim),
              weight_dtype,
              kernel_in_axis,
              kernel_out_axis,
          ),
          out_sharding=self.wi_kernel_axes,
      )
      self.wi_1 = nnx.Param(
          self.kernel_init(
              self.rngs.params(),
              (num_experts, self.moe_expert_input_dim, moe_intermediate_dim),
              weight_dtype,
              kernel_in_axis,
              kernel_out_axis,
          ),
          out_sharding=self.wi_kernel_axes,
      )
      self.wo = nnx.Param(
          self.kernel_init(
              self.rngs.params(),
              (self.num_experts, self.intermediate_dim, self.moe_expert_input_dim),
              self.weight_dtype,
              kernel_in_axis,
              kernel_out_axis,
          ),
          out_sharding=self.wo_kernel_axes,
      )

    if self.config.mlp_bias:
      wi_bias_axes = ("exp", "activation_mlp")
      wo_bias_axes = ("exp", "activation_embed")
      wi_bias_shape = (self.num_experts, self.intermediate_dim)
      wo_bias_shape = (self.num_experts, self.moe_expert_input_dim)
      self.wi_0_bias = nnx.Param(
          default_bias_init(self.rngs.params(), wi_bias_shape, self.weight_dtype),
          out_sharding=wi_bias_axes,
      )
      self.wi_1_bias = nnx.Param(
          default_bias_init(self.rngs.params(), wi_bias_shape, self.weight_dtype),
          out_sharding=wi_bias_axes,
      )
      self.wo_bias = nnx.Param(
          default_bias_init(self.rngs.params(), wo_bias_shape, self.weight_dtype),
          out_sharding=wo_bias_axes,
      )
    else:
      self.wi_0_bias = None
      self.wi_1_bias = None
      self.wo_bias = None

    if self.config.decoder_block == ctypes.DecoderBlockType.GEMMA4:
      self.per_expert_scale = nnx.Param(
          jnp.ones((self.num_experts,), dtype=self.weight_dtype),
          out_sharding=("exp",),
      )
    else:
      self.per_expert_scale = None

    # Scale the output projection ahead of time during inference for higher generation throughput.
    if (
        self.per_expert_scale is not None
        and self.config.model_call_mode == "inference"
        and self.config.fuse_expert_scales
    ):
      self.wo.value = self.wo.value * self.per_expert_scale.value[:, None, None]


  def _logical_to_mesh_axes(self, logical_name):
    logical_rules = None if self.config.using_pipeline_parallelism else self.config.logical_axis_rules
    return logical_to_mesh_axes(logical_name, mesh=self.mesh, rules=logical_rules)

  def _maybe_shard_with_pspec(self, inputs, pspec: jax.sharding.PartitionSpec | None):
    return maybe_shard_with_pspec(
        inputs,
        pspec,
        mesh=self.mesh,
        shard_mode=self.config.shard_mode,
        debug_sharding=self.config.debug_sharding,
        extra_stack_level=1,
    )

  def get_expert_parallelism_size(self):
    # When expert parallelism has more than one physical axes, take product of their shapes
    if isinstance(self._expert_parallelism_name, tuple):
      return math.prod(self.mesh.shape.get(name, 1) for name in self._expert_parallelism_name)
    return self.mesh.shape.get(self._expert_parallelism_name, 1)




  def should_update_load_balance(self):
    """Determines if loss-free load balancing updates should be applied."""
    return self.config.routed_bias and self.config.routed_bias_update_rate > 0.0

  def get_topk(self, gate_logits, pre_bias_logits, rngs=None):
    """get topk."""
    # shape of top_k_weights & top_k_indices:
    # (batch, sequence, num_experts_per_tok).
    if self.config.use_random_routing:
      if rngs is None:
        raise ValueError("The random key cannot be None for random routing.")
      # Reuse the 'params' RNG stream to ensure random routing
      rng = rngs.params()
      top_k_weights, top_k_indices = random_routing(rng, gate_logits, self.num_experts_per_tok)
      return top_k_weights, top_k_indices

    if self.config.model_name.startswith("deepseek3"):
      top_k_weights, top_k_indices = self.deepseek_routing(gate_logits, pre_bias_logits)
    elif self.config.decoder_block == ctypes.DecoderBlockType.GEMMA4:
      router_probs = jax.nn.softmax(gate_logits.astype(jnp.float32), axis=-1)
      _, top_k_indices = jax.lax.top_k(gate_logits, self.num_experts_per_tok)
      top_k_weights = jnp.take_along_axis(router_probs, top_k_indices, axis=-1).astype(self.dtype)
    else:
      top_k_weights, top_k_indices = jax.lax.top_k(gate_logits, self.num_experts_per_tok)

    if self.config.decoder_block == ctypes.DecoderBlockType.DEEPSEEK:
      top_k_weights = self.deepseek_scale_weights(top_k_weights)
    elif self.config.decoder_block not in (ctypes.DecoderBlockType.LLAMA4, ctypes.DecoderBlockType.GEMMA4):
      top_k_weights = jax.nn.softmax(top_k_weights.astype(jnp.float32), axis=-1).astype(self.dtype)

    # Normalization of router weights (e.g. used by Qwen3, Gemma4).
    if self.config.norm_topk_prob:
      top_k_weights /= top_k_weights.sum(axis=-1, keepdims=True)

    return top_k_weights, top_k_indices

  def deepseek_scale_weights(self, weights):
    """Scales weights according to DeepSeek's v3 reference implementation."""
    # https://github.com/deepseek-ai/DeepSeek-V3/blob/2f7b80eecebf3d1c84da5a0d465f6639ea175012/inference/model.py#L592-L594.
    if self.config.routed_score_func == "sigmoid":
      weights /= weights.sum(-1, keepdims=True)
    weights *= self.config.routed_scaling_factor
    return weights

  def expert_group_mask(self, gate_logits: jax.Array) -> jax.Array:
    """Returns a mask that selects only the top-k groups of experts.

    Groups of experts are selected based on the sum of the top-2 expert scores
    for each group.

    Args:
      gate_logits: Array of shape `(batch, seq, num_experts)`.

    Returns:
      Array of shape `(batch, seq, num_experts)` that is 1 for experts in the
      top-k groups and 0 elsewhere.
    """
    # Find top groups based on each group's top-2 expert scores, where
    # `scores_grouped.shape =
    # (batch * seq, n_routing_groups, experts_per_group)`.
    scores_grouped = jnp.reshape(
        gate_logits,
        gate_logits.shape[:-1] + (self.config.n_routing_groups, -1),
    )
    top2_in_group_vals, _ = jax.lax.top_k(scores_grouped, k=2)
    group_scores = jnp.sum(jnp.astype(top2_in_group_vals, jnp.float32), axis=-1)
    _, group_idx = jax.lax.top_k(group_scores, k=self.config.topk_routing_group)

    # Mask selected groups so that only those experts are considered.
    group_mask = jax.nn.one_hot(group_idx, num_classes=self.config.n_routing_groups, dtype=jnp.float32)
    group_mask = jnp.sum(group_mask, axis=-2)

    # Apply masks and get top-k indices.
    score_mask_expanded = jnp.broadcast_to(
        group_mask[..., None],
        group_mask.shape + (self.num_experts // self.config.n_routing_groups,),
    )
    return jnp.reshape(
        score_mask_expanded,
        score_mask_expanded.shape[:-2] + (self.num_experts,),
    )

  def deepseek_routing(self, gate_logits: jax.Array, pre_bias_logits: jax.Array) -> tuple[jax.Array, jax.Array]:
    """DeepSeek routing logit.

    If the configuration does not specify routing groups (`n_routing_groups` is
    -1), we use a standard top-k routing mechanism. Otherwise, we force all
    selected experts to be from the a subset of the highest rated expert groups.

    The selection process uses post_bias logits, while the return weights use
    pre_bias logits.

    Args:
      gate_logits: Array of shape `(batch, seq, num_experts)`.
      pre_bias_logits: Array of shape `(batch, seq,num_experts)`.

    Returns:
      - top_k_weights: `(batch, seq, num_experts_per_tok)` array of weight values for
        each selected expert.
      - top_k_indices: `(batch, seq, num_experts_per_tok)` array of indices
        identifying the selected experts for each token.
    """
    expert_mask = 1 if self.config.n_routing_groups == -1 else self.expert_group_mask(gate_logits)
    _, top_k_indices = jax.lax.top_k(
        jnp.where(expert_mask > 0, gate_logits, -jnp.inf),
        k=self.num_experts_per_tok,
    )
    top_k_weights = jnp.take_along_axis(pre_bias_logits, top_k_indices, axis=-1)
    return top_k_weights, top_k_indices

  def apply_ffn_activation(self, layer_w0, layer_w1):
    """Applies FFN activation function."""
    pass

  def permute(self, inputs, gate_logits, pre_bias_logits, use_custom_sort_vjp=True, rngs=None, roll_to_expert_id=None):
    """Permute tokens to group by expert to fit gmm call."""
    # reshape inputs (batch, sequence, emb) to (batch * sequence, emb)
    inputs_shape = inputs.shape
    bsz_times_seq_len = inputs_shape[0] * inputs_shape[1]
    inputs_2d = jnp.reshape(inputs, (bsz_times_seq_len, inputs_shape[2]))
    weights, selected_experts = self.get_topk(gate_logits, pre_bias_logits, rngs)
    lb_loss = None
    if self.config.load_balance_loss_weight > 0.0:
      softmax_probs = jax.nn.softmax(gate_logits.astype(jnp.float32), axis=-1).astype(self.dtype)
      lb_loss = self.load_balance_loss(selected_experts, softmax_probs)

    if self.should_update_load_balance():
      bias_updates = calculate_load_balance_updates(
          selected_experts, self.config.num_experts, self.config.routed_bias_update_rate
      )
    else:
      bias_updates = None

    if self.config.decoder_block == ctypes.DecoderBlockType.LLAMA4:
      # weights will be of shape (batch_size, seq_len, num_experts_per_tok)
      router_scores = jax.nn.sigmoid(weights.astype(jnp.float32))  # weights are top_k_weights here
      # Squeeze router_scores to (batch_size * seq_len, num_experts_per_tok)
      inputs_2d = inputs_2d * router_scores.reshape(bsz_times_seq_len, -1)

    num_expert_parallelism = self.get_expert_parallelism_size()
    # The ragged-kernel path inside permute()/unpermute() is only correct for
    # the ring-of-experts strategy: each shard's output is masked to its own
    # [start, end) range within a globally-sorted layout. When ring of experts
    # is disabled, the buffer must instead carry all tokens for the subsequent
    # ragged-all-to-all, so we keep the standard argsort + sort path here and
    # let local_permute()/local_unpermute apply the ragged kernels on the
    # local prefix of valid rows.
    use_ragged_in_permute = self.config.use_ragged_sort and self.config.use_ring_of_experts
    if use_ragged_in_permute:
      topk_indices_2d = jnp.reshape(selected_experts, (bsz_times_seq_len, selected_experts.shape[2]))
      # roll_to_expert_id is not directly used in the kernel, ep axis id is directly called
      sorted_inputs, group_size, sorted_selected_experts = ring_ragged_sort(
          inputs_2d,
          topk_indices_2d,
          self.config.num_experts,
          self.num_experts_per_tok,
          self._expert_parallelism_name,
          num_expert_parallelism,
      )
    else:
      flatten_selected_experts = jnp.ravel(selected_experts)

      if roll_to_expert_id is not None:
        flatten_selected_experts = (flatten_selected_experts - roll_to_expert_id) % self.num_experts
      sorted_selected_experts = jnp.argsort(flatten_selected_experts)
      # sort inputs for number of selected experts
      replicated_inputs_2d = jnp.repeat(inputs_2d, self.num_experts_per_tok, axis=0)
      sorted_inputs = _sort_activations(replicated_inputs_2d, sorted_selected_experts, use_custom_sort_vjp).astype(
          self.dtype
      )
      group_size = jnp.bincount(flatten_selected_experts, length=self.num_experts)
      # Return the experts for each sorted input.
    expert_indices = jnp.arange(self.num_experts)
    sorted_experts = jnp.repeat(
        expert_indices,
        repeats=group_size,
        total_repeat_length=math.prod(selected_experts.shape),
    )
    return (
        sorted_inputs,
        sorted_selected_experts,
        weights,
        group_size,
        sorted_experts,
        lb_loss,
        bias_updates,
    )

  def unpermute(
      self,
      intermediate,
      sorted_selected_experts,
      weights,
      batch_size,
      sequence_length,
      use_custom_sort_vjp=True,
      group_sizes=None,
  ):
    """Unpermute tokens to original order and combine weights."""
    pass

  @staticmethod
  def local_permute(
      inputs,
      global_group_sizes,
      local_expert_size,
      shard_index,
      is_offset=False,
      global_sorted_experts=None,
      use_custom_sort_vjp=True,
      use_ragged_sort=False,
  ):
    """Permutes tokens locally within an expert shard.

    This function prepares the input tokens for processing by the experts
    located
    on the current shard. It groups the tokens by their assigned local expert
    index (0 to local_expert_size - 1).

    Args:
      inputs: The input data (tokens) assigned to the experts on this shard.
        Shape `[tokens, emb_dim]`.
      global_group_sizes: The count of tokens assignments for each global expert
        across all the batch shards. Shape `[num_batch_shards, num_experts].
      local_expert_size: The number of experts handled by the current shard.
      shard_index: The index of the current expert shard (0 to
        num_expert_parallelism - 1).
      is_offset: If True, assumes `inputs` are pre-sorted by global expert ID
        and selects the slice relevant to this shard's assigned experts. If
        False, assumes that `inputs` corresponding to the shard's experts start
        from the beginning of the tensor but need to be permuted by expert ID.
      global_sorted_experts: Global expert IDs for the `inputs` used when
        `is_offset` is True. Shape `[total_tokens_for_this_shard]`.
      use_custom_sort_vjp: Whether to use the explicit custom-VJP gather/scatter
        for the standard sort path. Ignored when `use_ragged_sort=True`.
      use_ragged_sort: When True, use the Pallas ragged-gather kernel
        (`a2a_ragged_sort`) to sort only the valid prefix of `inputs`. The
        ragged buffer can be much larger than the actually-routed token count,
        so this avoids touching the padded tail in both forward and backward.

    Returns:
      A tuple containing:
        sorted_inputs: Input data permuted local expert ID.
        sorted_indices: Indices used to permute the inputs.
        local_group_size: Number of tokens assigned to each local expert on this
          shard.
        sorted_experts_ids: expert ID corresponding to each token of the permuted
        inputs.
    """
    pass

  @staticmethod
  def get_all_to_all_params(
      all_shards_group_sizes,
      shard_id,
      num_expert_parallelism,
      is_batch_sharded=True,
  ):
    """Generates input offsets, send sizes, output offsets, and receive sizes used for ragged_all_to_all."""
    pass

  def transform_bias(self, experts_index, *biases):
    """Selects bias values for a variable number of bias tensors based on chosen experts."""
    pass

  @staticmethod
  def get_ragged_buffer_size(local_batch, ep_degree, global_experts, top_k, ragged_buffer_factor):
    """Calculates the token batch size of the ragged buffer.
    When explicitly setting ragged_buffer_factor>0, this is balanced_size * ragged_buffer_factor, which can drop tokens.
    Otherwise this will be worst case size to ensure no dropping.

    Inputs:
      local_batch: local token batch (batch*seq blown up by top_k) shard on this device (e.g. inside shard_map)
      ep_degree: degree of expert parallelism, generally equal to ici_expert_parallelism
      global_experts: unsharded expert count, e.g. 256 for deepseek
      top_k: aka num_experts_per_tok, 8 for deepseek.
      ragged_buffer_factor: When set > 0, the buffer is balanced_size * ragged_buffer_factor.
        The value 1.0 will be dropless only in the perfectly balanced case, else tokens will be dropped.
    Outputs:
      The ragged buffer's token batch size.
    """
    pass

  def sparse_matmul(
      self,
      inputs,
      gate_logits,
      pre_bias_logits,
      w0_kernel,
      w1_kernel,
      wo_kernel,
      w0_bias,
      w1_bias,
      wo_bias,
  ):
    """Perform sparse matrix multiplication of inputs and Experts."""
    pass

  def reshape_and_update_weights(self, weights, indices):
    """reshape and update weights."""
    pass


  def generate_masks_subgroup(self, top_k_indices, softmax_probs):
    """Subgroup mask generation for inference only."""
    pass

  def generate_masks(self, top_k_indices, softmax_probs):
    """Generate masks."""
    pass

  # See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details.
  def load_balance_loss(self, top_k_indices, logits) -> jax.Array:
    """Compute the load balance loss."""
    expert_mask = jax.nn.one_hot(top_k_indices, num_classes=self.num_experts, dtype=jnp.int32)
    summed_expert_mask = jnp.sum(expert_mask, axis=2)
    # Get fraction of tokens dispatched to each expert
    density = jnp.mean(summed_expert_mask, axis=1)
    # get fraction of probability allocated to each expert
    density_prob = jnp.mean(logits, axis=1)
    loss = jnp.mean(density * density_prob) * (self.num_experts**2) * self.config.load_balance_loss_weight
    return loss

  def get_einsum(
      self,
      rhs_mesh_axes: Tuple[Optional[str], ...] = (),
      einsum_name: str | None = None,
  ):
    """Get the Einstein summation."""
    pass

  def maybe_all_gather_kernel_weight_in_expert_parallelism(
      self, kernel: jax.Array, kernel_axes: Tuple[Optional[str], ...]
  ):
    """All-gather kernel weight in expert parallelism if needed."""
    pass

  def dense_matmul(
      self,
      inputs,
      gate_logits,
      pre_bias_logits,
      w0_kernel,
      w1_kernel,
      wo_kernel,
      w0_bias,
      w1_bias,
      wo_bias,
  ) -> tuple[jax.Array, Optional[jax.Array], Optional[jax.Array]]:
    """Dense matrix multiplication."""
    pass

  def fused_moe_matmul(
      self,
      inputs,
      gate_logits,
      wo_kernel,
      w0_kernel=None,
      w1_kernel=None,
      fused_kernel=None,
  ) -> tuple[jax.Array, None, None]:
    """Fused MoE via tpu_inference fused_moe_func (vllm_rpa path only).

    fused_moe_func handles routing, GMM, and weighted combination internally.
    It does not compute lb_loss or bias_updates (inference-only).
    """
    pass

  def retrieve_quantized_weight(
      self,
      inputs,
      gate_logits,
      pre_bias_logits,
      w0_kernel,
      w1_kernel,
      wo_kernel,
      w0_bias,
      w1_bias,
      wo_bias,
  ) -> tuple[aqt.QTensor, aqt.QTensor, aqt.QTensor]:
    """Retrieve quantized weights."""
    pass

  def __call__(
      self, inputs: jax.Array, gate_inputs: jax.Array | None = None, out_sharding: NamedSharding | None = None
  ) -> tuple[jax.Array, Optional[jax.Array], Optional[jax.Array]]:
    cfg = self.config
    inputs = inputs.astype(cfg.dtype)
    gate_dtype = jnp.float32 if cfg.float32_gate_logits else cfg.dtype
    routing_inputs = inputs if gate_inputs is None else gate_inputs.astype(gate_dtype)
    gate_logits, pre_bias_logits = self.gate(routing_inputs)

    wo_kernel = jnp.asarray(self.wo[...], self.dtype)

    fused_kernel = None
    w0_kernel = None
    w1_kernel = None
    if cfg.prefuse_moe_weights and cfg.attention == "vllm_rpa":
      fused_kernel = jnp.asarray(self.wi[...], self.dtype)
    else:
      w0_kernel = jnp.asarray(self.wi_0[...], self.dtype)
      w1_kernel = jnp.asarray(self.wi_1[...], self.dtype)

    # Only apply per expert scales if we have not fused with the out-projections at init time.
    if self.per_expert_scale is not None and cfg.model_call_mode != "inference" and not cfg.fuse_expert_scales:
      wo_kernel = wo_kernel * jnp.asarray(self.per_expert_scale[...], self.dtype)[:, None, None]

    if self.wi_0_sparsity_module is not None:
      _, w0_kernel = self.wi_0_sparsity_module(jnp.zeros_like(w0_kernel), w0_kernel)
      _, w1_kernel = self.wi_1_sparsity_module(jnp.zeros_like(w1_kernel), w1_kernel)
      _, wo_kernel = self.wo_sparsity_module(jnp.zeros_like(wo_kernel), wo_kernel)
    if cfg.mlp_bias:
      w0_bias = jnp.asarray(self.wi_0_bias[...], self.dtype)
      w1_bias = jnp.asarray(self.wi_1_bias[...], self.dtype)
      wo_bias = jnp.asarray(self.wo_bias[...], self.dtype)
    else:
      w0_bias, w1_bias, wo_bias = None, None, None

    # vllm_rpa codepath uses fused_moe_func from tpu_inference for optimized inference.
    if cfg.attention == "vllm_rpa":
      output, lb_loss, bias_updates = self.fused_moe_matmul(
          inputs, gate_logits, wo_kernel, w0_kernel=w0_kernel, w1_kernel=w1_kernel, fused_kernel=fused_kernel
      )
    elif cfg.sparse_matmul:
      if quantizations.in_serve_mode(self.quant):
        w0_kernel, w1_kernel, wo_kernel = self.retrieve_quantized_weight(
            inputs,
            gate_logits,
            pre_bias_logits,
            w0_kernel,
            w1_kernel,
            wo_kernel,
            w0_bias,
            w1_bias,
            wo_bias,
        )
      output, lb_loss, bias_updates = self.sparse_matmul(
          inputs, gate_logits, pre_bias_logits, w0_kernel, w1_kernel, wo_kernel, w0_bias, w1_bias, wo_bias
      )
    else:
      output, lb_loss, bias_updates = self.dense_matmul(
          inputs, gate_logits, pre_bias_logits, w0_kernel, w1_kernel, wo_kernel, w0_bias, w1_bias, wo_bias
      )
    return output, lb_loss, bias_updates


class RoutedAndSharedMoE(nnx.Module):
  """Implements a block which combines shared and routed experts."""

  def __init__(
      self,
      config: ctypes.Config,
      mesh: jax.sharding.Mesh,
      kernel_init: NdInitializer,
      kernel_axes: Tuple[Optional[str], ...],
      rngs: nnx.Rngs,
      weight_dtype: ctypes.DType = jnp.float32,
      dtype: ctypes.DType = jnp.float32,
      quant: Optional[quantizations.AqtQuantization] = None,
  ):
    """Initializes the RoutedAndSharedMoE module.

    Attributes:
      config: The main config setting.
      mesh: Mesh, device mesh.
      kernel_init: The initializer function for the kernel weight matrix.
      kernel_axes: A tuple of logical axis names for partitioning the kernel.
      rngs: An `nnx.Rngs` object used for initializing parameters.
      weight_dtype: The data type of the kernel weights.
      dtype: The data type for the computation.
      quant: The quantization configuration. If None, no quantization is applied.
    """
    self.config = config
    self.mesh = mesh
    self.kernel_init = kernel_init
    self.kernel_axes = kernel_axes
    self.weight_dtype = weight_dtype
    self.dtype = dtype
    self.quant = quant
    self.rngs = rngs
    self.moe_expert_input_dim = (
        self.config.emb_dim if self.config.moe_expert_input_dim <= 0 else self.config.moe_expert_input_dim
    )

    # NOTE: the name MoeBlock_0 is to ensure reverse compatibility with
    # existing checkpoints for routed experts.
    self.MoeBlock_0 = RoutedMoE(
        config=self.config,
        num_experts=self.config.num_experts,
        num_experts_per_tok=self.config.num_experts_per_tok,
        mesh=self.mesh,
        kernel_init=self.kernel_init,
        kernel_axes=("embed_moe", None),
        intermediate_dim=self.config.moe_mlp_dim,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        quant=self.quant,
        rngs=self.rngs,
    )

    shared_expert_mlp_dim = (
        self.config.mlp_dim if self.config.decoder_block == ctypes.DecoderBlockType.GEMMA4 else self.config.moe_mlp_dim
    )
    self.shared_experts = linears.MlpBlock(
        mesh=self.mesh,
        in_features=self.moe_expert_input_dim,
        intermediate_dim=self.config.shared_experts * shared_expert_mlp_dim,
        activations=self.config.mlp_activations,
        kernel_init=self.kernel_init,
        intermediate_dropout_rate=self.config.dropout_rate,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        config=self.config,
        quant=self.quant,
        rngs=self.rngs,
    )


  def __call__(
      self,
      inputs: jax.Array,
      original_inputs: jax.Array | None = None,
      gate_inputs: jax.Array | None = None,
      intermediate_sharding: NamedSharding | None = None,
      out_sharding: NamedSharding | None = None,
  ) -> tuple[jax.Array, Optional[jax.Array], Optional[jax.Array]]:
    routed_experts, load_balance_loss, moe_bias_updates = self.routed_moe(
        inputs, gate_inputs=gate_inputs, out_sharding=out_sharding
    )
    shared_experts = self.shared_experts(inputs, intermediate_sharding=intermediate_sharding, out_sharding=out_sharding)
    return routed_experts + shared_experts, load_balance_loss, moe_bias_updates


def get_gate_logit(
    inputs_shape: tuple[int, ...],
    out_features_shape: Union[Iterable[int], int],
    model_name: str,
    axis: Union[Iterable[int], int] = -1,
    weight_dtype: ctypes.DType = jnp.float32,
    dtype: ctypes.DType = jnp.float32,
    kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
    kernel_axes: Tuple[Optional[str], ...] = (),
    use_bias: bool = False,
    score_func: str = "",
    quant: Optional[quantizations.AqtQuantization] = None,
    matmul_precision: str = "default",
    name: Optional[str] = None,
):
  """Creates a GateLogit Linen module."""
  pass


def get_routed_moe(
    config: ctypes.Config,
    num_experts: int,
    num_experts_per_tok: int,
    mesh: jax.sharding.Mesh,
    kernel_init: NdInitializer,
    kernel_axes: Tuple[Optional[str], ...],
    intermediate_dim: int = 2048,
    weight_dtype: ctypes.DType = jnp.float32,
    dtype: ctypes.DType = jnp.float32,
    quant: Optional[quantizations.AqtQuantization] = None,
    name: Optional[str] = None,
):
  """Creates a RoutedMoE Linen module."""
  pass


def get_routed_and_shared_moe(
    config: ctypes.Config,
    mesh: jax.sharding.Mesh,
    kernel_init: NdInitializer,
    kernel_axes: Tuple[Optional[str], ...],
    weight_dtype: ctypes.DType = jnp.float32,
    dtype: ctypes.DType = jnp.float32,
    quant: Optional[quantizations.AqtQuantization] = None,
    name: Optional[str] = None,
):
  """Creates a RoutedAndSharedMoE Linen module."""
  pass
