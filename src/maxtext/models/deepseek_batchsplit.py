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


"""Alternative DeepSeek model definition with batch-split schedule.

The model logic and optimizations are very explicit in this implementation.
Weights are explicitly pre-fetched and gathered in the forward pass and gradients
are explicitly reduced and post-scattered in the backward pass. Optimization
barriers are used to enforce ordering of both large blocks of operations (e.g.
attention, dispatch, etc) and individual operations (e.g. AG+gather within
dispatch). In order to control remat, residuals from the forward pass are
explicitly stored and passed to the backward pass in a custom VJP over the
entire layer scan. The backward pass comprises of remat/bwd functions for each
forward pass function, with relevant residuals passed between them.
"""

import contextlib
import functools
import math
from typing import Sequence

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask

from maxtext.kernels import attention, sort_activations, megablox
from maxtext.layers import quantizations




def fetch_weights(params, dtype):
  """Fetches weights from params in the proper format for batch-split schedule."""
  pass


@jax.named_scope("deepseek_batchsplit_split")
def split(x, split_factor=2):
  """Splits the input into `split_factor` parts along the batch dimension."""
  if split_factor == 1:
    return [x]
  if x is None:
    return [None] * split_factor
  else:
    x = jnp.reshape(x, (-1, split_factor) + x.shape[1:])
    return [x[:, i, ...] for i in range(split_factor)]


@jax.named_scope("deepseek_batchsplit_merge")
def merge(x, split_factor=2):
  """Merges the input microbatches back into a single tensor."""
  if split_factor == 1:
    return x[0]
  x = jnp.stack(x, axis=1)
  return jnp.reshape(x, (-1,) + x.shape[2:])


def extract_layer_weights(all_weights, layer_idx, layer_axis):
  """Extracts the weights for given layer."""
  pass


def insert_layer_ws_grad(all_ws_grad, ws_grad, layer_idx, layer_axis):
  """Inserts the weight gradients for given layer."""
  pass


def gather_weights(weights, mesh):
  """all-gathers FSDP sharded weights."""
  pass


def reduce_scatter_ws_grad(ws_grad, mesh):
  """reduce-scatters weight gradients to FSDP sharding."""
  pass


def all_reduce_ws_grad_dcn(ws_grad, mesh):
  """all-reduces weight gradients across DCN axes."""
  pass


def init_splash_kernel(config):
  """Initializes the Splash kernel."""
  pass


def tpu_flash_attention(
    query,
    key,
    value,
    mesh,
    splash_kernel,
    activation_pspec,
):
  """TPU Flash Attention."""
  # Transpose to ('batch', 'heads', 'length', 'kv')
  query = jnp.transpose(query, axes=(0, 2, 1, 3))
  key = jnp.transpose(key, axes=(0, 2, 1, 3))
  value = jnp.transpose(value, axes=(0, 2, 1, 3))

  q_pspec = jax.sharding.PartitionSpec(*activation_pspec + (None,))
  kv_pspec = jax.sharding.PartitionSpec(*activation_pspec + (None,))
  lse_pspec = activation_pspec

  @functools.partial(
      jax.shard_map,
      mesh=mesh,
      in_specs=(
          q_pspec,
          kv_pspec,
          kv_pspec,
      ),
      out_specs=(
          q_pspec,
          lse_pspec,
      ),
      check_vma=False,
  )
  def wrap_flash_attention_manual(query, key, value):
    attention_output, logsumexp = jax.vmap(splash_kernel.manual_fwd, in_axes=(0, 0, 0, None, None), out_axes=(0, 0))(
        query,
        key,
        value,
        None,
        None,
    )
    return attention_output, logsumexp

  attention_output, logsumexp = wrap_flash_attention_manual(
      query,
      key,
      value,
  )
  return jnp.transpose(attention_output, axes=(0, 2, 1, 3)), logsumexp


def tpu_flash_attention_bwd(
    attention_out_grad,
    query,
    key,
    value,
    attention_output,
    logsumexp,
    mesh,
    splash_kernel,
    activation_pspec,
):
  """TPU Flash Attention backward."""
  pass


def batch_split_layer(
    inputs,
    params,
    positions,
    *,
    mesh,
    cfg,
):
  """Processes a single layer with batch-split schedule."""
  pass


def scan_batch_split_layers(
    inputs,
    params,
    positions,
    *,
    mesh,
    cfg,
    num_layers,
):
  """Scans the layers with batch-split schedule."""
  pass


def batch_split_schedule(
    inputs,
    weights,
    positions,
    *,
    mesh,
    cfg,
    splash_kernel,
    activation_pspec,
    pairwise_swap_and_negate_mask,
):
  """Applies the DeepSeek MoE layer with batch-split schedule."""
  pass


def batch_split_schedule_bwd(
    residuals,
    outputs_grad,
    weights,
    positions,
    *,
    mesh,
    cfg,
    splash_kernel,
    activation_pspec,
    pairwise_swap_and_negate_mask,
):
  """Performs the backward pass for a single layer."""
  pass


def staggered_call(fn, xs):
  """Calls a function in a staggered manner while accumulating residuals."""
  pass


def dot(x, y, axes=1):
  return jnp.tensordot(x, y, axes=axes)


def mla_with_norms(
    inputs,
    weights,
    yarn_freqs,
    *,
    mesh,
    config,
    splash_kernel,
    normalization_layer_epsilon,
    kv_lora_rank,
    qk_nope_head_dim,
    qk_rope_head_dim,
    num_query_heads,
    max_position_embeddings,
    original_max_position_embeddings,
    rope_factor,
    mscale,
    pairwise_swap_and_negate_mask,
    dtype,
    activation_pspec,
):
  """Performs MLA with pre-normalization."""
  pass


def mla_with_norms_remat(
    residuals,
    weights,
    yarn_freqs,
    *,
    mesh,
    config,
    splash_kernel,
    normalization_layer_epsilon,
    kv_lora_rank,
    qk_nope_head_dim,
    qk_rope_head_dim,
    num_query_heads,
    max_position_embeddings,
    original_max_position_embeddings,
    rope_factor,
    mscale,
    pairwise_swap_and_negate_mask,
    dtype,
    activation_pspec,
):
  """Performs remat for the mla_with_norms function."""
  pass


def mla_with_norms_bwd(
    outputs_grad,
    bwds,
):
  """Performs the backward pass for the mla_with_norms function."""
  pass


def mla(
    inputs,
    yarn_freqs,
    weights,
    *,
    epsilon,
    kv_lora_rank,
    kv_norm_epsilon,
    qk_nope_head_dim,
    qk_rope_head_dim,
    num_query_heads,
    max_position_embeddings,
    original_max_position_embeddings,
    rope_factor,
    mscale,
    config,
    splash_kernel,
    pairwise_swap_and_negate_mask,
    dtype,
    mesh,
    activation_pspec,
):
  """Performs MLA."""
  (
      wq_a_weights,
      wq_b_weights,
      q_norm_scale_weights,
      wkv_a_weights,
      wkv_b_weights,
      kv_norm_scale_weights,
      out_weights,
  ) = weights
  query = query_projection(
      inputs,
      yarn_freqs,
      wq_a_weights,
      wq_b_weights,
      q_norm_scale_weights,
      epsilon=epsilon,
      qk_rope_head_dim=qk_rope_head_dim,
      max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings,
      rope_factor=rope_factor,
      pairwise_swap_and_negate_mask=pairwise_swap_and_negate_mask,
      dtype=dtype,
      qk_nope_head_dim=qk_nope_head_dim,
      mscale=mscale,
      config=config,
      mesh=mesh,
      activation_pspec=activation_pspec,
  )
  key, value = kv_projection(
      inputs,
      yarn_freqs,
      wkv_a_weights,
      wkv_b_weights,
      kv_norm_scale_weights,
      kv_lora_rank=kv_lora_rank,
      kv_norm_epsilon=kv_norm_epsilon,
      pairwise_swap_and_negate_mask=pairwise_swap_and_negate_mask,
      dtype=dtype,
      qk_nope_head_dim=qk_nope_head_dim,
      num_query_heads=num_query_heads,
      config=config,
      mesh=mesh,
      activation_pspec=activation_pspec,
  )
  attn_out, lse = tpu_flash_attention(
      query,
      key,
      value,
      mesh=mesh,
      splash_kernel=splash_kernel,
      activation_pspec=activation_pspec,
  )
  out = dot(attn_out, out_weights, axes=2)
  return out, {"attn_out": attn_out, "lse": lse}


def mla_remat(
    residuals,
    yarn_freqs,
    weights,
    *,
    epsilon,
    kv_lora_rank,
    kv_norm_epsilon,
    qk_nope_head_dim,
    qk_rope_head_dim,
    num_query_heads,
    max_position_embeddings,
    original_max_position_embeddings,
    rope_factor,
    mscale,
    config,
    splash_kernel,
    pairwise_swap_and_negate_mask,
    dtype,
    mesh,
    activation_pspec,
):
  """Performs remat for the mla function."""
  pass


def mla_bwd(
    out_grad,
    bwds,
):
  """Performs the backward pass for the mla function."""
  pass


def query_projection(
    inputs_q,
    yarn_freqs,
    wq_a_weights,
    wq_b_weights,
    q_norm_scale_weights,
    *,
    epsilon,
    qk_nope_head_dim,
    qk_rope_head_dim,
    max_position_embeddings,
    original_max_position_embeddings,
    rope_factor,
    pairwise_swap_and_negate_mask,
    dtype,
    mscale,
    config,
    mesh,
    activation_pspec,
):
  """Performs query projection."""
  # Set softmax scaling.
  qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
  softmax_scale = qk_head_dim**-0.5
  if max_position_embeddings > original_max_position_embeddings:
    m = 0.1 * mscale * math.log(rope_factor) + 1.0
    softmax_scale = softmax_scale * m * m

  # LoRA path
  low_rank_q = dot(inputs_q, wq_a_weights)
  low_rank_q = rms_norm(
      low_rank_q,
      q_norm_scale_weights,
      epsilon=epsilon,
      dtype=dtype,
      out_sharding=jax.sharding.NamedSharding(mesh, activation_pspec),
  )
  q = dot(low_rank_q, wq_b_weights)

  # Split into non-positional and rotary parts.
  q_nope, q_pe = jnp.split(q, [qk_nope_head_dim], axis=-1)
  q_pe = yarn(
      q_pe,
      yarn_freqs,
      pairwise_swap_and_negate_mask=pairwise_swap_and_negate_mask,
      fprop_dtype=dtype,
  )
  query = jnp.concatenate([q_nope, q_pe], axis=-1) * softmax_scale
  return query


def kv_projection(
    inputs,
    yarn_freqs,
    wkv_a_weights,
    wkv_b_weights,
    kv_norm_scale_weights,
    *,
    kv_lora_rank,
    kv_norm_epsilon,
    pairwise_swap_and_negate_mask,
    dtype,
    qk_nope_head_dim,
    num_query_heads,
    config,
    mesh,
    activation_pspec,
):
  """Performs KV projection."""
  low_rank = dot(inputs, wkv_a_weights)
  low_rank_main, low_rank_rope = jnp.split(low_rank, [kv_lora_rank], axis=-1)
  low_rank_main = rms_norm(
      low_rank_main,
      kv_norm_scale_weights,
      epsilon=kv_norm_epsilon,
      dtype=dtype,
      out_sharding=jax.sharding.NamedSharding(mesh, activation_pspec),
  )
  key_rope = jnp.expand_dims(low_rank_rope, axis=2)
  key_rope = yarn(
      key_rope,
      yarn_freqs,
      pairwise_swap_and_negate_mask=pairwise_swap_and_negate_mask,
      fprop_dtype=dtype,
  )

  return get_key_value(
      low_rank_main,
      key_rope,
      wkv_b_weights,
      qk_nope_head_dim=qk_nope_head_dim,
      num_query_heads=num_query_heads,
  )


def get_key_value(low_rank_main, key_rope, wkv_b_weights, *, qk_nope_head_dim, num_query_heads):
  """Gets key and value from compressed KV latent vector and key rope."""
  kv_out = dot(low_rank_main, wkv_b_weights)

  # Split kv_out into key_nope and value parts.
  key_nope, value = jnp.split(kv_out, [qk_nope_head_dim], axis=-1)
  key_rope = jnp.broadcast_to(
      key_rope,
      (
          key_nope.shape[0],
          key_nope.shape[1],
          num_query_heads,
          key_rope.shape[3],
      ),
  )

  key = jnp.concatenate([key_nope, key_rope], axis=-1)

  return key, value


def rms_norm(x, scale, *, epsilon, dtype, out_sharding=None):
  """RMS normalization."""
  x = jnp.asarray(x, jnp.float32)
  mean2 = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
  y = jnp.asarray(x * jax.lax.rsqrt(mean2 + epsilon), dtype)
  return jnp.einsum("i...k,...k->i...k", y, scale, out_sharding=out_sharding)


def initialize_yarn_mask(embedding_dims):
  """Initializes YaRN mask."""
  pass


def initialize_yarn_freqs(
    positions,
    embedding_dims,
    rope_theta,
    max_position_embeddings,
    original_max_position_embeddings,
    beta_fast,
    beta_slow,
    rope_factor,
    mesh,
    activation_pspec,
):
  """Initializes YaRN frequencies."""
  pass


def yarn(
    inputs,
    freqs,
    *,
    pairwise_swap_and_negate_mask,
    fprop_dtype,
):
  """Performs YaRN rotary embedding."""
  # inputs @ mask: [B, S, N, embedding_dims] @ [embedding_dims, embedding_dims] -> [B, S, N, embedding_dims]
  output = inputs * jnp.cos(freqs) + jnp.matmul(inputs, pairwise_swap_and_negate_mask) * jnp.sin(freqs)
  return output.astype(fprop_dtype)


def shared_expert_and_route(
    inputs,
    post_attn_scale,
    shared_w0,
    shared_w1,
    shared_wo,
    gate_kernel,
    gate_bias,
    *,
    num_experts,
    num_experts_per_tok,
    routed_scaling_factor,
    n_routing_groups,
    topk_routing_group,
    top_k_in_group,
    expert_axis_name,
    use_gather_mosaic_kernel,
    config,
    normalization_layer_epsilon,
    dtype,
):
  """Computes the shared expert and routes the activations."""
  pass




# NOTE: Consider deduplicating this logic which is also in `moe.py`.
def expert_group_mask(gate_logits, *, n_routing_groups, topk_routing_group, top_k_in_group):
  """Computes expert group mask for node-limited routing."""
  num_experts = gate_logits.shape[-1]
  # Find top groups based on each group's top-2 expert scores, where
  # `scores_grouped.shape =
  # (batch * seq, n_routing_groups, experts_per_group)`.
  scores_grouped = jnp.reshape(
      gate_logits,
      gate_logits.shape[:-1] + (n_routing_groups, -1),
  )
  top2_in_group_vals, _ = jax.lax.top_k(scores_grouped, k=top_k_in_group)
  group_scores = jnp.sum(jnp.astype(top2_in_group_vals, jnp.float32), axis=-1)
  _, group_idx = jax.lax.top_k(group_scores, k=topk_routing_group)

  # Mask selected groups so that only those experts are considered.
  group_mask = jax.nn.one_hot(group_idx, num_classes=n_routing_groups, dtype=jnp.float32)
  group_mask = jnp.sum(group_mask, axis=-2)

  # Apply masks and get top-k indices.
  score_mask_expanded = jnp.broadcast_to(
      group_mask[..., None],
      group_mask.shape + (num_experts // n_routing_groups,),
  )
  return jnp.reshape(
      score_mask_expanded,
      score_mask_expanded.shape[:-2] + (num_experts,),
  )


def expert_indices_and_weights(
    gate_logits: jax.Array,
    pre_bias_logits: jax.Array,
    num_experts_per_tok: int,
    routed_scaling_factor: float,
    n_routing_groups: int,
    topk_routing_group: int,
    top_k_in_group: int,
) -> tuple[jax.Array, jax.Array]:
  """Computes expert indices for each token and their corresponding weights."""
  pass


def expert_selection(
    x,
    routing_kernel,
    routing_bias,
    *,
    num_experts: int,
    num_experts_per_tok: int,
    routed_scaling_factor: float,
    n_routing_groups: int,
    topk_routing_group: int,
    top_k_in_group: int,
):
  """Selects experts for each token and calculates group sizes for each expert."""
  pass


def route(
    x,
    selected_experts,
    weights,
    group_sizes,
    *,
    expert_axis_name,
    use_gather_mosaic_kernel,
):
  """All-gather tokens and then perform local routing."""
  pass


def unroute(
    x,
    selected_experts,
    *,
    expert_axis_name,
    use_gather_mosaic_kernel,
):
  """Undo `route()`."""
  pass




def route_impl_fwd(x, selected_experts, expert_axis_name, use_gather_mosaic_kernel):
  """Routes the activations and all-gathers across the expert axis."""
  pass






def unroute_impl_fwd(x, selected_experts, expert_axis_name, use_gather_mosaic_kernel):
  """Unroutes the activations and reduce-scatters across the expert axis."""
  pass




route_impl.defvjp(route_impl_fwd, route_impl_bwd)
unroute_impl.defvjp(unroute_impl_fwd, unroute_impl_bwd)


def gmm(
    inputs,
    kernel,
    group_sizes,
    preferred_element_type,
    config,
):
  """Performs a Grouped Matrix Multiplication (GMM).

  This function can use either a quantized Megablox kernel or a standard
  jax.lax.ragged_dot for the GMM operation, based on the configuration.

  Args:
    inputs: The left-hand side operand of the GMM.
    kernel: The right-hand side operand (kernel) of the GMM.
    group_sizes: An array indicating the size of each group.
    preferred_element_type: The preferred element type for the computation.
    config: Configuration object containing model settings, including
      `use_qwix_quantization` and `merge_gating_gmm`.

  Returns:
    The result of the grouped matrix multiplication.
  """
  if config.quantization:
    output = megablox.gmm(
        lhs=inputs,
        rhs=kernel,
        group_sizes=group_sizes,
        preferred_element_type=preferred_element_type,
        use_qwix_quantization=True,
        use_tokamax_backend=True,
        qwix_rule=quantizations.get_fp8_full_qwix_rule_w_sparsity(config)[0],
        use_manual_quantization=True,
    )
  else:
    output = jax.lax.ragged_dot(
        lhs=inputs,
        rhs=kernel,
        group_sizes=group_sizes,
        precision=jax.lax.Precision.DEFAULT,
        preferred_element_type=preferred_element_type,
    )
  return output


def compute_gating(x, w0, w1, group_sizes, *, dtype, config):
  """Computes the gating GMMs."""
  pass


def compute_linear(layer_w0, layer_w1, wo, group_sizes, weights, *, dtype, config):
  """Combines the outputs of the gating GMMs and computes the final GMM."""
  pass


def route_compute_unroute(
    xs,
    weights,
    *,
    num_experts,
    num_experts_per_tok,
    routed_scaling_factor,
    n_routing_groups,
    topk_routing_group,
    top_k_in_group,
    expert_axis_name,
    use_gather_mosaic_kernel,
    normalization_layer_epsilon,
    dtype,
    config,
):
  """Routes, processes, and unroutes activations."""
  pass


def unroute_ubatch_shard_mapped(
    moe_inputs,
    routed_expert_out,
    shared_expert_out,
    selected_experts,
    *,
    expert_axis_name,
    use_gather_mosaic_kernel,
    target_length,
    mesh,
    activation_pspec,
):
  """Performs the unroute operation for a single microbatch in a shard map."""
  pass


def unroute_ubatch_fn(
    moe_inputs,
    routed_expert_out,
    shared_expert_out,
    selected_experts,
    *,
    expert_axis_name,
    use_gather_mosaic_kernel,
    target_length,
):
  """Performs the unroute operation for a single microbatch."""
  pass


def unroute_ubatch_remat_and_bwd_shard_mapped(
    selected_experts,
    outputs_grad,
    *,
    expert_axis_name,
    use_gather_mosaic_kernel,
    mesh,
    activation_pspec,
):
  """Performs remat and backward pass for unroute_ubatch in a shard map."""
  pass








def route_compute_unroute_bwd(
    residuals,
    outputs_grad,
    weights,
    *,
    num_experts,
    num_experts_per_tok,
    routed_scaling_factor,
    n_routing_groups,
    topk_routing_group,
    top_k_in_group,
    expert_axis_name,
    use_gather_mosaic_kernel,
    normalization_layer_epsilon,
    dtype,
    config,
):
  """Performs the backward pass for route_compute_unroute."""
  pass


def moe(
    xs,
    weights,
    *,
    mesh,
    num_experts,
    num_experts_per_tok,
    routed_scaling_factor,
    n_routing_groups,
    topk_routing_group,
    top_k_in_group,
    expert_axis_name,
    use_gather_mosaic_kernel,
    config,
    normalization_layer_epsilon,
    dtype,
    activation_pspec,
):
  """Performs dropless MoE with tensor/expert parallelism."""
  return jax.shard_map(
      functools.partial(
          route_compute_unroute,
          num_experts=num_experts,
          num_experts_per_tok=num_experts_per_tok,
          routed_scaling_factor=routed_scaling_factor,
          n_routing_groups=n_routing_groups,
          topk_routing_group=topk_routing_group,
          top_k_in_group=top_k_in_group,
          expert_axis_name=expert_axis_name,
          use_gather_mosaic_kernel=use_gather_mosaic_kernel,
          normalization_layer_epsilon=normalization_layer_epsilon,
          dtype=dtype,
          config=config,
      ),
      mesh=mesh,
      in_specs=(
          [activation_pspec] * config.batch_split_factor,
          (
              jax.sharding.PartitionSpec(None),
              (
                  jax.sharding.PartitionSpec(None, None, reduced={"data", "fsdp", "expert"}),
                  jax.sharding.PartitionSpec(None),
              ),
              (
                  jax.sharding.PartitionSpec(None, None, expert_axis_name, reduced={"data", "fsdp"}),
                  jax.sharding.PartitionSpec(None, None, expert_axis_name, reduced={"data", "fsdp"}),
                  jax.sharding.PartitionSpec(None, expert_axis_name, None, reduced={"data", "fsdp"}),
              ),
              (
                  jax.sharding.PartitionSpec(None, None, reduced={"data", "fsdp", "expert"}),
                  jax.sharding.PartitionSpec(None, None, reduced={"data", "fsdp", "expert"}),
                  jax.sharding.PartitionSpec(None, None, reduced={"data", "fsdp", "expert"}),
              ),
          ),
      ),
      out_specs=(
          [
              activation_pspec,
              (
                  activation_pspec,
                  jax.sharding.PartitionSpec(*activation_pspec[:-1]),
                  jax.sharding.PartitionSpec(*activation_pspec[:-1]),
                  jax.sharding.PartitionSpec(activation_pspec[0], None),
              ),
          ],
          {
              "mlpwi_0": [jax.sharding.PartitionSpec(*activation_pspec[:-1])] * config.batch_split_factor,
              "mlpwi_1": [jax.sharding.PartitionSpec(*activation_pspec[:-1])] * config.batch_split_factor,
          },
      ),
      check_vma=True,
  )([x.astype(config.dtype) for x in xs], weights)


def moe_bwd(
    residuals,
    outputs_grad,
    weights,
    *,
    mesh,
    num_experts,
    num_experts_per_tok,
    routed_scaling_factor,
    n_routing_groups,
    topk_routing_group,
    top_k_in_group,
    expert_axis_name,
    use_gather_mosaic_kernel,
    config,
    normalization_layer_epsilon,
    dtype,
    activation_pspec,
):
  """Performs the backward pass for the moe function."""
  pass
