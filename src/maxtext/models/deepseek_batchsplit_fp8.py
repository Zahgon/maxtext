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


"""Alternative DeepSeek model definition with batch-split schedule."""

import dataclasses
import functools
import math
from typing import Any, Sequence

from flax import linen as nn
import jax
import jax.numpy as jnp
from maxtext.kernels import megablox, sort_activations
from maxtext.layers import attention_op
from maxtext.layers import moe as moe_lib
from maxtext.layers import quantizations
import qwix.pallas as qpl
import tokamax


@functools.partial(
    jax.custom_vjp,
    nondiff_argnums=(
        1,
        2,
        3,
    ),
)
def quantized_psum_scatter(x: jax.Array, axis_name: str, scatter_dimension: int, tiled: bool) -> jax.Array:
  """Forward: Standard BF16 Reduce-Scatter.

  Backward: Quantized FP8 All-Gather (DeepSeek optimization).

  Args:
    x: The input tensor.
    axis_name: The axis name for the psum_scatter/all_gather operation.
    scatter_dimension: The dimension along which to scatter.
    tiled: Whether the scatter/gather is tiled.

  Returns:
    The result of the reduce-scatter operation.
  """
  pass




def _q_psum_scatter_bwd(
    axis_name: str,
    scatter_dimension: int,
    tiled: bool,
    res: Any,
    grads: jax.Array,
) -> tuple[jax.Array]:  # pylint: disable=g-one-element-tuple
  """Backward pass for quantized_psum_scatter.

  Performs a quantized All-Gather of the gradients.

  Args:
    axis_name: The axis name for the all_gather operation.
    scatter_dimension: The dimension along which the scatter occurred in the
      forward pass.
    tiled: Whether the gather is tiled.
    res: The residuals from the forward pass (_q_psum_scatter_fwd), containing
      (axis_name, scatter_dimension, tiled).
    grads: The gradients from the next layer, which are in BF16.

  Returns:
    The dequantized and all-gathered gradients.
  """
  pass


quantized_psum_scatter.defvjp(_q_psum_scatter_fwd, _q_psum_scatter_bwd)


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


def gather_weights(weights, mesh):
  """all-gathers FSDP sharded weights."""
  pass


def scan_batch_split_layers(
    inputs,
    params,
    positions,
    segment_ids,
    *,
    model_mode,
    mesh,
    quant,
    cfg,
    policy,
):
  """Scans the layers with batch-split schedule."""
  pass


def batch_split_schedule(
    inputs,
    weights,
    positions,
    segment_ids,
    *,
    model_mode,
    mesh,
    quant,
    cfg,
):
  """Applies the DeepSeek MoE layer with batch-split schedule."""
  pass




def with_data_parallel_constraint(x, mesh):
  activation_pspec = jax.sharding.PartitionSpec(
      ("data", "fsdp", "fsdp_transpose", "expert", "context"),
      None,
      None,
  )
  return jax.lax.with_sharding_constraint(x, jax.NamedSharding(mesh, activation_pspec))


def dot(x, y, quant=None, axes=1):
  """Computes the dot product of two arrays, optionally using quantization."""
  if quant is not None:
    # Convert axes to jax.lax.dot_general dimension_numbers
    if isinstance(axes, int):
      x_contract = tuple(range(x.ndim - axes, x.ndim))
      y_contract = tuple(range(axes))
    else:
      x_contract, y_contract = axes
    dimension_numbers = ((x_contract, y_contract), ((), ()))
    # Instantiate and call qwix dot_general
    custom_dot = quant.dot_general_cls()()
    return custom_dot(lhs=x, rhs=y, dimension_numbers=dimension_numbers)

  # Unquantized
  return jnp.tensordot(x, y, axes=axes)


def mla_with_norms(
    inputs,
    weights,
    decoder_positions,
    decoder_segment_ids,
    *,
    mesh,
    model_mode,
    attn_op,
    normalization_layer_epsilon,
    kv_lora_rank,
    qk_nope_head_dim,
    qk_rope_head_dim,
    rope_max_timescale,
    num_query_heads,
    max_position_embeddings,
    original_max_position_embeddings,
    beta_fast,
    beta_slow,
    rope_factor,
    mscale,
    dtype,
    quant,
):
  """Performs MLA with pre- and post-normalization."""
  pass


def mla(
    inputs,
    positions,
    segment_ids,
    weights,
    *,
    model_mode,
    epsilon,
    kv_lora_rank,
    kv_norm_epsilon,
    qk_nope_head_dim,
    qk_rope_head_dim,
    num_query_heads,
    rope_theta,
    max_position_embeddings,
    original_max_position_embeddings,
    beta_fast,
    beta_slow,
    rope_factor,
    mscale,
    attention_op_fn,
    dtype,
    quant,
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
      positions,
      wq_a_weights,
      wq_b_weights,
      q_norm_scale_weights,
      epsilon=epsilon,
      qk_rope_head_dim=qk_rope_head_dim,
      rope_theta=rope_theta,
      max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings,
      beta_fast=beta_fast,
      beta_slow=beta_slow,
      rope_factor=rope_factor,
      dtype=dtype,
      qk_nope_head_dim=qk_nope_head_dim,
      mscale=mscale,
      quant=quant,
  )
  query = jax.ad_checkpoint.checkpoint_name(query, "query_proj")
  key, value = kv_projection(
      inputs,
      positions,
      wkv_a_weights,
      wkv_b_weights,
      kv_norm_scale_weights,
      kv_lora_rank=kv_lora_rank,
      kv_norm_epsilon=kv_norm_epsilon,
      qk_rope_head_dim=qk_rope_head_dim,
      rope_theta=rope_theta,
      max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings,
      beta_fast=beta_fast,
      beta_slow=beta_slow,
      rope_factor=rope_factor,
      dtype=dtype,
      qk_nope_head_dim=qk_nope_head_dim,
      num_query_heads=num_query_heads,
      quant=quant,
  )
  key = jax.ad_checkpoint.checkpoint_name(key, "key_proj")
  value = jax.ad_checkpoint.checkpoint_name(value, "value_proj")
  out = attention_op_fn(
      query,
      key,
      value,
      segment_ids,
      model_mode,
      cached_values=[None, None],
  )
  out = jax.ad_checkpoint.checkpoint_name(out, "attention_out")
  out = dot(out, out_weights, quant=quant, axes=2)
  out = jax.ad_checkpoint.checkpoint_name(out, "out_proj")
  return out


def query_projection(
    inputs_q,
    inputs_positions,
    wq_a_weights,
    wq_b_weights,
    q_norm_scale_weights,
    *,
    epsilon,
    qk_nope_head_dim,
    qk_rope_head_dim,
    rope_theta,
    max_position_embeddings,
    original_max_position_embeddings,
    beta_fast,
    beta_slow,
    rope_factor,
    dtype,
    mscale,
    quant,
):
  """Performs query projection."""
  # Set softmax scaling.
  qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
  softmax_scale = qk_head_dim**-0.5
  if max_position_embeddings > original_max_position_embeddings:
    m = 0.1 * mscale * math.log(rope_factor) + 1.0
    softmax_scale = softmax_scale * m * m

  # LoRA path
  low_rank_q = dot(inputs_q, wq_a_weights, quant=quant)
  low_rank_q = rms_norm(
      low_rank_q,
      q_norm_scale_weights,
      epsilon=epsilon,
      dtype=dtype,
  )
  low_rank_q = jax.ad_checkpoint.checkpoint_name(low_rank_q, "mla_q")
  q = dot(low_rank_q, wq_b_weights, quant=quant)

  # Split into non-positional and rotary parts.
  q_nope, q_pe = jnp.split(q, [qk_nope_head_dim], axis=-1)
  q_pe = yarn(
      q_pe,
      inputs_positions,
      embedding_dims=qk_rope_head_dim,
      rope_theta=rope_theta,
      max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings,
      beta_fast=beta_fast,
      beta_slow=beta_slow,
      rope_factor=rope_factor,
      fprop_dtype=dtype,
  )
  query = jnp.concatenate([q_nope, q_pe], axis=-1) * softmax_scale
  return query


def kv_projection(
    inputs,
    inputs_positions,
    wkv_a_weights,
    wkv_b_weights,
    kv_norm_scale_weights,
    *,
    kv_lora_rank,
    kv_norm_epsilon,
    qk_rope_head_dim,
    rope_theta,
    max_position_embeddings,
    original_max_position_embeddings,
    beta_fast,
    beta_slow,
    rope_factor,
    dtype,
    qk_nope_head_dim,
    num_query_heads,
    quant,
):
  """Performs KV projection."""
  low_rank = dot(inputs, wkv_a_weights, quant=quant)
  low_rank_main, low_rank_rope = jnp.split(low_rank, [kv_lora_rank], axis=-1)
  low_rank_main = rms_norm(
      low_rank_main,
      kv_norm_scale_weights,
      epsilon=kv_norm_epsilon,
      dtype=dtype,
  )
  low_rank_main = jax.ad_checkpoint.checkpoint_name(low_rank_main, "mla_kv")
  key_rope = jnp.expand_dims(low_rank_rope, axis=2)
  key_rope = yarn(
      key_rope,
      inputs_positions,
      embedding_dims=qk_rope_head_dim,
      rope_theta=rope_theta,
      max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings,
      beta_fast=beta_fast,
      beta_slow=beta_slow,
      rope_factor=rope_factor,
      fprop_dtype=dtype,
  )

  return get_key_value(
      low_rank_main,
      key_rope,
      wkv_b_weights,
      qk_nope_head_dim=qk_nope_head_dim,
      num_query_heads=num_query_heads,
      quant=quant,
  )


def get_key_value(low_rank_main, key_rope, wkv_b_weights, *, qk_nope_head_dim, num_query_heads, quant):
  """Gets key and value from compressed KV latent vector and key rope."""
  kv_out = dot(low_rank_main, wkv_b_weights, quant=quant)

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


def rms_norm(x, scale, *, epsilon, dtype):
  """RMS normalization."""
  x = jnp.asarray(x, jnp.float32)
  mean2 = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
  y = jnp.asarray(x * jax.lax.rsqrt(mean2 + epsilon), dtype)
  return jnp.einsum("i...k,...k->i...k", y, scale)


def yarn(
    inputs,
    positions,
    *,
    embedding_dims,
    rope_theta,
    max_position_embeddings,
    original_max_position_embeddings,
    beta_fast,
    beta_slow,
    rope_factor,
    fprop_dtype,
):
  """Performs YaRN rotary embedding."""
  # Initialize the swap and negate mask.
  indices = jnp.arange(embedding_dims)
  # [1, 0, 3, 2, 5, 4, ...]
  swap_indices = jnp.where(indices % 2 == 0, indices + 1, indices - 1)
  negation_mask = jnp.where(indices % 2 == 0, -1, 1)
  identity = jnp.eye(embedding_dims, dtype=jnp.int32)
  pairwise_swap_and_negate_mask = identity[swap_indices] * negation_mask

  # Calculate the frequencies.
  half_dim = embedding_dims // 2
  # Compute base frequencies for each (even-indexed) dimension.
  # (Note: We use jnp.arange with float32 for precision.)
  freqs = 1.0 / (rope_theta ** (2.0 * jnp.arange(0, half_dim, dtype=jnp.float32) / embedding_dims))

  low = (
      embedding_dims * math.log(original_max_position_embeddings / (beta_fast * 2 * math.pi)) / (2 * math.log(rope_theta))
  )
  high = (
      embedding_dims * math.log(original_max_position_embeddings / (beta_slow * 2 * math.pi)) / (2 * math.log(rope_theta))
  )
  low = max(math.floor(low), 0)
  high = min(math.ceil(high), embedding_dims - 1)
  diff = high - low if high > low else 0.001
  linear_func = (jnp.arange(half_dim, dtype=jnp.float32) - low) / diff
  smooth = 1 - jnp.clip(linear_func, 0, 1)
  # The corrected frequency is a weighted mix of the scaled and base values.
  freqs = freqs / rope_factor * (1 - smooth) + freqs * smooth

  # Precompute frequencies for all positions by taking the outer product.
  t = jnp.arange(max_position_embeddings, dtype=jnp.float32)  # shape [max_position_embeddings]
  # This gives a [max_position_embeddings, half_dim] tensor with rows as time steps.
  freqs = jnp.outer(t, freqs)

  # Lookup the precomputed frequencies using the position indices.
  # self.freqs has shape [max_position_embeddings, half_dim] so we use jnp.take along axis 0.
  # After indexing, shape becomes [B, S, half_dim]; we then add an axis for the heads.
  freqs = jnp.take(freqs, positions, axis=0)  # shape: [B, S, half_dim]
  freqs = freqs[:, :, jnp.newaxis, :]  # shape: [B, S, 1, half_dim]
  freqs = jnp.repeat(freqs, 2, axis=-1)  # shape: [B, S, 1, embedding_dims]
  # inputs @ mask: [B, S, N, embedding_dims] @ [embedding_dims, embedding_dims] -> [B, S, N, embedding_dims]
  output = inputs * jnp.cos(freqs) + jnp.matmul(inputs, pairwise_swap_and_negate_mask) * jnp.sin(freqs)
  return output.astype(fprop_dtype)


def moe(
    inputs,
    weights,
    *,
    mesh,
    num_experts,
    num_experts_per_tok,
    routed_scaling_factor,
    expert_axis_name,
    use_gather_mosaic_kernel,
    config,
    quant,
):
  """Performs dropless MoE with tensor/expert parallelism."""
  xs, ys = list(zip(*inputs))
  ys = with_data_parallel_constraint(
      process_activations(
          ys,
          weights,
          mesh=mesh,
          num_experts=num_experts,
          num_experts_per_tok=num_experts_per_tok,
          routed_scaling_factor=routed_scaling_factor,
          expert_axis_name=expert_axis_name,
          use_gather_mosaic_kernel=use_gather_mosaic_kernel,
          config=config,
          quant=quant,
      ),
      mesh,
  )
  return [x + y for x, y in zip(xs, ys)]


def expert_indices_and_weights(
    gate_logits: jax.Array,
    pre_bias_logits: jax.Array,
    num_experts_per_tok: int,
    routed_scaling_factor: float,
) -> tuple[jax.Array, jax.Array]:
  """Computes expert indices for each token and their corresponding weights."""
  pass


def expert_selection(
    x,
    routing_kernel,
    routing_bias,
    *,
    num_experts,
    num_experts_per_tok,
    routed_scaling_factor,
    quant,
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


def compute(x, w0, w1, wo, group_sizes, weights, *, config, mesh):
  """Processes routed tokens through the MLP."""

  def gmm(
      inputs,
      kernel,
      tiling,
      group_sizes,
      preferred_element_type,
      weight_gather_axes,
  ):
    if config.use_qwix_quantization:
      output = megablox.gmm(
          lhs=inputs,
          rhs=kernel,
          group_sizes=group_sizes,
          preferred_element_type=preferred_element_type,
          tiling=tiling,
          use_qwix_quantization=config.use_qwix_quantization,
          use_tokamax_backend=config.use_tokamax_gmm,
          weight_gather_axes=weight_gather_axes,
          qwix_rule=quantizations.get_fp8_full_qwix_rule_w_sparsity(config)[0],
      )
    else:
      output = tokamax.ragged_dot(
          lhs=inputs,
          rhs=kernel,
          group_sizes=tokamax.RaggedDotGroupSizes(group_sizes, len(inputs)),
          precision=jax.lax.Precision.DEFAULT,
          preferred_element_type=preferred_element_type,
          implementation="mosaic",
      )
    return output

  gmm_fn = functools.partial(gmm, group_sizes=group_sizes, preferred_element_type=config.dtype)
  wi_gather_axes = []
  wo_gather_axes = []

  wi_tile_size = (
      config.wi_tile_fwd_batch_seq,  # m (LHS batch)
      config.wi_tile_fwd_embed_dim,  # k  (contracting)
      config.wi_tile_fwd_mlp_dim,  # n (RHS batch)
      config.wi_tile_dlhs_batch_seq,  # m (LHS batch)
      config.wi_tile_dlhs_mlp_dim,  # k (contracting)
      config.wi_tile_dlhs_embed_dim,  # n (RHS batch)
      config.wi_tile_drhs_batch_seq,  # Called m in megablox, but this is contracting
      config.wi_tile_drhs_embed_dim,  # Called k in megablox, but this is LHS batch dim
      config.wi_tile_drhs_mlp_dim,  # Called n in megablox, and indeed is the RHS batch dim
  )

  wo_tile_size = (
      config.wo_tile_fwd_batch_seq,  # m (LHS batch)
      config.wo_tile_fwd_mlp_dim,  # k (contracting)
      config.wo_tile_fwd_embed_dim,  # n (RHS batch)
      config.wo_tile_dlhs_batch_seq,  # m (LHS batch)
      config.wo_tile_dlhs_embed_dim,  # k (contracting)
      config.wo_tile_dlhs_mlp_dim,  # n (RHS)
      config.wo_tile_drhs_batch_seq,  # Called m in megablox, but this is contracting
      config.wo_tile_drhs_mlp_dim,  # Called k in megablox, but this is LHS batch dim
      config.wo_tile_drhs_embed_dim,  # Called n in megablox, and indeed is the RHS batch dim
  )

  if config.use_qwix_quantization:
    gating_pspec, linear_pspec = moe_lib.get_batchsplit_init_kernel_axes()
    w0_pspec = nn.logical_to_mesh_axes(gating_pspec)
    wo_pspec = nn.logical_to_mesh_axes(linear_pspec)
    ignored_axes = ("expert", "tensor", "tensor_transpose")

    def get_active_sharding_axes(pspec_dim_axes, tensor_dim_index):
      if pspec_dim_axes is None:
        return []
      axes = (pspec_dim_axes,) if isinstance(pspec_dim_axes, str) else pspec_dim_axes
      active = []
      for ax in axes:
        if ax and ax not in ignored_axes and mesh.shape.get(ax, 1) > 1:
          active.append((ax, tensor_dim_index))
      return active

    wi_gather_axes.extend(get_active_sharding_axes(w0_pspec[0], 0))
    wi_gather_axes.extend(get_active_sharding_axes(w0_pspec[2], 2))

    wo_gather_axes.extend(get_active_sharding_axes(wo_pspec[0], 0))
    wo_gather_axes.extend(get_active_sharding_axes(wo_pspec[1], 1))
  if config.merge_gating_gmm:
    w01 = jnp.concatenate([w0, w1], axis=-1)
    layer_w01 = gmm_fn(
        x,
        w01,
        tiling=wi_tile_size,
        weight_gather_axes=wi_gather_axes,
    )
    layer_w0, layer_w1 = jnp.split(layer_w01, 2, axis=-1)
  else:
    layer_w0 = gmm_fn(
        x,
        w0,
        tiling=wi_tile_size,
        weight_gather_axes=wi_gather_axes,
    )
    layer_w1 = gmm_fn(
        x,
        w1,
        tiling=wi_tile_size,
        weight_gather_axes=wi_gather_axes,
    )
  layer_w0 = jax.ad_checkpoint.checkpoint_name(layer_w0, "mlpwi_0")
  layer_w1 = jax.ad_checkpoint.checkpoint_name(layer_w1, "mlpwi_1")
  intermediate_layer = jax.nn.silu(layer_w0) * layer_w1
  intermediate_layer *= weights[:, None]
  layer_wo = gmm_fn(
      intermediate_layer,
      wo,
      tiling=wo_tile_size,
      weight_gather_axes=wo_gather_axes,
  )
  return layer_wo


def route_compute_unroute(
    xs,
    weights,
    *,
    num_experts,
    num_experts_per_tok,
    routed_scaling_factor,
    expert_axis_name,
    use_gather_mosaic_kernel,
    config,
    mesh,
    quant,
):
  """Routes, processes, and unroutes activations."""
  pass


def process_activations(
    xs,
    weights,
    *,
    mesh,
    num_experts,
    num_experts_per_tok,
    routed_scaling_factor,
    expert_axis_name,
    use_gather_mosaic_kernel,
    config,
    quant,
):
  """Processes activations, which are fully sharded on the batch axis, with tensor/expert sharded weights."""
  activation_pspec = jax.sharding.PartitionSpec(
      ("data", "fsdp", "fsdp_transpose", "expert", "context"),
      None,
      None,
  )
  if config.use_qwix_quantization:
    gating_pspec, linear_pspec = moe_lib.get_batchsplit_init_kernel_axes()
    gating_pspec = nn.logical_to_mesh_axes(gating_pspec)
    linear_pspec = nn.logical_to_mesh_axes(linear_pspec)
  else:
    gating_pspec = jax.sharding.PartitionSpec(None, None, expert_axis_name)
    linear_pspec = jax.sharding.PartitionSpec(None, expert_axis_name, None)
  return jax.shard_map(
      functools.partial(
          route_compute_unroute,
          num_experts=num_experts,
          num_experts_per_tok=num_experts_per_tok,
          routed_scaling_factor=routed_scaling_factor,
          expert_axis_name=expert_axis_name,
          use_gather_mosaic_kernel=use_gather_mosaic_kernel,
          config=config,
          mesh=mesh,
          quant=quant,
      ),
      mesh=mesh,
      in_specs=(
          [activation_pspec] * len(xs),
          (
              (
                  jax.sharding.PartitionSpec(None, None),
                  jax.sharding.PartitionSpec(None),
              ),
              (
                  gating_pspec,
                  gating_pspec,
                  linear_pspec,
              ),
              (
                  jax.sharding.PartitionSpec(None, None),
                  jax.sharding.PartitionSpec(None, None),
                  jax.sharding.PartitionSpec(None, None),
              ),
          ),
      ),
      out_specs=activation_pspec,
      check_vma=False,
  )([x.astype(config.dtype) for x in xs], weights)
