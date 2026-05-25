# Copyright 2025-2026 Google LLC
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

"""Functions for vocabulary tiling (VT)"""

import functools

from flax import linen as nn
from flax import nnx

import jax
import jax.numpy as jnp
from maxtext.utils.sharding import (
    maybe_shard_with_name,
    all_gather_over_fsdp,
    create_sharding,
)
from maxtext.common.common_types import ShardMode
from maxtext.utils import max_utils


def vocab_tiling_linen_loss(
    hidden_states,
    data,
    config,
    model,
    params,
    is_train,
):
  """Calculates cross-entropy loss using vocab tiling for Linen models.

  This function implements a memory-efficient approach for calculating loss when the
  vocabulary is too large to fit in memory. It works by breaking the computation
  into chunks (tiles) and processing them sequentially using `jax.lax.scan`.
  A custom VJP rule is defined to handle the backward pass efficiently.

  Args:
    hidden_states: The final hidden states from the decoder.
    data: A dictionary containing the input data, including 'targets' and 'targets_segmentation'.
    config: The model and training configuration.
    model: The Linen model instance.
    params: The model parameters.
    is_train: A boolean indicating if the model is in training mode.
  Returns:
    A tuple of (total_loss, total_z_loss) computed via vocab tiling.
  """
  labels = data["targets"]
  segmentation = data["targets_segmentation"]
  deterministic = not config.enable_dropout if is_train else True

  param_spec = nn.get_partition_spec(params)
  hidden_spec = create_sharding(
      model.mesh,
      ("activation_embed_and_logits_batch", "activation_length", "activation_embed"),
  )
  label_spec = create_sharding(
      model.mesh,
      ("activation_embed_and_logits_batch", "activation_length"),
  )
  reshaped_hidden_spec = create_sharding(
      model.mesh,
      ("num_tile", "activation_embed_and_logits_batch_sequence", "activation_embed"),
  )
  reshaped_data_spec = create_sharding(
      model.mesh,
      ("num_tile", "activation_embed_and_logits_batch_sequence"),
  )
  chunked_hidden_spec = create_sharding(
      model.mesh,
      ("activation_embed_and_logits_batch_sequence", "activation_embed"),
  )
  chunked_data_spec = create_sharding(
      model.mesh,
      ("activation_embed_and_logits_batch_sequence",),
  )
  chunked_logits_spec = create_sharding(
      model.mesh,
      ("activation_embed_and_logits_batch_sequence", "activation_vocab"),
  )

  _maybe_shard_with_name = functools.partial(
      maybe_shard_with_name,
      shard_mode=config.shard_mode,
      debug_sharding=config.debug_sharding,
      extra_stack_level=1,
  )

  def _reshape(inputs, out_shape, out_sharding):
    reshape_out_sharding = out_sharding if config.shard_mode == ShardMode.EXPLICIT else None
    inputs = jax.lax.reshape(inputs, out_shape, out_sharding=reshape_out_sharding)
    return _maybe_shard_with_name(inputs, out_sharding)

  hidden_states = _maybe_shard_with_name(hidden_states, hidden_spec)
  labels = _maybe_shard_with_name(labels, label_spec)
  segmentation = _maybe_shard_with_name(segmentation, label_spec)
  # TODO (chengnuojin) all gather only embedding table instead of all params after NNX module is enabled
  gathered_params = all_gather_over_fsdp(params, param_spec, model.mesh, config.logical_axis_rules, config.shard_mode)

  # Customized forward and backward maps for the embedding tiling
  @jax.custom_vjp
  def chunked_cross_entropy_loss(gathered_params, hidden_states, labels, segmentation):
    """
    Calculates the total cross-entropy loss using vocab tiling.
    """
    (total_loss, total_z_loss), _ = _chunked_cross_entropy_loss_fwd(gathered_params, hidden_states, labels, segmentation)
    return total_loss, total_z_loss

  def _chunked_cross_entropy_loss_fwd(gathered_params, hidden_states, labels, segmentation):
    batch_size, seq_len, emb_dim = hidden_states.shape
    vocab_tile_size = (batch_size * seq_len) // config.num_vocab_tiling

    reshaped_hidden_states = _reshape(
        hidden_states, (config.num_vocab_tiling, vocab_tile_size, emb_dim), reshaped_hidden_spec
    )
    reshaped_labels = _reshape(labels, (config.num_vocab_tiling, vocab_tile_size), reshaped_data_spec)
    reshaped_segmentation = _reshape(segmentation, (config.num_vocab_tiling, vocab_tile_size), reshaped_data_spec)

    # Scan body accumulates loss from each tile given chunked hidden states and labels

    initial_acc = (0.0, 0.0)
    (total_loss, total_z_loss), _ = jax.lax.scan(
        _fwd_scan_body, initial_acc, (reshaped_hidden_states, reshaped_labels, reshaped_segmentation)
    )
    residuals = (
        gathered_params,
        reshaped_hidden_states,
        reshaped_labels,
        reshaped_segmentation,
        batch_size,
        seq_len,
        emb_dim,
    )

    return (total_loss, total_z_loss), residuals


  chunked_cross_entropy_loss.defvjp(_chunked_cross_entropy_loss_fwd, _chunked_cross_entropy_loss_bwd)

  total_loss, total_z_loss = chunked_cross_entropy_loss(
      gathered_params,
      hidden_states,
      labels,
      segmentation,
  )

  return total_loss, total_z_loss


def vocab_tiling_nnx_loss(model, hidden_states, data, config, is_train):
  """Computes cross-entropy loss with vocab tiling for NNX models.

  NNX equivalent of ``vocab_tiling_linen_loss``. Scans the vocab dimension
  and calls ``model.logits_from_hidden_states_for_vocab_tiling`` per chunk. The NNX model
  carries its own parameters, so no explicit gather is needed.

  Uses default autograd; a custom_vjp for backward memory savings can be
  added later if needed.

  Args:
    model: NNX model exposing ``logits_from_hidden_states_for_vocab_tiling``.
    hidden_states: Final hidden states from the decoder.
    data: Dict with ``targets`` and ``targets_segmentation``.
    config: Model and training config.
    is_train: Whether the model is in training mode.

  Returns:
    A tuple ``(total_loss, total_z_loss)``.
  """
  labels = data["targets"]
  segmentation = data["targets_segmentation"]
  deterministic = not config.enable_dropout if is_train else True
  model_mode = "train"

  hidden_spec = create_sharding(
      model.mesh,
      ("activation_embed_and_logits_batch", "activation_length", "activation_embed"),
  )
  label_spec = create_sharding(
      model.mesh,
      ("activation_embed_and_logits_batch", "activation_length"),
  )
  reshaped_hidden_spec = create_sharding(
      model.mesh,
      ("num_tile", "activation_embed_and_logits_batch_sequence", "activation_embed"),
  )
  reshaped_data_spec = create_sharding(
      model.mesh,
      ("num_tile", "activation_embed_and_logits_batch_sequence"),
  )
  chunked_hidden_spec = create_sharding(
      model.mesh,
      ("activation_embed_and_logits_batch_sequence", "activation_embed"),
  )
  chunked_data_spec = create_sharding(
      model.mesh,
      ("activation_embed_and_logits_batch_sequence",),
  )
  chunked_logits_spec = create_sharding(
      model.mesh,
      ("activation_embed_and_logits_batch_sequence", "activation_vocab"),
  )

  _maybe_shard_with_name = functools.partial(
      maybe_shard_with_name,
      shard_mode=config.shard_mode,
      debug_sharding=config.debug_sharding,
      extra_stack_level=1,
  )

  def _reshape(inputs, out_shape, out_sharding):
    reshape_out_sharding = out_sharding if config.shard_mode == ShardMode.EXPLICIT else None
    inputs = jax.lax.reshape(inputs, out_shape, out_sharding=reshape_out_sharding)
    return _maybe_shard_with_name(inputs, out_sharding)

  hidden_states = _maybe_shard_with_name(hidden_states, hidden_spec)
  labels = _maybe_shard_with_name(labels, label_spec)
  segmentation = _maybe_shard_with_name(segmentation, label_spec)

  batch_size, seq_len, emb_dim = hidden_states.shape
  vocab_tile_size = (batch_size * seq_len) // config.num_vocab_tiling

  reshaped_hidden_states = _reshape(
      hidden_states, (config.num_vocab_tiling, vocab_tile_size, emb_dim), reshaped_hidden_spec
  )
  reshaped_labels = _reshape(labels, (config.num_vocab_tiling, vocab_tile_size), reshaped_data_spec)
  reshaped_segmentation = _reshape(segmentation, (config.num_vocab_tiling, vocab_tile_size), reshaped_data_spec)

  # Rebuild the model per chunk inside the scan: the output head pulls an rng stream, and
  # mutating the outer model's rng inside scan's sub-trace raises TraceContextError.
  # nnx.merge(..., copy=True) makes fresh Variables local to each iteration.
  graphdef, model_state = nnx.split(model)


  initial_acc = (jnp.zeros((), dtype=hidden_states.dtype), jnp.zeros((), dtype=hidden_states.dtype))
  (total_loss, total_z_loss), _ = jax.lax.scan(
      _scan_body, initial_acc, (reshaped_hidden_states, reshaped_labels, reshaped_segmentation)
  )
  return total_loss, total_z_loss
