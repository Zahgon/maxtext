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

""" Pipeline layer wrapping a decoder layer(s). Supports circular pipelining """

import functools
from typing import Any

import numpy as np

from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import jax
import jax.ad_checkpoint

from flax.core import meta
from flax import linen as nn
from flax.linen.spmd import LogicallyPartitioned

from maxtext.common.common_types import Config, MODEL_MODE_TRAIN, ShardMode
from maxtext.utils.sharding import (
    maybe_shard_with_logical,
    maybe_shard_with_name,
    create_sharding,
    logical_to_mesh_axes,
    logical_to_mesh,
)


class Pipeline(nn.Module):
  """Module that implements pipelining across stages.

  This module will loop over microbatches and execute the main body with a vmap for both the inputs and weights.
  This will produce a pipeline pattern if the stage dimension is sharded.

  Supports circular pipelines, and multiple layers per stage are used when a module that executes multiple layers
  is passed as the layers input.
  """

  #: Importantly contains num_pipeline_microbatches, num_pipeline_repeats.
  config: Config
  #: A module instance that each stage can execute. It can either be a single layer such as a LlamaDecoderLayer instance or
  #: scanned/looped set of decoder layers to execute multiple layers per stage. The name of this property (layers) is
  #: reflected in the state pytree and thus also checkpoints.
  layers: nn.Module
  #: The device mesh of the system.
  mesh: Mesh
  #: Remat policy to use for the loop iterations
  remat_policy: Any = None

  def setup(self):  # pylint: disable=missing-function-docstring
    self.num_stages = self.config.ici_pipeline_parallelism * self.config.dcn_pipeline_parallelism
    self.forwarding_delay = 2 if self.config.pipeline_delay_activation_forwarding else 1
    self.pipeline_microbatch_size = self.config.micro_batch_size_to_train_on // self.config.num_pipeline_microbatches
    microbatches_per_stage = self.config.num_pipeline_microbatches // self.num_stages
    self.microbatches_per_stage = microbatches_per_stage
    self.use_circ_storage = self.need_circ_storage()

    self.batch_axis_name = "activation_batch"
    self.seq_len_axis_name = "activation_length"

    # TODO(b/470167805): replace self.spmd_axis_name with "stage" when JAX >= 0.8.2.
    self.spmd_axis_name = "stage" if self.config.shard_mode == ShardMode.AUTO else None

    self.stages_in_logical = ("activation_stage", self.batch_axis_name, self.seq_len_axis_name, "activation_embed")
    self.stages_in_spec = logical_to_mesh_axes(self.stages_in_logical, self.mesh, rules=self.config.logical_axis_rules)
    self.stages_in_sharding = (
        NamedSharding(self.mesh, self.stages_in_spec) if self.config.shard_mode == ShardMode.EXPLICIT else None
    )

    self.state_io_logical = ("activation_stage", None, self.batch_axis_name, self.seq_len_axis_name, "activation_embed")
    self.state_io_spec = logical_to_mesh_axes(self.state_io_logical, self.mesh, rules=self.config.logical_axis_rules)
    self.state_io_sharding = (
        NamedSharding(self.mesh, self.state_io_spec) if self.config.shard_mode == ShardMode.EXPLICIT else None
    )
    self.input_sharding = (
        create_sharding(
            self.mesh,
            (None, self.batch_axis_name, self.seq_len_axis_name, "activation_embed"),
            rules=self.config.logical_axis_rules,
        )
        if self.config.shard_mode == ShardMode.EXPLICIT
        else None
    )
    self.output_sharding = (
        create_sharding(
            self.mesh,
            (self.batch_axis_name, self.seq_len_axis_name, "activation_embed"),
            rules=self.config.logical_axis_rules,
        )
        if self.config.shard_mode == ShardMode.EXPLICIT
        else None
    )

  def need_circ_storage(self):
    return (
        self.config.num_pipeline_repeats > 1
        and self.config.num_pipeline_microbatches > self.num_stages * self.forwarding_delay
    )



  def _maybe_shard_with_logical(self, inputs, logical_axes):
    """Wrapper of maybe_shard_with_logical"""
    pass

  def _maybe_shard_with_name(self, inputs, sharding_name):
    """Wrapper of maybe_shard_with_name"""
    return maybe_shard_with_name(
        inputs,
        sharding_name,
        shard_mode=self.config.shard_mode,
        debug_sharding=self.config.debug_sharding,
    )

  def init_states(self, inputs):
    """Initialize components of state: state_io, shift, circular_storage and circular_storage_mover
    Assumes input has already been reshaped into microbatches: [num_micro_batches, micro_batch_size, sequence, embed]

    Returns a dictionary with properties
      shift: zeros shape [num_stages, micro_size, sequence, embed]
      prev_outputs: same shape as shift, only used when pipeline_delay_activation_forwarding is set to true, else None
      state_io: reshaped inputs [num_stages, microbatches/stages, micro_size, sequence, embed]
      circ_storage: zeros [num_stages, microbatches, micro_size, sequence, embed] when needed, else None
      circ_storage_mover: zeros[num_stages, micro_size, sequence, embed] when needed, else None
      loop_iteration: scalar set initially to 0.
    """
    pass

  def get_iteration_inputs(self, loop_iteration, state_io, circ_storage, shift):
    """
    Construct stages_in: the global array that is operated on for this iteration, shape same as
    shift=[stages, micro_size, sequence, embed]
    This is almost a rotated version of the last outputs, except for the first stage which must grab a new batch from
    state_io or an old one from circ_storage
    """
    pass

  def shard_dim_by_stages(self, x, dim: int, physical_partition_spec: P | None, is_stage_weight: bool = False):
    """Shards x using the provided partition_spec, but adds the "stage" mesh axis to the existing sharding at
    the specified dimension."""
    pass

  def get_microbatch_and_repeat_ids(self, loop_iteration):
    """Gets the microbatch_ids and repeat_ids for all stages on this loop_iteration. Works for both circular and
    non-circular"""
    pass

  def vmap_parallel_gather(
      self, weights, physical_partition_spec, repeat_ids, repeat_dim_in_weights, stages_dim_in_weights
  ):
    """Use vmap to implement a sharded parallel gather.
    Parallel gather means each stage has its own weights, and gets one slice from it.
    Args:
      weights: Per-stage data to be gathered from.
      physical_partition_spec: Physical partition spec of the input weight.
      repeat_ids: Integer tensor of shape [num_stages], the repeats of the stages.
      repeat_dim_in_weights: The dimension in weights where repeat_ids are applied. The output will not
        have this dimension.
      stages_dim_in_weights: The dimension in weights that represents parallel stages.

    Returns:
      The per-stage gathered values. The shape is weights.shape but with repeat_dim_in_weights
        removed.
    """
    pass

  def vmap_gather(self, xs, ids, ids_dim):
    """Use vmap to implement a stage-wise sharded gather.

    The stages share the same input, but they have different offsets.

    Args:
      xs: Data shared by all stages, to be gathered from.
      ids: Integer tensor of shape [num_stages], the offsets of the stages.
      ids_dim: The dimension in xs where ids are applied. In the output, this
        dimension will be [num_stages], since each stage gets one slice.

    Returns:
      The per-stage gathered values. The shape is xs.shape but with ids_dim size
        replaced with [num_stages].
    """
    pass

  def get_new_loop_state(self, output, loop_state):
    """
    Update the various buffers given the output of the most recent iteration
    * state_io: rotates left/up by 1 (the whole created in the last slot is filled with the most recent pipeline output)
       * Pushing inputs up from top of state_io into first stage of shift
       * Pulling outputs up from last stage of shift into bottom of state_io
    * shift: rotate output (or prev_outputs if using delay) right/down by 1 - we imagine the pipeline moves to
               right/down
    * circ_storage: pushes circ_storage_mover (the output of the previous iteration) into rotating index of circ_storage
    * circ_storage_mover: assigned to rotated output and pushed into circ_storage on the next iteration
    * prev_outputs: is set to the current output
    """
    pass


  def get_current_stage_weights(self, pipeline_weights, loop_iteration, physical_partition_spec=None):
    """
    Gets the current weights used for one iteration. Outputs a pytree whose arrays have leading dimension of stages, e.g.
    {'mlp': 'wo': [stages, mlp, embed]}. Stage 0 will use the 0th index of this pytree, Stage 1 the 1st index, etc.
    For non-circular pipelines, this simply returns all weights - every weight is used in every iteraiton. However
    for circular pipelines each stage grabs only the weights corresponding to the current repeat.
    """
    pass

  def get_current_repeat_from_stages(self, weights, loop_iteration, physical_partition_spec=None):
    """get current repeat from stages"""
    pass

  def get_vmap_func_for_init(self):
    """This vmap func is used to initialize the weights only on init."""
    pass

  def get_main_vmap_func_for_iterations(self):
    """
    Returns main stage function vmapped by number of stages.
    This becomes a vmap over a single layer instance if body_instance is a single layer,
    else a set of layers if body_instance is a set of layers.
    """
    pass

  def run_one_iteration(
      self,
      loop_state,
      pipeline_weights,
      positions,
      segment_ids,
      deterministic,
      model_mode,
      decoder_layer_instance,
      logical_partition_spec=None,
  ):
    """Run one loop iteration - gets weights and inputs for each stage, run the stages in parallel,
    and update the loop state."""
    pass

  def get_pipeline_remat_policy(self):
    """Returns the pipeline remat policy for this pipeline."""
    pass

  def get_weight_sharding(self, *init_args):
    """get weight sharding function for this pipeline."""
    pass


  # TODO(chengnuojin) Remove this function and its usage after pipeline nnx migration

  @staticmethod
  def _remove_fsdp_from_physical_partition_spec(pps):
    """Removes 'fsdp' and 'fsdp_transpose' from a physical PartitionSpec."""
    pass

  def all_gather_over_fsdp(self, variables, logical_partition_spec):
    physical_partition_spec = logical_to_mesh(
        logical_partition_spec, mesh=self.mesh, rules=self.config.logical_axis_rules
    )
    physical_partition_spec_no_fsdp = jax.tree.map(
        self._remove_fsdp_from_physical_partition_spec, physical_partition_spec
    )
    return jax.tree.map(
        lambda w, p: self._maybe_shard_with_name(w, NamedSharding(self.mesh, p)),
        variables,
        physical_partition_spec_no_fsdp,
    )

  @nn.compact
  def __call__(
      self,
      inputs: jnp.ndarray,
      segment_ids: jnp.ndarray,
      positions: jnp.ndarray,
      deterministic: bool,
      model_mode=MODEL_MODE_TRAIN,
      logical_partition_spec=None,  # Pytree of sharding specifications of the weights (aka self.layers.variables)
  ) -> jnp.ndarray:
    """The main method that maps the series of decoder layer inputs to final layer outputs.
    Has the same signature of a single decoder layer, and expects the same shapes, e.g. the inputs should have shape
    [global_batch], and internally this will be reshapped into microbatches.
    """
    # Reshape inputs of [global_batch, ...] to [microbatches, pipeline_microbatch_sizes, ...]
    inputs = inputs.reshape(
        (
            self.config.num_pipeline_microbatches,
            self.pipeline_microbatch_size,
            self.config.max_target_length,
            self.config.emb_dim,
        ),
        out_sharding=self.input_sharding,
    )

    example_inputs = jax.lax.broadcast(inputs[0], [self.num_stages])  # dummy inputs fed to initialize the module
    ag_sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec(None, None))
    if positions is not None:
      # AG positions
      positions = self._maybe_shard_with_name(positions, ag_sharding)

      positions = positions.reshape(
          (self.config.num_pipeline_microbatches, self.pipeline_microbatch_size, self.config.max_target_length)
      )
      example_position = jax.lax.broadcast(positions[0], [self.num_stages])
      position_idx = 0
    else:
      example_position = None
      position_idx = None
    if segment_ids is not None:
      segment_ids = self._maybe_shard_with_name(segment_ids, ag_sharding)
      segment_ids = segment_ids.reshape(
          (self.config.num_pipeline_microbatches, self.pipeline_microbatch_size, self.config.max_target_length)
      )
      example_segmentation = jax.lax.broadcast(segment_ids[0], [self.num_stages])
      segment_idx = 0
    else:
      example_segmentation = None
      segment_idx = None

    loop_state = self.init_states(inputs)

    # Each microbatch should go through each stage (with repeats) - so there is num_micro * (num_stages * repeats)
    # compute to perform
    # Each iteration is vmapped by num_stages, so the number of iterations should be
    # num_micro * num_stages * repeats / num_stages = num_micro * repeats
    # However due to the pipeline bubble some iterations process less than num_stages microbatches. It takes
    # num_micro * repeat iterations for the last microbatch to start the final repeat, then an additional
    # num_stages - 1 to finish the final repeat.
    # Thus the total iterations is num_micro * repeat + num_stages - 1, & we may consider the num_stages - 1 as bubble.
    # The bubble doubles when we use forwarding delay.
    bubble_iterations = self.forwarding_delay * (self.num_stages - 1)
    real_iterations = self.config.num_pipeline_microbatches * self.config.num_pipeline_repeats
    total_iterations = real_iterations + bubble_iterations

    if self.is_initializing():
      vmap_func = self.get_vmap_func_for_init()

      if self.config.num_pipeline_repeats > 1:
        # To shard the weights on initialization for the circular pipeline we create weights of
        # shape [num_repeat, num_stages, ...] (e.g. [num_repeat, num_stages, embed, mlp]) and shard the num_stages axis.
        # We wrap the main stage vmap with a num_repeat vmap to generate this axis only for parameter initialization.
        vmap_func = nn.vmap(
            vmap_func,
            in_axes=(0, segment_idx, position_idx, None, None),
            variable_axes={
                "params": 0,
                "_overwrite_with_gradient": 0,
                "non_trainable": 0,
                "hyper_params": 0,
            },
            split_rngs={"params": True, "dropout": self.config.enable_dropout},
            metadata_params={
                nn.PARTITION_NAME: "circular_repeats",
                "sub_weight_split_dims_mapping": (None,),
                "is_initializing": True,
                "x_times": self.config.num_pipeline_repeats,
                "optimizer_dims_mapping": None,
            },
        )

        example_inputs = jax.lax.broadcast(example_inputs, [self.config.num_pipeline_repeats])
        example_segmentation = (
            jax.lax.broadcast(example_segmentation, [self.config.num_pipeline_repeats])
            if example_segmentation is not None
            else None
        )
        example_position = (
            jax.lax.broadcast(example_position, [self.config.num_pipeline_repeats])
            if example_position is not None
            else None
        )
      # We only need to run one set of stages to initialize the variables, instead of looping over all microbatches for
      # the full total_iterations.
      example_inputs = self._maybe_shard_with_logical(example_inputs, (None, None, None, None))
      stage_outputs = vmap_func(
          self.layers, example_inputs, example_segmentation, example_position, deterministic, model_mode
      )
      if self.config.scan_layers:
        stage_outputs = stage_outputs[0]

      # We return something of the correct shape (global_batch, sequence, embed) by reshaping a single stages output
      # which has shape [pipeline_microbatch_size, sequence, embed]
      if self.config.num_pipeline_repeats > 1:
        stage_outputs = stage_outputs[0]  # Remove extra dimension created for the circular vmap
      broadcasted_stage_outpus = jax.lax.broadcast(
          stage_outputs[0], [self.config.micro_batch_size_to_train_on // self.pipeline_microbatch_size]
      )

      return jnp.reshape(
          broadcasted_stage_outpus,
          [self.config.micro_batch_size_to_train_on, self.config.max_target_length, self.config.emb_dim],
          out_sharding=self.output_sharding,
      )

    if self.config.pipeline_fsdp_ag_once:
      variables = self._remove_logically_partition(self.layers.variables)
      all_pipeline_weights = self.all_gather_over_fsdp(variables, logical_partition_spec)
    else:
      all_pipeline_weights = self.layers.variables

    logical_partition_spec = self.get_logical_spec_repeats_removed(logical_partition_spec)


    if self.config.set_remat_policy_on_pipeline_iterations:
      run_iteration_scannable = nn.remat(
          run_iteration_scannable,
          prevent_cse=not self.config.scan_pipeline_iterations,  # prevent_cse not used with scan
          policy=self.get_pipeline_remat_policy(),
      )

    # The scan cannot be used on init since it broadcasts the weights, which aren't yet initialized.
    if self.config.scan_pipeline_iterations:
      variable_carry = []
      variable_broadcast = [
          "params",
          "_overwrite_with_gradient",
      ]  # All loop iterations need the weights for the full pipeline.
      if self.is_mutable_collection("non_trainable"):
        variable_carry.append("non_trainable")
      else:
        variable_broadcast.append("non_trainable")
      run_all_iterations_scanned = nn.scan(
          run_iteration_scannable,
          variable_axes={
              "summaries": 0,
              "aux_loss": 0,
              "intermediates": 0,
              "hyper_params": 0,
          },
          variable_broadcast=variable_broadcast,
          variable_carry=variable_carry,
          # Dropout/aqt keys will be split for each iteration.
          split_rngs={"random": True},
          length=total_iterations,
      )
      loop_state, _ = run_all_iterations_scanned(self, loop_state, None)
    else:
      for _ in range(total_iterations):
        loop_state, _ = run_iteration_scannable(self, loop_state, None)

    # The final output is located in the input/output array, however the output microbatches may be permuted relative to
    # the input
    final_output = self.permute_output_micro_per_stage_dim(loop_state["state_io"])

    # reshape outputs to match input shape of total batch instead of microbatches [batch, sequence, embed]
    final_output = jnp.reshape(
        final_output,
        (self.config.micro_batch_size_to_train_on, self.config.max_target_length, self.config.emb_dim),
        out_sharding=self.output_sharding,
    )

    return final_output
