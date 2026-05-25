# Copyright 2023-2026 Google LLC
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

"""Pipeline layer wrapping a decoder layer(s). Supports circular pipelining."""

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
from maxtext.utils import pipeline_utils


class PipelineBase(nn.Module):
  """Base module that implements shared pipelining logic across stages."""

  config: Config
  layers: nn.Module
  mesh: Mesh
  remat_policy: Any = None

  def setup(self):
    """Initializes the configuration, calculating num_stages, delay, axes, and partition specs."""
    self.num_stages = self.config.ici_pipeline_parallelism * self.config.dcn_pipeline_parallelism
    self.forwarding_delay = 2 if self.config.pipeline_delay_activation_forwarding else 1
    self.pipeline_microbatch_size = self.config.micro_batch_size_to_train_on // self.config.num_pipeline_microbatches
    microbatches_per_stage = self.config.num_pipeline_microbatches // self.num_stages
    self.microbatches_per_stage = microbatches_per_stage
    self.use_circ_storage = self.need_circ_storage()

    self.batch_axis_name = "activation_batch"
    self.seq_len_axis_name = "activation_length"

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

  def get_iteration_inputs(self, loop_iteration, state_io, circ_storage, shift):
    """
    Construct stages_in: the global array that is operated on for this iteration, shape same as
    shift=[stages, micro_size, sequence, embed]
    This is almost a rotated version of the last outputs, except for the first stage which must grab a new batch from
    state_io or an old one from circ_storage
    """
    pass

  def get_microbatch_and_repeat_ids(self, loop_iteration):
    """Gets the microbatch_ids and repeat_ids for all stages on this loop_iteration. Works for both circular and
    non-circular"""
    pass

  def get_pipeline_remat_policy(self):
    """Returns the pipeline remat policy for this pipeline."""
    pass

  def get_weight_sharding(self, *init_args):
    """get weight sharding function for this pipeline."""
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

  def _run_weight_initialization(
      self, example_inputs, example_segmentation, example_position, segment_idx, position_idx, deterministic, model_mode
  ):
    """Runs the initialization sequence mapping layers appropriately based on pipeline settings."""
    pass

  @staticmethod
  def _remove_fsdp_from_physical_partition_spec(pps):
    """Removes 'fsdp' and 'fsdp_transpose' from physical partition spec."""
    pass


class Pipeline(PipelineBase):
  """Original Pipeline implementation."""

  def init_states(self, inputs):
    """Initialize components of state: state_io, shift, circular_storage and circular_storage_mover
    Assumes input has already been reshaped into microbatches: [num_micro_batches, micro_batch_size, sequence, embed]

    Returns a dictionary with properties
      shift: zeros shape [num_stages, micro_size, sequence, embed]
      prev_outputs: same shape as shift, only used when pipeline_delay_activation_forwarding is set to true, else None
      state_io: reshaped inputs [num_stages, microbatches/stages, micro_size, sequence, embed]
      circ_storage: zeros [num_stages, microbatches, micro_size, sequence, embed] when needed, else None
      circ_storage_mover: zeros[num_stages, micro_size, sequence, embed] when needed, else None
      loop_iteration: scalar set initially to 0
      bsw: pytree of identical structure as weights with leaf arrays leading dimension of num_repeats replaced by 2, e.g.
        a leaf of shape [num_repeats, stages, mlp, embed] is mapped to [2, num_stages, mlp, embed].
    """
    pass

  def shard_dim_by_stages(self, x, dim: int, physical_partition_spec: P | None, is_stage_weight: bool = False):
    """Shards x using the provided partition_spec, but adds the "stage" mesh axis to the existing sharding at
    the specified dimension."""
    pass

  def vmap_parallel_gather(
      self, weights, physical_partition_spec, repeat_ids, repeat_dim_in_weights, stages_dim_in_weights
  ):
    """Use vmap to implement a sharded parallel gather.
    Parallel gather means each stage has its own weights, and gets one slice from it.
    Args:
      weights: Per-stage data to be gathered from.
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

  def permute_output_micro_per_stage_dim(self, output):
    """
    Permutes the output microbatches to match the input order.

    The pipeline execution introduces a delay (bubble) for each stage.
    Consequently, the first microbatch (index 0) finishes after a certain number of iterations
    and lands at a shifted position in the output buffer (`state_io`).
    This function calculates the offset (`microbatch_0_idx`) and permutes the output
    along the microbatch dimension so that microbatch 0 is at index 0, microbatch 1 at index 1, etc.
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
    """Fetches the weights for the current repeat from the stages."""
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
    and update the loop state.

    Args:
      loop_state: Dictionary containing the current state of the pipeline (state_io, shift, etc.)
      positions: Positional encodings.
      segment_ids: Segment IDs for packed sequences.
      deterministic: Boolean indicating if execution should be deterministic (e.g. for dropout).
      model_mode: Current model mode (train/predict).
      logical_partition_spec: Logical partition specification for weights.
    """
    pass

  @staticmethod
  def get_logical_spec_repeats_removed(full_logical):
    """Returns a new logical spec with 'circular_repeats' removed."""
    pass

  @staticmethod
  def _remove_logically_partition(weights):
    """Removes LogicallyPartitioned wrappers from the variables."""
    pass

  def all_gather_over_fsdp(self, variables, logical_partition_spec):
    """Gathers FSDP partitioned variables to reconstruct them fully."""
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
    inputs = inputs.reshape(
        (
            self.config.num_pipeline_microbatches,
            self.pipeline_microbatch_size,
            self.config.max_target_length,
            self.config.emb_dim,
        ),
        out_sharding=self.input_sharding,
    )
    example_inputs = jax.lax.broadcast(inputs[0], [self.num_stages])
    ag_sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec(None, None))

    if positions is not None:
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
      return self._run_weight_initialization(
          example_inputs, example_segmentation, example_position, segment_idx, position_idx, deterministic, model_mode
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
          variable_axes={"summaries": 0, "aux_loss": 0, "intermediates": 0, "hyper_params": 0},
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


class CircularPipeline(PipelineBase):
  """Implements an circular pipeline schedule with asynchronous weight prefetching.

  Circular pipelining reduces the pipeline "bubble" by interleaving multiple pipeline
  stages on the same physical devices. To hide the communication overhead of Fully
  Sharded Data Parallelism (FSDP), this module utilizes a Buffer Sliding Window (BSW)
  to prefetch and all-gather the weights for the *next* pipeline repeat while the
  *current* repeat is executing.
  """

  def init_states(self, inputs):
    """Initializes the pipeline execution state and communication buffers.

    This sets up the memory needed to pass activations between pipeline stages
    (`state_io` and `shift`) and allocates the empty Buffer Sliding Window (BSW)
    that will hold the gathered FSDP weights.
    """
    pass

  def gather_weights_across_stages_vmap(self, weights, repeat_ids, repeat_dim_in_weights, stages_dim_in_weights):
    """Uses jax.vmap to dynamically slice and gather weights for specific pipeline repeats."""
    pass

  def gather_microbatch_inputs_vmap(self, xs, ids, ids_dim):
    """Slices out the specific sequence inputs (e.g., positions, segments) for the current microbatch."""
    pass

  def advance_circular_buffers(self, output, loop_state):
    """Rotates pipeline activations to the next physical device stage.

    Uses `jax.lax.ppermute` to perform cross-device ring communication, shifting
    the forward activations (`state_io` and `shift`) from stage $i$ to stage $i+1$.
    """
    pass

  def realign_output_microbatches(self, output):
    """Reorders the output tensor to reverse the circular shifts applied during execution.

    Because the pipeline operates circularly, the output microbatches are shifted
    out of order by the time the final stage is completed. This rolls them back
    into their original sequential layout.
    """
    pass

  def fetch_active_stage_weights(self, bsw, loop_iteration, physical_partition_spec=None, is_initializing=None):
    """The module fetches the actively prefetched weights
    from the Buffer Sliding Window to avoid mid-iteration FSDP all-gathers.
    """
    pass

  def get_current_weights_from_bsw(self, bsw, loop_iteration, physical_partition_spec, is_initializing=None):
    """Pulls the fully gathered parameters for the current repeat from the BSW dual-buffer."""
    pass

  def from_all_variables_to_repeat_weights(self, weights, loop_iteration):
    """Gathers weights corresponding to the repeat IDs for current iteration."""
    pass

  def from_repeat_weights_to_bsw(
      self,
      repeat_weights,
      physical_partition_spec,
      axes_to_gather=("fsdp", "fsdp_transpose", "context", "expert"),
      # TODO (chengnuojin) set use_shardmap=true after JAX >= 10.0.0 and use all_gather(..., to='invarying')
      use_shardmap=False,  # using shardmap produces additional reduce-scatter in backward pass
  ):
    """Executes the FSDP-like all-gathers to fully materialize a block of weights for the BSW."""
    pass

  def weight_prefetching(self, weights, physical_partition_spec, loop_iteration):
    """Triggers asynchronous FSDP-like all-gathers for the next pipeline steps.

    By gathering weights for `loop_iteration + 1` right now, the network communication
    can overlap with the compute happening in `loop_iteration`.
    """
    pass

  def run_one_iteration(self, loop_state, bsw, positions, segment_ids, deterministic, model_mode, logical_partition_spec):
    """Executes the forward/backward logic for a single microbatch inside the pipeline.

    This acts as the core step function that our `jax.lax.scan` wrappers call. It routes
    the active BSW weights, sequences, and position IDs into the layer blocks, and then
    advances the pipeline communication buffers via `advance_circular_buffers`.
    """
    pass

  @nn.compact
  def __call__(
      self,
      inputs: jnp.ndarray,
      segment_ids: jnp.ndarray,
      positions: jnp.ndarray,
      deterministic: bool,
      model_mode=MODEL_MODE_TRAIN,
      logical_partition_spec=None,
  ) -> jnp.ndarray:
    """Entry point for the Pipeline Module. Sets up microbatch schedules and executes scans."""
    inputs = inputs.reshape(
        (
            self.config.num_pipeline_microbatches,
            self.pipeline_microbatch_size,
            self.config.max_target_length,
            self.config.emb_dim,
        ),
        out_sharding=self.input_sharding,
    )
    example_inputs = jax.lax.broadcast(inputs[0], [self.num_stages])
    ag_sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec(None, None))

    if positions is not None:
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

    loop_state, bsw = self.init_states(inputs)
    physical_partition_spec = logical_to_mesh(
        logical_partition_spec, mesh=self.mesh, rules=self.config.logical_axis_rules
    )

    bubble_iterations = self.forwarding_delay * (self.num_stages - 1)

    if self.is_initializing():
      return self._run_weight_initialization(
          example_inputs, example_segmentation, example_position, segment_idx, position_idx, deterministic, model_mode
      )

    logical_partition_spec = pipeline_utils.strip_pipeline_repeat_logical_axis(logical_partition_spec)


    if self.config.set_remat_policy_on_pipeline_iterations:
      run_iteration_scannable = nn.remat(
          run_iteration_scannable,
          prevent_cse=not self.config.scan_pipeline_iterations,
          policy=self.get_pipeline_remat_policy(),
      )

    # base scannable function used twice for real and bubble runs
    base_scannable = functools.partial(
        pipeline_utils.create_pipeline_stage,
        deterministic=deterministic,
        model_mode=model_mode,
        logical_partition_spec=logical_partition_spec,
        physical_partition_spec=physical_partition_spec,
        positions=positions,
        segment_ids=segment_ids,
    )

    run_one_repeat_scannable = base_scannable(length=self.config.num_pipeline_microbatches)
    run_bubbles_scannable = base_scannable(length=bubble_iterations)

    run_repeats_scanned = pipeline_utils.create_flax_pipeline_scan(
        pipeline_stage_fn=run_one_repeat_scannable,
        length=self.config.num_pipeline_repeats,
        remat_policy=self.get_pipeline_remat_policy(),
        use_scan=self.config.scan_pipeline_repeats,
    )
    run_bubbles_scanned = pipeline_utils.create_flax_pipeline_scan(
        pipeline_stage_fn=run_bubbles_scannable,
        length=1,
        remat_policy=self.get_pipeline_remat_policy(),
        use_scan=self.config.scan_pipeline_repeats,
    )
    initial_carry_repeats = (loop_state, bsw[0], self.layers.variables)
    (loop_state, w_curr, pipeline_weights), _ = run_repeats_scanned(self, initial_carry_repeats)
    initial_carry_bubbles = (loop_state, w_curr, pipeline_weights)
    (loop_state, _, pipeline_weights), _ = run_bubbles_scanned(self, initial_carry_bubbles)

    final_output = self.realign_output_microbatches(loop_state["state_io"])
    final_output = jnp.reshape(
        final_output,
        (self.config.micro_batch_size_to_train_on, self.config.max_target_length, self.config.emb_dim),
        out_sharding=self.output_sharding,
    )
    return final_output


def create_pipeline(config: Config, layers: nn.Module, mesh: Mesh, remat_policy: Any = None) -> PipelineBase:
  """Factory function to instantiate the correct Pipeline module based on config."""

  if config.pipeline_fsdp_ag_per_repeat:
    return CircularPipeline(config=config, layers=layers, mesh=mesh, remat_policy=remat_policy)

  return Pipeline(config=config, layers=layers, mesh=mesh, remat_policy=remat_policy)
