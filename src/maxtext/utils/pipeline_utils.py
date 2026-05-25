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

"""Pipeline layer wrapping a decoder layer(s). Supports circular pipelining"""

import functools
import jax
from jax.sharding import PartitionSpec as P
from flax import linen as nn
from flax.linen.spmd import LogicallyPartitioned
import jax.numpy as jnp


def get_mesh_axis_dim_indices(physical_partition_spec, axis_name="fsdp"):
  """Finds the tensor dimension index sharded across a specific physical mesh axis.

  In JAX sharding, a PartitionSpec maps tensor dimensions to physical device mesh axes.
  This utility traverses a PyTree of PartitionSpecs and returns the integer index of the
  tensor dimension that is mapped to the target `axis_name` (e.g., finding which dimension
  is FSDP-sharded to prepare for an all-gather operation).

  Args:
    physical_partition_spec: A PyTree where leaves are `jax.sharding.PartitionSpec` objects.
    axis_name: The physical mesh axis string to search for (defaults to "fsdp").

  Returns:
    A PyTree of the exact same structure where the leaves are integers representing the
    dimension index of the target axis, or -1 if the axis is not found in that spec.
  """
  pass


def derive_stage_weight_partition_specs(physical_partition_spec, axes_to_remove):
  """Derives the physical partition specs for weights inside the scanned pipeline loop.

  When weights enter the inner `jax.lax.scan` loop for microbatch execution, their
  sharding requirements change. This function modifies the base weight specs by:
  1. Removing physical axes that will be all-gathered during the forward pass
     (e.g., FSDP axes, and Expert axes outside of the routed MoE block).
  2. Slicing off the first dimension `[1:]`, which typically represents the
     outer pipeline stage or layer-repeat dimension that the scan operates over.

  Args:
    physical_partition_spec: A PyTree of `PartitionSpec` objects for the full model weights.
    axes_to_remove: list of physical axes to remove.

  Returns:
    A PyTree of `PartitionSpec` objects tailored for the inner scanned execution block.
  """
  pass


def remove_gathered_mesh_axes(pps, is_moe_block_0, axes_to_remove):
  """Strips FSDP and specific MoE mesh axes from a PartitionSpec.

  When FSDP or Expert-sharded weights are all-gathered for computation, the resulting
  tensor is no longer sharded across those physical mesh dimensions. This function
  removes those axes from the PartitionSpec, replacing them with `None` to indicate
  replication across that mesh dimension.

  Args:
    pps: A single `jax.sharding.PartitionSpec` object.
    is_moe_block_0: Boolean indicating if the target is the routed MoE block. The 'expert'
                    mesh axis is only retained for the routed block.
    axes_to_remove: physical axes that we should remove from current physical partition axes

  Returns:
    A new `PartitionSpec` with the gathered axes removed, or the original object if it
    was not a PartitionSpec.
  """
  pass


def strip_pipeline_repeat_logical_axis(full_logical_spec):
  """Removes 'circular_repeats' from a logical PartitionSpec PyTree.

  Args:
    full_logical_spec: A PyTree of logical PartitionSpecs (strings like 'vocab', 'embed').

  Returns:
    A PyTree with 'circular_repeats' filtered out of all logical partition tuples.
  """
  pass


# TODO(chengnuojin) Remove this function and its usage after pipeline nnx migration
def remove_logically_partition(weights):
  """Removes LogicallyPartitioned wrapper from weights."""
  pass


def create_gradient_accumulation_scan(
    model,
    length,
    deterministic=True,
    model_mode=None,
    logical_partition_spec=None,
):
  """Creates a memory-efficient `jax.lax.scan` loop for pipeline microbatches with a custom VJP.

  In pipeline parallelism, scanning over microbatches normally forces JAX to save
  heavy parameter states for every iteration to compute the backward pass. This helper
  defines a custom Vector-Jacobian Product (VJP) to solve this by:

  1. Forward pass: Separating transient activations (`lightweight_state`) from heavy
     parameters (`bsw` and `weights`).
  2. Rematerialization: Recomputing the forward pass of individual steps during the backward
     pass to save memory.
  3. Backward pass: Manually accumulating gradients (`d + g`) onto the heavy parameters
     across the scanned iterations, rather than letting the standard autodiff trace them linearly.

  Args:
    model: The model instance containing the `run_one_iteration` and rematerialization logic.
    length: The number of microbatch iterations to scan over.
    deterministic: Whether to run the model in a deterministic mode (e.g., disable dropout).
    model_mode: The operational mode of the model (e.g., 'train', 'eval').
    logical_partition_spec: Rules for logical partitioning in standard tensor parallelism.

  Returns:
    A JAX custom_vjp function that executes the `length` pipeline iterations.
  """
  pass


def create_pipeline_stage(
    length,
    deterministic,
    model_mode,
    logical_partition_spec,
    physical_partition_spec,
    positions,
    segment_ids,
):
  """Builds an execution block for a single pipeline stage.

  This function prepares the state for a specific chunk of pipeline execution by:
  1. Prefetching the required weights (e.g., FSDP-gathered) for the current stage/loop iteration.
  2. Executing `length` microbatches using a memory-efficient `jax.lax.scan` via a custom VJP
     that manages collective communication overlap.

  Args:
    length: The number of microbatches to process in this stage.
    deterministic: Whether to run deterministically (e.g., disable dropout).
    model_mode: The operational mode (e.g., 'train').
    logical_partition_spec: Rules for logical tensor sharding.
    physical_partition_spec: Rules for physical device mesh mappings (used in prefetching).
    positions: Position IDs for the sequence.
    segment_ids: Segment/Attention routing IDs for the sequence.

  Returns:
    A function that takes `(model, carry)` and returns the updated `carry` and `None` for the scan outputs.
  """
  pass


def create_flax_pipeline_scan(pipeline_stage_fn, length, remat_policy, use_scan=True):
  """Wraps the pipeline stage execution in `flax.linen.remat` and `flax.linen.scan`.

  This explicitly wraps the pipeline step in a gradient checkpointing policy
  and then lifts it so it can be repeated sequentially over the specified length.
  It safely handles Flax-specific state collections, ensuring that metrics, intermediate
  values, and PRNG keys do not collide or overwrite each other across loop iterations.

  Args:
    pipeline_stage_fn: The function representing a single pipeline stage
                       (usually created by `create_pipeline_stage`).
    remat_policy: The checkpointing policy used by `nn.remat` to manage activation memory.
    length: The total number of pipeline stages/repeats to scan over.
    use_scan: Whether to use `jax.lax.scan` (True) or unroll the loop (False).

  Returns:
    A Flax scanned function that executes the full pipeline schedule.
  """
  pass
