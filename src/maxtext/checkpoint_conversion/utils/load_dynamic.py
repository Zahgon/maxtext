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

"""Dynamic loading of HuggingFace checkpoints during training/eval workloads directly in the target format."""

import jax
from flax import traverse_util
from flax import nnx
from orbax.checkpoint import v1 as ocp_v1
from orbax.checkpoint._src.arrays import sharding as sharding_utils

from maxtext.utils import max_logging
from maxtext.checkpoint_conversion.utils.tensor_handling import _get_hf_loading_function
from maxtext.checkpoint_conversion.utils import param_mapping
from maxtext.checkpoint_conversion.utils.hf_model_configs import HF_MODEL_CONFIGS
import time


def get_hf_config_and_mappings(maxtext_config):
  """Gets HF config and parameter mapping based on the MaxText config."""
  model_key = maxtext_config.model_name
  if "-Instruct" in model_key:
    model_key = model_key.replace("-Instruct", "")
  hf_config_obj = HF_MODEL_CONFIGS[model_key]
  hf_config_dict = hf_config_obj.to_dict()

  param_map_mt_to_hf = param_mapping.PARAM_MAPPING[model_key](
      hf_config_dict, maxtext_config, scan_layers=maxtext_config.scan_layers
  )
  hook_fn_map_mt = param_mapping.HOOK_FNS[model_key](
      hf_config_dict, maxtext_config, scan_layers=maxtext_config.scan_layers, saving_to_hf=False
  )
  return param_map_mt_to_hf, hook_fn_map_mt


def load_sharded_hf_state(path):
  """Loads HF state with maximal sharding across TPU mesh to avoid host OOM."""
  t0 = time.time()
  context = ocp_v1.Context(checkpoint_layout=ocp_v1.options.CheckpointLayout.SAFETENSORS)
  with context:
    metadata = ocp_v1.pytree_metadata(path)
    simple_abstract_state = metadata.metadata
    
    # Distributed Sharded Download: Tell JAX to shard the HF Safetensors download 
    # across the TPU mesh as much as mathematically possible to avoid Host OOM and GCS throttling!
    shardings = sharding_utils.construct_maximal_shardings(simple_abstract_state)

    def combine_sharding(sds, single_sharding):
      return jax.ShapeDtypeStruct(shape=sds.shape, dtype=sds.dtype, sharding=single_sharding)

    sharded_abstract_state = jax.tree.map(combine_sharding, simple_abstract_state, shardings)
    
    max_logging.log("Reading raw Safetensors into memory (Distributed Sharded GCS Download)...")
    hf_state = ocp_v1.load_pytree(path, sharded_abstract_state)
    max_logging.log(f"load_sharded_hf_state took {time.time() - t0:.2f}s")
    return hf_state


def transform_hf_state_to_mt_state(
    hf_state, target_tree, param_map_mt_to_hf, hook_fn_map_mt, maxtext_config
):
  """Transforms HF state into MaxText state by applying param mappings and mathematical hooks."""
  t0 = time.time()
  def tensor_getter(key):
    return hf_state[key]

  flat_target = traverse_util.flatten_dict(target_tree, sep=".")
  flat_restored = flat_target.copy()

  mapped_count = 0
  keys_missed = []
  max_logging.log("Starting fast in-memory Distributed Transformations...")

  for mt_key, hf_source in param_map_mt_to_hf.items():
    mt_name = mt_key.replace("params-", "").replace("-", ".")
    
    # Determine the correct key in flat_target
    check_name = mt_name
    if check_name not in flat_target:
        if ("params." + mt_name) in flat_target:
            check_name = "params." + mt_name
        elif mt_key.replace("-", ".") in flat_target:
            check_name = mt_key.replace("-", ".")
    
    if check_name not in flat_target:
      keys_missed.append(mt_name)
      continue
      
    target_shape = flat_target[check_name].shape
    hook_fn = hook_fn_map_mt.get(mt_key)

    load_fn = _get_hf_loading_function(
        hf_source,
        tensor_getter,
        hook_fn,
        target_shape,
        maxtext_config,
    )

    # Execute transformation and assign to flat_restored
    t_layer = time.time()
    unsharded_array = load_fn()
    
    # Ensure it's Sharded explicitly matching the JAX model expectations
    target_sharding = flat_target[check_name].sharding
    flat_restored[check_name] = jax.device_put(unsharded_array, device=target_sharding)
    
    max_logging.log(f"Transformed {check_name} from {hf_source} in {time.time() - t_layer:.4f}s")
    mapped_count += 1
      
  if mapped_count == 0:
    max_logging.log(f"All transformations missed! Sample missed mt_names: {keys_missed[:5]}")
    max_logging.log(f"Sample flat_target keys: {list(flat_target.keys())[:5]}")
  
  max_logging.log(f"Successfully mapped {mapped_count} parameters.")
  restored_params = traverse_util.unflatten_dict(flat_restored, sep=".")
  
  if "params" in restored_params:
      restored_params = restored_params["params"]
  
  max_logging.log(f"transform_hf_state_to_mt_state took {time.time() - t0:.2f}s")
  
  return {"params": restored_params}


def load_safetensors_dynamic_state(path, abstract_unboxed_pre_state, maxtext_config):
  """Main entry point to dynamically build and load safetensors into MaxText format.
  
  Splits execution into:
  1. Deriving Mappings
  2. Loading Sharded arrays directly to TPUs
  3. Processing the transformations natively on TPUs
  """
  if maxtext_config is None:
    raise ValueError("maxtext_config must be provided for safetensors_dynamic loading.")
  
  t_total = time.time()
  param_map_mt_to_hf, hook_fn_map_mt = get_hf_config_and_mappings(maxtext_config)
  max_logging.log(f"[1/3] Mappings derived in {time.time() - t_total:.2f}s")

  target_tree = (
      abstract_unboxed_pre_state.to_pure_dict()
      if isinstance(abstract_unboxed_pre_state, nnx.State)
      else abstract_unboxed_pre_state.params
  )
  
  t1 = time.time()
  hf_state = load_sharded_hf_state(path)
  max_logging.log(f"[2/3] Distributed Sharded GCS load completed in {time.time() - t1:.2f}s")
  
  t2 = time.time()
  restored_params = transform_hf_state_to_mt_state(
      hf_state, target_tree, param_map_mt_to_hf, hook_fn_map_mt, maxtext_config
  )
  max_logging.log(f"[3/3] CPU Transformations completed in {time.time() - t2:.2f}s")
  max_logging.log(f"Total safetensors_dynamic duration: {time.time() - t_total:.2f}s")
  
  return None, restored_params
