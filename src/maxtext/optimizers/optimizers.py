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

# pylint: disable=bare-except, consider-using-generator, too-many-positional-arguments
""" Utils that are only interesting to MaxText. """

import re
import jax
import jax.numpy as jnp

import optax
from optax.contrib._muon import muon
from maxtext.utils.muon_utils import get_muon_weight_dimension_numbers


def _get_path_mask_fn(patterns, match_returns_true=True):
  """Helper to create a mask function from a list of regex patterns."""
  if not patterns:
    return None

  compiled_patterns = [re.compile(pattern) for pattern in patterns]


  return mask_fn


def get_adamw_mask(config):
  """Create a mask function for AdamW optimizer to exclude certain parameters from weight decay."""
  return _get_path_mask_fn(getattr(config, "adamw_mask", None), match_returns_true=False)


def _compute_rolling_stats(arr: jax.Array, count: jax.Array, interval: int):
  """Computes mean and unbiased std (Bessel's correction) over a rolling window."""
  valid_elements = jnp.minimum(count, interval)
  safe_elements = jnp.maximum(1, valid_elements)
  mask = jnp.arange(interval) < valid_elements

  mean = jnp.sum(jnp.where(mask, arr, 0.0)) / safe_elements
  sq_diff = jnp.where(mask, (arr - mean) ** 2, 0.0)

  # Use Bessel's correction (N - 1) for unbiased variance to align with torch.std
  variance = jnp.sum(sq_diff) / jnp.maximum(1, valid_elements - 1)
  std = jnp.sqrt(variance)
  return mean, std


def skip_step_on_spikes(
    inner_opt: optax.GradientTransformation, interval: int, scaling_factor: float
) -> optax.GradientTransformationExtraArgs:
  """Wrapper that skips updates when loss or grad_norm spike.

  This wrapper calculates a rolling mean and standard deviation (using
  Bessel's correction) over the last `interval` steps for both the loss
  and the gradient norm. If the current step's loss or gradient norm
  exceeds `mean + scaling_factor * std`, the update is zeroed and the
  optimizer state is not advanced, effectively skipping the step.

  Reference implementation:
  https://github.com/allenai/OLMo-core/blob/c757b7c3c15197154c753d883330afbfa4869dcc/src/olmo_core/optim/skip_step_optimizer.py#L12

  Args:
    inner_opt: The inner Optax gradient transformation to wrap.
    interval: The number of recent steps to use for calculating mean and std.
    scaling_factor: The multiplier for standard deviation to set the spike threshold.

  Returns:
    An optax.GradientTransformationExtraArgs that skips spikes.
  """

  def init_fn(params):
    return {
        "inner_state": inner_opt.init(params),
        "losses": jnp.zeros(interval, dtype=jnp.float32),
        "grad_norms": jnp.zeros(interval, dtype=jnp.float32),
        "count": jnp.zeros((), dtype=jnp.int32),
        "is_skipped": jnp.array(False, dtype=jnp.bool_),
    }


  return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def get_optimizer(config, learning_rate_schedule, model=None):
  """Create optimizer."""
  if config.opt_type == "adamw":
    # Create AdamW Optimizer following Llama2's training details, see https://arxiv.org/pdf/2307.09288.pdf section 2.2
    base_opt = optax.adamw(
        learning_rate_schedule,
        b1=config.adam_b1,
        b2=config.adam_b2,
        eps=config.adam_eps,
        eps_root=config.adam_eps_root,
        weight_decay=config.adam_weight_decay,
        mu_dtype=config.mu_dtype,
        mask=get_adamw_mask(config),
    )
  elif config.opt_type == "adam_pax":
    base_opt = adam_pax(
        learning_rate_schedule,
        beta1=config.adam_b1,
        beta2=config.adam_b2,
        epsilon=config.adam_eps,
        epsilon_root=config.adam_eps_root,
        weight_decay=config.adam_weight_decay,
        mask=get_adamw_mask(config),
    )
  elif config.opt_type == "sgd":
    base_opt = optax.sgd(learning_rate_schedule)
  elif config.opt_type == "muon":
    # extract muon dimension number from model structure
    if model is not None:
      muon_weight_dimension_numbers = get_muon_weight_dimension_numbers(model, config)
    else:
      raise ValueError("Please specify model to extract muon dimension number.")
    muon_kwargs = {
        # Shared parameters: "nesterov" uses default
        "learning_rate": learning_rate_schedule,
        "eps": config.adam_eps,
        "mu_dtype": config.mu_dtype,
        # Muon-specific parameters: "ns_coeffs", "ns_steps", "weight_decay_mask", "adaptive" uses default
        "beta": config.muon_beta,
        "weight_decay": config.muon_weight_decay,
        "muon_weight_dimension_numbers": muon_weight_dimension_numbers,
        "consistent_rms": config.muon_consistent_rms,
        # AdamW-specific parameters
        "adam_b1": config.adam_b1,
        "adam_b2": config.adam_b2,
        "adam_eps_root": config.adam_eps_root,
        "adam_weight_decay": config.adam_weight_decay,
    }
    base_opt = muon(**muon_kwargs)
  else:
    raise ValueError(f"{config.opt_type=} is not a supported.")

  if getattr(config, "skip_step_on_spikes", False):
    base_opt = skip_step_on_spikes(
        base_opt,
        interval=config.skip_step_interval,
        scaling_factor=config.skip_step_scaling_factor,
    )

  # If a whitelist of trainable parameters is provided, freeze everything else.
  # When trainable_parameters_mask is empty, freeze_mask_fn is None and all parameters are trained.
  trainable_patterns = getattr(config, "trainable_parameters_mask", None)
  freeze_mask_fn = _get_path_mask_fn(trainable_patterns, match_returns_true=False)
  if freeze_mask_fn is not None:
    # Use optax.multi_transform to explicitly map frozen parameters to a stateless set_to_zero() optimizer.
    # If we simply wrapped base_opt in optax.masked() or chained it, Optax would still allocate
    # massive states (momentum, variance) for the entire model before zeroing the updates.
    # By using multi_transform, only the trainable parameters get states allocated.
    return optax.multi_transform(
        {"trainable": base_opt, "frozen": optax.set_to_zero()},
        lambda params: jax.tree_util.tree_map(lambda x: "frozen" if x else "trainable", freeze_mask_fn(params)),
    )

  return base_opt


def adam_pax(
    learning_rate_fn: optax.Schedule,
    beta1: float,
    beta2: float,
    epsilon: float,
    epsilon_root: float,
    weight_decay: float,
    mask=None,
) -> optax.GradientTransformation:
  """Standard Adam optimizer that supports weight decay.

  Follows the implementation in pax/praxis sharded_adam
  https://github.com/google/praxis/blob/545e00ab126b823265d70c715950d39333484f38/praxis/optimizers.py#L621

  Args:
    learning_rate_fn: a callable that given the current training step, returns
      the learning rate to apply.
    beta1: decay rate to track the first moment.
    beta2: decay rate to track the second moment.
    epsilon: Small constant applied to the denominator outside of the square
      root to avoid dividing by zero when rescaling.
    epsilon_root: Small constant applied to the denominator inside of the square
      root to avoid dividing by zero when rescaling.
    weight_decay: If > 0, weight decay to apply.

  Returns:
    A `optax.GradientTransformation`.
  """

  def init_fn(params):
    mu = jax.tree_util.tree_map(jnp.zeros_like, params)  # First moment
    nu = jax.tree_util.tree_map(jnp.zeros_like, params)  # Second moment
    return optax.ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def bias_corrected_decay(step: jnp.int32, decay: float):
    """Incorporates bias correction into decay.

    Please see section 7.1 in https://arxiv.org/pdf/1804.04235.pdf for the
    derivation of the formulas below. With bias-corrected decay, we can simply
    do

    m_{t} = decay1 * m_{t-1} + (1 - decay1) * g
    v_{t} = decay2 * v_{t-1} + (1 - decay2) * g ^ 2

    without further bias correction.

    Args:
      step: current step, 0-based.
      decay: the raw decay. As t -> infinity, bias corrected decay converges to
        this value.

    Returns:
      Bias corrected decay.
    """
    t = step.astype(jnp.float32) + 1.0
    return decay * (1.0 - jnp.power(decay, t - 1.0)) / (1.0 - jnp.power(decay, t))


  return optax.GradientTransformation(init_fn, update_fn)
