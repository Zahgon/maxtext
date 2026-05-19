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

"""Unit tests for the NNX branches of load_state_if_possible and the params adapter."""

import os
import tempfile
import unittest
from unittest import mock

from etils import epath
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import nnx

from maxtext.common import checkpointing
from maxtext.layers import train_state_nnx


class _Model(nnx.Module):
  """Tiny single-linear NNX model for restore tests."""

  def __init__(self, rngs: nnx.Rngs):
    self.linear = nnx.Linear(2, 1, rngs=rngs)


def _abstract_nnx_state():
  """Build an nnx.State from a TrainStateNNX — same shape that pre_train passes in."""
  model = _Model(rngs=nnx.Rngs(0))
  optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
  return nnx.state(train_state_nnx.TrainStateNNX(model, optimizer))


def _nnx_params_abstract():
  """The nnx.State of just the Param group — what load_params_from_path gets for NNX."""
  model = _Model(rngs=nnx.Rngs(0))
  _, params, _ = nnx.split(model, nnx.Param, ...)
  return params


def _concrete_weights():
  return {
      "linear": {
          "kernel": jnp.arange(2, dtype=jnp.float32).reshape(2, 1),
          "bias": jnp.array([7.0], dtype=jnp.float32),
      }
  }


def _save_tree(path, tree):
  ckptr = ocp.PyTreeCheckpointer(use_ocdbt=True, use_zarr3=True)
  ckptr.save(epath.Path(path), tree, force=True)


class TestLoadStateIfPossibleNNX(unittest.TestCase):
  """Cover the NNX branches in load_state_if_possible."""

  def test_load_parameters_from_path_splits_nnx_state_for_param_view(self):
    """When abstract_unboxed_pre_state is an nnx.State, the function must call
    nnx.split(model, nnx.Param, ...) to get the params and forward them to load_params_from_path."""
    abstract = _abstract_nnx_state()
    sentinel_restored = {"linear": {"kernel": jnp.ones((2, 1)), "bias": jnp.zeros((1,))}}

    with mock.patch.object(checkpointing, "load_params_from_path", return_value=sentinel_restored) as m:
      full, params = checkpointing.load_state_if_possible(
          checkpoint_manager=None,
          data_iterator=None,
          load_parameters_from_path="gs://does-not-exist/params",
          load_full_state_from_path="",
          checkpoint_storage_concurrent_gb=8,
          abstract_unboxed_pre_state=abstract,
      )

    self.assertIsNone(full)
    self.assertIs(params, sentinel_restored)
    m.assert_called_once()
    forwarded_params = m.call_args[0][1]  # second positional arg = abstract_unboxed_params
    # The forwarded params come from nnx.split(..., nnx.Param, ...) — same key shape as the model.
    leaves = jax.tree.leaves(forwarded_params)
    self.assertEqual(len(leaves), 2)  # linear.kernel + linear.bias

  def test_load_parameters_from_path_uses_state_params_for_linen(self):
    """For Linen TrainState, the function must use state.params (not nnx.split)."""
    fake_state = mock.Mock(spec=["params"])
    fake_state.params = {"layer": {"kernel": jnp.ones((2, 2))}}
    sentinel = object()

    with mock.patch.object(checkpointing, "load_params_from_path", return_value=sentinel) as m:
      full, params = checkpointing.load_state_if_possible(
          checkpoint_manager=None,
          data_iterator=None,
          load_parameters_from_path="gs://does-not-exist/params",
          load_full_state_from_path="",
          checkpoint_storage_concurrent_gb=8,
          abstract_unboxed_pre_state=fake_state,
      )

    self.assertIsNone(full)
    self.assertIs(params, sentinel)
    forwarded_params = m.call_args[0][1]
    self.assertIs(forwarded_params, fake_state.params)

  def test_no_paths_returns_none_none(self):
    """Sanity: with no checkpoint manager and no load paths, the function returns (None, None)."""
    full, params = checkpointing.load_state_if_possible(
        checkpoint_manager=None,
        data_iterator=None,
        load_parameters_from_path="",
        load_full_state_from_path="",
        checkpoint_storage_concurrent_gb=8,
        abstract_unboxed_pre_state=_abstract_nnx_state(),
    )
    self.assertIsNone(full)
    self.assertIsNone(params)


class TestParamsAdapter(unittest.TestCase):
  """Round-trip tests for the Linen<->NNX params adapter in load_params_from_path."""

  def setUp(self):
    self._tmp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
    self.addCleanup(self._tmp.cleanup)
    self.weights = _concrete_weights()
    self.linen_path = os.path.join(self._tmp.name, "linen_ckpt")
    self.nnx_path = os.path.join(self._tmp.name, "nnx_ckpt")
    # Linen on-disk: params/params/<weights>; the extra `step` exercises partial_restore.
    _save_tree(self.linen_path, {"params": {"params": self.weights}, "step": jnp.array(3)})
    # NNX on-disk: model/<weights>; the extra `optimizer` exercises partial_restore.
    _save_tree(self.nnx_path, {"model": self.weights, "optimizer": {"step": jnp.array(3)}})

  def _assert_weights_match(self, got):
    self.assertEqual(set(got.keys()), {"linear"})
    self.assertTrue(jnp.array_equal(got["linear"]["kernel"], self.weights["linear"]["kernel"]))
    self.assertTrue(jnp.array_equal(got["linear"]["bias"], self.weights["linear"]["bias"]))

  def test_linen_disk_into_nnx_target(self):
    """A Linen-format checkpoint loads into an NNX target."""
    restored = checkpointing.load_params_from_path(self.linen_path, _nnx_params_abstract(), 8)
    self.assertIsInstance(restored, nnx.State)
    self._assert_weights_match(restored.to_pure_dict())

  def test_nnx_disk_into_linen_target(self):
    """An NNX-format checkpoint loads into a Linen target."""
    restored = checkpointing.load_params_from_path(self.nnx_path, {"params": self.weights}, 8)
    self.assertIn("params", restored)
    self._assert_weights_match(restored["params"])

  def test_nnx_disk_into_nnx_target(self):
    """An NNX-format checkpoint loads into an NNX target."""
    restored = checkpointing.load_params_from_path(self.nnx_path, _nnx_params_abstract(), 8)
    self.assertIsInstance(restored, nnx.State)
    self._assert_weights_match(restored.to_pure_dict())

  def test_is_nnx_format_on_disk(self):
    """Format detection distinguishes NNX from Linen on disk."""
    # pylint: disable=protected-access
    self.assertTrue(checkpointing._is_nnx_format_on_disk(self.nnx_path, True, True))
    self.assertFalse(checkpointing._is_nnx_format_on_disk(self.linen_path, True, True))


class TestAdapterHelpers(unittest.TestCase):
  """Unit tests for the adapter's tree-shaping helpers."""

  def test_rebuild_nnx_with_values_leaf_mismatch_raises(self):
    """A leaf-count mismatch between abstract and restored arrays raises."""
    abstract = _nnx_params_abstract()  # two Variables: linear.kernel + linear.bias
    with self.assertRaises(ValueError):
      checkpointing._rebuild_nnx_with_values(abstract, {"linear": {"kernel": jnp.ones((2, 1))}})  # pylint: disable=protected-access


if __name__ == "__main__":
  unittest.main()
