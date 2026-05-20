"""Unit tests targeting prefill packing processor compiled paths on CPU backend."""

import functools
import os
import unittest
import jax
import jax.numpy as jnp
import pytest
from maxtext.configs import pyconfig
from maxtext.inference.page_manager import PageManager
from maxtext.input_pipeline.packing.prefill_packing import BatchedPrefillProcessor
from maxtext.input_pipeline.packing.prefill_packing import PrefillProcessor

# Decorate all tests as cpu_only to be eagerly selected by GHA CPU runners
pytestmark = [pytest.mark.cpu_only]

# Force CPU execution
os.environ["JAX_PLATFORMS"] = "cpu"
jax.config.update("jax_default_prng_impl", "unsafe_rbg")


@jax.tree_util.register_pytree_node_class
class MockToken:
  """A first-class JAX PyTree representing a sampled token, traceable inside JIT compilations."""

  def __init__(self, data, log_prob):
    self.data = data
    self.log_prob = log_prob

  def tree_flatten(self):
    children = (self.data, self.log_prob)
    aux_data = None
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(*children)


class MockEngine:
  """A mock MaxEngine returning registered MockToken PyTrees and supporting fully aligned batch layout structures."""

  def __init__(self, config):
    self.config = config
    self.param_layouts = {"params": None}

    # Fully aligned structures matching AOT compile & runtime layout dictionaries
    self.decode_state_layouts = {"cache": {"dummy": None}, "prompt_logp": None}
    self.decode_state_shapes = {
        "cache": {"dummy": jax.ShapeDtypeStruct((1, 1), jnp.float32)},
        "prompt_logp": jax.ShapeDtypeStruct((1,), jnp.float32),
    }

    if config.attention == "paged":
      self.page_manager = PageManager(config)
      self.page_state = self.page_manager.get_initial_page_state()
    else:
      self.page_manager = None
      self.page_state = None

    # Mock jitted prefill executor returning registered MockToken
    @functools.partial(jax.jit, static_argnames=("return_prompt_logp",))
    def mock_prefill_jit(params, padded_tokens, true_length, rng, page_state=None, slot=0, return_prompt_logp=False):
      prefill_result = {"cache": {"dummy": jnp.ones((1, 1))}}
      if return_prompt_logp:
        prefill_result["prompt_logp"] = jnp.zeros((1,), dtype=jnp.float32)
      first_token = MockToken(jnp.ones((1, 1), dtype=jnp.int32), jnp.zeros((1, 1), dtype=jnp.float32))
      return prefill_result, first_token

    # Mock jitted insert executor
    @jax.jit
    def mock_insert_jit(prefix, decode_state, slot, request_id=None, page_state_in=None):
      return {"cache": {"dummy": jnp.ones((1, 1))}, "prompt_logp": jnp.zeros((1,))}

    # Mock jitted prefill_concat executor returning lists of valid JAX tuples
    @functools.partial(jax.jit, static_argnames=("num_prompts", "return_prompt_logp"))
    def mock_prefill_concat_jit(
        params,
        padded_tokens,
        decoder_positions,
        decoder_segment_ids,
        start_pos,
        true_lengths,
        num_prompts,
        return_prompt_logp=False,
    ):
      cache = {"dummy": jnp.ones((1, 1))}
      prefix_state = {"cache": cache}
      if return_prompt_logp:
        prefix_state["prompt_logp"] = jnp.zeros((1,), dtype=jnp.float32)
      tokens_list = [
          (jnp.ones((1, 1), dtype=jnp.int32), jnp.zeros((1, 1), dtype=jnp.float32)) for _ in range(num_prompts)
      ]
      return cache, prefix_state, tokens_list

    # Mock jitted insert_partial executor
    @functools.partial(
        jax.jit,
        static_argnames=("num_prompts", "seq_len"),
    )
    def mock_insert_partial_jit(
        prefix,
        decode_state,
        cache,
        slots,
        start_indices,
        num_prompts,
        seq_len,
    ):
      return {"cache": {"dummy": jnp.ones((1, 1))}, "prompt_logp": jnp.zeros((1,))}

    self._prefill_jit = mock_prefill_jit
    self._insert_jit = mock_insert_jit
    self._prefill_concat_jit = mock_prefill_concat_jit
    self._insert_partial_jit = mock_insert_partial_jit

  # Positional standard method wrappers matching MaxEngine API
  def prefill(
      self,
      params,
      padded_tokens,
      true_length,
      rng,
      page_state=None,
      slot=0,
      return_prompt_logp=False,
  ):
    """Mock public prefill API delegating positionally to the JIT executor."""
    return self._prefill_jit(
        params,
        padded_tokens,
        true_length,
        rng,
        page_state,
        slot,
        return_prompt_logp,
    )

  def insert(self, prefix, decode_state, slot, request_id=None):
    """Mock public insert API delegating positionally to the JIT executor."""
    return self._insert_jit(prefix, decode_state, slot, request_id, None)

  def prefill_concat(
      self,
      params,
      padded_tokens,
      decoder_positions,
      decoder_segment_ids,
      start_pos,
      true_lengths,
      num_prompts,
      return_prompt_logp=False,
  ):
    """Mock public prefill_concat API delegating positionally to the JIT executor."""
    cache, prefix_state, tokens_list = self._prefill_concat_jit(
        params,
        padded_tokens,
        decoder_positions,
        decoder_segment_ids,
        start_pos,
        true_lengths,
        num_prompts,
        return_prompt_logp,
    )
    first_tokens = [MockToken(t[0], t[1]) for t in tokens_list]
    return cache, prefix_state, first_tokens

  def insert_partial(
      self,
      prefix,
      decode_state,
      cache,
      slots,
      start_indices,
      num_prompts,
      seq_len,
  ):
    """Mock public insert_partial API delegating positionally to the JIT executor."""
    return self._insert_partial_jit(
        prefix,
        decode_state,
        cache,
        slots,
        start_indices,
        num_prompts,
        seq_len,
    )


class PrefillPackingCPUTest(unittest.TestCase):

  def setUp(self):
    """Initialize test configurations and default keys eagerly."""
    super().setUp()
    self.config_args = {
        "max_prefill_predict_length": 8,
        "max_target_length": 16,
        "pagedattn_num_pages": 16,
        "pagedattn_tokens_per_page": 4,
        "pagedattn_max_pages_per_group": 4,
        "per_device_batch_size": 1.0,
        "global_batch_size_to_load": 1,
    }
    self.rng = jax.random.PRNGKey(0)

  def init_pyconfig(self, **kwargs):
    """Initialize pyconfig using designated path and configs override."""
    args = [None, "src/maxtext/configs/base.yml"]
    merged = {**self.config_args, **kwargs}
    for k, v in merged.items():
      args.append(f"{k}={v}")
    return pyconfig.initialize(args)

  def test_prefill_packing_non_paged_coverage_cpu(self):
    """Verifies prefill_packing process and AOT compilation under default non-paged attention on CPU using MockEngine."""
    config = self.init_pyconfig(attention="dot_product")
    engine = MockEngine(config)

    prefill_processor = PrefillProcessor(engine)
    params = {"params": None}

    # Trigger aot_compile coverage for non-paged (is_paged=False)
    prefill_processor.aot_compile(params, config.max_prefill_predict_length)

    # Trigger process coverage for non-paged (is_paged=False)
    decode_state = {"cache": {"dummy": jnp.ones((1, 1))}, "prompt_logp": jnp.zeros((1,))}
    input_tokens = jnp.ones((config.max_prefill_predict_length,), dtype=jnp.int32)

    first_token, decode_state, page_state = prefill_processor.process(
        model_params=params,
        decode_state=decode_state,
        decode_slot=0,
        input_tokens_padded=input_tokens,
        input_true_length=4,
        rng=self.rng,
        page_state=None,
    )

    self.assertIsNotNone(first_token)
    self.assertIsNotNone(decode_state)
    self.assertIsNone(page_state)

  def test_prefill_packing_paged_coverage_cpu(self):
    """Verifies prefill_packing process and AOT compilation under paged attention on CPU using MockEngine."""
    config = self.init_pyconfig(attention="paged")
    engine = MockEngine(config)

    prefill_processor = PrefillProcessor(engine)
    params = {"params": None}

    # Trigger aot_compile coverage for paged (is_paged=True)
    prefill_processor.aot_compile(params, config.max_prefill_predict_length)

    # Trigger process coverage for paged (is_paged=True)
    decode_state = {"cache": {"dummy": jnp.ones((1, 1))}, "prompt_logp": jnp.zeros((1,))}
    page_state = engine.page_state
    input_tokens = jnp.ones((config.max_prefill_predict_length,), dtype=jnp.int32)

    first_token, decode_state, page_state = prefill_processor.process(
        model_params=params,
        decode_state=decode_state,
        decode_slot=0,
        input_tokens_padded=input_tokens,
        input_true_length=4,
        rng=self.rng,
        page_state=page_state,
    )

    self.assertIsNotNone(first_token)
    self.assertIsNotNone(decode_state)
    self.assertIsNotNone(page_state)

  def test_prefill_packing_with_logp_coverage_cpu(self):
    """Verifies prefill_packing process and AOT compilation with return_prompt_logp=True on CPU using MockEngine."""
    config = self.init_pyconfig(attention="dot_product")
    engine = MockEngine(config)

    prefill_processor = PrefillProcessor(engine)
    params = {"params": None}

    # AOT compile with return_prompt_logp=True
    prefill_processor.aot_compile(params, config.max_prefill_predict_length)

    decode_state = {"cache": {"dummy": jnp.ones((1, 1))}, "prompt_logp": jnp.zeros((1,))}
    input_tokens = jnp.ones((config.max_prefill_predict_length,), dtype=jnp.int32)

    # Process with return_prompt_logp=True
    first_token, decode_state, page_state = prefill_processor.process(
        model_params=params,
        decode_state=decode_state,
        decode_slot=0,
        input_tokens_padded=input_tokens,
        input_true_length=4,
        rng=self.rng,
        return_prompt_logp=True,
        page_state=None,
    )
    self.assertIsNotNone(first_token)
    self.assertIsNotNone(decode_state)
    self.assertIsNone(page_state)

  def test_prefill_packing_paged_assertion_coverage_cpu(self):
    """Verifies that passing page_state=None under attention='paged' triggers AssertionErrors for complete coverage."""
    config = self.init_pyconfig(attention="paged")
    engine = MockEngine(config)

    prefill_processor = PrefillProcessor(engine)
    params = {"params": None}

    prefill_processor.aot_compile(params, config.max_prefill_predict_length)
    decode_state = {"cache": {"dummy": jnp.ones((1, 1))}, "prompt_logp": jnp.zeros((1,))}
    input_tokens = jnp.ones((config.max_prefill_predict_length,), dtype=jnp.int32)

    # Trigger assertion checks
    with self.assertRaises(AssertionError):
      prefill_processor.process(
          model_params=params,
          decode_state=decode_state,
          decode_slot=0,
          input_tokens_padded=input_tokens,
          input_true_length=4,
          rng=self.rng,
          page_state=None,
      )

  def test_batched_prefill_processor_coverage_cpu(self):
    """Verifies BatchedPrefillProcessor AOT compilation, bucket processing, and flushing on CPU using MockEngine."""
    config = self.init_pyconfig(attention="dot_product")
    engine = MockEngine(config)

    # Instantiate BatchedPrefillProcessor with MockEngine
    batched_processor = BatchedPrefillProcessor(engine, max_batch_size=4)
    params = {"params": None}

    # 1. Trigger aot_compile coverage
    batched_processor.aot_compile(params, input_padding=8, capacity=32, num_prompts=4)

    # 2. Trigger process bucket queue allocation
    decode_state = {"cache": {"dummy": jnp.ones((1, 1))}, "prompt_logp": jnp.zeros((1,))}
    input_prompt = jnp.ones((6,), dtype=jnp.int32)

    done_called = [False]

    def prefill_done_callback(prefill_result, row_ids, state):
      del prefill_result, row_ids, state  # Unused params inside test callback
      done_called[0] = True

    # Add first prompt (length 6, fits in unallocated capacity)
    batched_processor.process(
        model_params=params,
        decode_state=decode_state,
        decode_slot=0,
        input_id=100,
        input_prompt=input_prompt,
        input_padding=8,
        capacity=8,  # Cap is 8
        prefill_done=prefill_done_callback,
    )
    # Done callback shouldn't trigger yet (capacity is not exceeded!)
    self.assertFalse(done_called[0])

    # Add prompt that exceeds capacity -> raises ValueError
    with self.assertRaises(ValueError):
      batched_processor.process(
          model_params=params,
          decode_state=decode_state,
          decode_slot=2,
          input_id=102,
          input_prompt=jnp.ones((10,), dtype=jnp.int32),
          input_padding=8,
          capacity=8,
          prefill_done=prefill_done_callback,
      )

    # Add second prompt (length 4, exceeds bucket unallocated capacity of 2!)
    # This should trigger _process_bucket!
    batched_processor.process(
        model_params=params,
        decode_state=decode_state,
        decode_slot=1,
        input_id=101,
        input_prompt=jnp.ones((4,), dtype=jnp.int32),
        input_padding=8,
        capacity=8,
        prefill_done=prefill_done_callback,
    )
    # Done callback should be triggered by bucket flush!
    self.assertTrue(done_called[0])

    # 3. Trigger flush coverage for remaining items
    done_called[0] = False
    # Flush remaining items
    batched_processor.flush(
        model_params=params,
        decode_state=decode_state,
        prefill_done=prefill_done_callback,
    )
    self.assertTrue(done_called[0])


if __name__ == "__main__":
  unittest.main()
