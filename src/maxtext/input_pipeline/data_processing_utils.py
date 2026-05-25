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

"""Utility functions for data processing pipelines."""

import functools

import jax
from grain.experimental import BestFitPackIterDataset, pick_performance_config
import grain.python as grain

from maxtext.input_pipeline import input_pipeline_utils
from maxtext.input_pipeline import tokenizer
from maxtext.utils import elastic_utils


def parse_and_keep_features(dataset, config, data_columns, tokenize):
  """Parse arrayrecord features or keep specified columns for other formats."""
  pass


def get_tokenizer_and_pad_id(config):
  """Builds tokenizer and extracts pad_id safely."""
  pass


def validate_and_configure_sft_columns(data_columns, tokenizer_model, chat_template=None):
  """Validates SFT data columns and configures the tokenizer chat template."""
  if chat_template and hasattr(tokenizer_model, "chat_template"):
    tokenizer_model.chat_template = chat_template

  supported_columns = [["prompt", "completion"], ["messages"], ["question", "answer"]]
  assert any(
      set(data_columns) == set(supported) for supported in supported_columns
  ), f"Dataset column names mismatch. Expected columns to match one of {supported_columns}, but got {data_columns}"


def get_local_batch_size(config):
  """Computes local batch size based on process count and expansion factor."""
  if config.elastic_enabled:
    batch_size = elastic_utils.get_local_batch_size(config)
  else:
    batch_size = config.global_batch_size_to_load // jax.process_count()
  if config.expansion_factor_real_data > 1:
    # global_batch_size_to_load has been expanded in pyconfig.py when expansion_factor_real_data > 1.
    # But when using Grain, we want to keep the batch_size consistent with that in the checkpoint.
    # We revert the batch_size expansion here, but load multiple batches per step in multihost_dataloading.py.
    batch_size = int(batch_size // config.expansion_factor_real_data)
  return batch_size


def format_and_batch(dataset, config, batch_size, pad_id, data_columns, tokenizer_model, shift=True):
  """Packs or pads the dataset, batches it, and optionally shifts tokens for next-token prediction.

  When `config.grain_use_elastic_iterator` is True, batching is skipped
  (ElasticIterator performs it internally) and, if `shift=True`, the shift is
  applied pre-batch on axis 0, which is equivalent to a post-batch axis=1 shift.

  `shift` should be False for pipelines that don't do next-token prediction
  (e.g. DPO, which scores full sequences).
  """
  pass


def apply_multiprocessing_and_prefetch(dataset, config, grain_worker_count, grain_per_worker_buffer_size):
  """Applies multiprocessing and prefetching configurations to the dataset."""
  pass
