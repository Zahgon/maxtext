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

"""Shared training and data loading hooks for post-training."""

from collections import defaultdict
import abc
from typing import override

import jax
import jax.numpy as jnp
import time

from flax import nnx

from tunix.sft import peft_trainer
from tunix.sft.hooks import DataHooks, TrainingHooks

from maxtext.input_pipeline.input_pipeline_interface import create_data_iterator
from maxtext.common.data_loader import DataLoader
from maxtext.common.goodput import GoodputEvent, record_goodput
from maxtext.common.metric_logger import MetricLogger, MetadataKey
from maxtext.utils import exceptions
from maxtext.utils import gcs_utils
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils import sharding


class BaseTrainingHooks(TrainingHooks, abc.ABC):
  """Shared training hooks for post-training."""

  def __init__(self, config, mesh, learning_rate_schedule, goodput_recorder):
    self.config = config
    self.mesh = mesh
    self.metric_logger = MetricLogger(self.config, learning_rate_schedule)
    self.goodput_recorder = goodput_recorder
    self.metadata = {}
    self.train_metadata = defaultdict(float)
    self.eval_metadata = defaultdict(float)
    self.step_start_time = 0.0

  @override
  def on_train_start(self, train_ctx: peft_trainer.PeftTrainer):
    """Called at the beginning of training."""
    pass

  @override
  def on_train_end(self, train_ctx: peft_trainer.PeftTrainer):  # pylint: disable=unused-argument
    """Called at the end of training."""
    pass

  @override
  def on_train_step_start(self, train_ctx: peft_trainer.PeftTrainer):
    """Called at the beginning of a training step."""
    pass

  @override
  def on_train_step_end(
      self,
      train_ctx: peft_trainer.PeftTrainer,
      train_step: int,
      train_loss: float,
      step_time: float = 0.0,  # No longer provided. See https://github.com/google/tunix/pull/1289.
  ):
    """Called at the end of training step."""
    pass

  @override
  def on_eval_step_start(self, train_ctx: peft_trainer.PeftTrainer):
    """Called at the beginning of an evaluation step."""
    pass

  @override
  def on_eval_step_end(self, train_ctx: peft_trainer.PeftTrainer, eval_loss: float):
    """Called at the end of evaluation step."""
    pass

  @abc.abstractmethod
  def get_total_weights(self, batch) -> jax.Array:
    """Calculate the number of non-padded tokens in the batch."""


class BaseDataHooks(DataHooks):
  """Shared data hooks for post-training."""

  def __init__(self, config, mesh, goodput_recorder):
    self.config = config
    self.train_data_iterator, self.eval_data_iterator = create_data_iterator(config, mesh)
    self.train_data_loader = DataLoader(config, mesh, self.train_data_iterator, goodput_recorder=goodput_recorder)
    self.train_batch = None
    self.eval_batch = None

  @override
  def load_next_train_batch(self, train_ctx: peft_trainer.PeftTrainer):  # pylint: disable=unused-argument
    """Loads the next batch of data for training."""
    pass

  @override
  def load_next_eval_batch(self, train_ctx: peft_trainer.PeftTrainer):
    """Loads the next batch of data for evaluation."""
    pass
