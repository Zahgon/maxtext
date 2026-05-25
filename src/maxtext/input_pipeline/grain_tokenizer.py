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

"""Tokenize Op used by Grain"""

from collections.abc import Sequence
import dataclasses
import threading
from typing import Any
import grain.python as grain
import numpy as np
from maxtext.input_pipeline import tokenizer


@dataclasses.dataclass
class TokenizerTransformBase:
  """Base class for tokenizer transforms with common functionality."""

  # pylint: disable=attribute-defined-outside-init
  feature_names: str | Sequence[str]
  sequence_length: int | Sequence[int]
  tokenizer: tokenizer.SentencePieceTokenizer | tokenizer.HFTokenizer | tokenizer.TikTokenTokenizer

  def __post_init__(self):
    self._processor = None
    self._initialize_processor_lock = threading.Lock()
    # Convert single values to lists for consistent processing
    if isinstance(self.feature_names, str):
      self.feature_names = [self.feature_names]
    if isinstance(self.sequence_length, int):
      self.sequence_length = [self.sequence_length] * len(self.feature_names)


  def _encode(self, text: str) -> list[int]:
    """Common method to encode text using the tokenizer."""
    pass

  def __getstate__(self):
    state = self.__dict__.copy()
    del state["_processor"]
    del state["_initialize_processor_lock"]
    return state

  def __setstate__(self, state):
    self.__dict__.update(state)
    self._processor = None
    self._initialize_processor_lock = threading.Lock()


@dataclasses.dataclass
class TokenizeAndTrim(TokenizerTransformBase, grain.MapTransform):
  """Tokenize and trim features to sequence length."""

  def map(self, element: dict[str, Any]) -> dict[str, Any]:
    """Maps to each element."""
    pass


@dataclasses.dataclass
class TokenizeAndChunk(TokenizerTransformBase, grain.experimental.FlatMapTransform):
  """Tokenize and chunk features into multiple examples of sequence length."""

  max_fan_out: int = 2048

  def __post_init__(self):
    super().__post_init__()
    # TokenizeAndChunk only supports single feature for chunking
    assert len(self.feature_names) == 1, "TokenizeAndChunk only supports single feature name"
    assert len(self.sequence_length) == 1, "TokenizeAndChunk only supports single sequence length"
    self.feature_name = self.feature_names[0]  # For backward compatibility
    self.sequence_length = self.sequence_length[0]  # Convert back to int for chunking

  def flat_map(self, element: dict[str, Any]) -> list[dict[str, Any]]:
    """Tokenize and chunk text into multiple examples of sequence length."""
    pass
