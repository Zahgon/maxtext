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

"""DPO specific input pipeline utilities."""

import dataclasses
import grain.python as grain
import numpy as np


@dataclasses.dataclass
class DPODataFormatting(grain.MapTransform):
  """Prepares DPO data.
  Renames input columns, extracts common prefix if needed, generates masks, and performs
  DPO-aware padding (left-padded prompts, right-padded responses).
  """

  pad_id: int
  max_target_length: int
  data_column_names: tuple[str, ...]
  max_prompt_length: int | None = None

  def map(self, element):
    "Apply the dataset transformations for DPO."
    pass

  def _pad(self, x, length, left=False):
    """Pads or trims an array to a specific length.

    When left=True (for prompts), trims from the left to keep the suffix (closest context).
    When left=False (for responses), trims from the right to keep the prefix.
    """
    pass
