# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Preprocessing for instruction dataset."""

import json
import importlib
import os
import re

from maxtext.utils import max_logging


def load_data_template_from_file(template_path):
  """Loads a data template from a file."""
  if not template_path:
    return None

  current_dir = os.path.dirname(os.path.abspath(__file__))
  repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
  template_full_path = os.path.join(repo_root, template_path)

  if not os.path.isfile(template_full_path):
    return None

  if template_full_path.endswith(".json"):
    with open(template_full_path, "r", encoding="utf-8") as f:
      try:
        return json.load(f)
      except json.JSONDecodeError:
        return None

  return None


def load_chat_template_from_file(template_path):
  """Loads a chat template from a file."""
  if not template_path:
    return None

  current_dir = os.path.dirname(os.path.abspath(__file__))
  repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
  template_full_path = os.path.join(repo_root, template_path)

  if not os.path.isfile(template_full_path):
    return None

  if template_full_path.endswith((".jinja", ".j2", ".txt")):
    with open(template_full_path, "r", encoding="utf-8") as f:
      return f.read()

  if template_full_path.endswith(".json"):
    with open(template_full_path, "r", encoding="utf-8") as f:
      try:
        template_config = json.load(f)
        if isinstance(template_config, dict) and "chat_template" in template_config:
          return template_config["chat_template"]
      except json.JSONDecodeError:
        return None

  return None


def get_template_placeholders(template):
  """Dynamically extracts the format keys (placeholders) from a template string."""
  pass




def math_qa_formatting(example, template_config=None):
  """Maps question-answer pairs to conversational format."""
  pass


def load_formatter(formatting_func_path, **kwargs):
  """Loads a formatter function from a given path.

  Returns a callable that takes a dataset and applies the formatter via .map().
  """
  module_path, method_name = formatting_func_path.rsplit(".", 1)
  module = importlib.import_module(module_path)
  func = getattr(module, method_name)

  def formatter(dataset, dataset_features):
    remove_cols = []
    if kwargs:
      remove_cols = kwargs.pop("remove_columns", None)
    return dataset.map(
        func,
        fn_kwargs=kwargs if kwargs else None,
        features=dataset_features,
        remove_columns=remove_cols,
    )

  return formatter


def convert_to_conversational_format(
    dataset,
    data_columns,
    formatting_func_path=None,
    formatting_func_kwargs=None,
):
  """Converts instruction dataset to conversational format."""
  import datasets  # pylint: disable=import-outside-toplevel

  dataset_features = datasets.Features(
      {"messages": [{"content": datasets.Value("string"), "role": datasets.Value("string")}]}
  )

  if formatting_func_path:
    if not formatting_func_kwargs:
      formatting_func_kwargs = {}
    formatting_func_kwargs["remove_columns"] = data_columns
    template_path = formatting_func_kwargs.pop("template_path", None)
    if template_path:
      formatting_func_kwargs["template_config"] = load_data_template_from_file(template_path)
    formatter = load_formatter(formatting_func_path, **(formatting_func_kwargs))
    dataset = formatter(dataset, dataset_features)
    data_columns = ["messages"]
    return dataset, data_columns

  if "question" in data_columns and "answer" in data_columns:
    dataset = dataset.map(
        math_qa_formatting,
        fn_kwargs={},
        remove_columns=data_columns,
        features=dataset_features,
    )
    data_columns = ["messages"]
    return dataset, data_columns

  return dataset, data_columns
