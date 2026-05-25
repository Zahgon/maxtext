# Copyright 2026 Google LLC
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

"""vLLM-TPU server lifecycle (in-process LLM + thin HTTP wrapper)."""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from typing import Any

import requests

logger = logging.getLogger(__name__)

_HEALTH_ENDPOINT = "/health"


def _build_app(llm: Any) -> Any:
  """Return a FastAPI app that wraps an in-process vLLM LLM instance."""
  import fastapi  # pylint: disable=import-outside-toplevel
  from vllm.sampling_params import SamplingParams  # pylint: disable=import-outside-toplevel

  globals()["fastapi"] = fastapi

  app = fastapi.FastAPI()



  @app.post("/v1/chat/completions")
  async def chat_completions(request: fastapi.Request):  # pylint: disable=unused-variable
    """OpenAI-compatible chat completions endpoint.

    Used by evalchemy and lm-eval chat tasks.
    """
    pass

  return app


class VllmServerManager:
  """Manages an in-process vLLM-TPU LLM with an OpenAI-compatible HTTP layer.

  Args:
    model_path: HF model ID or local path.
    checkpoint_path: MaxText orbax checkpoint path.
    maxtext_model_name: MaxText model name (e.g. "llama3.1-8b").
    host: Hostname the HTTP server binds to (rank-0 only).
    port: Port the HTTP server listens on.
    tensor_parallel_size: Total number of chips.
    expert_parallel_size: Chips allocated to the expert mesh axis (EP).
    max_model_len: Maximum sequence length.
    dtype: Activation dtype string passed to vLLM (e.g. "bfloat16").
    max_num_batched_tokens: Tokens per scheduler step (None = vLLM default).
    max_num_seqs: Max concurrent sequences (None = vLLM default).
    startup_timeout: Seconds to wait for /health to return healthy.
    hbm_memory_utilization: Fraction of HBM reserved for KV cache.
    env: Optional environment-variable overrides.
    additional_vllm_kwargs: Extra kwargs merged into the vLLM LLM() constructor.
  """

  def __init__(
      self,
      model_path: str,
      checkpoint_path: str | None = None,
      maxtext_model_name: str | None = None,
      host: str = "localhost",
      port: int = 8000,
      tensor_parallel_size: int = 4,
      expert_parallel_size: int = 1,
      data_parallel_size: int = 1,
      max_model_len: int = 4096,
      dtype: str = "bfloat16",
      max_num_batched_tokens: int | None = None,
      max_num_seqs: int | None = None,
      startup_timeout: int = 600,
      hbm_memory_utilization: float = 0.3,
      env: dict[str, str] | None = None,
      additional_vllm_kwargs: dict | None = None,
  ):
    if checkpoint_path and not maxtext_model_name:
      raise ValueError("maxtext_model_name is required when checkpoint_path is set.")
    if tensor_parallel_size % expert_parallel_size != 0:
      raise ValueError(
          f"tensor_parallel_size ({tensor_parallel_size}) is not divisible by "
          f"expert_parallel_size ({expert_parallel_size})."
      )
    self.model_path = model_path
    self.checkpoint_path = checkpoint_path
    self.maxtext_model_name = maxtext_model_name
    self.host = host
    self.port = port
    self.tensor_parallel_size = tensor_parallel_size
    self.expert_parallel_size = expert_parallel_size
    self.data_parallel_size = data_parallel_size
    self.max_model_len = max_model_len
    self.dtype = dtype
    self.max_num_batched_tokens = max_num_batched_tokens
    self.max_num_seqs = max_num_seqs
    self.startup_timeout = startup_timeout
    self.hbm_memory_utilization = hbm_memory_utilization
    self.env = env
    self.additional_vllm_kwargs = additional_vllm_kwargs or {}

    self._llm: Any | None = None
    self._uvicorn_server: Any | None = None
    self._server_thread: threading.Thread | None = None


  def start(self) -> None:
    """Initialize the in-process vLLM LLM and start the HTTP server."""

    # Disable V1 multiprocessing to make EngineCore run in-process.
    # JAX initialized exactly once inside LLM() in this process.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("NEW_MODEL_DESIGN", "1")
    os.environ.setdefault("SKIP_JAX_PRECOMPILE", "1")
    from vllm import LLM  # pylint: disable=import-outside-toplevel

    if self.env:
      os.environ.update(self.env)

    # total chips = ici_tensor_parallelism * ici_expert_parallelism.
    ici_tp = self.tensor_parallel_size // self.expert_parallel_size
    ici_ep = self.expert_parallel_size

    vllm_kwargs: dict = {
        "model": self.model_path,
        "tensor_parallel_size": ici_tp,
        "data_parallel_size": self.data_parallel_size,
        "max_model_len": self.max_model_len,
        "dtype": self.dtype,
        "gpu_memory_utilization": self.hbm_memory_utilization,
    }
    if self.max_num_batched_tokens is not None:
      vllm_kwargs["max_num_batched_tokens"] = self.max_num_batched_tokens
    if self.max_num_seqs is not None:
      vllm_kwargs["max_num_seqs"] = self.max_num_seqs

    if self.checkpoint_path:
      vllm_kwargs["additional_config"] = {
          "maxtext_config": {
              "model_name": self.maxtext_model_name,
              "load_parameters_path": self.checkpoint_path,
              "log_config": False,
              "ici_tensor_parallelism": ici_tp,
              "ici_expert_parallelism": ici_ep,
          },
          "sharding": {
              "sharding_strategy": {},
          },
      }
      if ici_ep > 1:
        vllm_kwargs["additional_config"]["sharding"]["sharding_strategy"]["expert_parallelism"] = ici_ep
    else:
      vllm_kwargs["load_format"] = "auto"

    if self.additional_vllm_kwargs:
      for _k, _v in self.additional_vllm_kwargs.items():
        if _k == "additional_config" and isinstance(_v, dict) and isinstance(vllm_kwargs.get("additional_config"), dict):
          for _sub_k, _sub_v in _v.items():
            if isinstance(_sub_v, dict) and isinstance(vllm_kwargs["additional_config"].get(_sub_k), dict):
              vllm_kwargs["additional_config"][_sub_k].update(_sub_v)
            else:
              vllm_kwargs["additional_config"][_sub_k] = _sub_v
        else:
          vllm_kwargs[_k] = _v

    logger.info(
        "Initializing in-process vLLM (tp=%d, ep=%d, dp=%d, max_len=%d)",
        ici_tp,
        ici_ep,
        self.data_parallel_size,
        self.max_model_len,
    )
    self._llm = LLM(**vllm_kwargs)

    import jax as _jax  # pylint: disable=import-outside-toplevel

    logger.info("Rank %d: vLLM LLM ready.", _jax.process_index())

    if _jax.process_index() == 0:
      import uvicorn  # pylint: disable=import-outside-toplevel

      app = _build_app(self._llm)
      config = uvicorn.Config(
          app,
          host=self.host,
          port=self.port,
          log_level="warning",
          workers=1,
      )
      self._uvicorn_server = uvicorn.Server(config)
      self._server_thread = threading.Thread(
          target=self._uvicorn_server.run,
          daemon=True,
          name="vllm-http-server",
      )
      self._server_thread.start()
      self._wait_until_healthy()

  def _wait_until_healthy(self) -> None:
    """Wait until the HTTP server returns 200 OK on /health."""
    deadline = time.time() + self.startup_timeout
    health_url = f"{self.base_url}{_HEALTH_ENDPOINT}"
    while time.time() < deadline:
      try:
        resp = requests.get(health_url, timeout=5)
        if resp.status_code == 200:
          logger.info("vLLM HTTP server is healthy at %s", self.base_url)
          return
      except requests.exceptions.ConnectionError:
        pass
      if self._server_thread is not None and not self._server_thread.is_alive():
        raise RuntimeError("vLLM HTTP server thread died before becoming healthy.")
      time.sleep(2)
    raise TimeoutError(f"vLLM HTTP server did not become healthy within {self.startup_timeout}s.")

  def stop(self) -> None:
    """Stop the HTTP server and release the LLM."""
    if self._uvicorn_server is not None:
      logger.info("Stopping vLLM HTTP server.")
      self._uvicorn_server.should_exit = True
      if self._server_thread is not None:
        self._server_thread.join(timeout=30)
        if self._server_thread.is_alive():
          logger.warning("vLLM HTTP server thread did not exit within 30 s.")
    self._llm = None
    self._uvicorn_server = None
    self._server_thread = None
    logger.info("VllmServerManager stopped.")

  def __enter__(self) -> "VllmServerManager":
    self.start()
    return self

  def __exit__(self, *_) -> None:
    self.stop()
