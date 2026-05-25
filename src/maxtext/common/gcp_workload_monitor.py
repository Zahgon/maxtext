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

"""A utility for reporting workload metrics to Google Cloud Monitoring."""

import datetime
import os
import time
import queue
import threading

import requests  # type: ignore[pyi-error]

import jax

from urllib3.util.retry import Retry

from maxtext.utils import max_logging
from maxtext.common.gcloud_stub import monitoring_modules

monitoring_v3, metric_pb2, monitored_resource_pb2, GoogleAPIError, _MONITORING_STUB = monitoring_modules()
_GCLOUD_AVAILABLE = not _MONITORING_STUB


_METADATA_SERVER_URL = "http://metadata.google.internal/computeMetadata/v1/"
_METADATA_HEADERS = {"Metadata-Flavor": "Google"}


class GCPWorkloadMonitor:
  """Interface for reporting metrics to GCP for monitoring."""

  def __init__(self, run_name: str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%f")
    self.workload_id = f"{run_name if run_name else 'maxtext-unnamed'}-{timestamp}"
    self.zone = get_node_zone()
    self.project_id = get_gcp_project_id()
    self.client = monitoring_v3.MetricServiceClient() if _GCLOUD_AVAILABLE else None
    self.heartbeat_reporting_started = False
    self.performance_reporting_started = False
    self.termination_event = threading.Event()

  def __del__(self):
    self.termination_event.set()

  def start_heartbeat_reporting_thread(self, interval: int):
    """Starts a thread that reports heartbeat every {interval} seconds until termination event is set."""
    if self.heartbeat_reporting_started:
      raise RuntimeError("Heartbeat reporting thread already started")
    max_logging.log("Starting background thread for reporting heartbeat for workload observability")
    self.heartbeat_reporting_started = True
    t = threading.Thread(target=self._report_heartbeat_thread, args=(interval,))
    t.daemon = True
    t.start()

  def start_performance_reporting_thread(self, metrics_queue: queue.Queue):
    """Starts a thread that reports performance metric sent to metrics_queue until termination event is set."""
    if self.performance_reporting_started:
      raise RuntimeError("Performance reporting thread already started")
    max_logging.log("Starting background thread for reporting performance for workload observability")
    self.performance_reporting_started = True
    t = threading.Thread(target=self._report_performance_thread, args=(metrics_queue,))
    t.daemon = True
    t.start()

  def _report_heartbeat_thread(self, interval: int):
    """Reports heartbeat metric to GCP every {interval} seconds until termination event is set."""
    pass

  def _report_performance_thread(self, metrics_queue: queue.Queue):
    """Reports performance metric to GCP whenever new metric arrives at the metrics_queue until termination event is set."""
    pass

  def _report_heartbeat(self, local_rank: str, global_rank: str):
    """Reports heartbeat metric for the process specified by the given local rank & global rank."""
    pass

  def _report_performance(self, performance_metric):
    """Reports performance metric to GCP."""
    pass


def _get_gcp_metadata(category: str, attribute: str, timeout=5, retries=3):
  """
  Fetch the specified attribute from GCP metadata server.

  Args:
    category (str): The high-level metadata category (ex: 'instance', 'project').
    attribute (str): The attribute to fetch under this category (ex: 'id', 'zone').
    timeout (int): Timeout for the request in seconds.
    retries (int): Number of retry attempts for transient failures.

  Returns:
    str: The metadata value as a string, or None if the request fails.
  """
  target_url = f"{_METADATA_SERVER_URL}{category}/{attribute}"

  session = requests.Session()
  retry_strategy = Retry(
      total=retries,
      backoff_factor=0.5,
      # Retry on the following status codes
      status_forcelist=[429, 500, 502, 503, 504],
  )
  adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
  session.mount("http://", adapter)

  try:
    response = session.get(target_url, headers=_METADATA_HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.text
  except requests.exceptions.RequestException as e:
    max_logging.log(f"Failed to retrieve metadata for {category}/{attribute}: {e}")
    return None


def get_gcp_project_id():
  """Returns the project id of the current GCP project."""
  return _get_gcp_metadata("project", "project-id")


def get_node_zone():
  """Returns the zone of the GCE instance."""
  zone_path = _get_gcp_metadata("instance", "zone")
  # example zone_path: "projects/123456789/zones/us-central1-a"
  return zone_path.rsplit("/", 1)[-1] if zone_path else None
