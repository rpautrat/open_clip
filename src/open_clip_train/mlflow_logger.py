"""Minimal MLflow metric logger over the MLflow REST API (standard library only, no `mlflow` package needed).

Configured from the standard MLflow environment variables:
- MLFLOW_TRACKING_URI (required): http(s):// server, or azureml:// as set inside Azure ML jobs.
- MLFLOW_TRACKING_TOKEN (optional): bearer token, set inside Azure ML jobs.
- MLFLOW_RUN_ID (optional): log into this existing run, e.g. the Azure ML job's run, so metrics show up in AML
  studio. Without it a run is created (experiment from --mlflow-experiment-name, run name from --name) and closed by
  finish().

Requests are sent from a background thread so that a slow or unreachable tracking server never stalls training;
failures are logged as warnings and the affected metrics dropped.
"""
import atexit
import json
import logging
import math
import os
import queue
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

_logger = logging.getLogger(__name__)

_MAX_METRICS_PER_BATCH = 1000  # MLflow log-batch limits
_MAX_PARAMS_PER_BATCH = 100
_MAX_PARAM_VALUE_LENGTH = 500
_INVALID_NAME_CHARS = re.compile(r"[^\w./\- ]")

_active_logger: Optional["MLflowLogger"] = None


def sanitize_name(name: str) -> str:
    """MLflow names allow only alphanumerics, '_', '-', '.', '/' and ' ' (e.g. 'R@1' -> 'R_at_1')."""
    return _INVALID_NAME_CHARS.sub("_", name.replace("@", "_at_"))


def _rest_base(uri: str) -> str:
    if uri.startswith("azureml://"):
        uri = "https://" + uri.split("://", 1)[1]
    return uri.rstrip("/") + "/api/2.0/mlflow/"


class MLflowLogger:
    def __init__(self, tracking_uri: str, token: Optional[str] = None, run_id: Optional[str] = None,
                 experiment_name: str = "open_clip", run_name: Optional[str] = None):
        if not tracking_uri.startswith(("http://", "https://", "azureml://")):
            raise ValueError(f"MLflow logging needs an http(s):// or azureml:// MLFLOW_TRACKING_URI, got {tracking_uri!r}")
        self._base = _rest_base(tracking_uri)
        self._token = token
        self._owns_run = run_id is None
        if run_id is None:
            experiment_id = self._get_or_create_experiment(experiment_name)
            body = {"experiment_id": experiment_id, "start_time": int(time.time() * 1000)}
            if run_name:
                body["run_name"] = run_name
            run_id = self._request("POST", "runs/create", body)["run"]["info"]["run_id"]
        self.run_id = run_id
        self._queue: queue.Queue = queue.Queue()
        self._warned: set = set()
        self._closed = False
        self._thread = threading.Thread(target=self._worker, name="mlflow-logger", daemon=True)
        self._thread.start()
        atexit.register(self.finish)

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        request = urllib.request.Request(
            self._base + path,
            data=None if body is None else json.dumps(body).encode(),
            method=method,
            headers=headers,
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read() or b"{}")

    def _get_or_create_experiment(self, name: str) -> str:
        try:
            query = urllib.parse.urlencode({"experiment_name": name})
            return self._request("GET", f"experiments/get-by-name?{query}")["experiment"]["experiment_id"]
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
        return self._request("POST", "experiments/create", {"name": name})["experiment_id"]

    def log(self, metrics: dict, step: Optional[int] = None) -> None:
        """Queue numeric metrics (non-numeric and non-finite values are skipped) for logging at `step`."""
        timestamp = int(time.time() * 1000)
        entries = []
        for key, value in metrics.items():
            if hasattr(value, "item"):
                value = value.item()
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                continue
            entries.append({"key": sanitize_name(key), "value": float(value), "timestamp": timestamp,
                            "step": int(step or 0)})
        if entries and not self._closed:
            self._queue.put(("metrics", entries))

    def log_params(self, params: dict[str, Any]) -> None:
        entries = [{"key": sanitize_name(str(k)), "value": str(v)[:_MAX_PARAM_VALUE_LENGTH]} for k, v in params.items()]
        if entries and not self._closed:
            self._queue.put(("params", entries))

    def _send(self, kind: str, entries: list) -> None:
        if kind == "params":
            # MLflow params are immutable: a restarted job logging into the same run (e.g. a preempted Azure ML job)
            # keeps the first start's values; resending a changed one would fail the whole batch.
            try:
                existing = self._request("GET", f"runs/get?run_id={self.run_id}")["run"]["data"].get("params", [])
                logged = {param["key"] for param in existing}
                entries = [entry for entry in entries if entry["key"] not in logged]
            except Exception as error:
                self._warn(error, "params")
                return
        limit = _MAX_METRICS_PER_BATCH if kind == "metrics" else _MAX_PARAMS_PER_BATCH
        for start in range(0, len(entries), limit):
            body = {"run_id": self.run_id, kind: entries[start:start + limit]}
            for attempt in range(3):
                try:
                    self._request("POST", "runs/log-batch", body)
                    break
                except Exception as error:
                    if attempt == 2:
                        self._warn(error, kind)
                    else:
                        time.sleep(2 ** attempt)

    def _warn(self, error: Exception, kind: str) -> None:
        detail = error.read().decode(errors="replace")[:500] if isinstance(error, urllib.error.HTTPError) else ""
        signature = (kind, type(error).__name__)
        if signature not in self._warned:  # Warn once per failure type to avoid flooding the log.
            self._warned.add(signature)
            _logger.warning(f"MLflow: failed to log {kind}: {type(error).__name__}: {error} {detail}".rstrip())

    def _worker(self) -> None:
        stop = False
        while not stop:
            # Drain everything queued so far into one request when the server is slower than the logging cadence.
            items = [self._queue.get()]
            while True:
                try:
                    items.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            batches = {"params": [], "metrics": []}
            for item in items:
                if item is None:
                    stop = True
                else:
                    batches[item[0]].extend(item[1])
            for kind, entries in batches.items():
                if entries:
                    self._send(kind, entries)

    def finish(self, timeout: float = 120.0) -> None:
        """Flush queued metrics (waiting up to `timeout` seconds) and close the run if this logger created it."""
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join(timeout)
        if self._owns_run:
            try:
                self._request("POST", "runs/update", {
                    "run_id": self.run_id, "status": "FINISHED", "end_time": int(time.time() * 1000)})
            except Exception as error:
                self._warn(error, "run status")


def init_mlflow_logger(args) -> MLflowLogger:
    """Create the process-wide MLflow logger (call on the master rank only) and log the run's arguments as params."""
    global _active_logger
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError("--report-to mlflow needs MLFLOW_TRACKING_URI (set automatically inside Azure ML jobs).")
    _active_logger = MLflowLogger(
        tracking_uri,
        token=os.environ.get("MLFLOW_TRACKING_TOKEN"),
        run_id=os.environ.get("MLFLOW_RUN_ID"),
        experiment_name=args.mlflow_experiment_name,
        run_name=args.name,
    )
    _logger.info(f"MLflow: logging to run {_active_logger.run_id}.")
    _active_logger.log_params(vars(args))
    return _active_logger


def get_mlflow_logger() -> Optional[MLflowLogger]:
    """The active MLflow logger, or None if MLflow logging is off (or this is not the master rank)."""
    return _active_logger
