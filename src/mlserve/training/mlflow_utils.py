"""MLflow helpers that make tracking work with zero infrastructure.

If MLFLOW_TRACKING_URI is unset, we log to a local `./mlruns` file store. That
means `uv run mlserve-train` works on a bare checkout — no server needed. When
the Docker stack is up, set MLFLOW_TRACKING_URI=http://mlflow:5000 and the same
code logs to the tracking server instead.
"""

from __future__ import annotations

import os

import mlflow

from mlserve.common.logging import get_logger
from mlserve.common.paths import ROOT

log = get_logger(__name__)


def configure_tracking() -> str:
    uri = os.getenv("MLFLOW_TRACKING_URI", f"file:{ROOT / 'mlruns'}")
    mlflow.set_tracking_uri(uri)
    log.info("MLflow tracking URI: %s", uri)
    return uri
