"""Inference runners — the pluggable execution engines.

`BaseRunner` defines the contract; `XGBoostRunner` runs predictions in-process
through the Python XGBoost API, and `ONNXRunner` runs them through the ONNX
Runtime (compiled C++) engine. Same weights, different engine — this pair is the
whole point of the platform's comparison story.

A runner takes a 2-D numpy array (rows x features) and returns a 1-D array of
scores: P(click) for classification, fare in USD for regression.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import onnxruntime as ort
import xgboost as xgb


class BaseRunner(ABC):
    runtime: str = "base"

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.meta = json.loads((model_dir / "model_meta.json").read_text())
        self.name: str = self.meta["endpoint"]
        self.task: str = self.meta["task"]
        self.datatype: str = self.meta["datatype"]
        self.feature_order: list[str] = self.meta["feature_order"]
        self.n_features: int = self.meta["n_features"]
        self._load()

    @abstractmethod
    def _load(self) -> None:
        """Load the model artifact into memory."""

    @abstractmethod
    def predict(self, arr: np.ndarray) -> np.ndarray:
        """Return a 1-D score array for the given (rows x features) input."""

    @property
    def is_classification(self) -> bool:
        return self.task == "classification"


class XGBoostRunner(BaseRunner):
    runtime = "xgboost"

    def _load(self) -> None:
        self._booster = xgb.Booster()
        self._booster.load_model(str(self.model_dir / "model.json"))

    def predict(self, arr: np.ndarray) -> np.ndarray:
        # binary:logistic -> P(class=1); reg:squarederror -> value. Either way 1-D.
        preds = self._booster.predict(xgb.DMatrix(arr.astype(np.float64)))
        return np.asarray(preds).reshape(-1)


class ONNXRunner(BaseRunner):
    runtime = "onnx"

    def _load(self) -> None:
        # log_severity_level=3 silences the benign "output shape {-1,1} vs {1,2}"
        # warnings the XGBoost->ONNX probabilities node emits on every call.
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self._sess = ort.InferenceSession(
            (self.model_dir / "model.onnx").read_bytes(),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._input = self._sess.get_inputs()[0].name

    def predict(self, arr: np.ndarray) -> np.ndarray:
        outputs = self._sess.run(None, {self._input: arr.astype(np.float32)})
        if not self.is_classification:
            return np.asarray(outputs[0]).reshape(-1)
        proba = outputs[1]
        if isinstance(proba, list):  # ONNX ZipMap -> list of {class: prob}
            return np.array([row[1] for row in proba], dtype=np.float64)
        proba = np.asarray(proba)
        return proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.reshape(-1)


def make_runner(model_dir: Path) -> BaseRunner:
    """Instantiate the right runner based on the model's metadata."""
    runtime = json.loads((model_dir / "model_meta.json").read_text())["runtime"]
    return {"xgboost": XGBoostRunner, "onnx": ONNXRunner}[runtime](model_dir)
