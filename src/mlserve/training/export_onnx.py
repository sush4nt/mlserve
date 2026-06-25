"""Stage 2b entrypoint: export a trained XGBoost booster to ONNX and validate
that ONNX Runtime reproduces the XGBoost predictions within tolerance.

    uv run mlserve-export --dataset avazu

The ONNX model serves identical weights to the Python model; only the execution
engine differs. That is what the Phase-6 load test measures.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import onnxruntime as ort
import xgboost as xgb
from onnxconverter_common.data_types import FloatTensorType
from onnxmltools.convert import convert_xgboost

from mlserve.common.logging import get_logger
from mlserve.common.paths import MODELS_DIR, PROCESSED_DIR
from mlserve.config.schema import load_config

log = get_logger(__name__)


def export_one(dataset: str, tol: float = 1e-3) -> None:
    config = load_config(dataset)
    if not config.onnx_endpoint:
        log.info("%s has no ONNX endpoint configured — skipping", dataset)
        return

    py_dir = MODELS_DIR / config.py_endpoint
    meta = json.loads((py_dir / "model_meta.json").read_text())
    n_features = meta["n_features"]

    booster = xgb.Booster()
    booster.load_model(str(py_dir / "model.json"))

    initial_types = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_xgboost(booster, initial_types=initial_types)

    onnx_dir = MODELS_DIR / config.onnx_endpoint
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / "model.onnx"
    onnx_path.write_bytes(onnx_model.SerializeToString())

    # --- Validate against the XGBoost predictions on real validation rows -----
    val = __import__("pyarrow.parquet", fromlist=["read_table"]).read_table(
        PROCESSED_DIR / dataset / "val.parquet"
    ).to_pandas()
    features = meta["feature_order"]
    sample = val[features].head(256).values.astype(np.float32)

    xgb_pred = booster.predict(xgb.DMatrix(sample))

    sess = ort.InferenceSession(onnx_path.read_bytes(), providers=["CPUExecutionProvider"])
    onnx_pred = _onnx_positive_proba(sess, sample, config.is_classification)

    max_diff = float(np.abs(xgb_pred - onnx_pred).max())
    log.info("%s: max |XGBoost - ONNX| = %.6g over %d rows", dataset, max_diff, len(sample))
    if max_diff > tol:
        raise AssertionError(
            f"ONNX/XGBoost mismatch {max_diff:.6g} > tol {tol}. Conversion suspect."
        )

    onnx_meta = {
        "endpoint": config.onnx_endpoint,
        "task": config.task,
        "target": config.target,
        "runtime": "onnx",
        "datatype": "FP32",          # ONNX Runtime expects float32
        "feature_order": features,
        "n_features": n_features,
        "exported_from": config.py_endpoint,
        "max_validation_diff": max_diff,
    }
    (onnx_dir / "model_meta.json").write_text(json.dumps(onnx_meta, indent=2))
    log.info("Saved ONNX model -> %s (validated)", onnx_path)


def _onnx_positive_proba(sess, sample: np.ndarray, is_classification: bool) -> np.ndarray:
    """Extract the comparable prediction from ONNX outputs.

    XGBoost->ONNX classification produces [label, probabilities]; we take P(class=1).
    Regression produces a single output tensor.
    """
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: sample})
    if not is_classification:
        return np.asarray(outputs[0]).reshape(-1)

    proba = outputs[1]
    # ONNX may return probabilities as an Nx2 array, or as a list of {class: prob} dicts.
    if isinstance(proba, list):
        return np.array([row[1] for row in proba], dtype=np.float64)
    proba = np.asarray(proba)
    return proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.reshape(-1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export models to ONNX.")
    ap.add_argument("--dataset", choices=["avazu", "nyc_taxi", "all"], default="avazu")
    args = ap.parse_args()
    targets = ["avazu", "nyc_taxi"] if args.dataset == "all" else [args.dataset]
    for ds in targets:
        export_one(ds)


if __name__ == "__main__":
    main()
