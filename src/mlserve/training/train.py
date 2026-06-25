"""Stage 2 entrypoint: train an XGBoost model, evaluate, log to MLflow, and save
the booster + a serving metadata sidecar.

    uv run mlserve-train --dataset avazu
    uv run mlserve-train --dataset all

One function handles both tasks; the config's `task` field selects the metrics.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import mlflow
import mlflow.xgboost
import numpy as np
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.metrics import (
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from mlserve.common.logging import get_logger
from mlserve.common.paths import MODELS_DIR, PROCESSED_DIR, ensure_dirs
from mlserve.config.schema import ModelConfig, load_config
from mlserve.training.mlflow_utils import configure_tracking

log = get_logger(__name__)
DATASETS = ["avazu", "nyc_taxi"]


@dataclass
class TrainResult:
    metrics: dict
    model_path: str
    best_iteration: int


def _load_xy(dataset: str, target: str):
    train = pq.read_table(PROCESSED_DIR / dataset / "train.parquet").to_pandas()
    val = pq.read_table(PROCESSED_DIR / dataset / "val.parquet").to_pandas()
    features = [c for c in train.columns if c != target]
    return (
        train[features].values, train[target].values,
        val[features].values, val[target].values,
        features,
    )


def _evaluate(task: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if task == "classification":
        y_bin = (y_pred > 0.5).astype(int)
        return {
            "val_auc": float(roc_auc_score(y_true, y_pred)),
            "val_logloss": float(log_loss(y_true, y_pred)),
            "val_f1": float(f1_score(y_true, y_bin)),
        }
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "val_rmse": rmse,
        "val_mae": float(mean_absolute_error(y_true, y_pred)),
        "val_r2": float(r2_score(y_true, y_pred)),
    }


def train_one(config: ModelConfig) -> TrainResult:
    X_tr, y_tr, X_val, y_val, features = _load_xy(config.name, config.target)

    params = dict(config.xgb_params)
    if config.is_classification:
        # Class imbalance handling for CTR.
        pos = max((y_tr == 1).sum(), 1)
        params["scale_pos_weight"] = float((y_tr == 0).sum() / pos)

    # NB: we deliberately do NOT attach feature_names to the DMatrix. The
    # onnxmltools XGBoost converter requires the default f0,f1,... naming, and
    # the canonical feature order is recorded in model_meta.json regardless.
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)

    configure_tracking()
    mlflow.set_experiment(config.mlflow_experiment)

    with mlflow.start_run(run_name="xgb-baseline") as run:
        mlflow.log_params(params)
        mlflow.log_param("num_boost_round", config.num_boost_round)

        booster = xgb.train(
            params, dtrain,
            num_boost_round=config.num_boost_round,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=config.early_stopping_rounds,
            verbose_eval=False,
        )

        y_pred = booster.predict(dval)
        metrics = _evaluate(config.task, y_val, y_pred)
        metrics["best_iteration"] = booster.best_iteration
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(
            booster, artifact_path="model",
            registered_model_name=config.registered_model_name,
        )

        # --- Save booster + serving metadata for the inference server ---------
        model_dir = MODELS_DIR / config.py_endpoint
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.json"
        booster.save_model(str(model_path))

        meta = {
            "endpoint": config.py_endpoint,
            "onnx_endpoint": config.onnx_endpoint,
            "task": config.task,
            "target": config.target,
            "runtime": "xgboost",
            "datatype": "FP64",
            "feature_order": features,
            "n_features": len(features),
            "mlflow_run_id": run.info.run_id,
        }
        (model_dir / "model_meta.json").write_text(json.dumps(meta, indent=2))

        log.info("%s metrics: %s", config.name, metrics)
        log.info("Saved booster -> %s", model_path)
        return TrainResult(metrics, str(model_path), booster.best_iteration)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train XGBoost models.")
    ap.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    args = ap.parse_args()

    ensure_dirs()
    targets = DATASETS if args.dataset == "all" else [args.dataset]
    for ds in targets:
        train_one(load_config(ds))


if __name__ == "__main__":
    main()
