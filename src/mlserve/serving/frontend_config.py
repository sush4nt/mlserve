"""Generate the frontend's model config from the trained artifacts.

This closes the loop the plan insists on: the feature order, count, and sensible
defaults all come from one source (the processed feature_meta.json written by
preprocessing). The frontend imports the generated JSON, so changing the feature
set anywhere propagates with one regeneration — no hand-edited feature lists.

    uv run mlserve-frontend-config
"""

from __future__ import annotations

import json

from mlserve.common.logging import get_logger
from mlserve.common.paths import FRONTEND_CONFIG, MODELS_DIR, PROCESSED_DIR
from mlserve.config.schema import ModelConfig, load_config

log = get_logger(__name__)

_DESCRIPTIONS = {
    "classification": "Predicts click-through probability for a display-ad impression.",
    "regression": "Predicts taxi fare amount in USD from trip features.",
}


def _fields(cfg: ModelConfig, feature_order: list[str], defaults: dict) -> list[dict]:
    """Build a field spec per feature: editable (from config) or hidden default."""
    by_name = {f.name: f for f in cfg.form_fields}
    out = []
    for name in feature_order:
        f = by_name.get(name)
        if f is not None:
            out.append({"name": name, "label": f.label, "default": f.default,
                        "min": f.min, "max": f.max, "editable": True})
        else:
            out.append({"name": name, "label": name,
                        "default": round(float(defaults.get(name, 0.0)), 4),
                        "min": None, "max": None, "editable": False})
    return out


def _entry(endpoint: str, cfg: ModelConfig, runtime: str, datatype: str,
           feature_order: list[str], defaults: dict, peer: str | None) -> dict:
    return {
        "id": endpoint,
        "displayName": {
            cfg.py_endpoint: f"{cfg.name.upper()} — Python XGBoost",
            cfg.onnx_endpoint: f"{cfg.name.upper()} — ONNX Runtime (C++)",
        }.get(endpoint, endpoint),
        "task": cfg.task,
        "dataset": cfg.name,
        "runtime": runtime,
        "runtimeBadge": runtime,
        "datatype": datatype,
        "description": _DESCRIPTIONS.get(cfg.task, ""),
        "outputLabel": "Click Probability" if cfg.is_classification else "Predicted Fare (USD)",
        "outputType": "probability" if cfg.is_classification else "value",
        "comparisonPeer": peer,
        "featureOrder": feature_order,
        "fields": _fields(cfg, feature_order, defaults),
    }


def main() -> None:
    models: list[dict] = []
    for ds in ["avazu", "nyc_taxi"]:
        cfg = load_config(ds)
        meta_path = PROCESSED_DIR / ds / "feature_meta.json"
        if not meta_path.exists():
            log.warning("%s missing — run prepare first; skipping", meta_path)
            continue
        meta = json.loads(meta_path.read_text())
        order, defaults = meta["feature_order"], meta["feature_defaults"]

        # Python endpoint (always exists once trained).
        if (MODELS_DIR / cfg.py_endpoint / "model_meta.json").exists():
            models.append(_entry(cfg.py_endpoint, cfg, "python", "FP64",
                                 order, defaults, cfg.onnx_endpoint))
        # ONNX endpoint (only if exported).
        if cfg.onnx_endpoint and (MODELS_DIR / cfg.onnx_endpoint / "model_meta.json").exists():
            models.append(_entry(cfg.onnx_endpoint, cfg, "onnx", "FP32",
                                 order, defaults, cfg.py_endpoint))

    FRONTEND_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    FRONTEND_CONFIG.write_text(json.dumps({"models": models}, indent=2))
    log.info("Wrote %d model entries -> %s", len(models), FRONTEND_CONFIG)


if __name__ == "__main__":
    main()
