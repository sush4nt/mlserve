"""Typed configuration for each dataset/model, loaded from a YAML file.

One YAML per model family (configs/avazu.yaml, configs/nyc_taxi.yaml) is the
single source of truth for: the task, the target column, the feature-engineering
recipe, the XGBoost hyperparameters, and the MLflow names. Loading it into
dataclasses (instead of passing dicts around) means typos surface immediately
and every consumer gets autocomplete + validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from mlserve.common.paths import CONFIG_DIR


@dataclass(frozen=True)
class FeatureField:
    """A user-facing input field exposed by the frontend form.

    `editable=False` fields are not shown; they are filled with `default` when
    assembling the full serving vector. This is how a human enters 6 friendly
    values yet the model still receives all 22.
    """

    name: str
    label: str
    default: float = 0.0
    min: float | None = None
    max: float | None = None
    editable: bool = True


@dataclass(frozen=True)
class ModelConfig:
    name: str                       # "avazu" | "nyc_taxi"
    task: str                       # "classification" | "regression"
    target: str
    mlflow_experiment: str
    registered_model_name: str

    # Feature engineering (training-time only).
    drop_cols: list[str] = field(default_factory=list)
    freq_encode_cols: list[str] = field(default_factory=list)
    label_encode_cols: list[str] = field(default_factory=list)

    # XGBoost.
    xgb_params: dict = field(default_factory=dict)
    num_boost_round: int = 300
    early_stopping_rounds: int = 30

    # Serving endpoints derived from this model family.
    py_endpoint: str = ""
    onnx_endpoint: str | None = None

    # Friendly form fields for the frontend.
    form_fields: list[FeatureField] = field(default_factory=list)

    @property
    def is_classification(self) -> bool:
        return self.task == "classification"


def load_config(name: str, config_dir: Path | None = None) -> ModelConfig:
    """Load configs/<name>.yaml into a ModelConfig."""
    config_dir = config_dir or CONFIG_DIR
    path = config_dir / f"{name}.yaml"
    raw = yaml.safe_load(path.read_text())

    fields = [FeatureField(**f) for f in raw.pop("form_fields", [])]
    return ModelConfig(form_fields=fields, **raw)
