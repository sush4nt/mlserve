"""Preprocessing abstraction.

`BasePreprocessor` owns the parts every dataset shares — the temporal split and
the feature-metadata file — and delegates the dataset-specific work (cleaning,
feature derivation, categorical encoding) to subclasses via a small template
method. This is the one place OOP clearly earns its keep: Avazu and Taxi share a
skeleton but differ in the middle.

Encoders are fit on the training split only (never the validation split) so no
future information leaks into training-set statistics.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from mlserve.common.logging import get_logger
from mlserve.config.schema import ModelConfig

log = get_logger(__name__)


@dataclass
class Processed:
    """Output bundle of a preprocessing run."""

    train: pd.DataFrame
    val: pd.DataFrame
    meta: dict


class BasePreprocessor(ABC):
    def __init__(self, config: ModelConfig, val_fraction: float = 0.2):
        self.config = config
        self.val_fraction = val_fraction
        self.encoders: dict = {}

    # --- Subclass hooks --------------------------------------------------------
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Row-level filtering. Default: pass through."""
        return df

    @abstractmethod
    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive features. Must return rows in time order and drop raw time cols."""

    def encode(
        self, train: pd.DataFrame, val: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fit encoders on train, apply to both. Default: no encoding."""
        return train, val

    # --- Template method -------------------------------------------------------
    def run(self, df_raw: pd.DataFrame) -> Processed:
        cfg = self.config
        df = self.clean(df_raw)
        df = df.drop(columns=[c for c in cfg.drop_cols if c in df.columns], errors="ignore")
        df = self.engineer(df).reset_index(drop=True)

        split = int(len(df) * (1 - self.val_fraction))
        train, val = df.iloc[:split].copy(), df.iloc[split:].copy()
        train, val = self.encode(train, val)

        feature_order = [c for c in train.columns if c != cfg.target]
        meta = {
            "name": cfg.name,
            "task": cfg.task,
            "target": cfg.target,
            "n_features": len(feature_order),
            "feature_order": feature_order,
            # Real medians from the training split -> sensible serving defaults.
            "feature_defaults": {
                c: float(train[c].median()) for c in feature_order
            },
            "encoders": self.encoders,
        }
        log.info(
            "%s: %d train / %d val rows | %d features",
            cfg.name, len(train), len(val), len(feature_order),
        )
        return Processed(train=train, val=val, meta=meta)


# --- Shared encoding helpers (used by subclasses) -----------------------------
def fit_frequency_map(series: pd.Series) -> dict:
    return series.value_counts().to_dict()


def apply_frequency_map(series: pd.Series, freq_map: dict) -> pd.Series:
    return series.map(freq_map).fillna(0).astype("int64")


def fit_label_map(series: pd.Series) -> dict:
    cats = series.astype("category").cat.categories
    return {str(v): i for i, v in enumerate(cats)}


def apply_label_map(series: pd.Series, label_map: dict) -> pd.Series:
    return series.astype(str).map(label_map).fillna(-1).astype("int64")


def write_meta(meta: dict, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2))
