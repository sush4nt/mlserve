"""Avazu preprocessing: parse the YYMMDDHH hour, frequency/label-encode
categoricals. Produces a fixed, deterministic 22-feature numeric frame."""

from __future__ import annotations

import pandas as pd

from mlserve.features.base import (
    BasePreprocessor,
    apply_frequency_map,
    apply_label_map,
    fit_frequency_map,
    fit_label_map,
)

# Canonical serving order. The model is trained on exactly this column order and
# the frontend assembles its vector in this order. Persisted to feature_meta.json.
FEATURE_ORDER = [
    "hour_of_day", "day_of_week", "banner_pos", "C1",
    "site_id", "site_domain", "site_category",
    "app_id", "app_domain", "app_category",
    "device_id", "device_model", "device_type", "device_conn_type",
    "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21",
]


class AvazuPreprocessor(BasePreprocessor):
    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        h = df["hour"].astype(str)
        df["hour_of_day"] = h.str[-2:].astype(int)
        df["day_of_week"] = h.str[4:6].astype(int) % 7  # YYMMDDHH -> day at pos 4:6
        df = df.drop(columns=["hour"])
        return df

    def encode(self, train, val):
        cfg = self.config
        for col in cfg.freq_encode_cols:
            fmap = fit_frequency_map(train[col])
            self.encoders[col] = {"type": "frequency", "map": {str(k): int(v) for k, v in fmap.items()}}
            train[col] = apply_frequency_map(train[col], fmap)
            val[col] = apply_frequency_map(val[col], fmap)
        for col in cfg.label_encode_cols:
            lmap = fit_label_map(train[col])
            self.encoders[col] = {"type": "label", "map": lmap}
            train[col] = apply_label_map(train[col], lmap)
            val[col] = apply_label_map(val[col], lmap)

        # Reorder to the canonical serving order (+ target at the end).
        ordered = [c for c in FEATURE_ORDER if c in train.columns]
        train = train[ordered + [cfg.target]]
        val = val[ordered + [cfg.target]]
        return train, val
