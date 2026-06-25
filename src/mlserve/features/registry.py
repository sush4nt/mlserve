"""Tiny factory mapping a dataset name to its preprocessor class."""

from __future__ import annotations

from mlserve.config.schema import ModelConfig
from mlserve.features.avazu import AvazuPreprocessor
from mlserve.features.base import BasePreprocessor
from mlserve.features.nyc_taxi import TaxiPreprocessor

_PREPROCESSORS = {"avazu": AvazuPreprocessor, "nyc_taxi": TaxiPreprocessor}


def make_preprocessor(config: ModelConfig, val_fraction: float = 0.2) -> BasePreprocessor:
    return _PREPROCESSORS[config.name](config, val_fraction=val_fraction)
