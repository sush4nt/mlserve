"""Load raw rows from one of two sources, behind a single function.

- "synthetic" (default): generated in-memory, no download. Great for dev/CI/demo.
- "kaggle": reads the real CSV from data/raw/<dataset>/train.csv (download it
  yourself with `make download-data`; needs a Kaggle account).

The rest of the pipeline does not care which source produced the rows.
"""

from __future__ import annotations

import pandas as pd

from mlserve.common.logging import get_logger
from mlserve.common.paths import RAW_DIR
from mlserve.data import synthetic

log = get_logger(__name__)

_SYNTHETIC = {
    "avazu": synthetic.make_avazu,
    "nyc_taxi": synthetic.make_nyc_taxi,
}

_KAGGLE_CSV = {
    "avazu": "train.gz",
    "nyc_taxi": "train.csv",
}

def _validate(dataset: str, source: str) -> None:
    if dataset not in _SYNTHETIC:
        raise ValueError(f"Unknown dataset: {dataset!r} (use 'avazu' or 'nyc_taxi')")
    if source not in {"synthetic", "kaggle"}:
        raise ValueError(f"Unknown source: {source!r} (use 'synthetic' or 'kaggle')")

def _kaggle_read(dataset: str, csv_path, n_rows: int) -> pd.DataFrame:
    log.info("Reading %d rows from %s", n_rows, csv_path)
    if dataset == "avazu":
        return pd.read_csv(csv_path, compression="gzip", nrows=n_rows)
    return pd.read_csv(csv_path, nrows=n_rows)  # nyc_taxi

def load_raw(dataset: str, source: str, n_rows: int) -> pd.DataFrame:
    _validate(dataset, source)

    if source == "synthetic":
        gen = _SYNTHETIC[dataset]
        log.info("Generating %d synthetic %s rows", n_rows, dataset)
        return gen(n_rows)

    # source == "kaggle"
    csv = RAW_DIR / dataset / _KAGGLE_CSV[dataset]
    if not csv.exists():
        raise FileNotFoundError(
            f"{csv} not found. Run `make download-data` first "
            f"(requires a Kaggle account and accepted competition rules)."
        )

    return _kaggle_read(dataset, csv, n_rows)