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


def load_raw(dataset: str, source: str, n_rows: int) -> pd.DataFrame:
    if source == "synthetic":
        gen = {"avazu": synthetic.make_avazu, "nyc_taxi": synthetic.make_nyc_taxi}[dataset]
        log.info("Generating %d synthetic %s rows", n_rows, dataset)
        return gen(n_rows)

    if source == "kaggle":
        csv = RAW_DIR / dataset / "train.csv"
        if not csv.exists():
            raise FileNotFoundError(
                f"{csv} not found. Run `make download-data` first "
                f"(requires a Kaggle account and accepted competition rules)."
            )
        log.info("Reading %d rows from %s", n_rows, csv)
        return pd.read_csv(csv, nrows=n_rows)

    raise ValueError(f"Unknown source: {source!r} (use 'synthetic' or 'kaggle')")
