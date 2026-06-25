"""Stage 1 entrypoint: prepare processed train/val parquet + feature metadata.

    uv run mlserve-prepare --dataset all --source synthetic --rows 100000
    uv run mlserve-prepare --dataset avazu --source kaggle --rows 5000000
"""

from __future__ import annotations

import argparse

import pyarrow as pa
import pyarrow.parquet as pq

from mlserve.common.logging import get_logger
from mlserve.common.paths import PROCESSED_DIR, ensure_dirs
from mlserve.config.schema import load_config
from mlserve.data.loader import load_raw
from mlserve.features.base import write_meta
from mlserve.features.registry import make_preprocessor

log = get_logger(__name__)
DATASETS = ["avazu", "nyc_taxi"]


def prepare_one(dataset: str, source: str, rows: int) -> None:
    config = load_config(dataset)
    df_raw = load_raw(dataset, source, rows)

    proc = make_preprocessor(config).run(df_raw)

    out = PROCESSED_DIR / dataset
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(proc.train), out / "train.parquet")
    pq.write_table(pa.Table.from_pandas(proc.val), out / "val.parquet")
    write_meta(proc.meta, out / "feature_meta.json")
    log.info("Wrote %s/{train,val}.parquet + feature_meta.json", out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare processed datasets.")
    ap.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    ap.add_argument("--source", choices=["synthetic", "kaggle"], default="synthetic")
    ap.add_argument("--rows", type=int, default=100_000,
                    help="Rows to generate (synthetic) or read (kaggle).")
    args = ap.parse_args()

    ensure_dirs()
    targets = DATASETS if args.dataset == "all" else [args.dataset]
    for ds in targets:
        prepare_one(ds, args.source, args.rows)


if __name__ == "__main__":
    main()
