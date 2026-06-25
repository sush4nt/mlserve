"""Central path resolution.

Every path in the project is derived from here so there are no scattered
relative-path assumptions. All locations can be overridden by environment
variables, which is what lets the same code run from a checkout (`uv run`),
a Docker Compose service, or a baked Hugging Face image without edits.
"""

from __future__ import annotations

import os
from pathlib import Path


def _root() -> Path:
    # MLSERVE_ROOT lets containers point at a fixed install dir.
    # Default: two levels up from this file (the repo root) when run from a checkout.
    env = os.getenv("MLSERVE_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[3]


ROOT = _root()

DATA_DIR = Path(os.getenv("MLSERVE_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = Path(os.getenv("MLSERVE_ARTIFACTS_DIR", ROOT / "artifacts"))
MODELS_DIR = ARTIFACTS_DIR / "models"

CONFIG_DIR = Path(os.getenv("MLSERVE_CONFIG_DIR", ROOT / "configs"))

FRONTEND_DIST = Path(os.getenv("MLSERVE_FRONTEND_DIST", ROOT / "frontend" / "dist"))
FRONTEND_CONFIG = ROOT / "frontend" / "src" / "config" / "models.generated.json"


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
