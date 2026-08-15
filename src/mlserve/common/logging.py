"""A single logger factory so every module logs the same way.

Every `mlserve.*` module calls `get_logger(__name__)` instead of configuring
its own handlers. That means one place controls: the log format, the level
(`MLSERVE_LOG_LEVEL`), and whether logs are also persisted to a file
(`MLSERVE_LOG_TO_FILE`) — so a training run, the serving app, and a one-off
CLI command all produce logs that look the same and land in the same place.

Console output stays exactly as before (short, human-readable, INFO+). The
optional file handler writes a slightly more detailed line (adds the process
ID, useful when `make serve`/uvicorn forks workers) to `logs/mlserve.log`,
rotated at 5 MB x 3 backups so it never grows unbounded across many runs.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

from mlserve.common.paths import LOGS_DIR

_CONFIGURED = False

_CONSOLE_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_FILE_FORMAT = "%(asctime)s | %(levelname)-7s | pid=%(process)d | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _truthy(env_var: str, default: str = "1") -> bool:
    return os.getenv(env_var, default).strip().lower() not in ("0", "false", "no", "")


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        level = os.getenv("MLSERVE_LOG_LEVEL", "INFO")
        root = logging.getLogger()
        root.setLevel(level)

        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt="%H:%M:%S"))
        root.addHandler(console)

        if _truthy("MLSERVE_LOG_TO_FILE"):
            try:
                LOGS_DIR.mkdir(parents=True, exist_ok=True)
                file_handler = RotatingFileHandler(
                    LOGS_DIR / "mlserve.log", maxBytes=5 * 1024 * 1024, backupCount=3
                )
                file_handler.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt=_DATEFMT))
                root.addHandler(file_handler)
            except OSError:
                # Read-only filesystem or permissions issue — console logging
                # still works, so don't crash the app over a nice-to-have.
                logging.getLogger(__name__).warning(
                    "Could not open %s for writing; file logging disabled", LOGS_DIR
                )

        _CONFIGURED = True
    return logging.getLogger(name)
