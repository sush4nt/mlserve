"""A single logger factory so every module logs the same way."""

from __future__ import annotations

import logging
import os

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=os.getenv("MLSERVE_LOG_LEVEL", "INFO"),
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        _CONFIGURED = True
    return logging.getLogger(name)
