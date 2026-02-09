"""Shared logging setup for scripts."""

from __future__ import annotations

import logging
import os
import sys

def setup_logging(default_level: str = "INFO") -> None:
    """Initialize consistent logging across scripts."""
    level_name = os.getenv("LOG_LEVEL", default_level).upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
