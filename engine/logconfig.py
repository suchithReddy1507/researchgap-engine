"""Shared 'engine' logger — writes every stage transition and validation
query/result to logs/engine.log (with timestamps) as well as stdout, so a
run's progress can always be inspected regardless of how the process's
stdout is captured.
"""
from __future__ import annotations

import logging
import pathlib

LOG_DIR = pathlib.Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("engine")
if not logger.handlers:
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(LOG_DIR / "engine.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)

    logger.propagate = False
