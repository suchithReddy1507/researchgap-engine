"""Shared retry-with-backoff wrapper for external calls (OpenAlex, Semantic
Scholar, Anthropic). A pipeline run must never hang indefinitely on one
call — every external call goes through here: a hard per-attempt timeout
(set by the caller) plus a bounded number of retries, then gives up and
returns None so the caller can skip that piece of work gracefully.
"""
from __future__ import annotations

import time
from typing import Callable, TypeVar

from .logconfig import logger

T = TypeVar("T")


def call_with_retry(fn: Callable[[], T], *, retries: int = 2, backoff: float = 2.0, label: str = "call") -> T | None:
    """Call fn() up to (retries + 1) times total. Returns fn()'s result, or
    None if every attempt raised — never re-raises, so a single flaky
    external call degrades gracefully instead of killing the whole run."""
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:  # deliberately broad: any external-call failure must degrade gracefully
            last_exc = e
            if attempt < retries:
                wait = backoff * (attempt + 1)
                logger.warning(f"{label} failed ({e}) — retry {attempt + 1}/{retries} in {wait:.0f}s")
                time.sleep(wait)
    logger.warning(f"{label} failed after {retries + 1} attempts, skipping: {last_exc}")
    return None
