"""Lifetime usage counters, persisted to stats.json at the project root.

run_pipeline() calls record_run() after every successful run (CLI and API
both go through run_pipeline, so this is the single place counters update).
"""
from __future__ import annotations

import json
import pathlib
import threading

STATS_PATH = pathlib.Path(__file__).resolve().parent.parent / "stats.json"
_lock = threading.Lock()

DEFAULTS = {
    "total_papers_analyzed": 0,
    "total_analyses": 0,
    "total_gaps_found": 0,
    "total_confirmed": 0,
}


def load_stats() -> dict:
    if STATS_PATH.exists():
        return {**DEFAULTS, **json.loads(STATS_PATH.read_text())}
    return dict(DEFAULTS)


def record_run(papers_count: int, gaps_found: int, confirmed: int) -> dict:
    """Increment lifetime counters by one completed run's worth and persist."""
    with _lock:
        stats = load_stats()
        stats["total_papers_analyzed"] += papers_count
        stats["total_analyses"] += 1
        stats["total_gaps_found"] += gaps_found
        stats["total_confirmed"] += confirmed
        STATS_PATH.write_text(json.dumps(stats, indent=2))
        return stats
