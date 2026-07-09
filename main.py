"""ResearchGap AI — engine runner.

Usage:
    python main.py "federated learning in healthcare"
    python main.py "quantum error correction" --papers 200
    python main.py "quantum error correction" --fresh   # bypass the cache

Every stage runs on live data fetched at execution time (or from the local
cache — see engine/fetch.py). There is no static or hardcoded result
anywhere in this pipeline.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

from dotenv import load_dotenv

load_dotenv()

from engine.pipeline import PipelineError, run_pipeline  # noqa: E402


def run(topic: str, n_papers: int, use_cache: bool = True) -> None:
    try:
        result = run_pipeline(topic, n_papers, use_cache=use_cache)
    except PipelineError as e:
        sys.exit(str(e))

    out = pathlib.Path("output")
    out.mkdir(exist_ok=True)
    (out / "gaps.json").write_text(json.dumps(result["gaps"], indent=2, default=str))
    (out / "report.md").write_text(result["report"])
    print(f"\nSaved: output/gaps.json, output/report.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("topic", help="research domain, e.g. 'federated learning healthcare'")
    ap.add_argument("--papers", type=int, default=300)
    ap.add_argument("--fresh", action="store_true", help="bypass the cache and re-fetch from APIs")
    args = ap.parse_args()
    run(args.topic, args.papers, use_cache=not args.fresh)
