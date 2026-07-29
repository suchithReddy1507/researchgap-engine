"""Standalone re-verification — independently checks a completed job's
numbers against fresh data. Separate from the main pipeline; never imported
by it, never imports gaps.py or agent.py.

Usage:
    python verify_job.py <job_id>
    python verify_job.py <job_id> --out report.json

Two different kinds of check, on purpose:

  Recent-hits (agent.py's validation numbers): a genuine independent
  re-check. Re-issues the exact verification_queries from the original run
  against OpenAlex right now, bypassing the cache, and compares to the
  claimed count. A mismatch may just mean new papers were published since
  the original run -- not necessarily a bug.

  Bridge-papers (gaps.py's detection numbers): NOT independently
  re-checkable. BERTopic's clustering here has no fixed random seed (see
  engine/nlp.py), so re-clustering the same cached papers can legitimately
  produce a different topic split and a different bridge count -- that
  would be clustering variance, not a bug. This script instead checks
  internal consistency against the evidence captured during the original
  run (jobs/<job_id>/verification_log.json) and prints the actual bridging
  papers for manual review.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

from dotenv import load_dotenv

load_dotenv()

from engine import fetch  # noqa: E402

JOBS_DIR = pathlib.Path(__file__).resolve().parent / "jobs"


def _load_job(job_id: str) -> dict:
    path = JOBS_DIR / f"{job_id}.json"
    if not path.exists():
        sys.exit(f"No job found at {path}")
    return json.loads(path.read_text())


def _load_verification_log(job_id: str) -> dict | None:
    path = JOBS_DIR / job_id / "verification_log.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _recheck_recent_hits(gap: dict, from_year: int) -> dict:
    queries = gap.get("verification_queries", [])
    hits = 0
    any_verified = False
    for q in queries:
        count = fetch.count_openalex(q, from_year, use_cache=False)
        if count is None:
            continue
        any_verified = True
        hits = max(hits, count)

    claimed = gap.get("recent_papers_found")
    return {
        "topic_a": gap["topic_a"],
        "topic_b": gap["topic_b"],
        "verification_queries": queries,
        "claimed_recent_papers_found": claimed,
        "fresh_recent_papers_found": hits if any_verified else None,
        "status": (
            "UNVERIFIABLE" if not any_verified
            else "MATCH" if hits == claimed
            else "MISMATCH"
        ),
    }


def _check_bridge_consistency(gap: dict, log: dict | None) -> dict:
    claimed = gap.get("bridge_papers")
    entry = None
    if log:
        entry = next(
            (
                e for e in log.get("bridge_evidence", [])
                if e["topic_a"] == gap["topic_a"] and e["topic_b"] == gap["topic_b"]
            ),
            None,
        )

    if entry is None:
        return {
            "topic_a": gap["topic_a"],
            "topic_b": gap["topic_b"],
            "claimed_bridge_papers": claimed,
            "status": "NO_EVIDENCE_CAPTURED",
            "note": "No verification_log.json for this job/gap -- job predates this feature, "
                    "or evidence wasn't captured. Cannot check even internal consistency.",
        }

    captured = entry["captured_bridge_papers"]
    return {
        "topic_a": gap["topic_a"],
        "topic_b": gap["topic_b"],
        "claimed_bridge_papers": claimed,
        "captured_bridge_papers": captured,
        "bridge_paper_evidence": entry["bridge_paper_evidence"],
        "status": "CONSISTENT" if captured == claimed else "INCONSISTENT",
        "note": "This is a consistency check against evidence captured during the original run, "
                "not a fresh independent recount -- BERTopic's clustering here isn't seeded, so "
                "re-clustering could legitimately produce a different split.",
    }


def verify(job_id: str) -> dict:
    job = _load_job(job_id)
    if job["status"] != "done":
        sys.exit(f"Job {job_id} is not done (status={job['status']}) -- nothing to verify yet.")

    log = _load_verification_log(job_id)
    from_year = log["from_year"] if log else None
    if from_year is None:
        print(
            f"WARNING: no verification_log.json for job {job_id} -- this job predates the "
            "verification feature (or evidence capture failed). Recent-hits re-check will use "
            "each query's original from_year is unavailable; skipping recent-hits re-check.",
            file=sys.stderr,
        )

    gaps = job["result"]["gaps"]
    recent_hits_report = [_recheck_recent_hits(g, from_year) for g in gaps] if from_year is not None else []
    bridge_report = [_check_bridge_consistency(g, log) for g in gaps]

    return {
        "job_id": job_id,
        "topic": job["topic"],
        "recent_hits_recheck": recent_hits_report,
        "bridge_consistency_check": bridge_report,
    }


def _print_summary(report: dict) -> None:
    print(f"\n=== Re-verification: job {report['job_id']} ('{report['topic']}') ===\n")

    print("-- Recent-hits (fresh independent re-check against OpenAlex) --")
    if not report["recent_hits_recheck"]:
        print("  skipped (no verification_log.json for this job)")
    for r in report["recent_hits_recheck"]:
        print(f"  [{r['status']}] {r['topic_a']} x {r['topic_b']}: "
              f"claimed={r['claimed_recent_papers_found']} fresh={r['fresh_recent_papers_found']}")

    print("\n-- Bridge-papers (consistency check against captured evidence, not a fresh recount) --")
    for r in report["bridge_consistency_check"]:
        captured = r.get("captured_bridge_papers", "n/a")
        print(f"  [{r['status']}] {r['topic_a']} x {r['topic_b']}: "
              f"claimed={r['claimed_bridge_papers']} captured={captured}")

    mismatches = [r for r in report["recent_hits_recheck"] if r["status"] == "MISMATCH"]
    inconsistent = [r for r in report["bridge_consistency_check"] if r["status"] == "INCONSISTENT"]
    print(f"\n{len(mismatches)} recent-hits mismatch(es), {len(inconsistent)} bridge inconsistency(ies).\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("job_id")
    ap.add_argument("--out", help="also write the full report to this JSON file")
    args = ap.parse_args()

    report = verify(args.job_id)
    _print_summary(report)

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(report, indent=2, default=str))
        print(f"Full report written to {args.out}")
