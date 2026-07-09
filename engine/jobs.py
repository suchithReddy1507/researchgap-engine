"""Async job tracking for /analyze.

POST /analyze must return instantly — the pipeline itself runs in a
background thread. Progress is written to jobs/{job_id}.json after every
stage transition so GET /jobs/{job_id} can be polled for live status
without holding an HTTP connection open for the whole run (the root cause
of the earlier "infinite timer" bug: a client-side fetch with no way to
observe real progress, timing out or getting orphaned mid-run).
"""
from __future__ import annotations

import json
import pathlib
import threading
import time
import uuid

from .logconfig import logger
from .pipeline import PipelineError, run_pipeline

JOBS_DIR = pathlib.Path(__file__).resolve().parent.parent / "jobs"
JOBS_DIR.mkdir(exist_ok=True)


def _job_path(job_id: str) -> pathlib.Path:
    return JOBS_DIR / f"{job_id}.json"


def _write_job(job_id: str, data: dict) -> None:
    """Write-then-rename so a concurrent GET never reads a half-written file."""
    path = _job_path(job_id)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.replace(path)


def get_job(job_id: str) -> dict | None:
    path = _job_path(job_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def create_job(topic: str, papers: int) -> str:
    """Writes the initial job record, starts the pipeline in a background
    thread, and returns immediately with the job_id."""
    job_id = uuid.uuid4().hex[:12]
    now = time.time()
    _write_job(
        job_id,
        {
            "job_id": job_id,
            "status": "pending",
            "topic": topic,
            "papers": papers,
            "stage": 0,
            "stage_label": "Queued",
            "created_at": now,
            "updated_at": now,
            "result": None,
            "error": None,
        },
    )
    thread = threading.Thread(target=_run, args=(job_id, topic, papers), daemon=True)
    thread.start()
    return job_id


def _update(job_id: str, **fields) -> None:
    job = get_job(job_id)
    if job is None:  # job file vanished — nothing sane to do but stop trying
        return
    job.update(**fields, updated_at=time.time())
    _write_job(job_id, job)


def _run(job_id: str, topic: str, papers: int) -> None:
    _update(job_id, status="running")

    def on_stage(n: int, label: str) -> None:
        _update(job_id, stage=n, stage_label=label)

    try:
        result = run_pipeline(topic, papers, on_stage=on_stage)
    except PipelineError as e:
        _update(job_id, status="error", error=str(e))
        return
    except Exception as e:  # a background thread must never die silently
        logger.exception(f"job {job_id} failed unexpectedly")
        _update(job_id, status="error", error=f"Unexpected error: {e}")
        return

    _update(
        job_id,
        status="done",
        stage=6,
        stage_label="Done",
        result={
            "topic": result["topic"],
            "papers_count": result["papers_count"],
            "topics": result["topics"],
            "gaps": result["gaps"],
            "confirmed_count": result["confirmed_count"],
            "weak_count": result["weak_count"],
            "rejected_count": result["rejected_count"],
            "unverified_count": result["unverified_count"],
            "report": result["report"],
        },
    )
