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

from . import landing
from .logconfig import logger
from .pipeline import PipelineError, run_pipeline

JOBS_DIR = pathlib.Path(__file__).resolve().parent.parent / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

# Stage 3 (UMAP inside BERTopic) runs Numba parallel regions. Numba's default
# "workqueue" threading layer is NOT threadsafe: two pipeline runs entering a
# parallel region concurrently abort the whole interpreter with SIGABRT
# (exit 134), killing the API server and orphaning every in-flight job. Runs
# are therefore serialized end-to-end -- a second analysis queues behind the
# first instead of taking the process down with it.
_PIPELINE_LOCK = threading.Lock()


def _job_path(job_id: str) -> pathlib.Path:
    return JOBS_DIR / f"{job_id}.json"


def _verification_log_path(job_id: str) -> pathlib.Path:
    return JOBS_DIR / job_id / "verification_log.json"


def _write_verification_log(job_id: str, evidence: dict) -> None:
    path = _verification_log_path(job_id)
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"job_id": job_id, **evidence}, indent=2, default=str))
    tmp.replace(path)


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


def get_verification_log(job_id: str) -> dict | None:
    """None if this job predates the verification feature (or the write
    never happened) -- callers that need real cited papers (gap_export.py)
    must treat that as "no citation data available", not "zero papers"."""
    path = _verification_log_path(job_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def reap_orphaned_jobs() -> int:
    """Mark jobs left mid-flight by a previous process as failed.

    Job state lives on disk but the thread running it does not survive a
    restart, so after a crash (or a plain Ctrl-C) GET /jobs/{id} keeps
    answering 200 {"status": "running"} forever and the UI polls a job that
    nothing is working on -- the "infinite spinner". Called once at import,
    i.e. once per server start, before any new job can be created.
    """
    reaped = 0
    for path in JOBS_DIR.glob("*.json"):
        try:
            job = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if job.get("status") not in ("running", "pending"):
            continue
        job.update(
            status="error",
            error="The engine restarted while this analysis was running. Please run it again.",
            updated_at=time.time(),
        )
        _write_job(job["job_id"], job)
        reaped += 1
    if reaped:
        logger.warning(f"marked {reaped} orphaned job(s) as failed at startup")
    return reaped


def create_job(topic: str, papers: int, idea: dict | None = None) -> str:
    """Writes the initial job record, starts the pipeline in a background
    thread, and returns immediately with the job_id.

    idea is only set for Idea Check jobs (extracted title/keywords/summary
    from an uploaded PDF) -- when present, _run() attaches an idea_landing
    comparison to the result once the pipeline completes. Normal /analyze
    jobs pass idea=None and are completely unaffected."""
    job_id = uuid.uuid4().hex[:12]
    now = time.time()
    _write_job(
        job_id,
        {
            "job_id": job_id,
            "status": "pending",
            "topic": topic,
            "papers": papers,
            "idea": idea,
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
    def on_stage(n: int, label: str) -> None:
        _update(job_id, stage=n, stage_label=label)

    # See _PIPELINE_LOCK: only one pipeline may be inside Numba's parallel
    # regions at a time. Queue rather than reject, so firing off a second
    # analysis stays a supported thing to do -- it just waits its turn. The
    # status stays "pending" while queued (an existing, already-handled
    # state) so neither the job schema nor the UI needs to learn a new one.
    if not _PIPELINE_LOCK.acquire(blocking=False):
        logger.info(f"job {job_id} queued behind a running analysis")
        _update(job_id, stage_label="Queued - waiting for the current analysis to finish...")
        _PIPELINE_LOCK.acquire()

    try:
        _update(job_id, status="running")
        try:
            result = run_pipeline(topic, papers, on_stage=on_stage)
        except PipelineError as e:
            _update(job_id, status="error", error=str(e))
            return
        except Exception as e:  # a background thread must never die silently
            logger.exception(f"job {job_id} failed unexpectedly")
            _update(job_id, status="error", error=f"Unexpected error: {e}")
            return

        # Written separately from the job record below (jobs/<job_id>/verification_log.json,
        # not jobs/<job_id>.json) so the existing job schema the UI already parses never changes.
        _write_verification_log(job_id, result["verification_evidence"])

        # Idea Check jobs only: compare the extracted idea's keywords against
        # this run's topics/gaps. Pure post-processing on run_pipeline()'s
        # existing output -- pipeline.py itself needs no changes for this.
        job = get_job(job_id)
        idea = job.get("idea") if job else None
        idea_landing = landing.compute_landing(idea, result["topics"], result["gaps"]) if idea else None

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
                # "llm" or "fallback_*" -- lets you tell at a glance whether this
                # report was actually written by Claude. See synth.synthesize.
                "report_source": result["report_source"],
                # Per-run measurements for the paper's Tables II, III and XI.
                "run_stats": result["run_stats"],
                "idea_landing": idea_landing,
            },
        )
    finally:
        _PIPELINE_LOCK.release()


# Runs once per server start, at import time -- before api.py can serve
# anything, so a client can never observe a stale "running" job from a
# previous process.
reap_orphaned_jobs()
