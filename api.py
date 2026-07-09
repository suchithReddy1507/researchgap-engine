"""ResearchGap AI — FastAPI wrapper around the same engine pipeline used by main.py.

Run with:
    uvicorn api:app --port 8000
"""
from __future__ import annotations

import json

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from engine import jobs as jobs_module  # noqa: E402
from engine import stats as stats_module  # noqa: E402

app = FastAPI(title="ResearchGap AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "http://127.0.0.1:3001"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    topic: str
    papers: int = 300


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def get_stats():
    return stats_module.load_stats()


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """Kicks off the pipeline in a background thread and returns immediately.
    Poll GET /jobs/{job_id} for progress and the final result."""
    job_id = jobs_module.create_job(req.topic, req.papers)
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs_module.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    # round-trip through json with default=str to guarantee everything is
    # JSON-serializable, same safety net main.py uses when writing gaps.json
    return json.loads(json.dumps(job, default=str))
