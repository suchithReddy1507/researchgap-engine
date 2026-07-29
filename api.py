"""ResearchGap AI — FastAPI wrapper around the same engine pipeline used by main.py.

Run with:
    uvicorn api:app --port 8000
"""
from __future__ import annotations

import json

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, HTTPException, Response, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from engine import chat as chat_module  # noqa: E402
from engine import gap_export  # noqa: E402
from engine import idea_extract  # noqa: E402
from engine import jobs as jobs_module  # noqa: E402
from engine import pdf_render  # noqa: E402
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


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []


class GapExportRequest(BaseModel):
    format: str


class PdfRenderRequest(BaseModel):
    content: str
    format: str
    gap_number: int


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


@app.post("/idea-check")
async def idea_check(file: UploadFile = File(...)):
    """Extracts a core topic/keywords from one uploaded PDF (via Claude's
    native PDF support), then kicks off the exact same pipeline /analyze
    uses, seeded with the extracted keywords. Returns immediately with a
    job_id to poll, same as /analyze -- extraction itself is a fast,
    blocking Claude call, done before the (1-3 min) pipeline job starts."""
    if file.content_type != "application/pdf" and not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Please upload a PDF file.")

    pdf_bytes = await file.read()
    try:
        idea = idea_extract.extract_idea_from_pdf(pdf_bytes)
    except idea_extract.MissingApiKeyError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except idea_extract.IdeaExtractionError as e:
        raise HTTPException(status_code=422, detail=str(e))

    job_id = jobs_module.create_job(idea["core_topic"], 300, idea=idea)
    return {"job_id": job_id, "idea": idea}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs_module.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    # round-trip through json with default=str to guarantee everything is
    # JSON-serializable, same safety net main.py uses when writing gaps.json
    return json.loads(json.dumps(job, default=str))


@app.post("/jobs/{job_id}/chat")
def job_chat(job_id: str, req: ChatRequest):
    """Answers a question about one completed job, grounded only in that
    job's own data. Stateless -- the caller sends the full conversation
    history each request; nothing is persisted server-side."""
    job = jobs_module.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    try:
        answer = chat_module.ask(job, req.message, [m.model_dump() for m in req.history])
    except chat_module.MissingApiKeyError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except chat_module.JobNotReadyError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except chat_module.ChatError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return {"answer": answer}


@app.post("/jobs/{job_id}/gaps/{gap_index}/export")
def export_gap(job_id: str, gap_index: int, req: GapExportRequest):
    """Generates fresh content scoped to one gap -- either today's report
    style (no citations needed) or a formatted academic-paper draft
    (ieee/acm/apa, which requires real cited papers from this job's
    verification_log.json and refuses rather than inventing references
    if that file doesn't exist)."""
    job = jobs_module.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    verification_log = jobs_module.get_verification_log(job_id)

    try:
        content = gap_export.generate(job, gap_index, req.format, verification_log)
    except gap_export.MissingApiKeyError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except gap_export.NoCitationDataError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except gap_export.InvalidGapError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except gap_export.GapExportError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return {"content": content, "format": req.format}


@app.post("/jobs/{job_id}/gaps/{gap_index}/export/pdf")
def export_gap_pdf(job_id: str, gap_index: int, req: PdfRenderRequest):
    """Renders already-generated gap-export content (the same text the
    caller already has, e.g. from /export) into a typeset PDF. Pure
    rendering -- no LLM call, so what downloads always matches exactly
    what was already shown on screen. job_id/gap_index stay in the URL
    for logging/REST consistency even though nothing is looked up from
    them here."""
    try:
        pdf_bytes = pdf_render.render_pdf(req.content, req.format, req.gap_number)
    except pdf_render.PdfRenderError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="gap-{req.gap_number}-{req.format}.pdf"'},
    )
