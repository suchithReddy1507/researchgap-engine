"""On-demand single-gap export: either today's report-style write-up
scoped to one gap, or a formatted academic-paper draft (IEEE/ACM/APA).

Separate from the main pipeline and from synth.py's existing full-report
path -- gaps.py, agent.py, nlp.py, pipeline.py, and synth.synthesize() are
never touched or imported here. Generates fresh content on demand; never
parses/extracts from an existing report string (LLM-authored markdown
structure isn't a stable contract to parse against -- see the audit that
preceded this feature).
"""
from __future__ import annotations

import json
import os

from .retry import call_with_retry

FORMATS = ("report_style", "ieee", "acm", "apa")

_STYLE_NAMES = {"ieee": "IEEE", "acm": "ACM", "apa": "APA"}


class GapExportError(Exception):
    """Message is safe to show directly to the user."""


class MissingApiKeyError(GapExportError):
    """Export hard-requires ANTHROPIC_API_KEY, same as Idea Check/chat."""


class InvalidGapError(GapExportError):
    """Bad format name, job not done, or gap_index out of range."""


class NoCitationDataError(GapExportError):
    """The requested format needs real cited papers, and this job has no
    verification_log.json (it predates the verification feature) -- refuse
    rather than let the model invent references."""


def _find_gap(job: dict, gap_index: int) -> dict:
    gaps = job["result"]["gaps"]
    if gap_index < 0 or gap_index >= len(gaps):
        raise InvalidGapError(f"No gap at index {gap_index} for this job.")
    return gaps[gap_index]


def _find_citations(verification_log: dict, gap: dict) -> dict:
    """Filters verification_log.json down to just title/doi/year for the
    one matching gap -- never the full file."""

    def matches(entry: dict) -> bool:
        return entry["topic_a"] == gap["topic_a"] and entry["topic_b"] == gap["topic_b"]

    bridge = next((e for e in verification_log.get("bridge_evidence", []) if matches(e)), None)
    validation = next((e for e in verification_log.get("validation_evidence", []) if matches(e)), None)

    bridge_papers = [
        {"title": p["title"], "doi": p.get("doi"), "year": p.get("year")}
        for p in (bridge["bridge_paper_evidence"] if bridge else [])
    ]
    validated_papers = []
    if validation:
        for matched in validation.get("matches_per_query", {}).values():
            for p in matched:
                validated_papers.append({"title": p["title"], "doi": p.get("doi"), "year": p.get("year")})

    return {"bridge_papers": bridge_papers, "validated_papers": validated_papers}


def _build_prompt(job: dict, gap: dict, gap_number: int, fmt: str, citations: dict | None) -> str:
    topics_by_label = {t["label"]: t for t in job["result"]["topics"]}
    topic_a = topics_by_label.get(gap["topic_a"], {})
    topic_b = topics_by_label.get(gap["topic_b"], {})

    context = {
        "domain": job["result"]["topic"],
        "gap_number": gap_number,
        "topic_a": {
            "label": gap["topic_a"],
            "keywords": gap["keywords_a"],
            "size": gap["size_a"],
            "mean_year": topic_a.get("mean_year"),
        },
        "topic_b": {
            "label": gap["topic_b"],
            "keywords": gap["keywords_b"],
            "size": gap["size_b"],
            "mean_year": topic_b.get("mean_year"),
        },
        "bridge_papers": gap["bridge_papers"],
        "prominence": gap["prominence"],
        "verification_queries": gap["verification_queries"],
        "recent_papers_found": gap["recent_papers_found"],
        "verdict": gap["verdict"],
        "confidence": gap["confidence"],
    }
    if citations is not None:
        context["cited_literature"] = citations

    if fmt == "report_style":
        instructions = (
            "Write this single gap's section in the same style as a research-gap report: "
            "a name for the gap, a 'Why it is a gap' paragraph grounded in the numbers "
            "below, and a 2-3 step research agenda. Natural prose, not templated-sounding."
        )
    else:
        style_name = _STYLE_NAMES[fmt]
        instructions = (
            f"Write a short academic paper draft in {style_name} style and structure "
            f"(correct section headers and in-text/reference citation conventions for "
            f"{style_name}) covering: title, abstract, introduction/background grounded "
            "in the real topic landscape data below, literature framing that cites ONLY "
            "the real papers listed in cited_literature (never invent a citation), "
            "methodology derived from the research agenda implied by the gap data, and a "
            "references section listing only the real cited papers. Natural academic "
            "prose, not robotic or templated-sounding."
        )

    return (
        f"{instructions}\n\nDo NOT invent the gap itself, its numbers, or the topics -- "
        "they are real, computed data. Ground every claim in what's given below.\n\n"
        f"DATA:\n{json.dumps(context, indent=2)}"
    )


def generate(job: dict, gap_index: int, fmt: str, verification_log: dict | None) -> str:
    if fmt not in FORMATS:
        raise InvalidGapError(f"Unknown format '{fmt}'.")
    if job.get("status") != "done" or not job.get("result"):
        raise InvalidGapError("This job hasn't finished yet.")

    gap = _find_gap(job, gap_index)

    citations = None
    if fmt != "report_style":
        if verification_log is None:
            raise NoCitationDataError(
                "No verified citation list available for this job -- it predates the "
                "verification feature. Try the report-style export instead, or re-run "
                "the analysis to generate one."
            )
        citations = _find_citations(verification_log, gap)

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise MissingApiKeyError("Export requires ANTHROPIC_API_KEY to be configured on the engine.")

    import anthropic

    # report_style is one section (comparable to a slice of the existing
    # full report); the paper-draft formats need a full academic structure
    # -- title through references -- which runs considerably longer, and
    # needs a correspondingly longer client timeout or it reliably times
    # out mid-generation (observed directly: 60s wasn't enough for 6000
    # tokens even once, let alone across retries).
    max_tokens = 2000 if fmt == "report_style" else 6000
    timeout = 30.0 if fmt == "report_style" else 150.0
    client = anthropic.Anthropic(timeout=timeout, max_retries=0)
    prompt = _build_prompt(job, gap, gap_index + 1, fmt, citations)

    def attempt():
        return client.messages.create(
            model=os.getenv("SYNTH_MODEL", "claude-sonnet-4-6"),
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

    resp = call_with_retry(attempt, retries=2, backoff=2.0, label="gap export")
    if resp is None:
        raise GapExportError("Could not generate the export right now -- try again shortly.")
    return resp.content[0].text
