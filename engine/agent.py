"""Stage 5 — Agentic validation.

For each candidate gap the agent:
  1. generates verification queries (LLM if key present, deterministic fallback otherwise)
  2. fires them at OpenAlex — a DIFFERENT source than the clustering data
  3. counts recent papers actually addressing that intersection
  4. decides: CONFIRMED / WEAK / REJECTED / UNVERIFIED

This is a real act-observe-decide loop per gap. Gaps visibly die here,
which is your proof in the demo that nothing is canned.

UNVERIFIED means every query for that gap failed after retries (OpenAlex
unreachable/rate-limited) — it is distinct from CONFIRMED (0 hits found)
because we genuinely don't know, and must not be presented as validated.
"""
from __future__ import annotations

import json
import os

from . import fetch
from .logconfig import logger
from .retry import call_with_retry

REJECT_AT = 25   # >= this many recent papers -> the gap is already an active field
WEAK_AT = 8      # some activity -> gap survives with lowered confidence
RECENT_YEARS = 3
QUERIES_PER_GAP = 2


def _llm_queries(gap: dict) -> list[str] | None:
    """Ask a small fast model for verification queries. Returns None if no
    key, or if the call fails after retries — callers fall back to
    deterministic queries so validation always proceeds."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    import anthropic

    # max_retries=0: the SDK's own built-in retry (default 2) would silently
    # stack under call_with_retry below, multiplying the effective timeout.
    client = anthropic.Anthropic(timeout=30.0, max_retries=0)
    prompt = (
        "Two research topics:\n"
        f"A: {', '.join(gap['keywords_a'])}\n"
        f"B: {', '.join(gap['keywords_b'])}\n"
        f"Write {QUERIES_PER_GAP} literature-search queries that would find papers combining BOTH "
        "topics. Each query must be exactly two double-quoted phrases: one core "
        "concept from topic A, one core concept from topic B, "
        'e.g. "graph neural networks" "drug discovery". '
        f"Respond ONLY with a JSON array of {QUERIES_PER_GAP} strings."
    )

    def attempt():
        return client.messages.create(
            model=os.getenv("AGENT_MODEL", "claude-haiku-4-5-20251001"),
            max_tokens=200,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )

    resp = call_with_retry(attempt, retries=2, backoff=2.0, label=f"Anthropic query-gen for {gap['topic_a']} x {gap['topic_b']}")
    if resp is None:
        return None
    try:
        text = resp.content[0].text.strip().removeprefix("```json").removesuffix("```").strip()
        queries = json.loads(text)
        return [q for q in queries if isinstance(q, str)][:QUERIES_PER_GAP]
    except (json.JSONDecodeError, IndexError, AttributeError):
        return None


def _fallback_queries(gap: dict) -> list[str]:
    """Deterministic queries from topic keywords — still real validation."""
    a, b = gap["keywords_a"], gap["keywords_b"]
    return [f"{a[0]} {b[0]}", f"{' '.join(a[:2])} {b[0]}"][:QUERIES_PER_GAP]


def validate_gaps(gaps: list[dict], current_year: int) -> list[dict]:
    from_year = current_year - RECENT_YEARS
    validated = []
    for gap in gaps:
        llm_queries = _llm_queries(gap)
        queries = llm_queries or _fallback_queries(gap)
        is_fallback = llm_queries is None
        hits = 0
        any_verified = False
        for q in queries:
            # Fallback queries are low-precision word pairs -- never trust
            # a cached count for one, and never let one poison the shared
            # cache for a future run.
            count = fetch.count_openalex(q, from_year=from_year, use_cache=not is_fallback)
            if count is None:
                logger.warning(f"[agent] query unverifiable ({q}) for {gap['topic_a']} x {gap['topic_b']}")
                continue
            any_verified = True
            hits = max(hits, count)
            logger.info(f"[agent] query='{q}' hits={count}")

        if not any_verified:
            verdict, confidence = "UNVERIFIED", 0.0
        elif hits >= REJECT_AT:
            verdict, confidence = "REJECTED", 0.0
        elif hits >= WEAK_AT:
            verdict, confidence = "WEAK", round(1 - hits / REJECT_AT, 2)
        else:
            verdict, confidence = "CONFIRMED", round(1 - hits / REJECT_AT, 2)

        validated.append(
            {
                **gap,
                "verification_queries": queries,
                "recent_papers_found": hits,
                "verdict": verdict,
                "confidence": confidence,
            }
        )
        logger.info(
            f"[agent] {gap['topic_a']} x {gap['topic_b']}: "
            f"{hits} recent papers -> {verdict}"
        )
    return validated
