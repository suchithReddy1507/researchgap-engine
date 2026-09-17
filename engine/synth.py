"""Stage 6 — LLM synthesis.

Takes the topics + validated gaps (all computed from live data) and asks the
LLM to VERBALIZE them — never to invent gaps. Falls back to a plain
programmatic report when no API key is set, so the pipeline always completes.
"""
from __future__ import annotations

import json
import os

from .logconfig import logger
from .retry import call_with_retry


def synthesize(topic: str, topics: dict, validated: list[dict]) -> tuple[str, str]:
    """Returns (report_markdown, report_source).

    report_source is "llm" when Claude wrote the report, or a "fallback_*"
    value when it did not. The fallback report is correct but visibly poorer
    (plain text, no prose synthesis), and it used to be indistinguishable
    from a real one at a glance -- so the reason is both logged loudly and
    returned to the caller for the job record.
    """
    survivors = [g for g in validated if g["verdict"] != "REJECTED"]
    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.warning(
            "=== REPORT IS FALLBACK (plain text, not LLM-written): "
            "ANTHROPIC_API_KEY is not set ==="
        )
        return _plain_report(topic, topics, validated), "fallback_no_api_key"

    import anthropic

    # max_retries=0: the SDK's own built-in retry (default 2) would silently
    # stack under call_with_retry below, multiplying the effective timeout.
    # 90s, not 30s: this call asks for up to 2000 tokens of report prose, and
    # a 30s ceiling was timing out on larger topic sets -- all three attempts
    # would fail and the run would silently degrade to _plain_report(), which
    # is correct but visibly worse output. Retries are still capped at 2 by
    # call_with_retry below, so worst case is bounded.
    client = anthropic.Anthropic(timeout=90.0, max_retries=0)
    payload = {
        "domain": topic,
        "topics_found": [
            {"label": t["label"], "size": t["size"], "mean_year": t["mean_year"]}
            for t in topics.values()
        ],
        "validated_gaps": survivors,
        "rejected_gaps": [
            {"topics": f"{g['topic_a']} x {g['topic_b']}", "recent_papers": g["recent_papers_found"]}
            for g in validated
            if g["verdict"] == "REJECTED"
        ],
    }
    prompt = (
        "You are the synthesis module of a research-gap finder. Below is REAL "
        "computed data: topic clusters from live literature and candidate gaps "
        "that survived agentic validation. Write a concise research-gap report:\n"
        "- one short paragraph on the domain's topic landscape\n"
        "- for each validated gap: a name, why it is a gap (use the bridge-paper "
        "and validation numbers), and a 2-3 step research agenda\n"
        "- one line acknowledging the rejected candidates\n"
        "Do NOT invent gaps that are not in the data. Ground every claim in the numbers.\n\n"
        f"DATA:\n{json.dumps(payload, indent=2)}"
    )
    def attempt():
        return client.messages.create(
            model=os.getenv("SYNTH_MODEL", "claude-sonnet-4-6"),
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )

    resp = call_with_retry(attempt, retries=2, backoff=2.0, label="Anthropic synthesis")
    if resp is None:
        logger.warning(
            "=== REPORT IS FALLBACK (plain text, not LLM-written): Anthropic "
            "synthesis failed every attempt (timeout, rate limit, or credit). "
            "The numbers are real; the prose is not LLM-generated. ==="
        )
        return _plain_report(topic, topics, validated), "fallback_llm_unavailable"
    return resp.content[0].text, "llm"


def _plain_report(topic: str, topics: dict, validated: list[dict]) -> str:
    lines = [f"RESEARCH GAP REPORT — {topic}", "=" * 60, ""]
    lines.append(f"Topics discovered: {len(topics)}")
    for t in topics.values():
        lines.append(f"  - {t['label']}  ({t['size']} papers, mean year {t['mean_year']})")
    lines.append("")
    for g in validated:
        lines.append(f"[{g['verdict']}] {g['topic_a']}  x  {g['topic_b']}")
        lines.append(
            f"    prominence={g['prominence']}  bridges={g['bridge_papers']}  "
            f"recent_papers_found={g['recent_papers_found']}  confidence={g['confidence']}"
        )
    return "\n".join(lines)
