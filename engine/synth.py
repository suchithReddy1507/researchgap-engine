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


def synthesize(topic: str, topics: dict, validated: list[dict]) -> str:
    survivors = [g for g in validated if g["verdict"] != "REJECTED"]
    if not os.getenv("ANTHROPIC_API_KEY"):
        return _plain_report(topic, topics, validated)

    import anthropic

    # max_retries=0: the SDK's own built-in retry (default 2) would silently
    # stack under call_with_retry below, multiplying the effective timeout.
    client = anthropic.Anthropic(timeout=30.0, max_retries=0)
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
        logger.warning("LLM synthesis unavailable after retries — falling back to plain report")
        return _plain_report(topic, topics, validated)
    return resp.content[0].text


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
