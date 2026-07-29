"""Chat about a completed job's results, grounded only in that job's own data.

Stateless: the caller sends the full conversation history each request,
nothing is persisted server-side. Separate from the main pipeline -- never
imports gaps.py, agent.py, nlp.py, or pipeline.py. Only ever reads a job
dict that's already fully computed by the time this runs.
"""
from __future__ import annotations

import json
import os

from .retry import call_with_retry


class ChatError(Exception):
    """Message is safe to show directly to the user."""


class MissingApiKeyError(ChatError):
    """Chat hard-requires ANTHROPIC_API_KEY, same as Idea Check."""


class JobNotReadyError(ChatError):
    """Job isn't done (or errored) yet -- nothing to chat about."""


def _build_system_prompt(job: dict) -> str:
    result = job["result"]
    core = {
        "topic": result["topic"],
        "papers_count": result["papers_count"],
        "topics": result["topics"],
        "gaps": result["gaps"],
        "confirmed_count": result["confirmed_count"],
        "weak_count": result["weak_count"],
        "rejected_count": result["rejected_count"],
        "unverified_count": result["unverified_count"],
        "report": result["report"],
    }
    return (
        "You are answering questions about ONE specific completed research-gap "
        "analysis job. Answer ONLY using the JOB DATA below -- do not use outside "
        "or general knowledge about the subject matter, do not search the web, and "
        "do not reference any other analysis or job. If a question asks about "
        "something this data doesn't cover (including anything unrelated to this "
        "analysis entirely), say so clearly rather than filling in with outside "
        "knowledge.\n\n"
        "Gaps are numbered in the order they appear below (gap 1, gap 2, ...) -- "
        "use that numbering when the user refers to \"gap N\".\n\n"
        f"JOB DATA:\n{json.dumps(core, indent=2)}"
    )


def ask(job: dict, message: str, history: list[dict]) -> str:
    if job.get("status") != "done" or not job.get("result"):
        raise JobNotReadyError("This job hasn't finished yet -- nothing to chat about.")

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise MissingApiKeyError("Chat requires ANTHROPIC_API_KEY to be configured on the engine.")

    import anthropic

    client = anthropic.Anthropic(timeout=30.0, max_retries=0)
    system = _build_system_prompt(job)
    messages = [*history, {"role": "user", "content": message}]

    def attempt():
        return client.messages.create(
            model=os.getenv("CHAT_MODEL", "claude-haiku-4-5-20251001"),
            max_tokens=1000,
            system=system,
            messages=messages,
        )

    resp = call_with_retry(attempt, retries=2, backoff=2.0, label="job chat")
    if resp is None:
        raise ChatError("Could not get a response right now -- try again shortly.")
    return resp.content[0].text
