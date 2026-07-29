"""Keyword-overlap comparison: where does an extracted idea land relative
to a completed run's topics/gaps?

v1, deliberately: pure token-Jaccard overlap, no embeddings. A future
embedding-based version (still not touching pipeline.py/gaps.py) is a
smaller, separate follow-up -- this stays lexical on purpose.
"""
from __future__ import annotations

import re

_STOPWORDS = {
    "a", "an", "the", "of", "for", "and", "or", "in", "on", "to", "with",
    "using", "based", "via", "by", "from", "into", "is", "are", "this",
    "that", "system", "approach", "method", "model", "study", "analysis",
}


def _tokenize(phrases: list[str]) -> set[str]:
    tokens: set[str] = set()
    for phrase in phrases:
        for word in re.findall(r"[a-zA-Z]+", phrase.lower()):
            if word not in _STOPWORDS and len(word) > 2:
                tokens.add(word)
    return tokens


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def compute_landing(idea: dict, topics: list[dict], gaps: list[dict]) -> dict:
    # related_terms (broader/synonymous concepts, e.g. "headache" for a
    # migraine idea) widen recall for this comparison only -- core_topic,
    # the actual search query already used to run the pipeline, is untouched.
    idea_tokens = _tokenize(idea["keywords"] + idea.get("related_terms", []))

    topic_scores = [
        {"label": t["label"], "score": round(_jaccard(idea_tokens, _tokenize(t["keywords"])), 3)}
        for t in topics
    ]
    topic_scores.sort(key=lambda s: s["score"], reverse=True)

    gap_scores = [
        {
            "topic_a": g["topic_a"],
            "topic_b": g["topic_b"],
            "verdict": g["verdict"],
            "prominence": g["prominence"],
            "score": round(_jaccard(idea_tokens, _tokenize(g["keywords_a"] + g["keywords_b"])), 3),
        }
        for g in gaps
    ]
    gap_scores.sort(key=lambda s: s["score"], reverse=True)

    return {
        "idea": idea,
        "closest_topic": topic_scores[0] if topic_scores else None,
        "topic_scores": topic_scores,
        "closest_gap": gap_scores[0] if gap_scores else None,
        "gap_scores": gap_scores,
    }
