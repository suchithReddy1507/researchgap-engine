"""Verification/audit capture — sits alongside the pipeline, never inside it.

Every function here re-derives or re-fetches evidence for a claim the
pipeline already made (bridge_papers count, recent_papers_found count) so
that claim can be manually traced back to actual papers. Nothing here
changes what gaps.py or agent.py compute or decide.
"""
from __future__ import annotations

import numpy as np

from . import fetch
from .gaps import BRIDGE_SIM


def capture_bridge_evidence(papers: list[dict], embeddings, topics: dict, candidates: list[dict]) -> list[dict]:
    """Re-derives the bridge-paper mask for each candidate gap using the same
    embeddings, topic centroids, and BRIDGE_SIM threshold gaps.detect_gaps()
    used — same arrays, same constant, same condition, just not thrown away
    this time. Returns the actual bridging papers per gap."""
    emb = np.asarray(embeddings)
    label_to_topic = {t["label"]: t for t in topics.values()}

    evidence = []
    for c in candidates:
        ta = label_to_topic[c["topic_a"]]
        tb = label_to_topic[c["topic_b"]]
        sim_a = emb @ ta["centroid"]
        sim_b = emb @ tb["centroid"]
        mask = (sim_a > BRIDGE_SIM) & (sim_b > BRIDGE_SIM)
        idxs = np.where(mask)[0]

        bridging_papers = [
            {
                "title": papers[i].get("title", ""),
                "doi": papers[i].get("doi"),
                "year": papers[i].get("year"),
                "authors": papers[i].get("authors", []),
            }
            for i in idxs
        ]
        evidence.append(
            {
                "topic_a": c["topic_a"],
                "topic_b": c["topic_b"],
                "claimed_bridge_papers": c["bridge_papers"],
                "captured_bridge_papers": len(bridging_papers),
                "bridge_paper_evidence": bridging_papers,
            }
        )
    return evidence


def capture_validation_evidence(validated_gaps: list[dict], from_year: int) -> list[dict]:
    """For each validated gap, re-fetches the actual OpenAlex records behind
    its recent_papers_found claim, using the exact verification_queries
    agent.validate_gaps() already generated and stored on the gap — queries
    are reused as-is, never regenerated."""
    evidence = []
    for gap in validated_gaps:
        matches_per_query = {}
        for q in gap.get("verification_queries", []):
            matches_per_query[q] = fetch.fetch_openalex_matches(q, from_year)

        evidence.append(
            {
                "topic_a": gap["topic_a"],
                "topic_b": gap["topic_b"],
                "verification_queries": gap.get("verification_queries", []),
                "claimed_recent_papers_found": gap.get("recent_papers_found"),
                "matches_per_query": matches_per_query,
            }
        )
    return evidence
