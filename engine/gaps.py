"""Stage 4 — Gap detection. Pure computation on the live knowledge map.

Method: cross-topic sparsity. For every pair of topics (A, B), count
"bridge papers" — papers semantically close to BOTH topic centroids.
Two healthy topics with almost no bridge papers = a candidate research gap
at their intersection.

Score design (this is the project's original contribution — tune it, own it):
    sparsity   = 1 / (1 + bridges)          # fewer bridges -> bigger gap
    mass       = sqrt(size_A * size_B)      # both topics must be substantial
    recency    = boost if both topics are actively published in
    prominence = sparsity * mass * recency  # the "gap prominence score"
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

BRIDGE_SIM = 0.45  # cosine sim threshold to count a paper as bridging a topic


def detect_gaps(papers: list[dict], embeddings, topics: dict, top_n: int = 8) -> list[dict]:
    emb = np.asarray(embeddings)  # normalized -> dot product == cosine sim
    current_year = max((p["year"] or 0) for p in papers)

    candidates = []
    for a, b in combinations(topics.keys(), 2):
        ta, tb = topics[a], topics[b]
        sim_a = emb @ ta["centroid"]
        sim_b = emb @ tb["centroid"]
        bridges = int(np.sum((sim_a > BRIDGE_SIM) & (sim_b > BRIDGE_SIM)))

        sparsity = 1.0 / (1.0 + bridges)
        mass = float(np.sqrt(ta["size"] * tb["size"]))
        years = [y for y in (ta["mean_year"], tb["mean_year"]) if y]
        recency = 1.0
        if years and current_year:
            # topics whose average paper is recent get up to a 1.5x boost
            recency = 1.0 + 0.5 * max(0.0, 1.0 - (current_year - min(years)) / 10.0)

        prominence = sparsity * mass * recency
        candidates.append(
            {
                "topic_a": ta["label"],
                "topic_b": tb["label"],
                "keywords_a": ta["keywords"],
                "keywords_b": tb["keywords"],
                "bridge_papers": bridges,
                "size_a": ta["size"],
                "size_b": tb["size"],
                "prominence": round(prominence, 2),
            }
        )

    candidates.sort(key=lambda c: c["prominence"], reverse=True)
    # Prefer truly sparse pairs, but never return nothing — if the domain is
    # tightly interconnected, pass the least-bridged pairs to the agent and
    # let validation (stage 5) make the kill decision.
    sparse = [c for c in candidates if c["bridge_papers"] <= 3]
    return (sparse or candidates)[:top_n]
