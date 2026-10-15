"""Shared pipeline runner — the single place stages 1-6 are wired together.

Both the CLI (main.py) and the API (api.py) call run_pipeline() so there is
one implementation of the engine's control flow.
"""
from __future__ import annotations

import time
from typing import Callable

from . import agent, fetch, gaps, nlp, stats, synth, verification
from .logconfig import logger

OnStage = Callable[[int, str], None]


class PipelineError(Exception):
    """Raised when the pipeline cannot produce a meaningful result."""


def _layout_topics(topics: dict) -> dict:
    """2D coords for the knowledge map: UMAP on topic centroids, normalized
    to 0..1. With only one topic there's nothing to project — center it."""
    import numpy as np

    ids = list(topics.keys())
    if len(ids) == 1:
        return {ids[0]: (0.5, 0.5)}

    import umap

    centroids = np.stack([topics[i]["centroid"] for i in ids])
    n_neighbors = max(2, min(15, len(ids) - 1))
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42, metric="cosine")
    coords = reducer.fit_transform(centroids)
    mins = coords.min(axis=0)
    spans = coords.max(axis=0) - mins
    spans = np.where(spans == 0, 1, spans)
    norm = (coords - mins) / spans
    return {tid: (float(norm[i, 0]), float(norm[i, 1])) for i, tid in enumerate(ids)}


def _stage(n: int, name: str, on_stage: OnStage | None = None) -> float:
    logger.info(f"[{n}/6] {name}")
    if on_stage:
        try:
            on_stage(n, name)
        except Exception:
            logger.exception("on_stage callback failed — continuing the run regardless")
    return time.monotonic()


def _done(start: float, timings: dict | None = None, key: str | None = None) -> None:
    elapsed = time.monotonic() - start
    if timings is not None and key:
        timings[key] = round(elapsed, 2)
    logger.info(f"      done in {elapsed:.1f}s")


def run_pipeline(
    topic: str,
    n_papers: int = 300,
    use_cache: bool = True,
    on_stage: OnStage | None = None,
) -> dict:
    run_start = time.monotonic()
    logger.info(f"=== ResearchGap AI engine — topic: '{topic}' ({n_papers} papers) ===")
    # Provenance for the paper's Table II: the only seed in the pipeline is
    # UMAP's random_state (nlp.py / _layout_topics); HDBSCAN is deterministic
    # given its input. Retrieval date is when the APIs were actually hit.
    retrieval_date = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    logger.info(f"      provenance: umap_random_state=42 retrieval_date={retrieval_date}")
    timings: dict = {}
    collection_stats: dict = {}

    t0 = _stage(1, "Input accepted.", on_stage)
    _done(t0)

    t0 = _stage(2, f"Fetching up to {n_papers} papers (Semantic Scholar + OpenAlex)...", on_stage)
    papers = fetch.collect_papers(topic, target=n_papers, use_cache=use_cache, stats=collection_stats)
    logger.info(f"      -> {len(papers)} unique papers with abstracts")
    _done(t0, timings, "collection")
    if len(papers) < 30:
        raise PipelineError("Too few papers for meaningful clustering. Try a broader topic.")

    t0 = _stage(3, "Cleaning, embedding (Sentence-BERT), clustering (BERTopic)...", on_stage)
    # Split the previously-merged stage-3 timings so Table XI can report
    # preprocessing, embedding and clustering separately. Same calls, same order.
    _t = time.monotonic()
    n_before_clean = len(papers)
    papers = nlp.clean_papers(papers)
    short_removed = n_before_clean - len(papers)
    _done(_t, timings, "preprocessing")

    _t = time.monotonic()
    embeddings = nlp.embed_papers(papers)
    _done(_t, timings, "embedding")

    _t = time.monotonic()
    result = nlp.cluster_topics(papers, embeddings, query=topic)
    _done(_t, timings, "clustering")
    topics = result["topics"]
    outliers = int(sum(1 for a in result["assignments"] if a == -1))
    logger.info(
        f"      corpus: merged={n_before_clean} short_or_missing_abstract_removed={short_removed} "
        f"final={len(papers)} topics={len(topics)} outliers={outliers}"
    )
    logger.info(f"      -> {len(topics)} topics found:")
    for t in topics.values():
        logger.info(f"         - {t['label']} ({t['size']} papers)")
    _done(t0)
    if len(topics) < 4:
        # 4, not 3: with exactly 3 topics, the knowledge-map layout step
        # below (_layout_topics) hits UMAP's spectral-init eigensolver at
        # its own edge case (needs N > n_components + 1 = 3 points) and
        # crashes with a raw scipy TypeError instead of failing gracefully
        # -- confirmed by direct reproduction, not a guess. Show the real
        # numbers rather than a generic message: papers were fine, they
        # just didn't separate into enough distinct topics.
        topic_summary = "; ".join(f"{t['label']} ({t['size']} papers)" for t in topics.values()) or "no coherent topics"
        raise PipelineError(
            f"Not enough distinct topics found for gap analysis on '{topic}' — "
            f"{len(papers)} papers were analyzed, but they only separated into "
            f"{len(topics)} topic cluster(s) ({topic_summary}). At least 4 distinct "
            f"topics are needed for reliable cross-topic gap analysis. Try a "
            f"broader or related keyword."
        )

    t0 = _stage(4, "Detecting gaps (cross-topic sparsity)...", on_stage)
    candidates = gaps.detect_gaps(papers, embeddings, topics)
    logger.info(f"      -> {len(candidates)} candidate gaps")
    bridge_evidence = verification.capture_bridge_evidence(papers, embeddings, topics, candidates)
    _done(t0, timings, "gap_detection")

    t0 = _stage(5, "Agentic validation against OpenAlex (independent source)...", on_stage)
    current_year = max((p["year"] or 0) for p in papers)
    validated = agent.validate_gaps(candidates, current_year)
    confirmed = [g for g in validated if g["verdict"] == "CONFIRMED"]
    weak = [g for g in validated if g["verdict"] == "WEAK"]
    rejected = [g for g in validated if g["verdict"] == "REJECTED"]
    unverified = [g for g in validated if g["verdict"] == "UNVERIFIED"]
    logger.info(
        f"      -> {len(confirmed)} confirmed, {len(weak)} weak, "
        f"{len(rejected)} rejected, {len(unverified)} unverified"
    )
    from_year = current_year - agent.RECENT_YEARS
    validation_evidence = verification.capture_validation_evidence(validated, from_year)
    _done(t0, timings, "agent_validation")

    t0 = _stage(6, "Synthesizing report...", on_stage)
    report, report_source = synth.synthesize(topic, topics, validated)
    _done(t0, timings, "synthesis")

    coords = _layout_topics(topics)
    topics_summary = [
        {
            "label": t["label"],
            "keywords": t["keywords"],
            "size": t["size"],
            "mean_year": t["mean_year"],
            "x": coords[tid][0],
            "y": coords[tid][1],
        }
        for tid, t in topics.items()
    ]

    stats.record_run(len(papers), len(validated), len(confirmed))

    logger.info(f"=== done in {time.monotonic() - run_start:.1f}s total ===")

    return {
        "topic": topic,
        "papers_count": len(papers),
        "topics": topics_summary,
        "gaps": validated,
        "confirmed_count": len(confirmed),
        "weak_count": len(weak),
        "rejected_count": len(rejected),
        "unverified_count": len(unverified),
        "report": report,
        "report_source": report_source,
        # Measurement only -- nothing here feeds back into the algorithm.
        "run_stats": {
            "query_string": topic,
            "papers_requested": n_papers,
            "retrieval_date": retrieval_date,
            "umap_random_state": 42,
            "collection": collection_stats,
            "merged_papers": n_before_clean,
            "short_or_missing_abstract_removed": short_removed,
            "final_corpus": len(papers),
            "topics": len(topics),
            "outliers": outliers,
            "candidates": len(candidates),
            "timings_sec": {**timings, "total": round(time.monotonic() - run_start, 2)},
        },
        "verification_evidence": {
            "bridge_evidence": bridge_evidence,
            "validation_evidence": validation_evidence,
            "current_year": current_year,
            "from_year": from_year,
        },
    }
