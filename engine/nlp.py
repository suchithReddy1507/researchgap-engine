"""Stage 3 — NLP core. Sentence-BERT embeddings + BERTopic clustering.

Input: papers fetched live in Stage 2.
Output: embeddings matrix, topic assignment per paper, topic metadata.
"""
from __future__ import annotations

import threading

import numpy as np

_model = None
_model_lock = threading.Lock()


def clean_papers(papers: list[dict], min_len: int = 200) -> list[dict]:
    """Drop papers with tiny/empty abstracts; strip whitespace noise."""
    cleaned = []
    for p in papers:
        abstract = " ".join(p["abstract"].split())
        if len(abstract) >= min_len:
            p = {**p, "abstract": abstract}
            cleaned.append(p)
    return cleaned


def embed_papers(papers: list[dict]):
    """384-dim Sentence-BERT vectors. Runs locally on CPU, no API.

    Model is loaded once and reused -- concurrent pipeline runs (each in
    their own background thread, see jobs.py) were previously each
    constructing their own SentenceTransformer instance, and racing through
    PyTorch's meta-device model materialization could corrupt global torch
    state badly enough to require a process restart. Loading once, and
    serializing every embed call behind the same lock, removes the race
    entirely -- this stage is a small fraction of a run's wall-clock time
    (dominated by network I/O), so serializing it costs nothing meaningful.
    """
    global _model
    from sentence_transformers import SentenceTransformer

    with _model_lock:
        if _model is None:
            _model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [f"{p['title']}. {p['abstract']}" for p in papers]
        return _model.encode(texts, show_progress_bar=True, normalize_embeddings=True)


def cluster_topics(papers: list[dict], embeddings, query: str = "") -> dict:
    """BERTopic (UMAP + HDBSCAN) on the precomputed embeddings.

    Returns {"assignments": [...], "topics": {topic_id: {...}}}.
    Topic -1 is BERTopic's outlier bucket and is excluded from topics.
    """
    from bertopic import BERTopic

    n = len(papers)
    # Small corpora need gentler clustering settings or HDBSCAN finds nothing.
    import re
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    stop = list(CountVectorizer(stop_words="english").get_stop_words())
    stop += [w.lower() for w in re.findall(r"[a-zA-Z]+", query)]

    # BERTopic's own default UMAP has no random_state, so the same
    # embeddings can produce a different topic split from run to run.
    # This is that exact default (every value matches BERTopic's internal
    # construction) with only random_state added for reproducibility.
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        low_memory=False,
        random_state=42,
    )

    model = BERTopic(
        umap_model=umap_model,
        vectorizer_model=CountVectorizer(stop_words=stop, ngram_range=(1, 2)),
        hdbscan_model=HDBSCAN(min_cluster_size=max(6, len(papers) // 40), min_samples=2),
        calculate_probabilities=False,
    )
    assignments, _ = model.fit_transform(
        [f"{p['title']}. {p['abstract']}" for p in papers], np.asarray(embeddings)
    )

    topics: dict[int, dict] = {}
    for tid in sorted(set(assignments)):
        if tid == -1:
            continue
        idxs = [i for i, a in enumerate(assignments) if a == tid]
        keywords = [w for w, _ in model.get_topic(tid)[:6]]
        years = [papers[i]["year"] for i in idxs if papers[i]["year"]]
        topics[tid] = {
            "keywords": keywords,
            "label": " / ".join(keywords[:3]),
            "size": len(idxs),
            "paper_idxs": idxs,
            "centroid": np.asarray(embeddings)[idxs].mean(axis=0),
            "mean_year": round(float(np.mean(years)), 1) if years else None,
        }
    return {"assignments": assignments, "topics": topics}
