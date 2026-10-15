"""Stage 2 — Paper collection. Live calls to Semantic Scholar and OpenAlex.

No static data anywhere in this module. Every paper comes from a real
HTTP request made at runtime.

Every external call is bounded: a hard per-attempt timeout plus a capped
number of retries (see engine.retry.call_with_retry), so a slow or
rate-limited upstream can delay a run but can never hang it forever.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import threading
import time

import httpx

from .logconfig import logger
from .retry import call_with_retry

S2_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
OPENALEX_URL = "https://api.openalex.org/works"
MAILTO = os.getenv("OPENALEX_MAILTO", "student@example.com")
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY")
CACHE_DIR = pathlib.Path("cache")
VALIDATION_CACHE_PATH = CACHE_DIR / "validation_cache.json"

REQUEST_TIMEOUT = 30.0
MAX_RETRIES = 2
RETRY_BACKOFF = 2.0
# S2's introductory rate limit is 1 request/sec even with an API key (per
# their key-request form) — throttle every call, keyed or not.
S2_MIN_INTERVAL = 3.0

_validation_cache: dict[str, int] | None = None
_validation_cache_lock = threading.Lock()


def _slugify(topic: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")
    return slug or "topic"


def _s2_headers() -> dict:
    key = os.getenv("S2_API_KEY")
    return {"x-api-key": key} if key else {}


def _openalex_params(extra: dict) -> dict:
    """mailto is always sent (polite pool); api_key is appended when set
    (falls back to unauthenticated + mailto-only if OPENALEX_API_KEY is
    missing, so this degrades gracefully rather than erroring)."""
    params = {**extra, "mailto": MAILTO}
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY
    return params


def _get(client: httpx.Client, url: str, params: dict, *, headers: dict | None = None, label: str) -> httpx.Response | None:
    """GET with a hard timeout + bounded retries. 429s count as a retryable
    failure (with backoff) rather than looping forever. Returns None — never
    raises — if every attempt fails, so callers can skip gracefully."""

    def attempt() -> httpx.Response:
        resp = client.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 429:
            raise httpx.HTTPStatusError("429 rate limited", request=resp.request, response=resp)
        resp.raise_for_status()
        return resp

    return call_with_retry(attempt, retries=MAX_RETRIES, backoff=RETRY_BACKOFF, label=label)


def search_semantic_scholar(query: str, limit: int = 100) -> list[dict]:
    """Fetch papers from Semantic Scholar (max 100 per request, paginated)."""
    papers: list[dict] = []
    offset = 0
    with httpx.Client() as client:
        while len(papers) < limit:
            batch = min(100, limit - len(papers))
            resp = _get(
                client,
                S2_URL,
                params={
                    "query": query,
                    "offset": offset,
                    "limit": batch,
                    "fields": "title,abstract,year,citationCount,externalIds,fieldsOfStudy,authors",
                },
                headers=_s2_headers(),
                label=f"S2 search '{query}' offset={offset}",
            )
            if resp is None:
                logger.warning(f"S2 search giving up early — returning {len(papers)} papers so far")
                break
            data = resp.json().get("data", [])
            if not data:
                break
            for p in data:
                if p.get("abstract"):
                    papers.append(
                        {
                            "title": p.get("title", ""),
                            "abstract": p["abstract"],
                            "year": p.get("year"),
                            "citations": p.get("citationCount", 0),
                            "doi": (p.get("externalIds") or {}).get("DOI"),
                            "authors": [a["name"] for a in (p.get("authors") or []) if a.get("name")],
                            "source": "semantic_scholar",
                        }
                    )
            offset += batch
            time.sleep(S2_MIN_INTERVAL)  # 1 req/sec, keyed or not
    return papers


def _rebuild_abstract(inverted: dict | None) -> str:
    """OpenAlex stores abstracts as an inverted index — rebuild the text."""
    if not inverted:
        return ""
    positions: dict[int, str] = {}
    for word, idxs in inverted.items():
        for i in idxs:
            positions[i] = word
    return " ".join(positions[i] for i in sorted(positions))


def _openalex_authors(work: dict) -> list[str]:
    """OpenAlex always includes authorships in a work's default response --
    no extra param needed, just extracting a field that was being
    discarded."""
    return [
        a["author"]["display_name"]
        for a in (work.get("authorships") or [])
        if a.get("author", {}).get("display_name")
    ]


def search_openalex(query: str, limit: int = 100, from_year: int | None = None) -> list[dict]:
    """Fetch papers from OpenAlex. Used both for collection and as the
    independent second source for agentic validation (Stage 5)."""
    papers: list[dict] = []
    page = 1
    filters = ["has_abstract:true"]
    if from_year:
        filters.append(f"from_publication_date:{from_year}-01-01")
    with httpx.Client() as client:
        while len(papers) < limit:
            resp = _get(
                client,
                OPENALEX_URL,
                params=_openalex_params(
                    {
                        "search": query,
                        "per-page": min(100, limit - len(papers)),
                        "page": page,
                        "filter": ",".join(filters),
                    }
                ),
                label=f"OpenAlex search '{query}' page={page}",
            )
            if resp is None:
                logger.warning(f"OpenAlex search giving up early — returning {len(papers)} papers so far")
                break
            results = resp.json().get("results", [])
            if not results:
                break
            for w in results:
                abstract = _rebuild_abstract(w.get("abstract_inverted_index"))
                if abstract:
                    papers.append(
                        {
                            "title": w.get("title") or "",
                            "abstract": abstract,
                            "year": w.get("publication_year"),
                            "citations": w.get("cited_by_count", 0),
                            "doi": (w.get("doi") or "").replace("https://doi.org/", "") or None,
                            "authors": _openalex_authors(w),
                            "source": "openalex",
                        }
                    )
            page += 1
    return papers


def _load_validation_cache() -> dict[str, int]:
    global _validation_cache
    if _validation_cache is None:
        if VALIDATION_CACHE_PATH.exists():
            try:
                _validation_cache = json.loads(VALIDATION_CACHE_PATH.read_text())
            except json.JSONDecodeError:
                _validation_cache = {}
        else:
            _validation_cache = {}
    return _validation_cache


def count_openalex(query: str, from_year: int, use_cache: bool = True) -> int | None:
    """Cheap count-only query — the validation agent uses this to check
    whether a candidate gap is already being actively researched.

    Cached to cache/validation_cache.json so the same (query, from_year)
    pair never hits the live API twice, across runs. Returns None (not 0!)
    if the request could not be completed after retries — callers must
    treat that as 'unknown', not 'zero hits', or a rate-limited network
    would silently look like a confirmed gap.

    use_cache=False bypasses the cache entirely -- no read, no write. Used
    by the standalone re-verification script (needs a genuinely fresh count,
    not a replay of the original run's cached answer) and by the agent for
    fallback-generated queries (low-precision word-pair queries must never
    be read from or written into the shared cache as if they were a real
    validated result).
    """
    key = f"{query}::{from_year}"
    if use_cache:
        with _validation_cache_lock:
            cache = _load_validation_cache()
            if key in cache:
                return cache[key]

    filters = [
        f"from_publication_date:{from_year}-01-01",
        f"title_and_abstract.search:{query}",
    ]
    with httpx.Client() as client:
        resp = _get(
            client,
            OPENALEX_URL,
            params=_openalex_params({"per-page": 1, "filter": ",".join(filters)}),
            label=f"OpenAlex count '{query}'",
        )
    if resp is None:
        return None

    count = resp.json().get("meta", {}).get("count", 0)
    if use_cache:
        with _validation_cache_lock:
            cache = _load_validation_cache()
            cache[key] = count
            CACHE_DIR.mkdir(exist_ok=True)
            VALIDATION_CACHE_PATH.write_text(json.dumps(cache, indent=2))
    return count


def fetch_openalex_matches(query: str, from_year: int, limit: int = 25) -> list[dict]:
    """Like count_openalex(), but returns the actual matching paper records
    (title, doi, year, authors) instead of just a count — used by the
    verification log to make a 'recent hits' claim traceable to real
    papers, not just a number. Always live (no cache), since it's
    evidence for one specific
    run rather than a reusable count."""
    filters = [
        f"from_publication_date:{from_year}-01-01",
        f"title_and_abstract.search:{query}",
    ]
    with httpx.Client() as client:
        resp = _get(
            client,
            OPENALEX_URL,
            params=_openalex_params({"per-page": limit, "filter": ",".join(filters)}),
            label=f"OpenAlex matches '{query}'",
        )
    if resp is None:
        return []

    matches = []
    for w in resp.json().get("results", []):
        matches.append(
            {
                "title": w.get("title") or "",
                "doi": (w.get("doi") or "").replace("https://doi.org/", "") or None,
                "year": w.get("publication_year"),
                "authors": _openalex_authors(w),
            }
        )
    return matches


def collect_papers(topic: str, target: int = 300, use_cache: bool = True,
                   stats: dict | None = None) -> list[dict]:
    """Primary collection: Semantic Scholar first, topped up with OpenAlex,
    deduplicated by DOI then by normalized title.

    Results are cached to disk at cache/{slug}_{target}.json. Pass
    use_cache=False to bypass a stale cache and re-fetch from the APIs
    (the cache file is refreshed with the new results either way)."""
    cache_path = CACHE_DIR / f"{_slugify(topic)}_{target}.json"
    if use_cache and cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if stats is not None:
            # A cache hit cannot report per-source or dedup counts: only the
            # merged survivors were persisted. Say so rather than inventing.
            stats.update(from_cache=True, s2_papers=None, openalex_papers=None,
                         raw_total=None, duplicates_removed=None, merged=len(cached))
        return cached

    s2 = search_semantic_scholar(topic, limit=target)
    oa = []
    if len(s2) < target:
        oa = search_openalex(topic, limit=target - len(s2))
    papers = s2 + oa

    seen_doi: set[str] = set()
    seen_title: set[str] = set()
    unique: list[dict] = []
    for p in papers:
        key_t = p["title"].lower().strip()
        if p["doi"] and p["doi"] in seen_doi:
            continue
        if key_t and key_t in seen_title:
            continue
        if p["doi"]:
            seen_doi.add(p["doi"])
        seen_title.add(key_t)
        unique.append(p)

    if stats is not None:
        stats.update(from_cache=False, s2_papers=len(s2), openalex_papers=len(oa),
                     raw_total=len(papers), duplicates_removed=len(papers) - len(unique),
                     merged=len(unique))
    logger.info(
        f"      collection: semantic_scholar={len(s2)} openalex={len(oa)} "
        f"raw={len(papers)} duplicates_removed={len(papers) - len(unique)} merged={len(unique)}"
    )
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path.write_text(json.dumps(unique, indent=2))
    return unique
