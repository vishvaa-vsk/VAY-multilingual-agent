"""
hybrid.py

Real hybrid retrieval: BM25 (sparse keyword) + ChromaDB (dense vector) with
reciprocal-rank-inspired score fusion, top-k reranked.

Why this file exists: `rag/bm25.py`'s BM25SearchEngine was a stub (it returned
whatever documents were passed in, in input order, with a fabricated
`0.80 - i*0.05` score -- no term matching at all), and nothing ever called it
from the live retrieval path (`manager_read.read()` only ever did a plain
`collection.query()` against ChromaDB). So despite `project_context.md` and
`context-rag-tts.md` both documenting "hybrid BM25 + vector search", the system
was actually running **vector-only retrieval** the whole time. This measurably
hurt precision on queries with exact numeric/keyword anchors (prices, data
amounts, plan names) that a small MiniLM embedding doesn't reliably separate
(e.g. "2GB daily plan for 300 rupees" scored the WRONG plan's chunk higher than
the exact-match plan -- see rag-tts-evaluuation.md for the measured before/after).

This module builds a real `rank_bm25.BM25Okapi` index per collection (cached,
invalidated automatically when the collection's chunk count changes), scores
the query against every chunk, and fuses that with ChromaDB's cosine similarity
via a weighted sum after min-max normalizing each signal over the candidate
pool. The fused score is converted back into a "distance" (`1 - score`) so
existing callers (`rag/retriever.py`'s `sim = 1 - dist`) don't need to change.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

# Weight given to the vector (dense) signal vs. the BM25 (sparse) signal when
# fusing scores. 0.5/0.5 is a reasonable default for short KB chunks where
# exact terms (prices, plan names, data amounts) matter as much as semantics.
VECTOR_WEIGHT = 0.5
BM25_WEIGHT = 0.5

_TOKEN_RE = re.compile(r"[a-zA-Z0-9஀-௿ऀ-ॿ]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase alnum tokenizer that also keeps Tamil/Devanagari codepoints
    intact (KB content is English, but queries can carry transliterated terms
    in normalized_query -- keeping the ranges doesn't hurt and future-proofs
    this for non-English KB content)."""
    return _TOKEN_RE.findall(text.lower())


@dataclass
class _CachedIndex:
    count: int
    ids: list[str]
    documents: list[str]
    metadatas: list[dict]
    bm25: BM25Okapi


_index_cache: dict[str, _CachedIndex] = {}


def _get_index(collection) -> _CachedIndex | None:
    """Return a cached BM25 index for this collection, rebuilding it if the
    collection's chunk count has changed since the index was last built (a
    cheap, good-enough invalidation signal for a hackathon-scale KB that's
    rebuilt via explicit admin scripts, not mutated per-request)."""
    name = collection.name
    count = collection.count()
    if count == 0:
        return None

    cached = _index_cache.get(name)
    if cached is not None and cached.count == count:
        return cached

    data = collection.get()  # all chunks: {"ids", "documents", "metadatas"}
    ids = data.get("ids") or []
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    if not documents:
        return None

    tokenized_corpus = [_tokenize(d) for d in documents]
    bm25 = BM25Okapi(tokenized_corpus)

    cached = _CachedIndex(count=count, ids=ids, documents=documents, metadatas=metadatas, bm25=bm25)
    _index_cache[name] = cached
    return cached


def invalidate_cache(collection_name: str | None = None) -> None:
    """Drop cached BM25 index(es) -- call after a bulk re-ingest so the very
    next query doesn't serve a stale index (count-based auto-invalidation
    already covers this in practice, but this makes intent explicit e.g. from
    scripts/manage_kb.py --rebuild)."""
    if collection_name is None:
        _index_cache.clear()
    else:
        _index_cache.pop(collection_name, None)


def _minmax_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def hybrid_query(collection, query: str, n_results: int, where: dict | None = None) -> dict:
    """Hybrid BM25 + vector query against a single ChromaDB collection.

    Returns the same shape ChromaDB's `.query()` returns
    (`{"documents": [[...]], "metadatas": [[...]], "distances": [[...]]}`)
    so every existing caller (rag/retriever.py, rag/manager_read.py) works
    unchanged -- `distance = 1 - fused_score`.
    """
    total = collection.count()
    if total == 0:
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    # --- Dense (vector) candidates: cast a wide net so BM25-only strong
    # matches that the embedding ranked poorly still have a chance to surface
    # once fused, then we truncate to n_results at the end. ---
    pool_n = min(max(n_results * 4, 10), total)
    vec_results = collection.query(query_texts=[query], n_results=pool_n, where=where)
    vec_docs = vec_results.get("documents", [[]])[0]
    vec_metas = vec_results.get("metadatas", [[]])[0]
    vec_dists = vec_results.get("distances", [[]])[0]
    vec_ids = vec_results.get("ids", [[]])[0] if vec_results.get("ids") else []

    candidates: dict[str, dict] = {}
    for i, (doc, meta, dist) in enumerate(zip(vec_docs, vec_metas, vec_dists)):
        cid = vec_ids[i] if i < len(vec_ids) else meta.get("content_hash", doc[:40])
        candidates[cid] = {"doc": doc, "meta": meta, "vector_sim": 1.0 - dist}

    # --- Sparse (BM25) signal, scored across the WHOLE collection (cheap at
    # this KB scale -- tens to low hundreds of chunks per collection) so a
    # strong exact-keyword match that the vector search missed entirely can
    # still enter the candidate pool. ---
    index = _get_index(collection)
    if index is not None:
        query_tokens = _tokenize(query)
        bm25_scores = index.bm25.get_scores(query_tokens) if query_tokens else []
        if len(bm25_scores):
            # Respect the same `where` filter as the vector leg, if any (post-filter
            # since BM25Okapi has no native metadata filtering).
            top_bm25_idx = sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])[:pool_n]
            for i in top_bm25_idx:
                if bm25_scores[i] <= 0:
                    continue
                meta = index.metadatas[i]
                if where and not _matches_where(meta, where):
                    continue
                cid = index.ids[i]
                if cid not in candidates:
                    candidates[cid] = {"doc": index.documents[i], "meta": meta, "vector_sim": 0.0}
                candidates[cid]["bm25_raw"] = bm25_scores[i]

    if not candidates:
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    # Fill in missing bm25_raw for vector-only candidates (score 0 relative to the batch)
    for c in candidates.values():
        c.setdefault("bm25_raw", 0.0)

    ids = list(candidates.keys())
    vec_sims = [candidates[i]["vector_sim"] for i in ids]
    bm25_raws = [candidates[i]["bm25_raw"] for i in ids]
    bm25_norms = _minmax_normalize(bm25_raws)

    fused = [
        VECTOR_WEIGHT * v + BM25_WEIGHT * b for v, b in zip(vec_sims, bm25_norms)
    ]

    ranked = sorted(zip(ids, fused), key=lambda pair: -pair[1])[:n_results]

    out_docs = [candidates[i]["doc"] for i, _ in ranked]
    out_metas = [candidates[i]["meta"] for i, _ in ranked]
    out_dists = [1.0 - score for _, score in ranked]

    return {"documents": [out_docs], "metadatas": [out_metas], "distances": [out_dists]}


def _matches_where(meta: dict, where: dict) -> bool:
    """Minimal support for the $eq / $and shapes manager_read.read() builds --
    enough to keep source_filter/language_filter working under hybrid search."""
    if "$and" in where:
        return all(_matches_where(meta, cond) for cond in where["$and"])
    for key, cond in where.items():
        if isinstance(cond, dict) and "$eq" in cond:
            if meta.get(key) != cond["$eq"]:
                return False
        elif meta.get(key) != cond:
            return False
    return True
