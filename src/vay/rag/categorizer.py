"""
content_manager.py

CRUD layer on top of chroma_setup.py.

  CREATE  create(source)   -> admin gives a URL or a PDF path.
                              Auto-detected, converted to Markdown, saved to
                              disk, chunked, embedded, and stored in ChromaDB.
  READ    read(query)      -> semantic search over stored chunks.
          list_sources()   -> every distinct source currently in the DB.
  UPDATE  update(source)   -> deletes the source's old chunks, re-ingests it.
                              Use after the underlying page/PDF has changed.
  DELETE  delete(source)   -> removes every chunk belonging to a source.

CLI:
    python content_manager.py create "https://example.com/article"
    python content_manager.py create "./reports/document.pdf"
    python content_manager.py create "https://example.com" --labels billing plans network
    python content_manager.py read "how do I reset my sim" --n-results 5
    python content_manager.py update "https://example.com/article"
    python content_manager.py delete "https://example.com/article"
    python content_manager.py list
    python content_manager.py inspect "https://example.com/article"
    python content_manager.py categories

Domain-agnostic design: works on any knowledge base (telecom, medical, legal,
HR, finance, …) without editing source code. Category labels are either
auto-discovered from the content (default) or supplied by the caller via
--labels / the `category_labels` argument.
"""

from __future__ import annotations

import sys as _sys_utf8

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from vay.rag.tfidf import compute_tfidf_descriptions
from vay.rag.vector_store import get_embedding_function

DEFAULT_TOP_LABELS = 2
DEFAULT_CLUSTER_K = 8


if hasattr(_sys_utf8.stdout, "reconfigure"):
    try:
        _sys_utf8.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def _embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of strings via the project's shared SentenceTransformer."""
    ef = get_embedding_function()
    # ChromaDB's SentenceTransformerEmbeddingFunction is callable
    raw = ef(texts)
    arr = np.array(raw, dtype=np.float32)
    return normalize(arr, norm="l2")  # unit-length for cosine sim


def categorize_chunks_guided(
    chunks: list[str],
    labels: list[str],
    top_k: int = DEFAULT_TOP_LABELS,
) -> list[str]:
    """
    FIX 1A — Label-guided: for each chunk, return the `top_k` most similar
    label strings (cosine similarity in embedding space).

    Works on ANY domain: just pass domain-appropriate label names.
    Examples:
      telecom  → ["billing", "plans", "network", "sim", "complaints"]
      medical  → ["diagnosis", "treatment", "medication", "surgery"]
      legal    → ["contract", "liability", "compliance", "dispute"]

    Returns a list of comma-separated label strings, one per chunk.
    """
    if not chunks or not labels:
        return ["general"] * len(chunks)

    chunk_embs = _embed_texts(chunks)  # (n_chunks, d)
    label_embs = _embed_texts(labels)  # (n_labels, d)

    results = []
    for cemb in chunk_embs:
        sims = [_cosine_sim(cemb, lemb) for lemb in label_embs]
        # Sort by similarity descending, take top_k
        ranked = sorted(zip(labels, sims), key=lambda x: -x[1])
        selected = [lbl for lbl, sim in ranked[:top_k] if sim > 0.05]
        results.append(",".join(selected) if selected else "general")
    return results


def categorize_chunks_unsupervised(
    chunks: list[str],
    top_k: int = DEFAULT_TOP_LABELS,
    k_clusters: int = DEFAULT_CLUSTER_K,
) -> list[str]:
    """
    FIX 1B — Unsupervised: cluster chunk embeddings (KMeans), then auto-label
    each cluster using TF-IDF centroid keywords. No domain knowledge required.

    Returns a list of comma-separated auto-label strings, one per chunk.
    """
    if not chunks:
        return []

    n = len(chunks)
    # Can't have more clusters than chunks
    k = min(k_clusters, n)

    embs = _embed_texts(chunks)  # (n, d)

    if k <= 1:
        # Single cluster — use top TF-IDF words as the one label
        desc = compute_tfidf_descriptions(chunks, top_n=top_k)
        return [desc[0].replace(", ", ",")] * n

    # KMeans clustering on embeddings
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels_idx = km.fit_predict(embs)  # cluster index per chunk

    # Auto-label each cluster: concatenate all its chunks, run TF-IDF
    cluster_texts: dict[int, list[str]] = {i: [] for i in range(k)}
    for chunk, cid in zip(chunks, labels_idx):
        cluster_texts[cid].append(chunk)

    cluster_labels: dict[int, str] = {}
    for cid, ctexts in cluster_texts.items():
        merged = " ".join(ctexts)
        desc = compute_tfidf_descriptions([merged], top_n=top_k)[0]
        # Use first top_k terms as the label (comma-separated)
        terms = [t.strip() for t in desc.split(",") if t.strip()][:top_k]
        cluster_labels[cid] = ",".join(terms) if terms else f"topic_{cid}"

    return [cluster_labels[cid] for cid in labels_idx]


def categorize_chunks(
    chunks: list[str],
    category_labels: list[str] | None = None,
    top_k: int = DEFAULT_TOP_LABELS,
    k_clusters: int = DEFAULT_CLUSTER_K,
) -> list[str]:
    """
    Dispatcher: use label-guided mode if `category_labels` is provided,
    otherwise use unsupervised KMeans discovery.
    """
    if category_labels:
        return categorize_chunks_guided(chunks, category_labels, top_k)
    return categorize_chunks_unsupervised(chunks, top_k, k_clusters)


# ============================================================================
# FIX 5 + FIX 6 + FIX 7 — SENTENCE-AWARE, HEADING-PROPAGATING, SIZE-SAFE CHUNKING
# ============================================================================
