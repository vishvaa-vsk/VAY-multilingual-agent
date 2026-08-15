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

import logging
import re


# ---------------------------------------------------------------------------
# Force UTF-8 on Windows consoles (cp1252 chokes on Unicode in rich content)
# ---------------------------------------------------------------------------
import sys as _sys_utf8
import warnings

try:
    import nltk
except ImportError:
    nltk = None

from sklearn.feature_extraction.text import TfidfVectorizer

if hasattr(_sys_utf8.stdout, "reconfigure"):
    try:
        _sys_utf8.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Silence noisy third-party warnings
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("langdetect").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# One-time NLTK data download (silent after first run)
# ---------------------------------------------------------------------------
def _ensure_nltk():
    if not nltk:
        return
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except Exception:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass



_ensure_nltk()

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MD_OUTPUT_DIR = "converted_md"
DEFAULT_CHUNK_SIZE = 500  # target chars per chunk (hard upper bound)
DEFAULT_CHUNK_OVERLAP = 100  # chars of sentence-boundary overlap between chunks
DEFAULT_TOP_LABELS = 2  # how many category labels to store per chunk
DEFAULT_CLUSTER_K = 5  # KMeans clusters for unsupervised topic discovery

# ---------------------------------------------------------------------------
# STOPWORDS (used by TF-IDF description)
# ---------------------------------------------------------------------------
STOPWORDS: set[str] = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "at",
    "by",
    "from",
    "up",
    "down",
    "as",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "into",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "can",
    "will",
    "just",
    "should",
    "now",
    "you",
    "your",
    "yours",
    "he",
    "she",
    "him",
    "her",
    "his",
    "hers",
    "they",
    "them",
    "their",
    "i",
    "me",
    "my",
    "we",
    "us",
    "our",
    "do",
    "does",
    "did",
    "doing",
    "have",
    "has",
    "had",
    "having",
    "which",
    "who",
    "whom",
    "what",
    "page",
    "section",
    "table",
    "contents",
    "edit",
    "also",
    "would",
    "could",
    "may",
    "might",
    "shall",
    "let",
    "get",
    "use",
    "used",
}


# ============================================================================
# FIX 4 — TF-IDF BATCH DESCRIPTION
# (replaces raw TF that made common words dominate every chunk's description)
# ============================================================================


def compute_tfidf_descriptions(chunks: list[str], top_n: int = 10) -> list[str]:
    """
    Compute a TF-IDF description for each chunk *relative to all chunks in
    this ingestion batch*. Words distinctive to a chunk score high; words
    that appear in every chunk (like brand names or headers) score low.

    Returns a list of comma-separated keyword strings, one per chunk.
    Falls back gracefully if there's only one chunk (TF only).
    """
    if not chunks:
        return []

    # Preprocess: lowercase, strip markdown, keep alphabetic tokens ≥3 chars
    def _clean(text: str) -> str:
        text = re.sub(r"[#*`_\[\]()>|~]", " ", text)
        words = re.findall(r"\b[a-zA-Z][a-zA-Z\-]{2,}\b", text.lower())
        return " ".join(w for w in words if w not in STOPWORDS)

    cleaned = [_clean(c) for c in chunks]

    if len(chunks) == 1:
        # Can't compute IDF with a single document — fall back to TF ranking
        counts: dict[str, int] = {}
        for w in cleaned[0].split():
            counts[w] = counts.get(w, 0) + 1
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:top_n]
        return [", ".join(w for w, _ in top)]

    try:
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),  # unigrams + bigrams for richer descriptions
            sublinear_tf=True,  # log(1+tf) dampens very frequent terms
        )
        matrix = vectorizer.fit_transform(cleaned)  # (n_chunks, n_features)
        feature_names = vectorizer.get_feature_names_out()
        descriptions = []
        for row in matrix:
            scores = zip(feature_names, row.toarray()[0])
            top = sorted(scores, key=lambda kv: -kv[1])[:top_n]
            descriptions.append(", ".join(term for term, score in top if score > 0))
        return descriptions
    except ValueError:
        # Empty vocabulary after cleaning (e.g. all numeric/symbol chunks)
        return [""] * len(chunks)


# ============================================================================
# FIX 1 — DOMAIN-AGNOSTIC CATEGORY TAGGING
# Two modes:
#   A) Label-guided  — caller supplies label strings; each chunk is scored by
#      cosine similarity of its embedding vs the label embedding.
#   B) Unsupervised  — KMeans clusters chunk embeddings, then auto-labels each
#      cluster with its top TF-IDF terms. No domain knowledge needed.
# ============================================================================


def _extract_heading(block: str) -> str | None:
    """Return the heading text if the block starts with a Markdown heading."""
    m = re.match(r"^(#{1,6})\s+(.+)", block.strip())
    return m.group(2).strip() if m else None
