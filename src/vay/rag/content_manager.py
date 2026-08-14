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

# ---------------------------------------------------------------------------
# Dynamic import for 'chroma_setup (1).py' — the (1) suffix prevents normal
# `import chroma_setup` from working, so we load it explicitly by file path
# and register it under the canonical name so downstream `from chroma_setup
# import …` statements work without modification.
# ---------------------------------------------------------------------------
import importlib.util as _ilu
import sys as _sys
from pathlib import Path as _Path

_chroma_setup_path = _Path(__file__).parent / "chroma_setup (1).py"
if not _chroma_setup_path.exists():
    _chroma_setup_path = _Path(__file__).parent / "chroma_setup.py"
_spec = _ilu.spec_from_file_location("chroma_setup", str(_chroma_setup_path))
_mod = _ilu.module_from_spec(_spec)
_sys.modules.setdefault("chroma_setup", _mod)
_spec.loader.exec_module(_mod)

import argparse
import hashlib
import logging
import re

# ---------------------------------------------------------------------------
# Force UTF-8 on Windows consoles (cp1252 chokes on Unicode in rich content)
# ---------------------------------------------------------------------------
import sys as _sys_utf8
import warnings
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import html2text
import nltk
import numpy as np
import requests
from bs4 import BeautifulSoup
from chroma_setup import COLLECTION_NAME as _DEFAULT_COLLECTION
from chroma_setup import get_collection, get_embedding_function
from langdetect import detect as _langdetect
from langdetect.lang_detect_exception import LangDetectException
from pypdf import PdfReader
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

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
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

_ensure_nltk()

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MD_OUTPUT_DIR = "converted_md"
DEFAULT_CHUNK_SIZE    = 500    # target chars per chunk (hard upper bound)
DEFAULT_CHUNK_OVERLAP = 100    # chars of sentence-boundary overlap between chunks
DEFAULT_TOP_LABELS    = 2      # how many category labels to store per chunk
DEFAULT_CLUSTER_K     = 5      # KMeans clusters for unsupervised topic discovery

# ---------------------------------------------------------------------------
# STOPWORDS (used by TF-IDF description)
# ---------------------------------------------------------------------------
STOPWORDS: set[str] = {
    "the", "a", "an", "and", "or", "but", "if", "is", "are", "was", "were",
    "be", "been", "being", "to", "of", "in", "on", "for", "with", "at",
    "by", "from", "up", "down", "as", "this", "that", "these", "those",
    "it", "its", "into", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "can", "will", "just", "should", "now", "you", "your", "yours",
    "he", "she", "him", "her", "his", "hers", "they", "them", "their",
    "i", "me", "my", "we", "us", "our", "do", "does", "did", "doing",
    "have", "has", "had", "having", "which", "who", "whom", "what",
    "page", "section", "table", "contents", "edit", "also", "would",
    "could", "may", "might", "shall", "let", "get", "use", "used",
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
            ngram_range=(1, 2),   # unigrams + bigrams for richer descriptions
            sublinear_tf=True,    # log(1+tf) dampens very frequent terms
        )
        matrix = vectorizer.fit_transform(cleaned)   # (n_chunks, n_features)
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

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def _embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of strings via the project's shared SentenceTransformer."""
    ef = get_embedding_function()
    # ChromaDB's SentenceTransformerEmbeddingFunction is callable
    raw = ef(texts)
    arr = np.array(raw, dtype=np.float32)
    return normalize(arr, norm="l2")   # unit-length for cosine sim


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

    chunk_embs = _embed_texts(chunks)           # (n_chunks, d)
    label_embs = _embed_texts(labels)           # (n_labels, d)

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

    embs = _embed_texts(chunks)   # (n, d)

    if k <= 1:
        # Single cluster — use top TF-IDF words as the one label
        desc = compute_tfidf_descriptions(chunks, top_n=top_k)
        return [desc[0].replace(", ", ",")] * n

    # KMeans clustering on embeddings
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels_idx = km.fit_predict(embs)   # cluster index per chunk

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

def _extract_heading(block: str) -> str | None:
    """Return the heading text if the block starts with a Markdown heading."""
    m = re.match(r"^(#{1,6})\s+(.+)", block.strip())
    return m.group(2).strip() if m else None


def chunk_markdown(
    markdown: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[tuple[str, str]]:
    """
    Structure-aware, sentence-boundary chunking.

    Returns a list of (chunk_text, heading_context) tuples.

    Fixes applied:
      - FIX 5: Splits are at sentence boundaries (nltk.sent_tokenize) — no mid-sentence cuts.
      - FIX 6: Last seen Markdown heading is propagated into orphan chunks.
      - FIX 7: Overlap is applied via sentence carry-over, never exceeding chunk_size.
    """
    # --- Step 1: Split into structural blocks (headings / paragraphs) ---
    blocks = re.split(r"\n(?=#{1,6}\s)|\n\n+", markdown)
    blocks = [b.strip() for b in blocks if b.strip()]

    # --- Step 2: For each block, segment into sentences ---
    # Track headings so every chunk knows its section context
    all_sentences: list[str] = []
    sentence_headings: list[str] = []   # heading in effect when sentence was added
    current_heading = ""

    for block in blocks:
        h = _extract_heading(block)
        if h:
            current_heading = h

        try:
            sents = nltk.sent_tokenize(block)
        except Exception:
            sents = [block]

        for s in sents:
            s = s.strip()
            if s:
                all_sentences.append(s)
                sentence_headings.append(current_heading)

    if not all_sentences:
        return []

    # --- Step 3: Greedy pack sentences into chunks (FIX 7 — size-safe) ---
    chunks: list[tuple[str, str]] = []
    cur_sents: list[str] = []
    cur_heading = ""
    cur_len = 0

    def _flush(sents: list[str], hdg: str) -> None:
        if sents:
            chunks.append((" ".join(sents), hdg))

    for sent, hdg in zip(all_sentences, sentence_headings):
        addition = len(sent) + (1 if cur_sents else 0)  # +1 for space

        if cur_len + addition <= chunk_size:
            cur_sents.append(sent)
            cur_len += addition
            if not cur_heading:
                cur_heading = hdg
        else:
            _flush(cur_sents, cur_heading)

            # If a SINGLE sentence exceeds chunk_size, hard-split it by words
            if len(sent) > chunk_size:
                words = sent.split()
                part, part_len = [], 0
                for w in words:
                    if part_len + len(w) + 1 > chunk_size and part:
                        chunks.append((" ".join(part), hdg))
                        part, part_len = [w], len(w)
                    else:
                        part.append(w)
                        part_len += len(w) + 1
                if part:
                    chunks.append((" ".join(part), hdg))
                cur_sents, cur_heading, cur_len = [], hdg, 0
            else:
                cur_sents = [sent]
                cur_heading = hdg
                cur_len = len(sent)

    _flush(cur_sents, cur_heading)

    if not chunks:
        return []

    # --- Step 4: Sentence-boundary overlap (FIX 7 continued) ---
    if chunk_overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped: list[tuple[str, str]] = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_text, _ = chunks[i - 1]
        curr_text, curr_hdg = chunks[i]

        # Take sentences from the END of the previous chunk until we hit overlap budget
        prev_sents = nltk.sent_tokenize(prev_text)
        tail_sents: list[str] = []
        tail_len = 0
        for s in reversed(prev_sents):
            if tail_len + len(s) + 1 > chunk_overlap:
                break
            tail_sents.insert(0, s)
            tail_len += len(s) + 1

        if tail_sents:
            new_text = " ".join(tail_sents) + " " + curr_text
        else:
            new_text = curr_text

        overlapped.append((new_text, curr_hdg))

    return overlapped


# ============================================================================
# LANGUAGE DETECTION
# ============================================================================

def detect_language(text: str) -> str:
    """
    Detect the primary language of a text using langdetect.
    Returns an ISO 639-1 code (e.g. 'en', 'ta', 'hi') or 'unknown'.
    Uses the first ~500 chars for speed — enough for reliable detection.
    """
    try:
        return _langdetect(text[:500])
    except (LangDetectException, Exception):
        return "unknown"


# ============================================================================
# CONVERSION: URL -> Markdown
# ============================================================================

def fetch_html(url: str, timeout: int = 15) -> str:
    headers = {"User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def url_to_markdown(url: str) -> tuple[str, str]:
    """Fetch a URL and convert its readable content to Markdown."""
    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer",
                     "header", "aside", "form", "iframe"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else url

    converter = html2text.HTML2Text()
    converter.ignore_images = False
    converter.ignore_links = False
    converter.body_width = 0
    converter.baseurl = url

    markdown = converter.handle(str(soup))
    markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()
    if title:
        markdown = f"# {title}\n\n{markdown}"
    return markdown, title


# ============================================================================
# CONVERSION: PDF -> Markdown
# ============================================================================

def pdf_to_markdown(pdf_path: str) -> tuple[str, str]:
    """Extract text page-by-page from a PDF and format it as Markdown."""
    reader = PdfReader(pdf_path)
    title = Path(pdf_path).stem
    meta_title = reader.metadata.title if reader.metadata else None
    if meta_title and meta_title.strip() and meta_title.strip().lower() not in ("untitled", ""):
        title = meta_title.strip()

    parts = [f"# {title}\n"]
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        parts.append(f"## Page {i}\n\n{text}")

    markdown = re.sub(r"\n{3,}", "\n\n", "\n\n".join(parts)).strip()
    return markdown, title


# ============================================================================
# SHARED: save markdown, slugify
# ============================================================================

def _slugify(source: str) -> str:
    if source.startswith(("http://", "https://")):
        parsed = urlparse(source)
        raw = f"{parsed.netloc}{parsed.path}"
    else:
        raw = Path(source).stem
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", raw).strip("-").lower()
    return (slug[:80] or "document")


def save_markdown(markdown: str, source: str, output_dir: str = MD_OUTPUT_DIR) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(output_dir) / f"{_slugify(source)}.md"
    filepath.write_text(markdown, encoding="utf-8")
    return str(filepath)


# ============================================================================
# FIX 3 — CONTENT-ADDRESSED CHUNK IDs (SHA-256)
# ============================================================================

def _chunk_id(chunk_text: str) -> str:
    """SHA-256 of the chunk content — same content always maps to the same ID.
    Makes upsert() fully idempotent: re-ingesting unchanged content is a no-op."""
    return hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()


# ============================================================================
# CREATE
# ============================================================================

def create(
    source: str,
    source_type: str = "auto",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    md_output_dir: str = MD_OUTPUT_DIR,
    category_labels: list[str] | None = None,
    top_labels: int = DEFAULT_TOP_LABELS,
    k_clusters: int = DEFAULT_CLUSTER_K,
    collection_name: str = _DEFAULT_COLLECTION,
) -> dict:
    """
    Ingest a URL or a PDF path:
      convert → save .md → sentence-aware chunk → compute TF-IDF descriptions
      → domain-agnostic category tagging → store in ChromaDB.

    Args:
        source:          URL (https://...) or path to a PDF file.
        source_type:     "url", "pdf", or "auto" (default).
        chunk_size:      Target max characters per chunk.
        chunk_overlap:   Sentence-level overlap chars between consecutive chunks.
        md_output_dir:   Where to save the .md intermediate file.
        category_labels: Optional list of topic labels for guided mode.
                         If None (default), topics are auto-discovered (unsupervised).
                         Example: ["billing", "plans", "network", "sim"]
        top_labels:      How many category labels to assign per chunk (default 2).
        k_clusters:      KMeans clusters for unsupervised discovery (default 8).
        collection_name: Which ChromaDB collection to store the chunks in.
                         Defaults to chroma_setup.COLLECTION_NAME. Pass one of
                         chroma_setup.KB_COLLECTIONS' values to target a scoped
                         RAG knowledge base instead of the shared collection.
    """
    if source_type == "auto":
        source_type = "url" if source.startswith(("http://", "https://")) else "pdf"

    mode = "guided" if category_labels else "unsupervised"
    print(f"[CREATE] type={source_type} | category_mode={mode} | collection={collection_name} | {source}")

    # --- Convert to Markdown ---
    if source_type == "url":
        markdown, title = url_to_markdown(source)
    elif source_type == "pdf":
        markdown, title = pdf_to_markdown(source)
    else:
        raise ValueError(f"Unknown source_type: {source_type!r} (use 'url', 'pdf', or 'auto')")

    md_path = save_markdown(markdown, source, md_output_dir)
    print(f"  Markdown saved -> {md_path}")

    return _ingest_markdown(
        markdown=markdown,
        source=source,
        source_type=source_type,
        title=title,
        md_path=md_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        category_labels=category_labels,
        top_labels=top_labels,
        k_clusters=k_clusters,
        collection_name=collection_name,
        mode=mode,
    )


def create_from_markdown(
    markdown: str,
    source_name: str,
    title: str,
    collection_name: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    category_labels: list[str] | None = None,
    top_labels: int = DEFAULT_TOP_LABELS,
    k_clusters: int = DEFAULT_CLUSTER_K,
) -> dict:
    """
    Ingest raw Markdown text directly (no URL fetch / PDF parse step) into a
    given ChromaDB collection. This is what local knowledge-base files like
    kb_docs/*.md go through — `create()` only accepts 'url'/'pdf' sources.

    Args:
        markdown:        Raw Markdown content to chunk and store.
        source_name:      A stable identifier for this document (used as the
                          `source` metadata field and for update()/delete()).
        title:           Human-readable document title (metadata `title`).
        collection_name: Target ChromaDB collection (e.g. one of
                         chroma_setup.KB_COLLECTIONS' values).
        chunk_size / chunk_overlap / category_labels / top_labels / k_clusters:
                         Same meaning as in create().
    """
    mode = "guided" if category_labels else "unsupervised"
    print(f"[CREATE] type=markdown | category_mode={mode} | collection={collection_name} | {source_name}")

    return _ingest_markdown(
        markdown=markdown,
        source=source_name,
        source_type="markdown",
        title=title,
        md_path=None,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        category_labels=category_labels,
        top_labels=top_labels,
        k_clusters=k_clusters,
        collection_name=collection_name,
        mode=mode,
    )


def _ingest_markdown(
    *,
    markdown: str,
    source: str,
    source_type: str,
    title: str,
    md_path: str | None,
    chunk_size: int,
    chunk_overlap: int,
    category_labels: list[str] | None,
    top_labels: int,
    k_clusters: int,
    collection_name: str,
    mode: str,
) -> dict:
    """Shared tail-end of create()/create_from_markdown(): chunk, describe,
    categorize, and upsert already-converted Markdown into ChromaDB."""
    # Detect document language from the first 1000 chars
    doc_language = detect_language(markdown[:1000])
    print(f"  Detected language: {doc_language}")

    # --- Chunk (FIX 5 + FIX 6 + FIX 7) ---
    chunk_tuples = chunk_markdown(markdown, chunk_size, chunk_overlap)
    chunk_texts   = [ct for ct, _ in chunk_tuples]
    chunk_headings = [ch for _, ch in chunk_tuples]
    print(f"  {len(chunk_texts)} chunks created")

    if not chunk_texts:
        print("  WARNING: no chunks produced — source may be empty.")
        return {"source": source, "source_type": source_type, "title": title,
                "markdown_path": md_path, "num_chunks": 0, "collection": collection_name}

    # --- TF-IDF descriptions (FIX 4) ---
    descriptions = compute_tfidf_descriptions(chunk_texts, top_n=10)

    # --- Domain-agnostic categories (FIX 1 + FIX 2) ---
    categories = categorize_chunks(chunk_texts, category_labels, top_labels, k_clusters)

    # --- Build ChromaDB records ---
    collection = get_collection(collection_name)
    ids, documents, metadatas = [], [], []
    now_utc = datetime.now(UTC).isoformat()
    total = len(chunk_texts)

    for i, (chunk_text, heading, description, category) in enumerate(
        zip(chunk_texts, chunk_headings, descriptions, categories)
    ):
        # FIX 3: content-addressed ID
        cid = _chunk_id(chunk_text)
        ids.append(cid)
        documents.append(chunk_text)
        metadatas.append({
            # --- Structural fields (hardcoded from input) ---
            "source":        source,
            "source_type":   source_type,
            "title":         title,
            "chunk_index":   i,
            "total_chunks":  total,
            # --- Algorithmic fields ---
            "category":      category,        # FIX 1+2: multi-label, domain-agnostic
            "description":   description,     # FIX 4: TF-IDF based
            "heading":       heading,         # FIX 6: section heading context
            "language":      doc_language,    # NEW: ISO 639-1 language code
            "num_words":     len(chunk_text.split()),  # NEW: word count
            "ingested_at":   now_utc,         # NEW: ingestion timestamp
            "content_hash":  cid,             # NEW: SHA-256 of content (=chunk ID)
        })

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    print(f"  Stored {len(ids)} chunks in ChromaDB collection '{collection_name}'")

    return {
        "source":        source,
        "source_type":   source_type,
        "title":         title,
        "language":      doc_language,
        "markdown_path": md_path,
        "num_chunks":    total,
        "category_mode": mode,
        "collection":    collection_name,
    }


# ============================================================================
# READ
# ============================================================================

def read(
    query: str,
    n_results: int = 5,
    source_filter: str | None = None,
    category_filter: str | None = None,
    language_filter: str | None = None,
    collection_name: str = _DEFAULT_COLLECTION,
) -> dict:
    """
    Semantic search over stored chunks. Filters are additive (AND).

    Args:
        query:           Free-text query.
        n_results:       Number of results to return.
        source_filter:   Restrict to a single source URL/path.
        category_filter: Restrict to chunks whose category contains this string.
                         (substring match applied in post-filter since ChromaDB
                          does exact-match on metadata strings)
        language_filter: Restrict to a specific language code, e.g. 'en', 'ta'.
        collection_name: Which ChromaDB collection to search (default: the
                         shared collection). Pass a chroma_setup.KB_COLLECTIONS
                         value to search a scoped RAG knowledge base.
    """
    collection = get_collection(collection_name)
    conditions: list[dict] = []

    if source_filter:
        conditions.append({"source": {"$eq": source_filter}})
    if language_filter:
        conditions.append({"language": {"$eq": language_filter}})

    where = None
    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    # Fetch more results than needed so we can post-filter on category substring
    fetch_n = n_results * 4 if category_filter else n_results
    fetch_n = max(fetch_n, n_results)

    results = collection.query(
        query_texts=[query],
        n_results=min(fetch_n, collection.count() or 1),
        where=where,
    )

    # Post-filter on category substring (handles multi-label "billing,plans")
    if category_filter and results["documents"][0]:
        cat_lower = category_filter.lower()
        filtered_docs, filtered_metas, filtered_dists = [], [], []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            if cat_lower in meta.get("category", "").lower():
                filtered_docs.append(doc)
                filtered_metas.append(meta)
                filtered_dists.append(dist)
                if len(filtered_docs) >= n_results:
                    break

        results["documents"][0]  = filtered_docs[:n_results]
        results["metadatas"][0]  = filtered_metas[:n_results]
        results["distances"][0]  = filtered_dists[:n_results]

    return results


# ============================================================================
# LIST SOURCES / CATEGORIES
# ============================================================================

def list_sources(collection_name: str = _DEFAULT_COLLECTION) -> list[dict]:
    """Every distinct source currently stored, with chunk count and categories."""
    collection = get_collection(collection_name)
    data = collection.get(include=["metadatas"])
    seen: dict[str, dict] = {}
    for meta in data["metadatas"]:
        src = meta["source"]
        if src not in seen:
            seen[src] = {
                "source":      src,
                "source_type": meta.get("source_type"),
                "title":       meta.get("title"),
                "language":    meta.get("language", "unknown"),
                "chunks":      0,
                "categories":  {},
            }
        seen[src]["chunks"] += 1
        for cat in meta.get("category", "general").split(","):
            cat = cat.strip()
            seen[src]["categories"][cat] = seen[src]["categories"].get(cat, 0) + 1
    return list(seen.values())


def list_categories(collection_name: str = _DEFAULT_COLLECTION) -> dict[str, int]:
    """Every category label currently in the DB, with total chunk counts."""
    collection = get_collection(collection_name)
    data = collection.get(include=["metadatas"])
    counts: dict[str, int] = {}
    for meta in data["metadatas"]:
        for cat in meta.get("category", "general").split(","):
            cat = cat.strip()
            counts[cat] = counts.get(cat, 0) + 1
    return counts


# ============================================================================
# UPDATE
# ============================================================================

def update(source: str, source_type: str = "auto", collection_name: str = _DEFAULT_COLLECTION, **kwargs) -> dict:
    """Re-ingest a source: delete old chunks, then create new ones."""
    delete(source, collection_name=collection_name)
    return create(source, source_type=source_type, collection_name=collection_name, **kwargs)


# ============================================================================
# DELETE
# ============================================================================

def delete(source: str, collection_name: str = _DEFAULT_COLLECTION) -> int:
    """Remove every chunk belonging to a given source (URL or PDF path)."""
    collection = get_collection(collection_name)
    existing = collection.get(where={"source": {"$eq": source}}, include=[])
    ids = existing["ids"]
    if ids:
        collection.delete(ids=ids)
    print(f"[DELETE] Removed {len(ids)} chunks for source: {source}")
    return len(ids)


# ============================================================================
# INSPECT
# ============================================================================

def inspect_source(source: str, text_preview_chars: int = 150, collection_name: str = _DEFAULT_COLLECTION) -> list[dict]:
    """Print every chunk for a source with its full metadata and text preview."""
    collection = get_collection(collection_name)
    data = collection.get(
        where={"source": {"$eq": source}},
        include=["metadatas", "documents"],
    )

    if not data["ids"]:
        print(f"No chunks found for source: {source}")
        print("Check the exact string with: python content_manager.py list")
        return []

    rows = sorted(
        zip(data["metadatas"], data["documents"]),
        key=lambda pair: pair[0].get("chunk_index", 0),
    )

    for meta, doc in rows:
        cats = meta.get("category", "general")
        print(f"\n--- chunk {meta.get('chunk_index')} / {meta.get('total_chunks')} ---")
        print(f"  heading:     {meta.get('heading', '')!r}")
        print(f"  category:    {cats}")
        print(f"  language:    {meta.get('language', 'unknown')}")
        print(f"  num_words:   {meta.get('num_words', '?')}")
        print(f"  description: {meta.get('description', '')}")
        print(f"  ingested_at: {meta.get('ingested_at', '')}")
        preview = doc[:text_preview_chars].replace("\n", " ")
        # Replace any character that can't encode on this terminal
        safe_preview = preview.encode("utf-8", errors="replace").decode("utf-8")
        print(f"  text:        {safe_preview}")

    return [{"metadata": m, "text": d} for m, d in rows]


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CRUD: URL/PDF → Markdown → ChromaDB (domain-agnostic)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- create ---
    p_create = sub.add_parser("create", help="Ingest a URL or PDF")
    p_create.add_argument("source", help="URL (https://...) or path to a PDF file")
    p_create.add_argument("--type", choices=["auto", "url", "pdf"], default="auto")
    p_create.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    p_create.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    p_create.add_argument(
        "--labels", nargs="+", default=None, metavar="LABEL",
        help=(
            "Optional topic labels for guided category mode. "
            "If omitted, categories are auto-discovered from content. "
            "Examples: --labels billing plans network sim  "
            "          --labels diagnosis treatment medication"
        ),
    )
    p_create.add_argument("--top-labels", type=int, default=DEFAULT_TOP_LABELS,
                          help="Max category labels per chunk (default 2)")
    p_create.add_argument("--k-clusters", type=int, default=DEFAULT_CLUSTER_K,
                          help="KMeans clusters for unsupervised discovery (default 8)")
    p_create.add_argument("--collection", default=_DEFAULT_COLLECTION,
                          help="Target ChromaDB collection (default: shared 'knowledge_base')")

    # --- read ---
    p_read = sub.add_parser("read", help="Semantic search over the knowledge base")
    p_read.add_argument("query")
    p_read.add_argument("--n-results", type=int, default=5)
    p_read.add_argument("--source", default=None)
    p_read.add_argument("--category", default=None,
                        help="Substring match on category label, e.g. 'billing'")
    p_read.add_argument("--language", default=None,
                        help="ISO 639-1 code, e.g. 'en', 'ta', 'hi'")
    p_read.add_argument("--preview-chars", type=int, default=300)
    p_read.add_argument("--collection", default=_DEFAULT_COLLECTION,
                        help="ChromaDB collection to search (default: shared 'knowledge_base')")

    # --- update ---
    p_update = sub.add_parser("update", help="Re-ingest a source (replaces its old chunks)")
    p_update.add_argument("source")
    p_update.add_argument("--type", choices=["auto", "url", "pdf"], default="auto")
    p_update.add_argument("--collection", default=_DEFAULT_COLLECTION)

    # --- delete ---
    p_delete = sub.add_parser("delete", help="Remove a source's chunks")
    p_delete.add_argument("source")
    p_delete.add_argument("--collection", default=_DEFAULT_COLLECTION)

    # --- inspect ---
    p_inspect = sub.add_parser("inspect", help="Show every chunk's metadata + text preview")
    p_inspect.add_argument("source")
    p_inspect.add_argument("--text-preview-chars", type=int, default=150)
    p_inspect.add_argument("--collection", default=_DEFAULT_COLLECTION)

    p_list = sub.add_parser("list", help="List every ingested source")
    p_list.add_argument("--collection", default=_DEFAULT_COLLECTION)

    p_cats = sub.add_parser("categories", help="List every category label in the DB")
    p_cats.add_argument("--collection", default=_DEFAULT_COLLECTION)

    args = parser.parse_args()

    if args.command == "create":
        result = create(
            args.source,
            source_type=args.type,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            category_labels=args.labels,
            top_labels=args.top_labels,
            k_clusters=args.k_clusters,
            collection_name=args.collection,
        )
        print(f"\n[DONE] {result['num_chunks']} chunks | lang={result.get('language','?')} "
              f"| mode={result.get('category_mode','?')} | title={result['title']!r}")

    elif args.command == "read":
        results = read(
            args.query,
            n_results=args.n_results,
            source_filter=args.source,
            category_filter=args.category,
            language_filter=args.language,
            collection_name=args.collection,
        )
        print(f"\nQuery: {args.query}")
        if not results["documents"][0]:
            print("  No results. Has anything been ingested yet? (see: content_manager.py list)")
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            sim = 1 - dist
            cats = meta.get("category", "general")
            hdg  = meta.get("heading", "")
            lang = meta.get("language", "?")
            print(f"\n--- similarity={sim:.3f} | [{cats}] | lang={lang} | "
                  f"heading={hdg!r} | {meta['source']} (chunk {meta['chunk_index']}) ---")
            print(doc[:args.preview_chars])

    elif args.command == "update":
        update(args.source, source_type=args.type, collection_name=args.collection)

    elif args.command == "delete":
        delete(args.source, collection_name=args.collection)

    elif args.command == "inspect":
        inspect_source(args.source, text_preview_chars=args.text_preview_chars, collection_name=args.collection)

    elif args.command == "list":
        sources = list_sources(collection_name=args.collection)
        if not sources:
            print("No sources ingested yet.")
        for s in sources:
            cat_summary = ", ".join(
                f"{cat}={n}" for cat, n in sorted(s["categories"].items())
            )
            print(f"  [{s['source_type']}] [{s.get('language','?')}] {s['title']} "
                  f"— {s['chunks']} chunks — {s['source']}")
            print(f"      categories: {cat_summary}")

    elif args.command == "categories":
        counts = list_categories(collection_name=args.collection)
        if not counts:
            print("No sources ingested yet.")
        for cat, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {cat:30} {n} chunks")


if __name__ == "__main__":
    main()
