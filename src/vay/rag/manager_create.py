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

import hashlib
import re
import sys as _sys_utf8
from pathlib import Path
from urllib.parse import urlparse

from vay.rag.manager_ingest import _ingest_markdown
from vay.rag.parsers import pdf_to_markdown, url_to_markdown
from vay.rag.vector_store import COLLECTION_NAME as _DEFAULT_COLLECTION
from vay.rag.vector_store import get_collection

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_LABELS = 2
DEFAULT_CLUSTER_K = 8
MD_OUTPUT_DIR = "kb_docs"

if hasattr(_sys_utf8.stdout, "reconfigure"):
    try:
        _sys_utf8.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass



def _slugify(source: str) -> str:
    if source.startswith(("http://", "https://")):
        parsed = urlparse(source)
        raw = f"{parsed.netloc}{parsed.path}"
    else:
        raw = Path(source).stem
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", raw).strip("-").lower()
    return slug[:80] or "document"


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
    print(
        f"[CREATE] type={source_type} | category_mode={mode} | collection={collection_name} | {source}"
    )

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
    print(
        f"[CREATE] type=markdown | category_mode={mode} | collection={collection_name} | {source_name}"
    )

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
