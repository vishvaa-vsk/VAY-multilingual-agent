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
from vay.rag.manager_create import get_collection


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
    chunk_texts = [ct for ct, _ in chunk_tuples]
    chunk_headings = [ch for _, ch in chunk_tuples]
    print(f"  {len(chunk_texts)} chunks created")

    if not chunk_texts:
        print("  WARNING: no chunks produced — source may be empty.")
        return {
            "source": source,
            "source_type": source_type,
            "title": title,
            "markdown_path": md_path,
            "num_chunks": 0,
            "collection": collection_name,
        }

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
        metadatas.append(
            {
                # --- Structural fields (hardcoded from input) ---
                "source": source,
                "source_type": source_type,
                "title": title,
                "chunk_index": i,
                "total_chunks": total,
                # --- Algorithmic fields ---
                "category": category,  # FIX 1+2: multi-label, domain-agnostic
                "description": description,  # FIX 4: TF-IDF based
                "heading": heading,  # FIX 6: section heading context
                "language": doc_language,  # NEW: ISO 639-1 language code
                "num_words": len(chunk_text.split()),  # NEW: word count
                "ingested_at": now_utc,  # NEW: ingestion timestamp
                "content_hash": cid,  # NEW: SHA-256 of content (=chunk ID)
            }
        )

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    print(f"  Stored {len(ids)} chunks in ChromaDB collection '{collection_name}'")

    return {
        "source": source,
        "source_type": source_type,
        "title": title,
        "language": doc_language,
        "markdown_path": md_path,
        "num_chunks": total,
        "category_mode": mode,
        "collection": collection_name,
    }


# ============================================================================
# READ
# ============================================================================
