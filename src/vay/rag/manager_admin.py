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
from vay.rag.manager_read import _DEFAULT_COLLECTION, delete, get_collection


def list_sources(collection_name: str = _DEFAULT_COLLECTION) -> list[dict]:
    """Every distinct source currently stored, with chunk count and categories."""
    collection = get_collection(collection_name)
    data = collection.get(include=["metadatas"])
    seen: dict[str, dict] = {}
    for meta in data["metadatas"]:
        src = meta["source"]
        if src not in seen:
            seen[src] = {
                "source": src,
                "source_type": meta.get("source_type"),
                "title": meta.get("title"),
                "language": meta.get("language", "unknown"),
                "chunks": 0,
                "categories": {},
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


def update(
    source: str, source_type: str = "auto", collection_name: str = _DEFAULT_COLLECTION, **kwargs
) -> dict:
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


def inspect_source(
    source: str, text_preview_chars: int = 150, collection_name: str = _DEFAULT_COLLECTION
) -> list[dict]:
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
