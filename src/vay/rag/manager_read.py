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

from vay.rag.vector_store import COLLECTION_NAME as _DEFAULT_COLLECTION
from vay.rag.vector_store import get_collection



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

        results["documents"][0] = filtered_docs[:n_results]
        results["metadatas"][0] = filtered_metas[:n_results]
        results["distances"][0] = filtered_dists[:n_results]

    return results


# ============================================================================
# LIST SOURCES / CATEGORIES
# ============================================================================
