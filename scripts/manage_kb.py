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

# ---------------------------------------------------------------------------
# Force UTF-8 on Windows consoles (cp1252 chokes on Unicode in rich content)
# ---------------------------------------------------------------------------
import sys as _sys_utf8

from chroma_setup import COLLECTION_NAME as _DEFAULT_COLLECTION

if hasattr(_sys_utf8.stdout, "reconfigure"):
    try:
        _sys_utf8.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from vay.rag.manager import (
    create,
    delete,
    inspect_source,
    list_categories,
    list_sources,
    read,
    update,
)


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
        "--labels",
        nargs="+",
        default=None,
        metavar="LABEL",
        help=(
            "Optional topic labels for guided category mode. "
            "If omitted, categories are auto-discovered from content. "
            "Examples: --labels billing plans network sim  "
            "          --labels diagnosis treatment medication"
        ),
    )
    p_create.add_argument(
        "--top-labels",
        type=int,
        default=DEFAULT_TOP_LABELS,
        help="Max category labels per chunk (default 2)",
    )
    p_create.add_argument(
        "--k-clusters",
        type=int,
        default=DEFAULT_CLUSTER_K,
        help="KMeans clusters for unsupervised discovery (default 8)",
    )
    p_create.add_argument(
        "--collection",
        default=_DEFAULT_COLLECTION,
        help="Target ChromaDB collection (default: shared 'knowledge_base')",
    )

    # --- read ---
    p_read = sub.add_parser("read", help="Semantic search over the knowledge base")
    p_read.add_argument("query")
    p_read.add_argument("--n-results", type=int, default=5)
    p_read.add_argument("--source", default=None)
    p_read.add_argument(
        "--category", default=None, help="Substring match on category label, e.g. 'billing'"
    )
    p_read.add_argument("--language", default=None, help="ISO 639-1 code, e.g. 'en', 'ta', 'hi'")
    p_read.add_argument("--preview-chars", type=int, default=300)
    p_read.add_argument(
        "--collection",
        default=_DEFAULT_COLLECTION,
        help="ChromaDB collection to search (default: shared 'knowledge_base')",
    )

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
        print(
            f"\n[DONE] {result['num_chunks']} chunks | lang={result.get('language', '?')} "
            f"| mode={result.get('category_mode', '?')} | title={result['title']!r}"
        )

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
            hdg = meta.get("heading", "")
            lang = meta.get("language", "?")
            print(
                f"\n--- similarity={sim:.3f} | [{cats}] | lang={lang} | "
                f"heading={hdg!r} | {meta['source']} (chunk {meta['chunk_index']}) ---"
            )
            print(doc[: args.preview_chars])

    elif args.command == "update":
        update(args.source, source_type=args.type, collection_name=args.collection)

    elif args.command == "delete":
        delete(args.source, collection_name=args.collection)

    elif args.command == "inspect":
        inspect_source(
            args.source, text_preview_chars=args.text_preview_chars, collection_name=args.collection
        )

    elif args.command == "list":
        sources = list_sources(collection_name=args.collection)
        if not sources:
            print("No sources ingested yet.")
        for s in sources:
            cat_summary = ", ".join(f"{cat}={n}" for cat, n in sorted(s["categories"].items()))
            print(
                f"  [{s['source_type']}] [{s.get('language', '?')}] {s['title']} "
                f"— {s['chunks']} chunks — {s['source']}"
            )
            print(f"      categories: {cat_summary}")

    elif args.command == "categories":
        counts = list_categories(collection_name=args.collection)
        if not counts:
            print("No sources ingested yet.")
        for cat, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {cat:30} {n} chunks")


if __name__ == "__main__":
    main()
