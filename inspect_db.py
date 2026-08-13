"""
inspect_db.py
-------------
Small CLI to browse/inspect the local Chroma DB built by build_chroma_kb.py,
without having to retype raw Python snippets each time.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
  # Overview: collection name, total chunk count, chunks per doc_type
  python inspect_db.py --info

  # Peek at N sample chunks (default 5)
  python inspect_db.py --peek
  python inspect_db.py --peek 10

  # List ALL chunks (doc + metadata), optionally truncated
  python inspect_db.py --list
  python inspect_db.py --list --full          # don't truncate chunk text

  # Filter by metadata field, e.g. only response_script chunks,
  # or only chunks from a specific source file / section / category
  python inspect_db.py --filter doc_type=response_script
  python inspect_db.py --filter source_file=NexaTel_Knowledge_Base.pdf
  python inspect_db.py --filter section="6.3 Lost, Stolen, or Damaged SIM"

  # Semantic similarity search (what actually gets retrieved at query time)
  python inspect_db.py --query "how do I port my number" 
  python inspect_db.py --query "roaming charges abroad" --n 5
  python inspect_db.py --query "duplicate SIM" --doc_type knowledge_base

  # Fetch one exact chunk by its id
  python inspect_db.py --id knowledge_base-12-a1b2c3d4

  # Point at a different persist dir / collection name if you changed defaults
  python inspect_db.py --persist_dir ./chroma_db --collection nexatel_kb --info
--------------------------------------------------------------------------
"""

import argparse
import textwrap

import chromadb
from chromadb.utils import embedding_functions


DEFAULT_PERSIST_DIR = "./chroma_db"
DEFAULT_COLLECTION = "nexatel_kb"
TRUNCATE_LEN = 220


def get_collection(persist_dir: str, collection_name: str):
    client = chromadb.PersistentClient(path=persist_dir)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    try:
        return client.get_collection(collection_name, embedding_function=embed_fn)
    except Exception as e:
        print(f"ERROR: could not open collection '{collection_name}' at '{persist_dir}'.")
        print(f"  ({e})")
        print("Did you run build_chroma_kb.py first, and are --persist_dir / --collection correct?")
        raise SystemExit(1)


def shorten(text: str, full: bool):
    text = text.replace("\n", " ").strip()
    if full or len(text) <= TRUNCATE_LEN:
        return text
    return text[:TRUNCATE_LEN].rstrip() + " ..."


def print_chunk(idx, doc_id, doc, meta, full=False, distance=None):
    header = f"[{idx}] id={doc_id}"
    if distance is not None:
        header += f"  distance={distance:.4f}  similarity~{1 - distance:.4f}"
    print(header)
    meta_line = "  " + " | ".join(f"{k}={v}" for k, v in meta.items())
    print(meta_line)
    print(textwrap.indent(shorten(doc, full), "  "))
    print()


def cmd_info(col):
    total = col.count()
    print(f"Collection: {col.name}")
    print(f"Total chunks: {total}\n")

    data = col.get(include=["metadatas"])
    counts = {}
    for meta in data["metadatas"]:
        dt = meta.get("doc_type", "unknown")
        counts[dt] = counts.get(dt, 0) + 1

    print("Chunks per doc_type:")
    for dt, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {dt:20s} {n}")

    sources = {}
    for meta in data["metadatas"]:
        sf = meta.get("source_file", "unknown")
        sources[sf] = sources.get(sf, 0) + 1
    print("\nChunks per source_file:")
    for sf, n in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {sf:45s} {n}")


def cmd_peek(col, n, full):
    data = col.peek(n)
    print(f"Peeking at {len(data['documents'])} chunks:\n")
    for i, (doc_id, doc, meta) in enumerate(zip(data["ids"], data["documents"], data["metadatas"])):
        print_chunk(i, doc_id, doc, meta, full=full)


def cmd_list(col, full, where=None):
    data = col.get(where=where, include=["documents", "metadatas"])
    print(f"{len(data['documents'])} chunk(s) found.\n")
    for i, (doc_id, doc, meta) in enumerate(zip(data["ids"], data["documents"], data["metadatas"])):
        print_chunk(i, doc_id, doc, meta, full=full)


def cmd_query(col, query_text, n_results, full, where=None):
    res = col.query(query_texts=[query_text], n_results=n_results, where=where)
    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    print(f"Top {len(ids)} result(s) for: \"{query_text}\"\n")
    for i, (doc_id, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists)):
        print_chunk(i, doc_id, doc, meta, full=full, distance=dist)


def cmd_get_id(col, chunk_id, full):
    data = col.get(ids=[chunk_id], include=["documents", "metadatas"])
    if not data["ids"]:
        print(f"No chunk found with id '{chunk_id}'")
        return
    doc_id, doc, meta = data["ids"][0], data["documents"][0], data["metadatas"][0]
    print_chunk(0, doc_id, doc, meta, full=full)


def parse_filter(filter_str: str):
    """Parses 'key=value' into a Chroma `where` dict."""
    if "=" not in filter_str:
        raise SystemExit("--filter must be in the form key=value, e.g. doc_type=response_script")
    key, value = filter_str.split("=", 1)
    key, value = key.strip(), value.strip().strip('"').strip("'")
    return {key: value}


def main():
    parser = argparse.ArgumentParser(
        description="Inspect the local NexaTel ChromaDB knowledge base.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--persist_dir", default=DEFAULT_PERSIST_DIR, help="Path to the persisted Chroma DB.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Chroma collection name.")
    parser.add_argument("--full", action="store_true", help="Don't truncate chunk text in output.")

    parser.add_argument("--info", action="store_true", help="Show collection overview & counts.")
    parser.add_argument("--peek", nargs="?", const=5, type=int, metavar="N", help="Peek at N sample chunks (default 5).")
    parser.add_argument("--list", action="store_true", help="List all chunks (respects --filter).")
    parser.add_argument("--filter", metavar="key=value", help="Metadata filter, e.g. doc_type=policy_guide")
    parser.add_argument("--query", metavar="TEXT", help="Run a similarity search against the DB.")
    parser.add_argument("--n", type=int, default=3, help="Number of results for --query (default 3).")
    parser.add_argument("--doc_type", help="Shortcut: restrict --query to a specific doc_type.")
    parser.add_argument("--id", metavar="CHUNK_ID", help="Fetch one exact chunk by id.")

    args = parser.parse_args()

    if not any([args.info, args.peek is not None, args.list, args.query, args.id]):
        parser.print_help()
        return

    col = get_collection(args.persist_dir, args.collection)

    where = parse_filter(args.filter) if args.filter else None
    if args.doc_type:
        where = {"doc_type": args.doc_type}

    if args.info:
        cmd_info(col)
        print()

    if args.peek is not None:
        cmd_peek(col, args.peek, args.full)

    if args.list:
        cmd_list(col, args.full, where=where)

    if args.query:
        cmd_query(col, args.query, args.n, args.full, where=where)

    if args.id:
        cmd_get_id(col, args.id, args.full)


if __name__ == "__main__":
    main()
