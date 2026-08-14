"""
chroma_setup.py

Owns the ChromaDB connection: config, client, collection, init, and reset.
Every other file imports get_collection() from here instead of creating
its own chromadb.PersistentClient — keeps the DB path / collection name /
embedding model in exactly ONE place.

Run directly to check status or wipe the DB:
    python chroma_setup.py              # show status (path, collection, chunk count)
    python chroma_setup.py --reset      # wipe the collection and start fresh
"""

import argparse
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

# ----------------------------------------------------------------------------
# CONFIG — change these here; every other file picks up the change automatically
# ----------------------------------------------------------------------------
PERSIST_DIRECTORY = "chroma_db"
COLLECTION_NAME = "knowledge_base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ----------------------------------------------------------------------------
# The 5 scoped Nexatel RAG knowledge-base collections (agentic architecture).
# Every module that needs one of these should import the name from here
# instead of hardcoding the string, so a rename only happens in one place.
# ----------------------------------------------------------------------------
KB_COLLECTIONS = {
    "billing_policy": "billing_policy",  # Billing & Payments Agent
    "product_catalog": "product_catalog",  # Plans & Offers Agent
    "support_faq": "support_faq",  # Complaints & Service-Request Agent
    "technical_kb": "technical_kb",  # Coverage & Technical Agent
    "compliance_policy": "compliance_policy",  # Guardrail layer (all agents)
}

# module-level cache so repeated calls in the same process reuse the same
# client/embedding-function/collections instead of reopening/reloading them
# every time. _collections is keyed by (persist_directory, collection_name)
# so switching between the 5 scoped RAG collections mid-session (as the
# orchestrator/sub-agents do) doesn't evict and reload the model each time.
_client = None
_embedding_function = None
_collections: dict = {}


def get_client(persist_directory: str = PERSIST_DIRECTORY):
    """Return a persistent ChromaDB client (creates the folder if needed)."""
    global _client
    if _client is None:
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=persist_directory)
    return _client


def get_embedding_function(model_name: str = EMBEDDING_MODEL):
    """SentenceTransformer embedding function ChromaDB calls automatically
    on every add/upsert/query. Cached so the model is only loaded once."""
    global _embedding_function
    if _embedding_function is None:
        _embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
    return _embedding_function


def get_collection(
    collection_name: str = COLLECTION_NAME, persist_directory: str = PERSIST_DIRECTORY
):
    """Open (or create) a collection. Call this from any other file.
    Safe to call repeatedly with different collection_name values (e.g. to
    switch between the 5 scoped KB_COLLECTIONS) — each is cached separately."""
    key = (persist_directory, collection_name)
    if key not in _collections:
        client = get_client(persist_directory)
        _collections[key] = client.get_or_create_collection(
            name=collection_name,
            embedding_function=get_embedding_function(),
            metadata={"hnsw:space": "cosine"},
        )
    return _collections[key]


def init_db(
    persist_directory: str = PERSIST_DIRECTORY, collection_name: str = COLLECTION_NAME
) -> "chromadb.Collection":
    """Initialize (or re-open) the DB and print a status summary."""
    collection = get_collection(collection_name, persist_directory)
    count = collection.count()
    print("ChromaDB ready.")
    print(f"  Path:        {Path(persist_directory).resolve()}")
    print(f"  Collection:  {collection_name}")
    print(f"  Chunks:      {count}")
    return collection


def reset_db(
    persist_directory: str = PERSIST_DIRECTORY, collection_name: str = COLLECTION_NAME
) -> "chromadb.Collection":
    """DANGER: deletes every chunk in the collection. Use for a clean slate."""
    client = get_client(persist_directory)
    try:
        client.delete_collection(name=collection_name)
        print(f"Deleted collection '{collection_name}'.")
    except Exception:
        pass  # collection didn't exist yet — fine
    _collections.pop((persist_directory, collection_name), None)
    return get_collection(collection_name, persist_directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChromaDB setup / status check")
    parser.add_argument("--reset", action="store_true", help="Wipe the collection and start fresh")
    args = parser.parse_args()

    if args.reset:
        confirm = input(
            f"This deletes ALL data in collection '{COLLECTION_NAME}'. Type 'yes' to confirm: "
        )
        if confirm.strip().lower() == "yes":
            reset_db()
            init_db()
        else:
            print("Cancelled.")
    else:
        init_db()
