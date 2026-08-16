"""
manage_kb.py

Admin CLI for the 5 scoped Nexatel RAG knowledge-base collections in ChromaDB.

FIXED (this session): this script was dead code -- it imported a nonexistent
local file `scripts/chroma_setup (1).py` (a leftover from before the project
was restructured into the `vay` package) and called functions
(`create`/`update`/`delete`/`list_sources`/etc. with a `DEFAULT_CHUNK_SIZE`
import) that no longer matched `vay.rag.manager`'s actual exports. It could
not run at all. Rewritten to use the current package directly.

    python scripts/manage_kb.py --status                  # chunk counts per collection
    python scripts/manage_kb.py --search "2gb daily plan" --collection product_catalog
    python scripts/manage_kb.py --rebuild product_catalog  # wipe + re-ingest one KB
    python scripts/manage_kb.py --rebuild all              # wipe + re-ingest all 5
"""

import argparse
from pathlib import Path

from vay.rag import manager as content_manager
from vay.rag import vector_store as chroma_setup
from vay.rag.hybrid import invalidate_cache

KB_DOCS_DIR = Path(__file__).parent.parent / "data" / "kb"

KB_SPEC = {
    chroma_setup.KB_COLLECTIONS["billing_policy"]: (
        "billing_policy.md",
        "Nexatel Billing & Payments Policy",
        ["tariff", "billing_cycle", "late_fee", "roaming", "refund", "charges", "payment"],
    ),
    chroma_setup.KB_COLLECTIONS["product_catalog"]: (
        "product_catalog.md",
        "Nexatel Product Catalog — Plans, Add-Ons & Offers",
        ["prepaid", "postpaid", "broadband", "add_on", "eligibility", "offer_terms"],
    ),
    chroma_setup.KB_COLLECTIONS["support_faq"]: (
        "support_faq.md",
        "Nexatel Support Knowledge Base — Troubleshooting, FAQs & Complaint Policy",
        ["troubleshooting", "complaint_policy", "sla", "escalation", "faq"],
    ),
    chroma_setup.KB_COLLECTIONS["technical_kb"]: (
        "technical_kb.md",
        "Nexatel Technical Knowledge Base — Devices, APN, SIM/eSIM & Coverage",
        ["apn_settings", "sim_esim", "volte_5g", "coverage", "device_settings"],
    ),
    chroma_setup.KB_COLLECTIONS["compliance_policy"]: (
        "compliance_policy.md",
        "Nexatel Compliance & Policy — Regulatory Scripts, Consent & Do/Don't-Say Rules",
        ["identity_verification", "consent_script", "do_dont_say", "escalation", "trai_regulation"],
    ),
}


def status() -> None:
    print(f"ChromaDB persist dir: {Path(chroma_setup.PERSIST_DIRECTORY).resolve()}\n")
    total = 0
    for name in chroma_setup.KB_COLLECTIONS.values():
        count = chroma_setup.get_collection(name).count()
        total += count
        print(f"  {name:20} {count:3} chunks")
    print(f"\nTotal chunks across all collections: {total}")


def rebuild(collection_key: str) -> None:
    targets = KB_SPEC.items() if collection_key == "all" else [
        (collection_key, KB_SPEC[collection_key])
    ]
    for collection_name, (filename, title, labels) in targets:
        md_path = KB_DOCS_DIR / filename
        if not md_path.exists():
            print(f"[SKIP] {collection_name}: {md_path} not found")
            continue
        chroma_setup.reset_db(collection_name=collection_name)
        invalidate_cache(collection_name)
        markdown = md_path.read_text(encoding="utf-8")
        result = content_manager.create_from_markdown(
            markdown=markdown,
            source_name=f"data/kb/{filename}",
            title=title,
            collection_name=collection_name,
            category_labels=labels,
        )
        print(f"  {collection_name:20} rebuilt -> {result.get('num_chunks', 0)} chunks")


def search(query: str, collection_name: str, n_results: int) -> None:
    results = content_manager.read(query, n_results=n_results, collection_name=collection_name)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    if not docs:
        print("No results.")
        return
    for doc, meta, dist in zip(docs, metas, dists):
        sim = 1 - dist
        print(f"\n--- sim={sim:.3f} | heading={meta.get('heading', '')!r} ---")
        print(doc[:300])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nexatel RAG knowledge-base admin CLI")
    parser.add_argument("--status", action="store_true", help="Show chunk counts per collection")
    parser.add_argument(
        "--rebuild",
        metavar="COLLECTION|all",
        choices=list(KB_SPEC.keys()) + ["all"],
        help="Wipe and re-ingest one collection (or 'all')",
    )
    parser.add_argument("--search", metavar="QUERY", help="Run a hybrid search against a collection")
    parser.add_argument(
        "--collection",
        default=chroma_setup.KB_COLLECTIONS["billing_policy"],
        choices=list(KB_SPEC.keys()),
        help="Which collection --search targets (default: billing_policy)",
    )
    parser.add_argument("--n-results", type=int, default=5)
    args = parser.parse_args()

    if args.rebuild:
        rebuild(args.rebuild)
    elif args.search:
        search(args.search, args.collection, args.n_results)
    else:
        status()
