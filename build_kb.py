"""
build_kb.py

Run this file to (re)build the entire Nexatel vector store: ingests each of
the 5 kb_docs/*.md knowledge-base documents into its own scoped ChromaDB
collection (see chroma_setup.KB_COLLECTIONS), with domain-relevant guided
category labels so retrieval/category filtering has useful signal from the
start.

    python build_kb.py            # (re)ingest all 5 KBs (idempotent upsert)
    python build_kb.py --reset    # wipe each collection first, then ingest

Each KB document is chunked, TF-IDF described, and category-tagged via the
existing content_manager.py pipeline (create_from_markdown), then upserted
into ChromaDB with the all-MiniLM-L6-v2 embedding model (chroma_setup.py).
"""

import argparse
from pathlib import Path

import chroma_setup
import content_manager

KB_DOCS_DIR = Path(__file__).parent / "kb_docs"

# collection name -> (markdown filename, human title, guided category labels)
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


def build(reset: bool = False) -> None:
    print(f"Nexatel KB build — persist dir: {Path(chroma_setup.PERSIST_DIRECTORY).resolve()}\n")

    summary = []
    for collection_name, (filename, title, labels) in KB_SPEC.items():
        md_path = KB_DOCS_DIR / filename
        if not md_path.exists():
            print(f"[SKIP] {collection_name}: {md_path} not found")
            continue

        if reset:
            chroma_setup.reset_db(collection_name=collection_name)

        markdown = md_path.read_text(encoding="utf-8")
        result = content_manager.create_from_markdown(
            markdown=markdown,
            source_name=f"kb_docs/{filename}",
            title=title,
            collection_name=collection_name,
            category_labels=labels,
        )
        summary.append((collection_name, result))
        print()

    print("=" * 60)
    print("Nexatel KB build summary")
    print("=" * 60)
    for collection_name, result in summary:
        count = chroma_setup.get_collection(collection_name).count()
        print(f"  {collection_name:20} {result.get('num_chunks', 0):3} chunks ingested "
              f"| {count:3} total in collection | lang={result.get('language', '?')}")
    print()
    print("Done. All 5 Nexatel RAG knowledge bases are ready in ChromaDB.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build/refresh the 5 Nexatel RAG knowledge-base collections.")
    parser.add_argument("--reset", action="store_true",
                        help="Wipe each collection before ingesting (clean rebuild).")
    args = parser.parse_args()
    build(reset=args.reset)
