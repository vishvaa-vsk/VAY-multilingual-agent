"""
build_chroma_kb.py
-------------------
Builds a local, persistent ChromaDB vector store from NexaTel's 3 knowledge PDFs:

  1. NexaTel_Knowledge_Base.pdf                       -> doc_type: knowledge_base
  2. Telecom_Customer_Care_Response_Scripts.pdf        -> doc_type: response_script
  3. Telecom_Customer_Queries_and_Policy_Guide.pdf     -> doc_type: policy_guide

Each chunk is stored with metadata (source file, doc_type, section/category,
query heading if applicable, chunk index) so the RAG retrieval step can filter
and cite results properly.

--------------------------------------------------------------------------
SETUP (run once)
--------------------------------------------------------------------------
    python -m venv venv
    source venv/bin/activate        # Windows: venv\\Scripts\\activate
    pip install chromadb pypdf

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
1. Put the 3 PDFs in a folder (default: ./pdfs) next to this script, named
   exactly as above, OR pass a custom folder with --pdf_dir.
2. Run:
    python build_chroma_kb.py
3. This creates a persistent Chroma store at ./chroma_db (change with
   --persist_dir). Re-running is safe — the collection is cleared and
   rebuilt each time so you don't get duplicate chunks.

The embedding model runs fully locally via Chroma's default embedding
function (sentence-transformers/all-MiniLM-L6-v2, auto-downloaded once from
Hugging Face on first run, then cached). No API key needed.
--------------------------------------------------------------------------
"""

import argparse
import os
import re
import sys
import uuid

from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions


# --------------------------------------------------------------------------
# Config: filename -> doc_type mapping
# --------------------------------------------------------------------------
FILE_DOC_TYPES = {
    "NexaTel_Knowledge_Base.pdf": "knowledge_base",
    "Telecom_Customer_Care_Response_Scripts.pdf": "response_script",
    "Telecom_Customer_Queries_and_Policy_Guide.pdf": "policy_guide",
}

COLLECTION_NAME = "nexatel_kb"
CHUNK_SIZE = 900          # characters, for the recursive fallback splitter
CHUNK_OVERLAP = 150


# --------------------------------------------------------------------------
# PDF text extraction
# --------------------------------------------------------------------------
def extract_pdf_text(path: str) -> str:
    """Extract raw text from a PDF, page by page, joined with page markers."""
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)


def clean_text(text: str) -> str:
    """Normalize whitespace/tabs that pypdf leaves behind in table-heavy PDFs."""
    text = text.replace("\t", " ")
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------
# Chunking strategies (one per document "shape")
# --------------------------------------------------------------------------
def recursive_split(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Simple fallback splitter: paragraph-aware, falls back to hard cuts."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 1 <= chunk_size:
            current = (current + "\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            if len(para) > chunk_size:
                # hard-cut an oversized paragraph with overlap
                start = 0
                while start < len(para):
                    end = start + chunk_size
                    chunks.append(para[start:end])
                    start = end - overlap
                current = ""
            else:
                current = para
    if current:
        chunks.append(current)
    return chunks


def split_knowledge_base(text: str):
    """
    Split NexaTel_Knowledge_Base.pdf by its numbered section headers
    (e.g. '3.2 Fair Usage Policy (FUP)') so each chunk carries a section
    title as metadata. Falls back to recursive_split for long sections.
    """
    header_pattern = re.compile(
        r"(?m)^(?P<num>\d+(?:\.\d+)?)\s+(?P<title>[A-Z][A-Za-z0-9 ,&/\-\(\)]{2,80})\s*$"
    )

    matches = list(header_pattern.finditer(text))
    records = []  # (section_title, section_text)

    if not matches:
        records.append(("NexaTel Knowledge Base", text))
    else:
        # anything before the first header
        if matches[0].start() > 0:
            preamble = text[: matches[0].start()].strip()
            if preamble:
                records.append(("Introduction", preamble))
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            title = f"{m.group('num')} {m.group('title')}".strip()
            body = text[start:end].strip()
            records.append((title, body))

    chunks = []  # (section_title, chunk_text)
    for title, body in records:
        for piece in recursive_split(body):
            chunks.append((title, piece))
    return chunks


def split_qa_style(text: str, qa_pattern: str, category_pattern: str = None):
    """
    Generic splitter for Q&A-style documents (response scripts, policy guide).
    Each match of qa_pattern starts a new chunk. Optionally tracks the most
    recent category/section heading seen (e.g. '1. Billing & Payments').
    """
    qa_re = re.compile(qa_pattern)
    cat_re = re.compile(category_pattern) if category_pattern else None

    matches = list(qa_re.finditer(text))
    if not matches:
        return [("General", None, text)]

    records = []
    current_category = "General"

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].strip()

        if cat_re:
            preceding = text[max(0, start - 400): start]
            cat_matches = list(cat_re.finditer(preceding))
            if cat_matches:
                current_category = cat_matches[-1].group(0).strip()

        # first line of the block = the question/heading
        first_line = block.split("\n", 1)[0].strip()
        records.append((current_category, first_line, block))

    return records


def split_response_scripts(text: str):
    """Telecom_Customer_Care_Response_Scripts.pdf: split on 'Q<number> "..."'."""
    records = split_qa_style(
        text,
        qa_pattern=r"(?m)^Q\d+\s",
        category_pattern=r"(?m)^\d{1,2}\.\s+[A-Z][A-Za-z ,&]+$",
    )
    chunks = []
    for category, question, block in records:
        for piece in recursive_split(block, chunk_size=1200, overlap=150):
            chunks.append((category, question, piece))
    return chunks


def split_policy_guide(text: str):
    """Telecom_Customer_Queries_and_Policy_Guide.pdf: split on 'Q: ...' lines."""
    records = split_qa_style(
        text,
        qa_pattern=r"(?m)^Q:\s",
        category_pattern=r"(?m)^\d\.\s+[A-Z][A-Za-z &,]+$",
    )
    chunks = []
    for category, question, block in records:
        for piece in recursive_split(block, chunk_size=1200, overlap=150):
            chunks.append((category, question, piece))
    return chunks


# --------------------------------------------------------------------------
# Main ingestion pipeline
# --------------------------------------------------------------------------
def build_documents(pdf_dir: str):
    """
    Returns three parallel lists: ids, documents (text), metadatas
    ready to hand to a Chroma collection.
    """
    ids, documents, metadatas = [], [], []

    for filename, doc_type in FILE_DOC_TYPES.items():
        path = os.path.join(pdf_dir, filename)
        if not os.path.exists(path):
            print(f"  [SKIP] {filename} not found in {pdf_dir}")
            continue

        print(f"  Extracting {filename} ...")
        raw = extract_pdf_text(path)
        text = clean_text(raw)

        if doc_type == "knowledge_base":
            section_chunks = split_knowledge_base(text)
            for idx, (section, chunk_text) in enumerate(section_chunks):
                chunk_text = chunk_text.strip()
                if len(chunk_text) < 20:
                    continue
                ids.append(f"{doc_type}-{idx}-{uuid.uuid4().hex[:8]}")
                documents.append(chunk_text)
                metadatas.append({
                    "source_file": filename,
                    "doc_type": doc_type,
                    "section": section,
                    "chunk_index": idx,
                })

        elif doc_type == "response_script":
            qa_chunks = split_response_scripts(text)
            for idx, (category, question, chunk_text) in enumerate(qa_chunks):
                chunk_text = chunk_text.strip()
                if len(chunk_text) < 20:
                    continue
                ids.append(f"{doc_type}-{idx}-{uuid.uuid4().hex[:8]}")
                documents.append(chunk_text)
                metadatas.append({
                    "source_file": filename,
                    "doc_type": doc_type,
                    "category": category,
                    "query": (question or "")[:150],
                    "chunk_index": idx,
                })

        elif doc_type == "policy_guide":
            qa_chunks = split_policy_guide(text)
            for idx, (category, question, chunk_text) in enumerate(qa_chunks):
                chunk_text = chunk_text.strip()
                if len(chunk_text) < 20:
                    continue
                ids.append(f"{doc_type}-{idx}-{uuid.uuid4().hex[:8]}")
                documents.append(chunk_text)
                metadatas.append({
                    "source_file": filename,
                    "doc_type": doc_type,
                    "category": category,
                    "query": (question or "")[:150],
                    "chunk_index": idx,
                })

        print(f"    -> {sum(1 for m in metadatas if m['source_file'] == filename)} chunks")

    return ids, documents, metadatas


def main():
    parser = argparse.ArgumentParser(description="Build local ChromaDB KB for NexaTel voice assistant.")
    parser.add_argument("--pdf_dir", default="./pdfs", help="Folder containing the 3 source PDFs.")
    parser.add_argument("--persist_dir", default="./chroma_db", help="Folder to persist the Chroma DB.")
    parser.add_argument("--collection", default=COLLECTION_NAME, help="Chroma collection name.")
    parser.add_argument("--batch_size", type=int, default=64, help="Embedding/insert batch size.")
    args = parser.parse_args()

    if not os.path.isdir(args.pdf_dir):
        print(f"ERROR: pdf_dir '{args.pdf_dir}' does not exist. "
              f"Create it and place the 3 PDFs inside, or pass --pdf_dir.")
        sys.exit(1)

    print("Step 1/3: Extracting & chunking PDFs")
    ids, documents, metadatas = build_documents(args.pdf_dir)

    if not documents:
        print("ERROR: No chunks produced. Check that the PDF filenames match exactly:")
        for f in FILE_DOC_TYPES:
            print(f"   - {f}")
        sys.exit(1)

    print(f"\nTotal chunks to embed: {len(documents)}")

    print("\nStep 2/3: Initializing local persistent ChromaDB client")
    client = chromadb.PersistentClient(path=args.persist_dir)

    # Local, free, no-API-key embedding model (downloaded once, cached after).
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Rebuild clean each run to avoid duplicate/stale chunks.
    try:
        client.delete_collection(args.collection)
        print(f"  (cleared existing collection '{args.collection}')")
    except Exception:
        pass

    collection = client.create_collection(
        name=args.collection,
        embedding_function=embed_fn,
        metadata={"description": "NexaTel customer-care knowledge base for RAG voice assistant"},
    )

    print("\nStep 3/3: Embedding & storing chunks in ChromaDB")
    batch_size = args.batch_size
    for i in range(0, len(documents), batch_size):
        batch_ids = ids[i:i + batch_size]
        batch_docs = documents[i:i + batch_size]
        batch_meta = metadatas[i:i + batch_size]
        collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_meta)
        print(f"  Embedded {min(i + batch_size, len(documents))}/{len(documents)} chunks")

    print(f"\nDone. Collection '{args.collection}' now has {collection.count()} chunks.")
    print(f"Persisted at: {os.path.abspath(args.persist_dir)}")

    # Quick sanity check query
    print("\n--- Sanity check query: 'how do I get a duplicate SIM' ---")
    result = collection.query(query_texts=["how do I get a duplicate SIM"], n_results=3)
    for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        print(f"\n[{meta.get('doc_type')} | {meta.get('section') or meta.get('category')}] (distance={dist:.4f})")
        print(doc[:200].replace("\n", " ") + "...")


if __name__ == "__main__":
    main()
