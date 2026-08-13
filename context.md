# System Context & Architecture Overview

This document provides a comprehensive technical context of the NexaTel Communications Voice RAG (Retrieval-Augmented Generation) & Knowledge Base System based on `build_chroma_kb.py`, `inspect_db.py`, and `voice_rag_pipeline.py`.

---

## 1. Executive Summary & Architecture

The system is a local, privacy-centric, low-latency Voice Assistant RAG pipeline designed for **NexaTel Communications** customer care. It transforms unstructured telecom PDFs into a structured ChromaDB vector store and powers a multi-turn conversational AI voice agent using Groq LLM inference (`llama-3.1-8b-instant`) and local sentence embeddings (`all-MiniLM-L6-v2`).

```
[ Customer Voice Utterance (Transcribed Text) ]
                        │
                        ▼
   ┌──────────────────────────────────────────┐
   │  LLM #1: Transcription Normalization &   │
   │  NLU (Groq API - llama-3.1-8b-instant)   │
   └────────────────────┬─────────────────────┘
                        │ Outputs JSON: Intent, Normalized Query, Entities,
                        │ Confidence, Suggested Doc Type, Call End Flag
                        ▼
   ┌──────────────────────────────────────────┐
   │    Intent-Aware Hybrid RAG Retrieval     │
   │     (ChromaDB Vector Database)           │
   └────────────────────┬─────────────────────┘
                        │
                        ▼
            [ Confidence Gate Check ]
            Similarity < min_similarity (0.25)?
            ├── YES ──► [ Human Handoff & End Session ]
            └── NO  ──► [ LLM #2: Response Generator ]
                              │
                              ▼
                [ Guardrailed Spoken Response ]
```

---

## 2. File-by-File Technical Analysis

### A. [build_chroma_kb.py](file:///c:/RAG+TTS/build_chroma_kb.py) — Knowledge Base Builder
* **Role**: Data ingestion, text processing, specialized document chunking, vector embedding, and ChromaDB database creation.
* **Input Documents**:
  1. `NexaTel_Knowledge_Base.pdf` (`doc_type: knowledge_base`): Plans, pricing, FUP, billing, SLA, SIM management.
  2. `Telecom_Customer_Care_Response_Scripts.pdf` (`doc_type: response_script`): Customer care agent word-for-word scripts categorized by topic.
  3. `Telecom_Customer_Queries_and_Policy_Guide.pdf` (`doc_type: policy_guide`): TRAI regulation & policy context.
* **Chunking Strategies**:
  * **Section-Based Header Splitter** (`split_knowledge_base`): Uses regex pattern matching (`\d+(\.\d+)? Header`) to partition formal documentation sections while preserving section titles in metadata.
  * **Q&A-Style Regex Splitters** (`split_qa_style`, `split_response_scripts`, `split_policy_guide`): Splits on questions (`Q1 ...`, `Q: ...`) and captures heading categories.
  * **Recursive Paragraph Splitter** (`recursive_split`): Paragraph-aware fallback with chunk size (900–1200 chars) and overlap (150 chars).
* **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (runs locally via standard Chroma embedding function).
* **Output Storage**: Persistent Chroma DB located at `./chroma_db` under collection `nexatel_kb`.

---

### B. [inspect_db.py](file:///c:/RAG+TTS/inspect_db.py) — Database Inspector & Debug Tool
* **Role**: Command-Line Interface (CLI) for querying, filtering, inspecting, and sanity-testing the Chroma vector store without altering code.
* **Key Capabilities**:
  * `--info`: Collection overview, overall chunk count, and distribution breakdown across `doc_type` and `source_file`.
  * `--peek [N]`: Peeks at N sample chunks (default: 5).
  * `--list`: Lists documents with metadata.
  * `--filter key=value`: Restricts chunk listing or search to specific metadata criteria (e.g., `doc_type=response_script`).
  * `--query "text"`: Runs cosine similarity semantic search with distance & similarity scores.
  * `--id "chunk_id"`: Fetches precise chunks by unique ID.

---

### C. [voice_rag_pipeline.py](file:///c:/RAG+TTS/voice_rag_pipeline.py) — Multi-Turn Voice RAG Pipeline
* **Role**: Execution engine handling live, simulated voice assistant call interactions in a multi-turn session loop.
* **Pipeline Workflow**:
  1. **NLU & Transcription Normalization (LLM #1)**:
     * Receives raw transcribed customer utterance along with recent conversation history (`max_history_turns=6`).
     * Resolves co-references ("that plan", "the same issue").
     * Produces structured JSON containing `language`, `intent`, `normalized_query`, `entities`, `confidence`, `suggested_doc_type`, and `call_end_requested`.
  2. **Call Termination Check**:
     * If `call_end_requested: true`, generates a quick closing message via LLM and gracefully terminates the session.
  3. **Intent-Aware Hybrid Retrieval**:
     * Queries ChromaDB using `normalized_query`. Applies metadata filtering based on `suggested_doc_type`, falling back to unfiltered retrieval if filtered search yields no results.
  4. **Confidence Gate & Human Handoff**:
     * Evaluates best match similarity (`1 - distance`).
     * If best match similarity < `min_similarity` (default `0.25`), the call triggers an immediate fallback message: *"I want to make sure I get this right for you..."* and transfers the call to a human agent, ending the AI session.
  5. **Guardrailed Response Generation (LLM #2)**:
     * Generates concise, natural, spoken-ready responses strictly grounded in the top-3 retrieved snippets.
     * Enforces **10 Guardrails**: Grounding (no hallucinated rates/plans), Insufficient Context Handling, Topic Scope, No Prompt Leakage, Prompt Injection Resistance, Sensitive Data Safety (PINs/cards), No Fabricated Ticket IDs, Channel Realism (KYC steps), Escalation Handling, and Spoken-Language Tone (no markdown headers/bullets).

---

## 3. Configuration & Parameters

| Parameter | Default Value | Description |
|---|---|---|
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq LLM model used for NLU and generation |
| `PERSIST_DIR` | `./chroma_db` | Storage path for vector database |
| `COLLECTION_NAME` | `nexatel_kb` | Name of Chroma collection |
| `TOP_K` | `3` | Number of context snippets retrieved |
| `MIN_SIMILARITY` | `0.25` | Minimum similarity threshold before triggering human handoff |
| `MAX_HISTORY_TURNS` | `6` | Max past exchange turns retained in LLM context window |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model for vector embeddings |

---

## 4. Dependencies & Prerequisites

* **Python Libraries**: `chromadb`, `pypdf`, `sentence-transformers`, `requests`
* **API Access**: Groq API Key (`GROQ_API_KEY` environment variable)
* **Local Data**: PDF files stored in `./pdfs`:
  * `NexaTel_Knowledge_Base.pdf`
  * `Telecom_Customer_Care_Response_Scripts.pdf`
  * `Telecom_Customer_Queries_and_Policy_Guide.pdf`
