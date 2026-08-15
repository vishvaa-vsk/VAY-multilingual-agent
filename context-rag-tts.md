# VAY — Multilingual GenAI Voice Assistant: Project Context

> **Last updated:** 2026-08-14  
> **Purpose:** Living reference document for any developer, AI agent, or team member joining this project mid-stream. Covers the full architecture, every file's role, and all changes made during the Aug 12–18 hackathon build window.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Technology Stack](#2-technology-stack)
3. [Full Directory Structure](#3-full-directory-structure)
4. [Pipeline Architecture](#4-pipeline-architecture)
5. [Module-by-Module Reference](#5-module-by-module-reference)
   - [5.1 ASR](#51-asr)
   - [5.2 Audio / VAD](#52-audio--vad)
   - [5.3 Normalization](#53-normalization)
   - [5.4 RAG](#54-rag)
   - [5.5 Graph (LangGraph Orchestration)](#55-graph-langgraph-orchestration)
   - [5.6 Tools (Backend / DB)](#56-tools-backend--db)
   - [5.7 TTS](#57-tts)
   - [5.8 Handoff](#58-handoff)
   - [5.9 UI](#59-ui)
   - [5.10 Scripts](#510-scripts)
   - [5.11 Tests](#511-tests)
6. [Data Models (types.py)](#6-data-models-typespy)
7. [Configuration](#7-configuration)
8. [Environment Variables](#8-environment-variables)
9. [Changes Made — Aug 14 Session](#9-changes-made--aug-14-session)
10. [Key Design Decisions](#10-key-design-decisions)
11. [Known Gotchas](#11-known-gotchas)
12. [How to Run](#12-how-to-run)

---

## 1. Project Overview

**Use Case:** Multilingual GenAI Voice Assistant for Telecom Customer Care  
**Operator:** Nexatel Communications (mock)  
**Hackathon:** Velammal-AIA Partnership / Cognizant, Use Case #15 of 18  
**Timeline:** Aug 12–18, 2026 (7 days), evaluation Aug 19  

The assistant handles telecom self-service across Tamil, Hindi, English, and 15+ other languages:
- Bill queries, plan changes, complaints, network coverage
- Graceful handoff to a human agent when the AI cannot confidently resolve the query
- Aggressive/abusive callers receive a warning (first offence) or call termination (second offence) — **not** a human handoff

**Entry point for the RAG + LLM pipeline:** `scripts/run_assistant.py`  
**Entry point for the web UI:** `python -m vay.ui.app`

---

## 2. Technology Stack

| Component | Technology | Notes |
|---|---|---|
| **ASR (Tier 1)** | `ai4bharat/indic-conformer-600m-multilingual` | Tamil + Hindi only. Must use `AutoModel`, NOT `pipeline()` |
| **ASR (Tier 2)** | `openai/whisper-large-v3-turbo` | English + 90 fallback languages |
| **Language ID** | Whisper encoder-only pass | No separate LID model — deliberate cost/latency decision |
| **VAD** | Silero / silence threshold | ~600–700ms silence = utterance boundary |
| **Normalization** | Groq LLM (llama-3.1-8b-instant) | Code-switch cleanup, intent tagging, entity extraction |
| **Vector DB** | ChromaDB (cosine, HNSW) | 5 scoped collections (billing / plans / complaints / coverage / compliance) |
| **Keyword search** | BM25 (rank-bm25) | Hybrid retrieval: BM25 + vector, top-k reranked |
| **Embedding** | `all-MiniLM-L6-v2` (SentenceTransformers) | ChromaDB embedding function |
| **LLM** | Groq API — `llama-3.1-8b-instant` | Orchestrator + 4 sub-agents + normalization |
| **Orchestration** | LangGraph (StateGraph) | Conditional branching, multi-turn state |
| **TTS** | edge-tts (Microsoft Neural) | 18 languages, neural quality — replaced gTTS Aug 14 |
| **Audio playback** | playsound3 | Cross-platform, graceful fallback |
| **Package manager** | uv | Python 3.11, `.venv` managed by uv |
| **Web UI** | Gradio | Demo interface |

---

## 3. Full Directory Structure

```
VAY-multilingual-agent/
│
├── .env                          # GROQ_API_KEY, GROQ_MODEL (not committed)
├── .python-version               # Pins Python 3.11
├── pyproject.toml                # uv project config + all dependencies
├── uv.lock                       # Locked dependency tree
├── README.md                     # Setup and developer guide
├── project_context.md            # Locked architecture source of truth
├── context.md                    # THIS FILE — living dev context
├── handoff_log.jsonl             # Appended by human_handoff_node at runtime
├── chroma_db/                    # ChromaDB persistent storage (5 collections)
├── data/                         # Raw KB docs, audio datasets
│
├── scripts/
│   ├── run_assistant.py          # Main call loop (transcript → LLM → TTS)
│   ├── build_kb.py               # Ingest KB markdown docs into ChromaDB
│   ├── manage_kb.py              # KB admin: list, delete, rebuild collections
│   └── manage_db.py              # CustomerDB admin: seed, inspect, reset
│
├── tests/
│   ├── __init__.py
│   ├── test_types.py             # Pydantic model creation tests
│   ├── test_routing.py           # ASR tier routing tests (Tamil→Indic, en→Whisper)
│   └── test_rag.py               # HybridRetriever initialization tests
│
└── src/
    └── vay/                      # Main Python package
        ├── __init__.py
        ├── py.typed               # PEP 561 marker
        ├── config.py              # Pydantic settings (model paths, thresholds)
        ├── types.py               # Core Pydantic data models
        │
        ├── audio/                 # Voice Activity Detection + audio utilities
        │   ├── __init__.py
        │   ├── vad.py             # Silence-based utterance boundary detection
        │   └── utils.py           # load_audio() helper (WAV → torch tensor)
        │
        ├── asr/                   # Automatic Speech Recognition
        │   ├── __init__.py
        │   ├── base.py            # ASREngine abstract base class
        │   ├── indic.py           # IndicConformer wrapper (Tier 1: ta, hi)
        │   ├── whisper.py         # Whisper wrapper (Tier 2: en + fallback)
        │   └── router.py          # Language detection + tier routing
        │
        ├── normalization/         # LLM transcript cleanup
        │   └── pass_llm.py        # [CHANGED Aug 14] Real LLM + prev-turn context
        │
        ├── rag/                   # Hybrid RAG retrieval
        │   ├── __init__.py
        │   ├── vector_store.py    # ChromaDB client, 5 KB collections, caching
        │   ├── bm25.py            # BM25 keyword index
        │   ├── tfidf.py           # TF-IDF scoring utilities
        │   ├── retriever.py       # [CHANGED Aug 14] HybridRetriever + RAG tools
        │   ├── manager.py         # Thin facade: read() delegates to manager_read
        │   ├── manager_read.py    # ChromaDB query execution
        │   ├── manager_create.py  # Collection + chunk creation
        │   ├── manager_ingest.py  # Markdown parsing → chunking → embedding
        │   ├── manager_admin.py   # Admin ops: delete, list, stats
        │   ├── chunking.py        # Text chunking strategies
        │   ├── parsers.py         # Markdown / plain-text parsers
        │   └── categorizer.py     # Auto-categorizes chunks to the right collection
        │
        ├── graph/                 # LangGraph orchestration
        │   ├── __init__.py
        │   ├── state.py           # [CHANGED Aug 14] GraphState TypedDict
        │   ├── nodes.py           # Legacy node stubs (ASR → Norm → RAG flow)
        │   ├── workflow.py        # [CHANGED Aug 14] Graph definition + edges
        │   ├── core_utils.py      # [CHANGED Aug 14] All prompts, templates, helpers
        │   ├── utils.py           # [CHANGED Aug 14] Facade re-exporting core_utils
        │   ├── tool_agent.py      # Bounded tool-calling loop (run_tool_agent)
        │   └── nodes/             # Production node implementations
        │       ├── __init__.py
        │       ├── orchestrator.py  # [CHANGED Aug 14] Orchestrator + sub-agent runner
        │       ├── agents.py        # billing/plans/complaints/coverage node wrappers
        │       └── utils.py         # [CHANGED Aug 14] guardrail/handoff/warning/TTS nodes
        │
        ├── tools/                 # Backend "API" tools + CustomerDB
        │   ├── billing.py         # build_billing_tools(session) — balance, bills, payment
        │   ├── plans.py           # build_plans_tools(session) — list, compare, changePlan
        │   ├── complaints.py      # build_complaints_tools(session) — tickets, SLA
        │   ├── coverage.py        # build_coverage_tools(session) — coverage, APN, outage
        │   ├── session.py         # [CHANGED Aug 14] SessionContext, consent logic
        │   ├── db_queries.py      # Raw SQLite queries (getBalance, getSubscription, etc.)
        │   ├── db_schema.py       # Schema definitions for nexatel_customers.db
        │   ├── db_seed_data.py    # Seed data for the mock customer database
        │   └── nexatel_customers.db  # SQLite mock customer database
        │
        ├── tts/                   # Text-to-Speech
        │   └── engine.py          # [CHANGED Aug 14] edge-tts, 18 languages
        │
        ├── handoff/               # Human escalation queue
        │   └── queue.py           # HandoffQueueManager (appends to handoff_log.jsonl)
        │
        └── ui/                    # Demo web interface
            └── app.py             # Gradio app (stub — to be expanded)
```

---

## 4. Pipeline Architecture

```
USER SPEAKS
  │
  ▼
[VAD] — Silero / silence threshold ~650ms
  │
  ▼
[Language ID] — Whisper encoder-only pass (no separate LID model)
  │
  ├─ Tamil / Hindi ──► [IndicConformer ASR] ──────────────────────────┐
  │                    ai4bharat/indic-conformer-600m-multilingual      │
  │                    AutoModel(wav_tensor, lang_code, "ctc")          │
  │                                                                     ▼
  └─ English / Other ► [Whisper ASR] + [Hallucination Filter] ────► [Raw Transcript]
                        openai/whisper-large-v3-turbo
  │
  ▼
[Normalization LLM Pass]  ← NEW: includes previous_intent + previous_normalized
  Groq LLM (llama-3.1-8b-instant)
  Outputs: { normalized_text, intent, entities, confidence, detected_language }
  │
  ▼
[ORCHESTRATOR NODE] (LangGraph)
  Groq LLM — NLU + routing
  Outputs JSON: { language, intent, route, normalized_query,
                  entities, confidence, sensitive, aggressive,     ← NEW field
                  call_end_requested }
  │
  ├─ aggressive=true, count=1 ──────────────────────► [WARNING NODE]
  │                                                     Speaks localized warning
  │                                                     (18 languages, hand-written)
  │                                                     → TTS → END
  │
  ├─ aggressive=true, count≥2 ──────────────────────► [CLOSING NODE]
  │                                                     Speaks call-cut message
  │                                                     → TTS → END
  │
  ├─ sensitive=true ────────────────────────────────► [HUMAN HANDOFF NODE]
  │  (billing dispute / cancellation / fraud)           Logs to handoff_log.jsonl
  │                                                     → TTS → END
  │
  ├─ route=unclear / low confidence (repeated) ──────► [HUMAN HANDOFF NODE]
  │
  ├─ route=unclear / low confidence (first time) ────► [CLARIFY NODE]
  │                                                     Fixed localized re-prompt
  │                                                     → TTS → END
  │
  └─ valid route ─────────────────────────────────────► SUB-AGENT
       │
       ├─ billing   ──► [BILLING NODE]   + billing_policy RAG
       ├─ plans     ──► [PLANS NODE]     + product_catalog RAG
       ├─ complaints──► [COMPLAINTS NODE]+ support_faq RAG
       └─ coverage  ──► [COVERAGE NODE]  + technical_kb RAG
             │
             ▼
       Tool-calling loop (max 6 iterations):
         - Domain tools (DB: getBalance, listPlans, logTicket, etc.)
         - Scoped RAG tool (ChromaDB similarity search)
         - Account context pre-injected in system prompt  ← NEW
         - Domain-switch history trimming                 ← NEW
             │
             ▼
       [GUARDRAIL NODE]
         - Low retrieval score → handoff
         - Customer requested human → handoff
         - Uncertainty + low score (BOTH needed) → handoff   ← CHANGED
         - PII leak in draft → handoff
         - Compliance check for consent-trigger keywords     ← NEW
             │
         ┌───┴────────────────┐
         ▼                    ▼
  [HUMAN HANDOFF]      [TTS NODE]
                         edge-tts — 18 neural voices  ← CHANGED (was gTTS)
                         playsound3 playback
                           │
                           ▼
                        AUDIO OUTPUT
                           │
                           └──── loop back to VAD for next utterance
```

---

## 5. Module-by-Module Reference

### 5.1 ASR

| File | Role |
|---|---|
| [`asr/base.py`](file:///c:/sample/VAY-multilingual-agent/src/vay/asr/base.py) | Abstract `ASREngine` base class |
| [`asr/indic.py`](file:///c:/sample/VAY-multilingual-agent/src/vay/asr/indic.py) | `IndicConformerEngine` — loads via `AutoModel.from_pretrained()`, NOT `pipeline()`. Supports `ta`, `hi`. Decodes with CTC. |
| [`asr/whisper.py`](file:///c:/sample/VAY-multilingual-agent/src/vay/asr/whisper.py) | `WhisperEngine` — `openai/whisper-large-v3-turbo`. Handles English + 90 language fallback. Includes hallucination/repetition filter. |
| [`asr/router.py`](file:///c:/sample/VAY-multilingual-agent/src/vay/asr/router.py) | `ASRRouter` — runs Whisper's encoder-only language detection, then routes `{ta, hi}` to IndicConformer and everything else to Whisper. |

**⚠️ Critical:** IndicConformer does NOT support English. Never route English through it.

---

### 5.2 Audio / VAD

| File | Role |
|---|---|
| [`audio/vad.py`](file:///c:/sample/VAY-multilingual-agent/src/vay/audio/vad.py) | Silence-based utterance boundary detection (~650ms silence threshold) |
| [`audio/utils.py`](file:///c:/sample/VAY-multilingual-agent/src/vay/audio/utils.py) | `load_audio(path)` — loads WAV/MP3 to a `[1, N]` float32 torch tensor at 16kHz mono |

---

### 5.3 Normalization

| File | Role |
|---|---|
| [`normalization/pass_llm.py`](file:///c:/sample/VAY-multilingual-agent/src/vay/normalization/pass_llm.py) | **[CHANGED Aug 14]** Real Groq LLM normalization pass. Cleans ASR output, resolves code-switching (Tanglish/Hinglish), extracts intent + entities. Injects `previous_intent` + `previous_normalized` for coreference resolution. Falls back to keyword heuristics if LLM unavailable. |

**Input:** `(raw_transcript: str, language: str, previous_intent?, previous_normalized?)`  
**Output:** `StructuredTranscript` — normalized_text, intent, entities, confidence, detected_language

---

### 5.4 RAG

#### 5 Scoped Knowledge Base Collections

| Collection Key | ChromaDB Name | Agent that uses it |
|---|---|---|
| `billing_policy` | `billing_policy` | Billing Agent |
| `product_catalog` | `product_catalog` | Plans Agent |
| `support_faq` | `support_faq` | Complaints Agent |
| `technical_kb` | `technical_kb` | Coverage Agent |
| `compliance_policy` | `compliance_policy` | Guardrail layer (all agents) |

#### RAG Files

| File | Role |
|---|---|
| [`rag/vector_store.py`](file:///c:/sample/VAY-multilingual-agent/src/vay/rag/vector_store.py) | ChromaDB client + collection management. Defines `KB_COLLECTIONS`. Module-level cache prevents re-loading embeddings per call. |
| [`rag/retriever.py`](file:///c:/sample/VAY-multilingual-agent/src/vay/rag/retriever.py) | **[CHANGED Aug 14]** `RetrievalTracker`, `_format_hits()`, `build_*_rag_tool()` factories (one per KB collection), `HybridRetriever` class. Default collection fixed to `billing_policy`. Confidence threshold: `0.75`. |
| [`rag/manager.py`](file:///c:/sample/VAY-multilingual-agent/src/vay/rag/manager.py) | Thin facade: `read(query, n_results, collection_name)` |
| [`rag/manager_read.py`](file:///c:/sample/VAY-multilingual-agent/src/vay/rag/manager_read.py) | ChromaDB `.query()` execution |
| [`rag/manager_create.py`](file:///c:/sample/VAY-multilingual-agent/src/vay/rag/manager_create.py) | Collection creation + chunk upsert |
| [`rag/manager_ingest.py`](file:///c:/sample/VAY-multilingual-agent/src/vay/rag/manager_ingest.py) | Markdown → chunks → embeddings pipeline |
| [`rag/manager_admin.py`](file:///c:/sample/VAY-multilingual-agent/src/vay/rag/manager_admin.py) | Delete, stats, list operations |
| [`rag/bm25.py`](file:///c:/sample/VAY-multilingual-agent/src/vay/rag/bm25.py) | BM25 keyword index |
| [`rag/tfidf.py`](file:///c:/sample/VAY-multilingual-agent/src/vay/rag/tfidf.py) | TF-IDF utilities |
| [`rag/chunking.py`](file:///c:/sample/VAY-multilingual-agent/src/vay/rag/chunking.py) | Text chunking (fixed-size + overlap) |
| [`rag/parsers.py`](file:///c:/sample/VAY-multilingual-agent/src/vay/rag/parsers.py) | Markdown / plain-text parsers |
| [`rag/categorizer.py`](file:///c:/sample/VAY-multilingual-agent/src/vay/rag/categorizer.py) | Routes ingested docs to the correct scoped collection |

---

### 5.5 Graph (LangGraph Orchestration)

#### State

**File:** [`graph/state.py`](file:///c:/sample/VAY-multilingual-agent/src/vay/graph/state.py) **[CHANGED Aug 14]**

```python
class GraphState(TypedDict, total=False):
    # Input (set once per turn by caller)
    phone_number: str
    language: str          # Updated each turn by orchestrator (language switches work)
    transcript: str        # Raw ASR output for this turn
    conversation_history: list  # LangChain message list — full call history

    # Orchestrator outputs
    intent: str
    entities: dict
    normalized_query: str  # Clean English query for RAG
    nlu_confidence: float
    sensitive: bool        # fraud / billing-dispute / cancellation → human handoff
    route: str             # billing | plans | complaints | coverage | unclear
    call_end_requested: bool
    session: Any           # SessionContext — created once per call, carried across turns

    # NEW Aug 14
    aggressive_count: int  # Incremented per abusive turn; warning on 1, cut on 2+
    previous_route: str    # Prior turn's route — used for domain-switch history trimming
    warning_reply: str     # Pre-built localized warning/call-cut text

    unclear_escalate: bool

    # Sub-agent outputs
    retrieval_score: float
    draft_reply: str
    final_reply: str

    # Handoff
    handoff: bool
    handoff_reason: str

    # Misc
    show_debug: bool
    min_similarity: float
```

#### Graph Workflow

**File:** [`graph/workflow.py`](file:///c:/sample/VAY-multilingual-agent/src/vay/graph/workflow.py) **[CHANGED Aug 14]**

```
START
  └─► orchestrator
        ├─► billing ──────────────► guardrail ─► human_handoff ─► tts ─► END
        ├─► plans ────────────────► guardrail ─► tts ─► END
        ├─► complaints ───────────► guardrail
        ├─► coverage ─────────────► guardrail
        ├─► warning (NEW) ────────────────────────────► tts ─► END
        ├─► human_handoff ────────────────────────────► tts ─► END
        ├─► clarify ──────────────────────────────────► tts ─► END
        └─► closing ──────────────────────────────────► tts ─► END
```

#### Node Files

| File | Nodes | Role |
|---|---|---|
| [`graph/nodes/orchestrator.py`](file:///c:/sample/VAY-multilingual-agent/src/vay/graph/nodes/orchestrator.py) **[CHANGED]** | `orchestrator_node`, `_run_subagent` | NLU + routing, aggressive handling, account context pre-fetch, domain-switch history trim |
| [`graph/nodes/agents.py`](file:///c:/sample/VAY-multilingual-agent/src/vay/graph/nodes/agents.py) | `billing_node`, `plans_node`, `complaints_node`, `coverage_node` | Thin wrappers calling `_run_subagent` with the right tools + RAG builder |
| [`graph/nodes/utils.py`](file:///c:/sample/VAY-multilingual-agent/src/vay/graph/nodes/utils.py) **[CHANGED]** | `guardrail_node`, `human_handoff_node`, `warning_node` (NEW), `clarify_node`, `closing_node`, `tts_node`, `route_after_orchestrator`, `route_after_guardrail` | All utility nodes + routing functions |
| [`graph/nodes.py`](file:///c:/sample/VAY-multilingual-agent/src/vay/graph/nodes.py) **[CHANGED]** | Legacy: `vad_node`, `asr_node`, `normalization_node`, `rag_node`, `handoff_gate_node`, `llm_generation_node`, `tts_node`, `human_handoff_node` | Used by the legacy ASR→Norm→RAG flow (not the main LangGraph agent path) |
| [`graph/tool_agent.py`](file:///c:/sample/VAY-multilingual-agent/src/vay/graph/tool_agent.py) | `run_tool_agent` | Bounded tool-calling loop (max 6 iterations). Handles STOP_AND_SAY sentinel, tool errors, and wrap-up prompt. |
| [`graph/core_utils.py`](file:///c:/sample/VAY-multilingual-agent/src/vay/graph/core_utils.py) **[CHANGED]** | Constants, prompts, templates | All system prompts, all localized message templates, helpers |
| [`graph/utils.py`](file:///c:/sample/VAY-multilingual-agent/src/vay/graph/utils.py) **[CHANGED]** | Facade | Re-exports everything from core_utils + run_tool_agent |

#### System Prompts (in core_utils.py)

**`ORCHESTRATOR_SYSTEM_PROMPT`** — Instructs the LLM to output strict JSON:
```json
{
  "language": "<ISO 639-1>",
  "intent": "<snake_case>",
  "route": "billing|plans|complaints|coverage|unclear",
  "normalized_query": "<clean English question>",
  "entities": {},
  "confidence": 0.0,
  "sensitive": false,
  "aggressive": false,
  "call_end_requested": false
}
```
> **Key distinction:** `sensitive=true` → human handoff. `aggressive=true` → warning/cut. Never conflated.

**`SUBAGENT_SYSTEM_PROMPT_TEMPLATE`** — Has `{account_context}` block (pre-fetched) and explicit rules:
- Telecom terms (`1 GB`, `4G`, `VoLTE`, `SIM`) stay in **English** even in Tamil/Hindi replies
- Never invent IDs, never escalate for things tools can resolve
- Account context is provided upfront — don't call tools redundantly for it

#### Message Templates (all localized, all 18 languages)

| Template Dict | Purpose |
|---|---|
| `HANDOFF_MESSAGE_TEMPLATES` | What to say when handing off to a human |
| `TOOL_LOOP_FAILURE_TEMPLATES` | When tool loop exhausts without a good answer |
| `CLOSING_FALLBACK_TEMPLATES` | Fallback closing line |
| `CLARIFY_TEMPLATES` | Re-prompt for unclear utterances |
| `AGGRESSIVE_WARNING_TEMPLATES` | **NEW** First-offence warning (18 languages) |
| `CALL_CUT_TEMPLATES` | **NEW** Second-offence call termination (18 languages) |

---

### 5.6 Tools (Backend / DB)

All tool factories take a `SessionContext` and return `@tool`-wrapped closures bound to the caller's phone number. The LLM never sees the phone number as an argument — it's baked in by the factory.

| File | Tools exposed |
|---|---|
| [`tools/billing.py`](file:///c:/sample/VAY-multilingual-agent/src/vay/tools/billing.py) | `getBalance`, `getBillDetails`, `getDueDate`, `sendPaymentLink`, `getPaymentHistory` |
| [`tools/plans.py`](file:///c:/sample/VAY-multilingual-agent/src/vay/tools/plans.py) | `listPlans`, `getPlanDetails`, `getActivePlan`, `getEligibleUpgrades`, `changePlan` (two-phase with consent) |
| [`tools/complaints.py`](file:///c:/sample/VAY-multilingual-agent/src/vay/tools/complaints.py) | `logComplaint`, `getTicketStatus`, `listOpenTickets`, `escalateToHuman` |
| [`tools/coverage.py`](file:///c:/sample/VAY-multilingual-agent/src/vay/tools/coverage.py) | `checkCoverage`, `getAPNSettings`, `checkOutage`, `troubleshootIssue` |
| [`tools/session.py`](file:///c:/sample/VAY-multilingual-agent/src/vay/tools/session.py) **[CHANGED]** | `SessionContext` dataclass, `confirm_pending_action()`, consent templates |
| [`tools/db_queries.py`](file:///c:/sample/VAY-multilingual-agent/src/vay/tools/db_queries.py) | Raw SQLite: `_connect()`, `get_customer()`, `get_active_subscription()`, `get_balance()`, etc. |

**`SessionContext` fields (per call, not per turn):**

```python
@dataclass
class SessionContext:
    phone_number: str
    verified: bool = False
    language: str = "en"
    escalation_requested: bool = False
    escalation_reason: str = ""
    pending_action: dict | None = None   # For two-phase changePlan consent
    consecutive_unclear: int = 0
    aggressive_count: int = 0            # NEW Aug 14
```

**Two-phase consent for sensitive actions (`changePlan`, `sendPaymentLink`):**
1. First tool call → stages `session.pending_action`, returns STOP_AND_SAY consent script
2. Next turn: `confirm_pending_action()` reads the CUSTOMER's transcript for "yes"/"no" (never the LLM's say-so) → executes or cancels

---

### 5.7 TTS

**File:** [`tts/engine.py`](file:///c:/sample/VAY-multilingual-agent/src/vay/tts/engine.py) **[CHANGED Aug 14 — full rewrite]**

**Replaced:** gTTS (Google Text-to-Speech)  
**With:** edge-tts (Microsoft Edge Neural Voices)

#### Voice Map (18 languages)

| Code | Voice | Code | Voice |
|---|---|---|---|
| `ta` | ta-IN-PallaviNeural | `te` | te-IN-ShrutiNeural |
| `hi` | hi-IN-SwaraNeural | `kn` | kn-IN-SapnaNeural |
| `en` | en-IN-NeerjaNeural | `ml` | ml-IN-SobhanaNeural |
| `fr` | fr-FR-DeniseNeural | `mr` | mr-IN-AarohiNeural |
| `de` | de-DE-KatjaNeural | `gu` | gu-IN-DhwaniNeural |
| `es` | es-ES-ElviraNeural | `ur` | ur-IN-GulNeural |
| `ja` | ja-JP-NanamiNeural | `ar` | ar-AE-FatimaNeural |
| `ko` | ko-KR-SunHiNeural | `it` | it-IT-ElsaNeural |
| `zh` | zh-CN-XiaoxiaoNeural | `ru` | ru-RU-SvetlanaNeural |

**Fallback:** `en-IN-NeerjaNeural` for any unknown language code.

#### API

```python
# Simple playback
speak(text, lang="ta")

# Save without playing
speak(text, lang="hi", output_path="out.mp3", play=False)

# Via class
engine = TTSEngine()
engine.speak(text, lang="en")
path = engine.synthesize(text, language="ta", output_path="out.mp3")
```

**Event-loop safety:** Uses `ThreadPoolExecutor` when called from inside an existing async loop (e.g. Gradio) to avoid `asyncio.run()` nesting errors.

---

### 5.8 Handoff

**File:** [`handoff/queue.py`](file:///c:/sample/VAY-multilingual-agent/src/vay/handoff/queue.py)

`HandoffQueueManager` — appends JSON entries to `handoff_log.jsonl`. Each entry contains:
- `phone_number`, `transcript`, `intent`, `entities`, `normalized_query`
- `route`, `reason`, `draft_reply_at_handoff`, `logged_at`

In the main agent path this is called by `human_handoff_node` in `graph/nodes/utils.py`.

---

### 5.9 UI

**File:** [`ui/app.py`](file:///c:/sample/VAY-multilingual-agent/src/vay/ui/app.py)

Stub currently — `main()` prints a startup message. Needs to be wired to `build_graph()` from `graph/workflow.py` and a Gradio audio input component.

Run: `python -m vay.ui.app`

---

### 5.10 Scripts

| File | Purpose |
|---|---|
| [`scripts/run_assistant.py`](file:///c:/sample/VAY-multilingual-agent/scripts/run_assistant.py) | Main entry point for a full voice call loop. Takes `--phone_number`, `--language`, optional `--show_debug`. Prompts for transcript input, runs the LangGraph, speaks the reply. |
| [`scripts/build_kb.py`](file:///c:/sample/VAY-multilingual-agent/scripts/build_kb.py) | Ingests KB markdown docs from `data/kb_docs/` into the 5 ChromaDB collections. Run once before first use. |
| [`scripts/manage_kb.py`](file:///c:/sample/VAY-multilingual-agent/scripts/manage_kb.py) | Admin: list chunks, delete collection, rebuild, check counts. |
| [`scripts/manage_db.py`](file:///c:/sample/VAY-multilingual-agent/scripts/manage_db.py) | Admin: seed `nexatel_customers.db`, inspect customers, reset. |

---

### 5.11 Tests

```
tests/
├── test_types.py      # ASRResult, StructuredTranscript Pydantic model creation
├── test_routing.py    # Tamil → IndicConformer, English → Whisper routing
└── test_rag.py        # HybridRetriever initialization and collection binding
```

Run: `uv run pytest tests/ -v`  
**Current status: 5/5 PASSED** ✅

---

## 6. Data Models (types.py)

```python
class LanguageTier(StrEnum):
    TIER_1 = "tier_1"   # Tamil, Hindi → IndicConformer
    TIER_2 = "tier_2"   # English + fallback → Whisper

class ASRResult(BaseModel):
    raw_text: str
    detected_language: str      # ISO 639-1
    language_tier: LanguageTier
    confidence: float = 1.0
    model_used: str

class StructuredTranscript(BaseModel):
    original_text: str
    normalized_text: str        # LLM-cleaned version
    detected_language: str
    intent: str                 # snake_case label
    entities: dict[str, Any]
    confidence: float

class Document(BaseModel):
    id: str
    content: str
    metadata: dict[str, Any]
    score: float                # Similarity (0.0–1.0)

class RetrievalResult(BaseModel):
    query: str
    documents: list[Document]
    confidence_score: float     # Best hit similarity
    is_high_confidence: bool    # score >= threshold (0.75)

class HandoffReason(StrEnum):
    SENSITIVE_INTENT = "sensitive_intent"
    LOW_RETRIEVAL_CONFIDENCE = "low_retrieval_confidence"
    HANDOFF_GATE_TRIGGERED = "handoff_gate_triggered"

class HandoffTicket(BaseModel):
    ticket_id: str
    user_id: str
    reason: HandoffReason
    transcript: str
    language: str
    context: dict[str, Any]
    created_at: str
```

---

## 7. Configuration

**File:** [`config.py`](file:///c:/sample/VAY-multilingual-agent/src/vay/config.py) — Pydantic settings (loaded from `.env`)

| Setting | Default | Notes |
|---|---|---|
| `indic_asr_model` | `ai4bharat/indic-conformer-600m-multilingual` | Tier 1 ASR |
| `whisper_asr_model` | `openai/whisper-large-v3-turbo` | Tier 2 ASR |
| `retrieval_confidence_threshold` | `0.80` | τ threshold — tune empirically |
| `top_k_results` | `5` | RAG top-k |
| `sample_rate` | `16000` | Audio sample rate (Hz) |
| `silence_duration_ms` | `650` | VAD utterance boundary |
| `tier1_languages` | `["ta", "hi"]` | Routes to IndicConformer |

Runtime graph config (in `core_utils.py`):

| Constant | Value | Notes |
|---|---|---|
| `DEFAULT_MIN_SIMILARITY` | `0.3` | Guardrail confidence floor |
| `DEFAULT_NLU_CONFIDENCE` | `0.4` | Orchestrator route threshold |
| `DEFAULT_MAX_HISTORY_TURNS` | `6` | Conversation history window |
| `MAX_TOOL_ITERATIONS` | `6` | Tool-calling loop cap |
| `UNCLEAR_ESCALATION_THRESHOLD` | `2` | Unclear turns before human handoff |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Overridable via `GROQ_MODEL` env var |

---

## 8. Environment Variables

```env
# Required
GROQ_API_KEY=gsk_...

# Optional — defaults shown
GROQ_MODEL=llama-3.1-8b-instant
```

Set in `.env` (project root) or as shell environment variables before running.

---

## 9. Changes Made — Aug 14 Session

All changes are in the **RAG + TTS pipeline only**. ASR and transcription pipeline are untouched.

### 9.1 TTS Engine — gTTS → edge-tts

**Files changed:** `tts/engine.py`, `pyproject.toml`

- Removed `gtts>=2.5.1` dependency
- Added `edge-tts>=6.1.9` (installed: 7.2.8) + `playsound3>=1.0.0` (installed: 3.3.2)
- Full rewrite of `tts/engine.py`:
  - 18-language `VOICES` dict with Microsoft Neural voices
  - `speak()` function: synthesize via edge-tts, play via playsound3, cleanup temp file
  - Event-loop safe: `ThreadPoolExecutor` fallback for async contexts (Gradio)
  - `TTSEngine` class: `speak()` and `synthesize()` methods (same API as before)
  - English fallback (`en-IN-NeerjaNeural`) for unknown language codes

### 9.2 Session Memory — Normalization with Previous Turn Context

**Files changed:** `normalization/pass_llm.py`, `graph/nodes.py`

- `pass_llm.py`: Complete rewrite. Real Groq LLM call (was keyword mock).
  - System prompt now includes `previous_intent` + `previous_normalized` from the prior turn
  - Resolves coreferences ("that plan", "same issue", "it") before orchestrator runs
  - Fallback to keyword heuristics when LLM unavailable
- `graph/nodes.py` `normalization_node`: extracts previous turn's `StructuredTranscript` from state and passes `previous_intent` + `previous_normalized` to `normalizer.normalize()`

### 9.3 Aggressive Caller — Warning + Call Cut (NOT Human Handoff)

**Files changed:** `graph/state.py`, `tools/session.py`, `graph/core_utils.py`, `graph/nodes/orchestrator.py`, `graph/nodes/utils.py`, `graph/workflow.py`

**New behaviour:**
- `sensitive=true` (fraud/dispute/cancellation) → human handoff (unchanged)
- `aggressive=true` (abuse/threats) → **warning on 1st offence, call-cut on 2nd** — no human wasted

**Orchestrator JSON schema change:** Added `"aggressive": bool` field, split from `"sensitive"`.  
**New state fields:** `aggressive_count: int`, `warning_reply: str`, `previous_route: str`  
**New session field:** `aggressive_count: int = 0`  
**New template dicts:** `AGGRESSIVE_WARNING_TEMPLATES`, `CALL_CUT_TEMPLATES` (18 languages each, hand-written, not LLM-generated)  
**New graph node:** `warning_node` → wired `orchestrator → warning → tts → END`  
**`route_after_orchestrator` priority order:**
1. Call cut (2nd offence) → closing
2. Warning (1st offence) → warning
3. Normal call end → closing
4. Sensitive → human_handoff
5. Unclear (escalated) → human_handoff
6. Unclear (first time) → clarify
7. Valid route → billing/plans/complaints/coverage

### 9.4 System Prompts Updated

**File:** `graph/core_utils.py`

**Orchestrator prompt:**
- Added `"aggressive"` to JSON schema with explicit definition
- Clear docs: `sensitive` ≠ `aggressive`

**Subagent prompt:**
- Added `{account_context}` placeholder (pre-fetched from CustomerDB)
- Added **telecom terms stay in English** rule with examples:
  - `"உங்கள் 499 plan-ல் 1 GB daily data மற்றும் unlimited calls கிடைக்கும்."`
  - `"आपके Smart 499 plan में 1 GB daily data और unlimited calls मिलते हैं।"`
- Escalation rule tightened: no escalation for rudeness — only for repeated unresolved issues, explicit human request, or tool refusal

### 9.5 Account Context Pre-fetch

**File:** `graph/nodes/orchestrator.py`

`_fetch_account_context(phone_number)` queries the mock CustomerDB before each sub-agent turn:
- Customer name (fixed SQL query crashing due to non-existent `email` column)
- Active subscription (fixed crashing due to incorrect `data_gb` and `calls` column names; updated to `data_limit` and `voice_minutes`)
- Outstanding balance
- Open tickets (last 3)

Result injected directly into the subagent system prompt as `{account_context}`. Saves one full tool-call round-trip per turn. Non-fatal: returns empty string on DB error.

**Subagent Prompt Fix (`graph/core_utils.py`):**
Updated the TOOL-USE RULES to force the LLM to use the `{account_context}` to answer direct questions like "What is my plan/balance?" without triggering unnecessary fallback or clarification loops.

### 9.6 Multi-Agent Domain Switching — History Trimming

**File:** `graph/nodes/orchestrator.py`

`_run_subagent` now detects when `previous_route != current_route` (domain switch):
- Strips `ToolMessage` objects and tool-calling `AIMessage` stubs from history
- Keeps only `HumanMessage` and clean `AIMessage` (no tool calls)
- Passes clean history to the new sub-agent
- Debug log: `[domain switch billing → plans: trimmed N tool messages from history]`

### 9.7 RAG Fixes

**File:** `rag/retriever.py`

- `HybridRetriever` default `collection_name`: was `"knowledge_base"` (non-existent generic collection) → fixed to `KB_COLLECTIONS["billing_policy"]`
- Default `confidence_threshold`: was `0.75` → explicitly `0.75` (no change, but now annotated with the project_context.md §8 rationale)

**File:** `graph/nodes/utils.py`

- `guardrail_node`: uncertainty alone no longer triggers handoff — must also have `retrieval_score < 0.5`
- Added compliance policy KB check for drafts containing consent-trigger words (`change plan`, `cancel`, `payment link`, etc.)

### 9.8 Handoff Routing Fix

**File:** `graph/nodes/orchestrator.py`

- Fixed an issue where answering "yes" to a human handoff suggestion resulted in an endless clarification loop.
- `explicit_human_request` now checks the LLM's `intent` output (`escalate`, `request_human_agent`) and `normalized_query`, rather than relying solely on an English regex matching the raw, localized transcript.

---

## 10. Key Design Decisions

| Decision | Rationale |
|---|---|
| No dedicated LID model | Whisper's own encoder does language detection. Deliberate cost/latency decision. |
| RAG queries English-normalized query, responds in-language | LLM is instructed to respond in the caller's language directly from English KB context — no separate translation step. |
| Retrieval confidence threshold τ ≈ 0.75–0.85 | Intentionally strict. Prioritizes safe handoff over confidently wrong answers (billing/account data at stake). |
| `aggressive` ≠ `sensitive` in orchestrator JSON | Abusive callers get a warning then call-cut, not a human agent — avoids wasting live agent capacity on abusive calls. |
| Previous intent injected into normalization | Resolves coreferences across turns without an extra LLM call at the orchestrator stage. |
| Telecom terms in English within localized replies | Natural speech — customers say "1 GB" and "4G" regardless of what language they're speaking. |
| Two-phase consent for changePlan / sendPaymentLink | LLM cannot be trusted to reliably wait for a "yes" — code enforces it instead. |
| Account context pre-fetched into system prompt | Saves one tool-call round-trip per turn; reduces latency and token cost. |
| Domain-switch history trimming | Prevents the plans agent from being confused by billing tool results from the previous turn. |
| No fine-tuning of any model | Pretrained only — deliberate scope decision for the 7-day hackathon window. |
| No barge-in / interruption handling | Explicitly out of scope. |

---

## 11. Known Gotchas

- **IndicConformer cannot be loaded via `pipeline()`** — use `AutoModel.from_pretrained()` only.
- **IndicConformer does NOT support English** — never route English through it.
- **Whisper hallucinates on Tamil/Hindi** — VAD silence trimming + post-ASR hallucination filter mitigates this.
- **Mozilla Common Voice Kaggle mirror filename collision** — only index MP3s from within a specific split folder (`cv-valid-test`), never by bare filename across the whole tree.
- **English WER (~3.8%)** is from a pre-bug run — do not present as validated until re-run after the Kaggle pairing fix.
- **`asyncio.run()` inside Gradio** will raise `RuntimeError: This event loop is already running` — the edge-tts engine handles this with a `ThreadPoolExecutor` fallback.
- **GROQ_API_KEY must be set** — `_llm()` raises `SystemExit` if missing.
- **ChromaDB collections must be built before first use** — run `scripts/build_kb.py`.
- **`nexatel_customers.db` must be seeded before tools work** — run `scripts/manage_db.py`.

---

## 12. How to Run

### Initial Setup

```powershell
# 1. Install uv (one-time)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Clone and sync
git clone https://github.com/vishvaa-vsk/VAY-multilingual-agent.git
cd VAY-multilingual-agent
uv sync

# 3. Set API key
$env:GROQ_API_KEY = "gsk_..."

# 4. Build the knowledge base (one-time)
uv run python scripts/build_kb.py

# 5. Seed the customer database (one-time)
uv run python scripts/manage_db.py --seed
```

### Running

```powershell
# Full call loop (text input mode — no microphone needed for testing)
uv run python scripts/run_assistant.py --phone_number 9876543210 --language ta

# With debug output
uv run python scripts/run_assistant.py --phone_number 9876543210 --language hi --show_debug

# Test TTS directly
uv run python -m vay.tts.engine

# Web UI (stub)
uv run python -m vay.ui.app

# Run tests
uv run pytest tests/ -v

# Type check
uv run mypy src

# Lint
uv run ruff check src tests
```

### KB Management

```powershell
# Check KB status
uv run python -m vay.rag.vector_store

# Rebuild a specific collection
uv run python scripts/manage_kb.py --rebuild billing_policy

# Check all collections
uv run python scripts/manage_kb.py --status
```
