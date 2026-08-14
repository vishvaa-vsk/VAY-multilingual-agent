# System Context & Architecture Overview

Comprehensive technical context for the **Nexatel Communications Multilingual Voice RAG
Assistant** — a LangGraph orchestrator + 4 domain sub-agents, each with its own scoped
RAG retriever and backend "API" tools, sitting behind a Groq-hosted LLM, driving an
edge-tts spoken reply. This is the second-generation architecture (agentic, multi-sub-agent)
replacing the earlier single-pipeline design.

---

## 1. Executive Summary

The system takes an already-transcribed customer utterance (transcript, language_code,
phone_number — ASR/VAD/Language-ID happen upstream and are out of scope for this repo) and:

1. Routes it via an **Orchestrator LLM node** to one of 4 domain **sub-agents** (Billing,
   Plans, Complaints, Coverage), or straight to human handoff if sensitive/unclear/low-confidence.
2. Each sub-agent runs a bounded **tool-calling loop** (Groq `llama-3.1-8b-instant` via
   `langchain_groq.ChatGroq`) over its own backend tools (SQLite-backed mock CRM/billing/network
   APIs) **plus** its own scoped **RAG retriever tool** (one of 5 ChromaDB collections).
3. A **guardrail node** applies a confidence gate on the RAG retrieval score, a PII-leak
   scan, an explicit "talk to a human" detector, and an uncertainty-phrase detector — any of
   which routes to human handoff instead of finalizing the reply.
4. The finalized (or handoff) reply is spoken back via **edge-tts** (`tts.py`), and the loop
   repeats for the next utterance within the same call (state persists via a per-call
   `SessionContext`).

Two sensitive actions (`changePlan`, `sendPaymentLink`) are **two-phase and enforced in code,
not by the LLM**: the first tool call only stages a `pending_action` and returns a fixed
consent script; the change is only committed once the *customer's own next-turn transcript*
contains a literal "yes", checked by regex in `agent_graph.py`, never trusted to the LLM.

---

## 2. LangGraph Architecture Diagram

```
                                   ┌─────────────────────────┐
                                   │          START           │
                                   └────────────┬─────────────┘
                                                │
                                                ▼
                              ┌──────────────────────────────────┐
                              │        orchestrator_node           │
                              │  Groq LLM -> STRICT JSON:           │
                              │  {language, intent, route,          │
                              │   normalized_query, entities,       │
                              │   confidence, sensitive,            │
                              │   call_end_requested}               │
                              │  (pending_action from a prior turn  │
                              │   force-routes a bare yes/no back   │
                              │   to its owning sub-agent)          │
                              │  ALSO tracks SessionContext.        │
                              │   consecutive_unclear and sets      │
                              │   unclear_escalate + handoff_reason │
                              └────────────────┬─────────────────┘
                                                │
                              route_after_orchestrator()
        ┌───────────┬─────────────┬─────────────┬───────────┬───────────┬────────────┐
        │           │             │             │           │           │            │
 call_end_requested │      sensitive OR         │           │           │     route="unclear" OR
        │           │      explicit "human"      │           │           │     confidence < 0.4
        │           │      request                │           │           │           │
        ▼           ▼             ▼             ▼           ▼           ▼           ▼
  ┌──────────┐  ┌────────┐   ┌──────────┐  ┌───────────┐ ┌─────────┐ ┌──────────┐   (unclear_escalate?)
  │ closing  │  │billing │   │  plans   │  │complaints │ │coverage │ │  human_  │    /            \
  │  node    │  │ node   │   │  node    │  │   node    │ │  node   │ │ handoff  │  yes             no
  └────┬─────┘  └───┬────┘   └────┬─────┘  └─────┬─────┘ └────┬────┘ └────┬─────┘   │               │
       │            │             │              │            │           ▼               ▼
       │       each sub-agent node runs a BOUNDED TOOL-CALLING LOOP:      │        ┌──────────┐  ┌─────────┐
       │       ┌──────────────────────────────────────────────────┐      │        │  human_  │  │ clarify │
       │       │ ChatGroq.bind_tools(domain_tools + rag_tool)      │      │        │ handoff  │  │  node   │
       │       │  loop (max MAX_TOOL_ITERATIONS=6 iters):          │      │        └────┬─────┘  └────┬────┘
       │       │   LLM -> tool_calls? -> invoke tool -> ToolMessage│      │             │             │
       │       │   "STOP_AND_SAY:" sentinel short-circuits         │      │             │             │
       │       │      (consent script returned verbatim, unseen    │      │             └──────┬──────┘
       │       │       by the LLM, for changePlan/sendPaymentLink) │      │                    tts_node
       │       │   no more tool_calls -> final reply, grounded in  │      │           (clarify_node does NOT set
       │       │     concrete tool/RAG facts (system prompt rule:  │      │            handoff=True -- call just
       │       │     state paid/no-dues status plainly, never      │      │            continues to the next
       │       │     unrelated chit-chat)                          │      │            utterance; only 2 CONSECUTIVE
       │       │                                                    │      │            unclear/low-confidence turns
       │       │  domain tools (tools.py, SQLite via customer_db.py):│     │            (SessionContext.
       │       │   billing:     getBalance, getBillBreakup,        │      │             consecutive_unclear >= 2,
       │       │                getDueDate, sendPaymentLink*,      │      │             reset the instant a turn
       │       │                explainCharge                     │      │             IS understood) escalate to
       │       │   plans:       listPlans, comparePlans,          │      │             human_handoff for real)
       │       │                changePlan*, activateAddOn,       │      │
       │       │                checkEligibility                  │      │
       │       │   complaints:  createComplaint, getTicketStatus, │      │
       │       │                runTroubleshootFlow,              │      │
       │       │                escalateToHuman (only for a       │      │
       │       │                  REQUIRED reason -- distress,    │      │
       │       │                  repeated issue, explicit human  │      │
       │       │                  request, failed verification -- │      │
       │       │                  never for an off-topic/rude     │      │
       │       │                  aside the agent could answer)   │      │
       │       │   coverage:    checkCoverage, getOutageStatus,   │      │
       │       │                getDeviceSettings, guideSimSwap   │      │
       │       │   (* = two-phase, code-enforced consent gate)    │      │
       │       │                                                    │      │
       │       │  + ONE scoped RAG tool per sub-agent (rag_tools.py│      │
       │       │    -> content_manager.read() -> its own ChromaDB  │      │
       │       │    collection); best similarity hit recorded on   │      │
       │       │    a RetrievalTracker for the guardrail node      │      │
       │       └──────────────────────────────────────────────────┘      │
       │            │             │              │            │           │
       │            └─────────────┴──────┬───────┴────────────┘           │
       │                                  ▼                                │
       │                       ┌────────────────────┐                      │
       │                       │   guardrail_node    │                      │
       │                       │ - retrieval_score    │                     │
       │                       │   < min_similarity?  │                     │
       │                       │ - "talk to a human"  │                     │
       │                       │   phrase detected?   │                     │
       │                       │ - uncertainty phrase │                     │
       │                       │   in draft reply?    │                     │
       │                       │ - PII/credential leak│                     │
       │                       │   pattern in draft?  │                     │
       │                       └──────────┬───────────┘                     │
       │                     route_after_guardrail()                        │
       │                        │                    │                      │
       │                     (ok)│              (handoff)                   │
       │                        ▼                    └─────────────────────►┤
       │                  final_reply=draft                                 │
       │                        │                                           │
       │                        │        human_handoff_node: logs full      │
       │                        │        context packet (transcript, intent,│
       │                        │        entities, route, reason, draft) to │
       │                        │        handoff_log.jsonl (mock escalation  │
       │                        │        queue); final_reply = HANDOFF_     │
       │                        │        MESSAGE_TEMPLATES[language] (fixed │
       │                        │        per-language text, NOT English-    │
       │                        │        only -- see below)                 │
       │                        │                                           │
       └────────────────────────┼───────────────────────────────────────────┘
                                              ▼
                                     ┌─────────────────┐
                                     │    tts_node       │
                                     │ tts.speak(reply,   │
                                     │  lang) -> edge-tts │
                                     │  synth + playback  │
                                     └────────┬──────────┘
                                              ▼
                                     ┌─────────────────┐
                                     │       END         │
                                     └─────────────────┘
                              (loop back to orchestrator for
                               the next utterance in the same call —
                               conversation_history + SessionContext
                               persist across turns in agent_graph.main())
```

### Why replies used to come back in English regardless of the caller's language

`tts_node` always spoke `final_reply` in `state["language"]` — the detected language was never
the bug. The bug was that several **fixed, code-level replies never varied with language in the
first place**: `HANDOFF_MESSAGE`, the tool-loop-failure fallback, the closing-call fallback, and
`tools.confirm_pending_action()`'s outcome strings were hardcoded English constants returned
verbatim (deliberately bypassing the LLM, for determinism on safety-critical text — same
rationale as the pre-existing `tools.CONSENT_TEMPLATES`). Fix: each now has its own
per-language template dict (`HANDOFF_MESSAGE_TEMPLATES`, `TOOL_LOOP_FAILURE_TEMPLATES`,
`CLOSING_FALLBACK_TEMPLATES`, `CLARIFY_TEMPLATES` in `agent_graph.py`;
`CONFIRM_DECLINED_TEMPLATES` / `CONFIRM_UNAVAILABLE_TEMPLATES` / `CONFIRM_CHANGED_TEMPLATES` in
`tools.py`), covering en/hi/ta with an English fallback for any other language — add more
entries rather than leaving a language silently in English. Sub-agent LLM replies were already
instructed to answer in `{language}` and are unaffected by this fix.

### Only handing off to a human when actually required

Previously *any* `route="unclear"` or low-NLU-confidence turn (garbled speech, an off-topic
remark, an insult directed at the bot) went straight to `human_handoff` — burning a live agent's
time on things the assistant could have just asked about. Now `unclear`/low-confidence turns go
to a new **`clarify_node`** (a localized "could you tell me a bit more?" re-prompt, no human
involved) unless `sensitive` is true, the caller explicitly asked for a human
(`HUMAN_REQUEST_PATTERNS`), or `SessionContext.consecutive_unclear` has hit
`UNCLEAR_ESCALATION_THRESHOLD` (2) — i.e. the assistant tried to clarify and still couldn't
understand. The counter resets to 0 the moment a turn IS understood. `sensitive` (billing
dispute, cancellation, fraud/security, distress) and explicit human requests still escalate
immediately, per the solution doc's Layer 4 triggers.

### Scoped RAG collections (one per sub-agent + one shared)

```
kb_docs/billing_policy.md     ──► ChromaDB "billing_policy"     ──► Billing agent
kb_docs/product_catalog.md    ──► ChromaDB "product_catalog"    ──► Plans agent
kb_docs/support_faq.md        ──► ChromaDB "support_faq"        ──► Complaints agent
kb_docs/technical_kb.md       ──► ChromaDB "technical_kb"       ──► Coverage agent
kb_docs/compliance_policy.md  ──► ChromaDB "compliance_policy"  ──► Guardrail layer (consent
                                                                     scripts / do-don't-say
                                                                     rules; via
                                                                     rag_tools.compliance_policy_search,
                                                                     not a bindable tool)
```

Rationale for per-sub-agent scoping (not one shared collection): precise retrieval, lower
hallucination risk, independently testable/evaluatable per domain.

---

## 3. File-by-File Technical Reference

### A. [agent_graph.py](agent_graph.py) — LangGraph orchestrator + entry point
* Builds and runs the graph above. Owns all system prompts (`ORCHESTRATOR_SYSTEM_PROMPT`,
  `SUBAGENT_SYSTEM_PROMPT_TEMPLATE`), the `GraphState` TypedDict, node functions, conditional
  routing functions, and `run_tool_agent()` (the bounded tool-calling loop shared by all 4
  sub-agent nodes).
* `main()` is a CLI REPL: prompts for phone number + language once per call, then loops
  reading transcribed utterances, invoking `graph.invoke(state)` each turn, printing the
  reply, and persisting `conversation_history` (trimmed to `--max_history_turns`) and the one
  `SessionContext` across turns until the customer ends the call or is handed off.
* CLI flags: `--min_similarity` (confidence gate, default `0.3`), `--max_history_turns`
  (default `6`), `--show_debug` (prints orchestrator JSON + tool calls), `--language`,
  `--phone` (skip interactive prompts).
* Guardrail regexes: `HUMAN_REQUEST_PATTERNS`, `UNCERTAINTY_PATTERNS`, `PII_LEAK_PATTERNS`,
  `AFFIRMATION_PATTERN`/`NEGATION_PATTERN` (the last two are the code-enforced yes/no
  consent check — deliberately English-literal so it works regardless of the call's spoken
  language; see `tools.consent_script()`).
* **Localized fixed-string templates** (`HANDOFF_MESSAGE_TEMPLATES`, `TOOL_LOOP_FAILURE_
  TEMPLATES`, `CLOSING_FALLBACK_TEMPLATES`, `CLARIFY_TEMPLATES`, looked up via `localized()`):
  the small set of replies that bypass the sub-agent LLM for determinism (handoff message, a
  hard tool-loop failure, the call-closing fallback, the clarify re-prompt) now vary with
  `state["language"]` (en/hi/ta, English fallback) instead of being hardcoded English —
  this, not language *detection*, was the actual cause of always-English hand-off audio.
* **`clarify_node`**: a human-agent-free re-prompt for an unclear/low-confidence turn — see
  `route_after_orchestrator()` and `SessionContext.consecutive_unclear` below. Keeps a single
  garbled/off-topic/rude utterance from consuming a live agent's time; only 2 consecutive
  unclear turns (`UNCLEAR_ESCALATION_THRESHOLD`) — or a `sensitive` intent, or an explicit
  human request — actually escalate to `human_handoff`.
* Escalations are appended as JSON lines to `handoff_log.jsonl` (mock human-agent queue —
  gitignored) via `log_handoff()`.
* Replaces the old `voice_rag_pipeline.py` single-pipeline entry point (deleted).

### B. [tools.py](tools.py) — Backend "API" tools for the 4 sub-agents
* One `build_<domain>_tools(session)` factory per sub-agent, each closing over a
  `SessionContext` (dataclass: `phone_number`, `verified`, `language`, `escalation_requested`,
  `escalation_reason`, `pending_action`, `consecutive_unclear`) so the **phone number is
  session-bound, never an LLM-fillable argument** (compliance requirement — the LLM only
  supplies content args like `plan_id`/`ticket_id`). `consecutive_unclear` is
  `agent_graph.orchestrator_node`'s counter of back-to-back unclear/low-confidence turns,
  driving the clarify-before-handoff logic (see file A above).
* `confirm_pending_action()`'s outcome strings (declined / plan-unavailable / changed) are
  localized per-language too (`CONFIRM_DECLINED_TEMPLATES` etc., en/hi/ta, English fallback) —
  same rationale as `CONSENT_TEMPLATES`: this path bypasses the LLM, so it can't rely on the
  sub-agent prompt's "reply in {language}" instruction to translate it.
* All tools read/write the mock SQLite DB via `customer_db._connect()`.
* Sensitive actions (`changePlan`, `sendPaymentLink`) refuse outright if `session.verified`
  is `False` (`SENSITIVE_DENIAL`), and are **two-phase**: first call stages
  `session.pending_action` and returns `"STOP_AND_SAY: " + consent_script(...)` (a sentinel
  `run_tool_agent()` recognizes and returns verbatim, bypassing the LLM entirely); the change
  is only committed by `confirm_pending_action()`, called by the graph itself from the
  *customer's own next-turn transcript*, never from an LLM judgment call.
* `consent_script()` / `CONSENT_TEMPLATES`: fixed, hand-written per-language (en/hi/ta)
  templates that always end with a literal English "yes"/"no" instruction, so the
  confirmation check never needs to enumerate affirmation phrases across every language.
* Static reference data: `TROUBLESHOOT_FLOWS` (5 issue types), `SLA_DAYS` (per ticket
  category), `DEVICE_SETTINGS` (Android/iPhone APN + VoLTE steps).

### C. [rag_tools.py](rag_tools.py) — Scoped RAG retriever tools
* `RetrievalTracker` dataclass: records the best similarity (`1 - distance`) seen across a
  sub-agent turn's RAG calls, read by `agent_graph.guardrail_node` for the confidence gate.
* `_make_retriever()` wraps `content_manager.read()` against one `chroma_setup.KB_COLLECTIONS`
  collection as a LangChain `@tool`.
* One `build_*_rag_tool(tracker)` factory per sub-agent (`build_billing_rag_tool`,
  `build_product_rag_tool`, `build_support_rag_tool`, `build_technical_rag_tool`) — each
  sub-agent gets **only its own** retriever tool.
* `compliance_policy_search()`: a direct (non-LangChain-tool) helper for the guardrail layer
  to pull consent-script / do-don't-say rules from the `compliance_policy` collection.

### D. [customer_db.py](customer_db.py) — Mock Nexatel operational database (SQLite)
* Stands in for real billing/CRM/network backend APIs. File: `nexatel_customers.db`.
* Tables: `customers`, `plans` (18 seeded plans across prepaid/postpaid/broadband),
  `subscriptions`, `bills`, `payments`, `tickets`, `coverage` (pincode → signal/outage).
* 10 seeded sample customers (phone `98765000xx`) covering every sub-agent demo path
  (overdue bill, roaming dispute, KYC-pending, youth-plan eligibility, network outage, etc.).
* CLI: `python customer_db.py` (create + seed if empty), `python customer_db.py --reset`
  (wipe and reseed).

### E. [chroma_setup.py](chroma_setup.py) — Centralized ChromaDB connection manager
* Single source of truth: `PERSIST_DIRECTORY="chroma_db"`, `EMBEDDING_MODEL="all-MiniLM-L6-v2"`,
  cosine distance (`hnsw:space: cosine`).
* `KB_COLLECTIONS`: the 5 scoped collection names (see diagram above) — every module imports
  these instead of hardcoding strings.
* `get_client()` / `get_embedding_function()` / `get_collection()` are module-level cached
  (keyed by `(persist_directory, collection_name)`) so switching between the 5 collections
  mid-session doesn't reload the embedding model each time.
* CLI: `python chroma_setup.py` (status), `--reset` (wipe, with a typed "yes" confirmation).

### F. [build_kb.py](build_kb.py) — Knowledge-base builder
* Ingests each of the 5 `kb_docs/*.md` files into its own scoped collection via
  `content_manager.create_from_markdown()`, with domain-relevant **guided category labels**
  per KB (`KB_SPEC` dict) rather than unsupervised clustering, so retrieval has useful
  category signal from the start.
* CLI: `python build_kb.py` (idempotent upsert), `--reset` (wipe each collection first).
* Replaces the old `build_chroma_kb.py` (PDF-based, deleted) — now markdown-doc-based.

### G. [content_manager.py](content_manager.py) — Domain-agnostic ingestion/retrieval engine
* Shared CRUD engine on top of `chroma_setup`; unchanged in role from the prior architecture
  (still the only file that talks to ChromaDB collections directly for ingest/read).
* Key entry points used by the new architecture: `create_from_markdown()` (used by
  `build_kb.py`), `read()` (used by `rag_tools.py`), plus `create()`, `update()`, `delete()`,
  `list_sources()`, `list_categories()`, `inspect_source()` for general CRUD/debugging.
* Category tagging: **guided** (cosine similarity against label embeddings, used by
  `build_kb.py`) or **unsupervised** (KMeans + TF-IDF centroid auto-labeling, default when no
  labels given) — see `categorize_chunks_guided` / `categorize_chunks_unsupervised`.
* Sentence-boundary chunking (`chunk_markdown`, NLTK `sent_tokenize`), heading-context
  propagation, content-addressed chunk IDs (`SHA-256` of chunk text, idempotent upserts),
  batch TF-IDF descriptions (`compute_tfidf_descriptions`), language detection (`langdetect`).
* Default chunk size ~500 chars / 100 char overlap, tuned for `all-MiniLM-L6-v2`'s 256-token
  window.

### H. [tts.py](tts.py) — Language-aware text-to-speech
* `speak(text, lang, output_path, play)`: synthesizes via `edge-tts` (Microsoft Edge neural
  voices), optionally plays via `playsound3`, then deletes the temp mp3. Never raises —
  synthesis/playback failures are logged and swallowed so a headless environment doesn't kill
  the call loop.
* `VOICES` dict: 18 language codes mapped to specific neural voices (ta, hi, en-IN, fr, de,
  es, ja, ko, zh, it, ru, ar, te, kn, ml, mr, gu, ur), falls back to `en-IN-NeerjaNeural`.
* Replaces the earlier IndicF5/gTTS approach referenced in the prior architecture doc.
* CLI (manual testing): `python tts.py` prompts for text + language code.

### I. [kb_docs/](kb_docs/) — Source knowledge-base markdown documents
* `billing_policy.md`, `product_catalog.md`, `support_faq.md`, `technical_kb.md`,
  `compliance_policy.md` — hand-authored Nexatel policy/product docs, one per scoped RAG
  collection, ingested by `build_kb.py`.

---

## 4. Configuration & Parameters

| Parameter | Default | Source | Description |
|---|---|---|---|
| `GROQ_MODEL` | `llama-3.1-8b-instant` | `agent_graph.py` (env `GROQ_MODEL`) | Groq LLM for orchestrator + all sub-agents |
| `GROQ_API_KEY` | *(required, no fallback)* | env var | Groq API key — hard error if unset |
| `DEFAULT_MIN_SIMILARITY` | `0.3` | `agent_graph.py` | Confidence gate on a sub-agent's best RAG hit |
| `DEFAULT_NLU_CONFIDENCE` | `0.4` | `agent_graph.py` | Orchestrator confidence floor before routing to a sub-agent |
| `DEFAULT_MAX_HISTORY_TURNS` | `6` | `agent_graph.py` | Max past exchange turns retained in LLM context |
| `MAX_TOOL_ITERATIONS` | `6` | `agent_graph.py` | Cap on tool-call round-trips per sub-agent turn (raised from 4 — a getBalance+getDueDate+RAG-search+sendPaymentLink turn routinely hit the old cap) |
| `UNCLEAR_ESCALATION_THRESHOLD` | `2` | `agent_graph.py` | Consecutive unclear/low-confidence turns before `clarify_node` gives up and hands off to a human |
| `HANDOFF_LOG_PATH` | `handoff_log.jsonl` | `agent_graph.py` | Mock human-escalation queue (gitignored) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | `chroma_setup.py` | SentenceTransformers embedding model |
| `PERSIST_DIRECTORY` | `chroma_db` | `chroma_setup.py` | ChromaDB storage path |
| `KB_COLLECTIONS` | 5 named collections | `chroma_setup.py` | billing_policy, product_catalog, support_faq, technical_kb, compliance_policy |
| `DEFAULT_CHUNK_SIZE` | ~500 chars | `content_manager.py` | Tuned to embedding model's 256-token limit |
| `DEFAULT_CHUNK_OVERLAP` | ~100 chars | `content_manager.py` | Sentence-boundary overlap |
| `DB_PATH` | `nexatel_customers.db` | `customer_db.py` | Mock SQLite customer/operational DB |

---

## 5. Dependencies & Setup

* **Python libraries** ([requirements.txt](requirements.txt)): `requests`, `beautifulsoup4`,
  `html2text`, `chromadb`, `sentence-transformers`, `pypdf`, `nltk`, `langdetect`,
  `scikit-learn`, `numpy`, `langgraph`, `langchain`, `langchain-core`, `langchain-groq`,
  `edge-tts`, `playsound3`, `python-dotenv`.
* **API access**: `GROQ_API_KEY` (env var / `.env`, gitignored), optional `GROQ_MODEL` override.
* **First-time setup** (per `agent_graph.py` module docstring):
  ```
  pip install -r requirements.txt
  python build_kb.py          # build the 5 Nexatel RAG collections
  python customer_db.py       # seed the mock customer database
  set GROQ_API_KEY=your_key_here      (Windows)
  export GROQ_API_KEY=your_key_here   (bash)
  ```
* **Run**: `python agent_graph.py` (`--show_debug`, `--min_similarity`, `--max_history_turns`,
  `--language`, `--phone` flags available).
* **Local data/output**: `kb_docs/*.md` (source KB docs) → `chroma_db/` (vector store,
  gitignored contents rebuild via `build_kb.py`); `nexatel_customers.db` (mock CRM, rebuilt
  via `customer_db.py`); `handoff_log.jsonl` (escalation packets, gitignored);
  `output.mp3` (transient TTS scratch file, gitignored, deleted after playback).

---

## 6. Superseded / Removed Files

The following files from the earlier single-pipeline architecture have been deleted and
replaced by the LangGraph multi-sub-agent design above — kept here only as a migration note:

| Old file | Replaced by |
|---|---|
| `voice_rag_pipeline.py` / `voice_rag_pipelinev1.py` | `agent_graph.py` |
| `build_chroma_kb.py` | `build_kb.py` |
| `inspect_db.py` | `content_manager.py`'s `inspect_source()` / `list_sources()` / `list_categories()` |
| `chroma_setup (1).py` | `chroma_setup.py` |
| `demo.py` | *(removed, no direct replacement)* |
| `pdfs/*.pdf` (NexaTel_Knowledge_Base, Response_Scripts, Policy_Guide) | `kb_docs/*.md` (hand-authored, per-domain) |
