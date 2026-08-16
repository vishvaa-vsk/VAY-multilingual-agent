# Project Context: Nexatel Communications Multilingual Voice RAG Assistant

**Purpose of this file**: This is the locked source of truth for any AI coding agent (Claude Code, Antigravity, or similar) working on this project. Everything in Sections 1-7 describes the ACTUAL EXISTING CODEBASE — file names, function names, config values, and behavior are ground truth, not aspirational design. Do not "improve," refactor, or second-guess these mechanisms without being asked — especially the compliance-critical ones marked below. Section 8 covers what's upstream but out of this repo's scope. Section 9 is genuinely open/unresolved — do not invent answers there.

---

## 1. What This Repo Is

A LangGraph orchestrator + 4 domain sub-agents (Billing, Plans, Complaints, Coverage), each with its own scoped RAG retriever and backend "API" tools, behind a Groq-hosted LLM, driving an edge-tts spoken reply. This is the **second-generation architecture** (agentic, multi-sub-agent) — an earlier single-pipeline design was fully replaced (see Section 7 for the deleted files).

**Scope boundary — important**: this repo takes an **already-transcribed** customer utterance as input: `(transcript, language_code, phone_number)`. ASR, VAD, and Language-ID happen in a separate upstream component and are OUT OF SCOPE here (see Section 8 for what's known about that upstream piece).

## 2. Locked Model/Service Choices — EXACT, DO NOT SUBSTITUTE

| Component | Choice | Notes |
|---|---|---|
| LLM (orchestrator + all 4 sub-agents) | Groq `llama-3.1-8b-instant` via `langchain_groq.ChatGroq` | Env var `GROQ_MODEL` overrides; `GROQ_API_KEY` required, hard error if unset, no fallback |
| Orchestration | LangGraph | See Section 3 for the exact node graph |
| TTS | `edge-tts` (Microsoft Edge neural voices) via `tts.py` | **This replaces an earlier plan to use IndicF5/gTTS** — if you see IndicF5 referenced anywhere (docs, diagrams, chat history), it's superseded; the actual code uses edge-tts |
| Vector DB | ChromaDB, 5 collections | `all-MiniLM-L6-v2` embeddings, cosine distance |
| Mock backend | SQLite (`nexatel_customers.db`) | Stands in for real billing/CRM/network APIs |

## 3. LangGraph Architecture — Node by Node

```
START
  → orchestrator_node
      Groq LLM outputs STRICT JSON:
      {language, intent, route, normalized_query, entities, confidence, sensitive, call_end_requested}
      (a pending_action from a prior turn force-routes a bare yes/no back to its owning sub-agent)
  → route_after_orchestrator():
      ├─ call_end_requested            → closing_node
      ├─ sensitive OR route="unclear"
      │  OR confidence < 0.4           → human_handoff_node
      └─ otherwise                     → billing_node | plans_node | complaints_node | coverage_node
  → [each sub-agent node runs a BOUNDED TOOL-CALLING LOOP, max 4 iterations]:
      ChatGroq.bind_tools(domain_tools + rag_tool)
      loop: LLM → tool_calls? → invoke tool → ToolMessage → (repeat or finalize)
      "STOP_AND_SAY:" sentinel short-circuits the loop entirely (used for consent scripts —
      see Section 4, this bypasses the LLM, it does not go through it)
      no more tool_calls → draft reply
  → guardrail_node checks, on the draft reply / retrieval:
      ├─ retrieval_score < DEFAULT_MIN_SIMILARITY (0.3)?
      ├─ "talk to a human" phrase detected? (HUMAN_REQUEST_PATTERNS)
      ├─ uncertainty phrase in draft? (UNCERTAINTY_PATTERNS)
      └─ PII/credential leak pattern in draft? (PII_LEAK_PATTERNS)
      any match → human_handoff_node ; otherwise → final_reply = draft
  → human_handoff_node (if reached from any path above):
      logs full context packet (transcript, intent, entities, route, reason, draft) to
      handoff_log.jsonl (mock escalation queue, gitignored) via log_handoff()
      final_reply = fixed HANDOFF_MESSAGE
  → tts_node: tts.speak(reply, lang) → edge-tts synth + playback
  → END, loop back to orchestrator_node for the next utterance in the same call
    (conversation_history + one SessionContext persist across turns in agent_graph.main())
```

## 4. Compliance-Critical Mechanisms — DO NOT WEAKEN OR "SIMPLIFY" THESE

These exist deliberately to keep sensitive actions out of the LLM's hands. Treat any change to them as requiring explicit sign-off, not a routine refactor.

- **Two-phase, code-enforced consent for sensitive actions** (`changePlan`, `sendPaymentLink`):
  1. First tool call only *stages* `session.pending_action` and returns `"STOP_AND_SAY: " + consent_script(...)`.
  2. `run_tool_agent()` recognizes the `STOP_AND_SAY:` sentinel and returns that text verbatim — the LLM never sees or paraphrases the consent script.
  3. The action is only actually committed by `confirm_pending_action()`, called by the graph itself from the **customer's own next-turn transcript** containing a literal "yes" — checked by regex (`AFFIRMATION_PATTERN`) in `agent_graph.py`. This is **never** an LLM judgment call.
  4. Both sensitive tools refuse outright if `session.verified` is `False` (`SENSITIVE_DENIAL`).
- **`consent_script()` / `CONSENT_TEMPLATES`**: fixed, hand-written per-language templates (en/hi/ta) that always end with a literal English "yes"/"no" instruction — deliberately English-literal so the confirmation regex doesn't need to enumerate affirmation phrases across every supported language.
- **Phone number is session-bound, never LLM-fillable**: `SessionContext` (dataclass: `phone_number`, `verified`, `language`, `escalation_requested`, `escalation_reason`, `pending_action`) is closed over by each `build_<domain>_tools(session)` factory. The LLM only ever supplies content arguments (`plan_id`, `ticket_id`, etc.), never the phone number itself — this is a compliance requirement, not an implementation convenience.

## 5. Per-Sub-Agent Tools and Scoped RAG

| Sub-agent | Domain tools (`tools.py`, SQLite via `customer_db.py`) | RAG collection |
|---|---|---|
| Billing | `getBalance`, `getBillBreakup`, `getDueDate`, `sendPaymentLink`*, `explainCharge` | `billing_policy` |
| Plans | `listPlans`, `comparePlans`, `changePlan`*, `activateAddOn`, `checkEligibility` | `product_catalog` |
| Complaints | `createComplaint`, `getTicketStatus`, `runTroubleshootFlow`, `escalateToHuman` | `support_faq` |
| Coverage | `checkCoverage`, `getOutageStatus`, `getDeviceSettings`, `guideSimSwap` | `technical_kb` |

(* = two-phase, code-enforced consent gate, see Section 4)

**A 5th collection, `compliance_policy`, exists but is NOT a bindable tool for any sub-agent** — it's read only by the guardrail layer via `compliance_policy_search()` (a direct, non-LangChain-tool helper) for consent-script and do/don't-say rules. Do not bind it to a sub-agent as a regular RAG tool.

Each sub-agent gets **only its own** retriever tool (`rag_tools.py`: `build_billing_rag_tool`, `build_product_rag_tool`, `build_support_rag_tool`, `build_technical_rag_tool`) — do not give a sub-agent access to another domain's collection. Rationale (per the team): precise retrieval, lower hallucination risk, independently testable/evaluatable per domain.

`RetrievalTracker` records the best similarity (`1 - distance`) seen across a sub-agent turn's RAG calls; this is what the guardrail node's confidence check reads.

## 6. Configuration — Current Values (confirm before changing)

| Parameter | Default | Source |
|---|---|---|
| `GROQ_MODEL` | `llama-3.1-8b-instant` | `agent_graph.py`, env `GROQ_MODEL` |
| `GROQ_API_KEY` | *(required)* | env var, no fallback |
| `DEFAULT_MIN_SIMILARITY` | **0.3** | `agent_graph.py` — see flag below |
| `DEFAULT_NLU_CONFIDENCE` | 0.4 | `agent_graph.py` — orchestrator confidence floor before routing to a sub-agent |
| `DEFAULT_MAX_HISTORY_TURNS` | 6 | `agent_graph.py` |
| `MAX_TOOL_ITERATIONS` | 4 | `agent_graph.py` |
| `HANDOFF_LOG_PATH` | `handoff_log.jsonl` | `agent_graph.py` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | `chroma_setup.py` |
| `PERSIST_DIRECTORY` | `chroma_db` | `chroma_setup.py` |
| `KB_COLLECTIONS` | 5 named collections | `chroma_setup.py` |
| `DEFAULT_CHUNK_SIZE` / overlap | ~500 chars / ~100 chars | `content_manager.py`, tuned for the embedding model's 256-token window |
| `DB_PATH` | `nexatel_customers.db` | `customer_db.py` |

**UNRESOLVED FLAG — do not silently "fix" either direction**: an earlier design decision (before this repo existed) set the retrieval confidence threshold at **~0.75-0.85** for a telecom self-service context, specifically to prioritize safe human hand-off over confidently-wrong billing/account answers. The actual code's `DEFAULT_MIN_SIMILARITY` is **0.3** — a much more permissive bar. This is a real, unresolved discrepancy between an earlier safety-motivated decision and the current implementation. If you're an agent working on this repo: do not change this value on your own judgment in either direction — surface it and ask, since the gap is large enough that it changes real behavior (how often the system answers vs. hands off).

## 7. File-by-File Reference

- **`agent_graph.py`** — orchestrator + entry point. Owns system prompts (`ORCHESTRATOR_SYSTEM_PROMPT`, `SUBAGENT_SYSTEM_PROMPT_TEMPLATE`), `GraphState` TypedDict, node functions, routing functions, `run_tool_agent()` (the bounded tool-calling loop shared by all 4 sub-agent nodes). `main()` is a CLI REPL — prompts for phone + language once, loops on transcribed utterances. CLI flags: `--min_similarity`, `--max_history_turns`, `--show_debug`, `--language`, `--phone`. Guardrail regexes: `HUMAN_REQUEST_PATTERNS`, `UNCERTAINTY_PATTERNS`, `PII_LEAK_PATTERNS`, `AFFIRMATION_PATTERN`/`NEGATION_PATTERN`.
- **`tools.py`** — backend tool factories per sub-agent, closing over `SessionContext`. All tools read/write mock SQLite via `customer_db._connect()`. Static reference data: `TROUBLESHOOT_FLOWS` (5 issue types), `SLA_DAYS` (per ticket category), `DEVICE_SETTINGS` (Android/iPhone APN + VoLTE steps).
- **`rag_tools.py`** — `RetrievalTracker`, `_make_retriever()` (wraps `content_manager.read()` as a LangChain `@tool`), one `build_*_rag_tool()` factory per sub-agent, `compliance_policy_search()`.
- **`customer_db.py`** — mock SQLite DB. Tables: `customers`, `plans` (18 seeded, prepaid/postpaid/broadband), `subscriptions`, `bills`, `payments`, `tickets`, `coverage` (pincode → signal/outage). 10 seeded sample customers (phone `98765000xx`) covering every sub-agent demo path. CLI: `python customer_db.py` (create+seed), `--reset`.
- **`chroma_setup.py`** — centralized ChromaDB connection manager, module-level cached `get_client()`/`get_embedding_function()`/`get_collection()`. CLI: `python chroma_setup.py` (status), `--reset` (typed "yes" confirmation).
- **`build_kb.py`** — ingests each `kb_docs/*.md` into its own scoped collection via `content_manager.create_from_markdown()`, guided category labels per KB (`KB_SPEC` dict). CLI: `python build_kb.py` (idempotent upsert), `--reset`.
- **`content_manager.py`** — shared CRUD engine, the only file that talks to ChromaDB collections directly. Sentence-boundary chunking (NLTK `sent_tokenize`), heading-context propagation, content-addressed chunk IDs (SHA-256, idempotent upserts), guided or unsupervised (KMeans+TF-IDF) category tagging, language detection (`langdetect`).
- **`tts.py`** — `speak(text, lang, output_path, play)`, synthesizes via edge-tts, optionally plays via `playsound3`, deletes the temp mp3. Never raises — failures are logged and swallowed so a headless environment doesn't kill the call loop. `VOICES` dict covers 18 language codes (ta, hi, en-IN, fr, de, es, ja, ko, zh, it, ru, ar, te, kn, ml, mr, gu, ur), falls back to `en-IN-NeerjaNeural`.
- **`kb_docs/`** — `billing_policy.md`, `product_catalog.md`, `support_faq.md`, `technical_kb.md`, `compliance_policy.md` — hand-authored Nexatel policy/product docs, one per scoped collection.

**Setup**: `pip install -r requirements.txt` → `python build_kb.py` → `python customer_db.py` → set `GROQ_API_KEY` → `python agent_graph.py`.

**Superseded/deleted files** (do not resurrect or reference as current): `voice_rag_pipeline.py`/`voice_rag_pipelinev1.py` (→ `agent_graph.py`), `build_chroma_kb.py` (→ `build_kb.py`), `inspect_db.py` (→ `content_manager.py` methods), `chroma_setup (1).py` (→ `chroma_setup.py`), `demo.py` (removed, no replacement), `pdfs/*.pdf` (→ `kb_docs/*.md`).

---

## 8. Upstream Component (separate from this repo, feeds it transcripts)

This repo consumes `(transcript, language_code, phone_number)` — it does not do ASR/VAD/LID itself. The following was decided/measured for that upstream piece, in an earlier phase of the project, and remains relevant context even though it's not in this codebase:

- **ASR, two-tier**: `ai4bharat/indic-conformer-600m-multilingual` for Tamil/Hindi (does not support English), `openai/whisper-large-v3-turbo` for English + fallback for any other language.
- **Measured WER (200 samples/language, Mozilla Common Voice)**: Tamil — Whisper 62.44%, IndicConformer 26.06%. Hindi — Whisper 35.10%, IndicConformer 12.00%. English — Whisper ~3.8% (a later run was invalidated by a Kaggle dataset filename-collision bug; that fix is documented but the number needs reconfirming).
- **IndicConformer loading gotcha**: must use `AutoModel.from_pretrained(..., trust_remote_code=True)` + direct call `model(wav, lang_code, "ctc")` — NOT `transformers.pipeline()`, which fails on its custom config class.
- **Whisper language-ID gotcha**: Whisper's LID confuses Hindi and Urdu heavily (documented, acoustically near-identical languages) — any routing logic in the upstream service must treat `hi`/`ur` predictions as equivalent, or Hindi speakers get misrouted away from IndicConformer roughly 3 out of 4 times.
- **Code-switching**: neither ASR model was trained primarily on code-switched (Tanglish/Hinglish) speech; a transcript normalization/cleanup pass is the intended mitigation, not fixed at the ASR layer.
- **Whether/how the upstream service is being built to feed this repo, and its own confidence-passing format, is not specified here** — if an agent needs this, it's a question for the team, not something to assume from the ASR-only facts above.

## 9. Explicit Non-Goals

- Do not attempt real-time barge-in/interruption handling.
- Do not fine-tune the LLM, ASR, or TTS models as part of this repo's scope.
- Do not give any sub-agent access to another domain's RAG collection or tools.
- Do not let the LLM judge or paraphrase consent-script text, or make the yes/no confirmation decision — both are code-enforced by design (Section 4).
- Do not build real human-agent telephony — `handoff_log.jsonl` is the intended mock escalation queue.

## 10. Genuinely Open / Unresolved

- **The 0.3 vs ~0.75-0.85 retrieval threshold discrepancy (Section 6)** — needs an explicit team decision, not an agent's guess.
- Whether the upstream ASR/VAD/LID service is being actively built, by whom, and its exact interface into this repo.
- Whether the invalidated English WER re-run (Section 8) has since been reconfirmed.
- Whether a Whisper LoRA fine-tune on Svarah (Indian-accented English) is still planned — last known status was "time-boxed, cut if not clearly better within a day," outcome not recorded here.