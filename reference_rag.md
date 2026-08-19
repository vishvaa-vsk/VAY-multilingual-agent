# Reference: RAG + TTS Pipeline (VAY Multilingual Agent)

This document explains, in detail, how the live pipeline actually works: from a
customer's spoken/typed utterance, through NLU/routing, sub-agents and their
tools, RAG retrieval, guardrails, handoff-to-human logic, and finally TTS
speech synthesis.

It is based on a direct read of the source under `src/vay/` as of this
writing, not on the older narrative docs (`context-rag-tts.md`,
`project_context.md`, `rag-tts-evaluvation.md`, `tts-opt.md`). Where those
docs disagree with the current code, the discrepancy is called out explicitly
in **⚠️ Discrepancy** notes below — trust this document and the code over
them.

> **Note on dead code**: `src/vay/graph/nodes.py` (flat file), `src/vay/rag/retriever.py::HybridRetriever`,
> `src/vay/normalization/pass_llm.py`, and `src/vay/handoff/queue.py` implement an
> earlier prototype of this same pipeline. None of it is called from
> `build_graph()` or `app.py`. It's described briefly at the end for
> orientation, but everything above that section is the **live** system.

---

## 1. End-to-end flow

```
Browser mic (JS component)
   │  base64 WAV
   ▼
app.py: decode_audio()              ── vay/audio/audio_handler.py
   │  mono float32 tensor
   ▼
ASRRouter.route_and_transcribe()    ── vay/asr/router.py
   │  transcript + detected language
   ▼
build_graph().invoke(state)         ── vay/graph/workflow.py (LangGraph)
   │
   ├─ orchestrator_node             ── NLU + routing (one LLM call, JSON out)
   │
   ├─ (route to) billing / plans / complaints / coverage
   │      └─ run_tool_agent()       ── bounded tool-calling loop
   │             ├─ domain tools (DB reads/writes)
   │             └─ one scoped RAG search tool
   │
   ├─ guardrail_node                ── confidence / PII / human-request gate
   │
   ├─ (or, bypassing sub-agents) human_handoff / warning / chitchat / clarify / closing
   │
   └─ tts_node                      ── speaks final_reply (no-op in Streamlit UI)
   ▼
app.py: generate_text_to_speech()   ── vay/tts/engine.py (edge-tts, MP3)
   │  base64 MP3
   ▼
Browser plays audio, UI/history updated, loop back to step 1
```

Every turn rebuilds a fresh `GraphState` dict (see §7); the only thing that
truly persists across turns within a call is the `SessionContext` object
(identity, consent state, counters).

---

## 2. The agent graph (LangGraph)

Built in `src/vay/graph/workflow.py::build_graph()` using
`langgraph.graph.StateGraph`. Nodes:

| Node | Function | File |
|---|---|---|
| `orchestrator` | `orchestrator_node` | `graph/nodes/orchestrator.py` |
| `billing` | `billing_node` | `graph/nodes/agents.py` |
| `plans` | `plans_node` | `graph/nodes/agents.py` |
| `complaints` | `complaints_node` | `graph/nodes/agents.py` |
| `coverage` | `coverage_node` | `graph/nodes/agents.py` |
| `guardrail` | `guardrail_node` | `graph/nodes/utils.py` |
| `human_handoff` | `human_handoff_node` | `graph/nodes/utils.py` |
| `warning` | `warning_node` | `graph/nodes/utils.py` |
| `chitchat` | `chitchat_node` | `graph/nodes/utils.py` |
| `clarify` | `clarify_node` | `graph/nodes/utils.py` |
| `closing` | `closing_node` | `graph/nodes/utils.py` |
| `identity_mismatch` | `identity_mismatch_node` | `graph/nodes/utils.py` |
| `tts` | `tts_node` | `graph/nodes/utils.py` |

**Edges**: `START → orchestrator` → conditional routing → one of
`{billing, plans, complaints, coverage, human_handoff, warning, chitchat, clarify, closing, identity_mismatch}`.
The four domain nodes always flow into `guardrail`, which conditionally
routes to `human_handoff` or `tts`. Every other terminal node (including
`identity_mismatch`) flows straight to `tts`. `tts → END`.

### 2.1 Orchestrator — what it does

`orchestrator_node` (`graph/nodes/orchestrator.py`) is the single NLU +
routing brain. Each turn it makes **one LLM call** with
`ORCHESTRATOR_SYSTEM_PROMPT` plus trimmed conversation history plus the raw
transcript, and demands strict JSON back:

```json
{
  "language": "...", "intent": "...", "route": "...",
  "normalized_query": "...", "entities": {...},
  "confidence": 0.0, "sensitive": false,
  "aggressive": false, "call_end_requested": false
}
```

It also:
- **Pre-fetches account context** (`_fetch_account_context`) — customer
  profile, active subscription, outstanding balance, last 3 tickets — direct
  from the mock DB, and hands it to whichever sub-agent gets picked, so the
  sub-agent doesn't need a redundant "look up my own account" tool call.
- **Double-gates aggression detection**: the LLM's `aggressive=true` is only
  trusted if the raw transcript *also* matches a hard-coded profanity/threat
  regex (`ABUSIVE_LANGUAGE_PATTERN`) — added because the LLM alone
  false-positived on caps-lock/exclamation-mark frustration.
- **Forces pending-action continuity**: if a sub-agent staged a
  `session.pending_action` (e.g. an unconfirmed plan change) last turn,
  routing is forced back to that agent (`PENDING_ACTION_ROUTE`), skipping
  LLM routing for this turn entirely.
- **PII-disclosure guardrail** (`_contains_sensitive_pii`, `core_utils.py`):
  scans the *raw customer transcript* — independent of route/confidence, and
  independent of a pending-action confirmation — for either an Aadhaar/PAN/
  CVV/IFSC/card/bank-account keyword, or a 9–19 digit run (10 digits
  excluded, so it never fires on an ordinary phone number) shaped like an
  Aadhaar/card/bank-account number, tolerant of spoken/transcribed spacing
  ("1234 5678 9012 3456"). A hit forces `sensitive = True` for the turn, so
  it rides the existing `sensitive → human_handoff` branch below — the call
  **never reaches a sub-agent, RAG, or any tool**. `human_handoff_node`
  additionally redacts the digit run before writing the transcript to
  `handoff_log.jsonl` (matched on the `"PII disclosure:"` reason prefix), so
  the guardrail doesn't just relocate the leak into a log file.
- **Identity-mismatch guardrail** (`_normalize_phone` +
  `IDENTITY_MISMATCH_TEMPLATES`, `core_utils.py`): every backend tool acts
  only on `session.phone_number`, the number bound once at call setup (see
  §2.2/§7) — it's never an LLM-fillable argument. If the NLU's extracted
  `entities.phone_number` (or `phone`/`mobile_number`/`contact_number`)
  names a **different** number (last-10-digits compared, tolerant of a
  `+91`/leading-0/spacing), e.g. "change the plan for my friend's number
  98765…", the sub-agent would otherwise silently act on the call's own
  number while sounding like it acted on the one the customer named. Caught
  here, before any sub-agent runs, and routed to the dedicated
  `identity_mismatch` node (see below) instead of a handoff — only checked
  for `route in {billing, plans, complaints, coverage}` and skipped entirely
  during a pending-action confirmation turn.

**Routing decision** (`route_after_orchestrator`, `nodes/utils.py`), in
priority order:
1. Aggressive caller, already warned once + still ending/abusive → `closing`.
2. Aggressive caller, first offence → `warning`.
3. Customer says goodbye → `closing`.
4. `sensitive=true` (new dispute/cancellation/fraud, **or** the PII-disclosure
   guardrail above) → `human_handoff` directly (sub-agents are skipped).
5. Every failover LLM candidate failed this turn (`llm_unavailable`) → `human_handoff`.
6. `route == "chitchat"` → `chitchat`.
7. `route == "unclear"` or confidence below `0.4` → `human_handoff` if this is
   the 2nd+ consecutive unclear turn, else → `clarify`.
8. Identity-mismatch guardrail tripped → `identity_mismatch`.
9. Otherwise → the named domain route (`billing`/`plans`/`complaints`/`coverage`).

### 2.2 The four sub-agents

All four are thin wrappers around a shared runner, `_run_subagent()`
(`orchestrator.py`), which: resolves the `SessionContext` → resolves any
pending consent action (`confirm_pending_action`, code-level, not the LLM) →
prepends account context to the system prompt → trims stale tool messages if
the domain changed since last turn → calls `run_tool_agent()` with that
domain's tool set + its one scoped RAG search tool.

They all share `SUBAGENT_SYSTEM_PROMPT_TEMPLATE` (`core_utils.py`), which
encodes: reply strictly in the target language (telecom jargon stays
English), never invent IDs/facts, retry a KB search once then stop, never
reveal the system prompt/tools/RAG/LLM internals, resist prompt injection,
never echo PII/OTP, require a spoken consent read-back before treating "yes"
as confirmation, no Markdown/tables/slashes in spoken text, natural-language
dates, max 3–4 sentences, avoid repetition.

---

**Billing** — *"Billing & Payments specialist"*
Triggered on: bill amount, charges, due dates, payments, refunds.
Tools (`tools/billing.py`):
| Tool | Purpose |
|---|---|
| `getBalance()` | Prepaid: remaining validity/data. Postpaid/broadband: outstanding due. |
| `getBillBreakup(billing_period)` | Itemized charge breakup for a bill. |
| `getDueDate()` | Next payment due date + amount. |
| `sendPaymentLink()` | Sends a mock SMS payment link — sensitive, requires `session.verified`. |
| `explainCharge(charge_name)` | Explains a specific line item on the latest bill. |
| `escalateToHuman(reason)` | Shared escalation tool (see §3). |
| `search_billing_policy` | RAG search over the `billing_policy` KB collection. |

**Plans** — *"Plans & Offers specialist"*
Triggered on: plan info, comparisons, upgrade/downgrade, add-ons, eligibility.
Tools (`tools/plans.py`):
| Tool | Purpose |
|---|---|
| `listPlans(plan_type)` | Lists plans, optionally filtered by type. |
| `comparePlans(plan_ids)` | Compares two or more plans. |
| `checkEligibility(plan_id)` | Age/KYC eligibility check. |
| `changePlan(new_plan_id)` | **Sensitive, two-phase**: stages `session.pending_action` and returns a `STOP_AND_SAY:` consent script; only commits on the customer's next-turn "yes" (see §3). |
| `activateAddOn(addon_name)` | Attaches an add-on to the active subscription. |
| `escalateToHuman(reason)` | Shared escalation tool. |
| `search_product_catalog` | RAG search over the `product_catalog` KB collection. |

**Complaints** — *"Complaints & Service-Request specialist"*
Triggered on: new complaints, status checks on any existing ticket/dispute,
SLA questions, and real-time troubleshooting ("internet is slow", "calls keep
dropping" — deliberately routed here, not to Coverage).
Tools (`tools/complaints.py`):
| Tool | Purpose |
|---|---|
| `createComplaint(category, description)` | Logs a ticket, computes SLA due date. |
| `getTicketStatus(ticket_id)` | Status of a specific ticket or the caller's recent tickets. |
| `runTroubleshootFlow(issue_type)` | Canned step lists for call_drop / slow_data / sms_issue / cannot_call / recharge_not_reflecting. |
| `escalateToHuman(reason)` | Shared escalation tool. |
| `search_support_kb` | RAG search over the `support_faq` KB collection. |

**Coverage** — *"Coverage & Technical specialist"*
Triggered on: signal/coverage checks by pincode, device/APN/VoLTE setup,
SIM/eSIM swap — not "why is my existing service degraded" (that's
Complaints).
Tools (`tools/coverage.py`):
| Tool | Purpose |
|---|---|
| `checkCoverage(pincode)` | Signal/technology lookup for a pincode. |
| `getOutageStatus(pincode)` | Known-outage lookup. |
| `getDeviceSettings(device_type)` | APN/VoLTE steps for android/iPhone. |
| `guideSimSwap()` | Canned SIM/eSIM replacement steps. |
| `getTicketStatus(ticket_id)` | Same shape as Complaints' version, for network tickets. |
| `escalateToHuman(reason)` | Shared escalation tool. |
| `search_technical_kb` | RAG search over the `technical_kb` KB collection. |

### 2.3 The tool-calling loop

`run_tool_agent()` (`graph/tool_agent.py`), `MAX_TOOL_ITERATIONS = 6`. Each
iteration: LLM call with bound tools → if no tool calls, finalize the reply;
otherwise execute each requested tool (deduping exact-repeat and
near-duplicate RAG queries so the model can't loop on the same search),
append results as `ToolMessage`s, and loop. A `STOP_AND_SAY:`-prefixed tool
result short-circuits immediately — this is how `changePlan`'s consent script
is delivered verbatim, without the LLM getting a chance to "helpfully"
declare the change already done.

Post-processing on the final reply:
- `_detoxify_repetition()` — strips repeated/duplicated sentences (tuned for
  the small `llama-3.1-8b-instant` model's tendency to loop in Tamil/Hindi).
- `_enforce_language()` — checks the reply's Unicode script matches the
  target language and forces a translation retry if not.

### 2.4 Non-domain nodes

- **`guardrail_node`** — runs after every sub-agent turn: (a) confidence gate
  — if `retrieval_score < min_similarity` (default **0.3**) → handoff; (b)
  human-request pattern matched against the *customer's* transcript only
  (never the agent's own draft, to avoid false triggers); (c) an uncertainty
  phrase in the draft *and* `retrieval_score < 0.5` → handoff; (d) PII-leak
  guard (`password|pin|otp` regex in the *assistant's draft reply*) →
  handoff; (e) a compliance-consent keyword check against the
  `compliance_policy` KB collection (logs only, doesn't block). Note this is
  a narrower, different check than the orchestrator's PII-disclosure
  guardrail (§2.1) — this one catches the assistant echoing a
  password/PIN/OTP back, the orchestrator's catches the *customer*
  disclosing an Aadhaar/card/bank-account number in the first place, before
  any sub-agent (and therefore this node) ever runs.
- **`human_handoff_node`** — logs full turn context and returns a fixed,
  localized (not LLM-generated) handoff message. If `handoff_reason` starts
  with `"PII disclosure:"`, the logged transcript is redacted
  (`_redact_pii`) before being written to `handoff_log.jsonl`.
- **`identity_mismatch_node`** — speaks the fixed, localized identity-
  mismatch refusal computed by `orchestrator_node` (`identity_mismatch_reply`
  in state) and logs the same context shape as a handoff, for an auditable
  record — but does **not** set `handoff=True`; no sub-agent or human agent
  is ever invoked, the call simply continues with the customer told to have
  the other person call in themselves.
- **`warning_node`** — speaks the pre-built localized warning for a first
  aggressive/abusive offence.
- **`chitchat_node`** / **`clarify_node`** — fixed template replies, no LLM
  call.
- **`closing_node`** — speaks the pre-built call-cut message if one was
  staged (2nd offence); otherwise makes one short LLM call for a "thanks for
  calling" line, with a hardcoded fallback if that call fails.
- **`tts_node`** — speaks `final_reply`. **No-op under the Streamlit UI**
  (`app.py` calls the TTS engine directly instead — see §5); does real
  playback in the non-Streamlit run mode.

### 2.5 LLM & failover

`_llm()` (`core_utils.py`) returns a `_FailoverLLM` wrapper around a list of
candidate models (`_llm_candidates()`), default Groq `llama-3.1-8b-instant`,
auto-discovering extra API keys (`GROQ_API_KEY_2`, `_3`, …) as failover
slots, and supporting arbitrary OpenAI-compatible providers via `base_url`.
On a 429/5xx/model-not-found it switches to the next candidate process-wide
and sticks with it. If every candidate fails on a given turn,
`llm_unavailable=True` is set, which routes straight to `human_handoff`.

---

## 3. Handover / handoff logic

There is exactly **one live handoff mechanism**: a state flag plus an
append-only audit log — not a queue that anything actively consumes.

### 3.1 What triggers a handoff

Any of the following sets `GraphState["handoff"] = True`:

1. Orchestrator marks the turn `sensitive` (new billing dispute, cancellation
   request, fraud report — **or** the PII-disclosure guardrail below firing)
   → routes straight to `human_handoff`, **skipping sub-agent/RAG entirely**.
   - Sub-case: the customer's transcript contains an Aadhaar/PAN/CVV/IFSC/
     card/bank-account keyword, or a 9–19 digit run shaped like one of those
     numbers (`_contains_sensitive_pii`, §2.1) — `handoff_reason` is set to
     `"PII disclosure: <reason>"` and the logged transcript is redacted.
2. Orchestrator detects an explicit human request — regex match on transcript
   ("human", "real person", "representative", "manager", "speak to
   someone…") or an LLM intent of `escalate`/`request_human_agent`/
   `human_handoff`/`speak_to_agent`.
3. `unclear_escalate` — 2 or more consecutive unclear/low-confidence turns
   (`session.consecutive_unclear >= 2`).
4. `llm_unavailable` — every failover LLM candidate exhausted this turn.
5. A sub-agent calls `escalateToHuman(reason)` — sets
   `session.escalation_requested`, surfaced into `GraphState["handoff"]` by
   `_run_subagent`.
6. The tool-calling loop degrades (LLM call fails outright, or produces
   nothing usable) — `_run_subagent` sets `handoff=True` defensively.
7. `guardrail_node` fires on: low `retrieval_score`, a human-request pattern
   in the transcript, an uncertainty phrase combined with a middling
   retrieval score, or a PII leak in the draft reply.

### 3.2 What gets handed off

`log_handoff()` (`core_utils.py`) appends a JSON line to `handoff_log.jsonl`
per handoff, containing: phone number, transcript, intent, entities,
normalized query, route, reason, the draft reply at the moment of handoff,
and a UTC timestamp — i.e. everything a human agent would need to pick up the
call cold, without re-asking the customer to repeat themselves. This is a
mock/stand-in for a real escalation queue; nothing in the live code currently
reads `handoff_log.jsonl` back out — it's an audit trail, not a live
dashboard feed. The customer hears a fixed, hand-written, per-language
template (`HANDOFF_MESSAGE_TEMPLATES`), deliberately *not* LLM-generated, so
the handoff message itself can never hallucinate.

**Language fallback for these fixed templates**: `localized()` (`core_utils.py`)
looks up `language` in the given `*_TEMPLATES` dict first; if that language
has no hand-written entry, it translates the dict's `en` entry via one LLM
call (`{placeholder}` tokens preserved verbatim) rather than silently
dropping the customer to English mid-conversation. Translations are cached
process-wide per `(text, language)` — a small, fixed, finite set of pairs —
so a given template/language combination only ever pays for one LLM round
trip, not one per turn. Falls back to the English source itself only if that
translation call fails outright. This applies to every `*_TEMPLATES` dict
looked up through `localized()`, including `tools/session.py`'s
`CONSENT_TEMPLATES`/`CONFIRM_*_TEMPLATES` (via a function-local import there,
to avoid a module-load-order issue between `tools` and `graph`).

### 3.3 A related but distinct mechanism: two-phase consent

Sensitive DB-mutating tools (`changePlan`, `sendPaymentLink`) don't hand off
to a human — they hand control back to the *code*, not the LLM, for
confirmation:
1. The tool call only **stages** the action (`session.pending_action`: tool
   name, args, a spoken summary, language) and returns a `STOP_AND_SAY:`
   sentinel with a fixed consent script.
2. On the customer's *next* turn, the orchestrator (via `PENDING_ACTION_ROUTE`)
   forces routing back to the owning agent, and `confirm_pending_action()`
   — code, not the LLM — checks the raw transcript for the literal word
   "yes" (`AFFIRMATION_PATTERN`) vs "no" (`NEGATION_PATTERN`) before actually
   committing the DB write.

Rationale stated in comments: a small, cheap LLM cannot be trusted to
reliably gate a real account mutation on genuine customer consent, so that
decision is pulled out of the model entirely.

### 3.4 A third distinct mechanism: identity-mismatch refusal

Neither a handoff (§3.1–3.2) nor consent (§3.3) — this is a **code-level
refusal that keeps the call going**, for when the customer's own words name
a phone number other than `session.phone_number` (the one verified at call
setup). Every backend tool (`build_*_tools(session)` factories in `tools/`)
closes over that one `SessionContext` and only ever reads/writes its bound
number; nothing about it is LLM-fillable. Without this guardrail, a request
like "change the plan for my friend's number 98765…" would still land on
`session.phone_number` while the sub-agent's reply reads as if it acted on
the number the customer named — a confusing, silent identity mismatch,
worse than an explicit refusal.

1. `orchestrator_node` normalizes the NLU's `entities.phone_number` (or
   `phone`/`mobile_number`/`contact_number`) and `state["phone_number"]` to
   their last 10 digits (`_normalize_phone`) and compares them — only for
   `route in {billing, plans, complaints, coverage}`, and skipped during a
   pending-action confirmation turn.
2. On a mismatch, `identity_mismatch_reply` is set to a fixed, localized
   refusal (`IDENTITY_MISMATCH_TEMPLATES`) and `route_after_orchestrator`
   sends the turn to `identity_mismatch_node` instead of the sub-agent — no
   RAG search, no tool call, no `handoff=True`.
3. `identity_mismatch_node` logs the context (phone number, transcript,
   entities, route) for audit purposes and speaks the refusal, telling the
   customer that person needs to call in themselves.

### 3.5 Legacy handoff (not wired in)

`src/vay/handoff/queue.py::HandoffQueueManager` is a plain in-memory list
(`enqueue_ticket()`/`get_pending_tickets()`, no persistence, no consumer) used
only by the orphaned `graph/nodes.py::human_handoff_node`. It's a simpler,
earlier prototype of §3.1–3.2 and is not reachable from `build_graph()`.
Safe to ignore unless resurrecting that code path.

---

## 4. RAG pipeline

### 4.1 Storage

- **Vector store**: ChromaDB (`chromadb.PersistentClient`), persisted to
  `./chroma_db` (`rag/vector_store.py`).
- **Embedding model**: `all-MiniLM-L6-v2` (SentenceTransformers), cosine
  similarity space (`hnsw:space: cosine`). Falls back to Chroma's default
  embedding function if SentenceTransformer fails to load.
- **5 scoped collections** (`KB_COLLECTIONS`): `billing_policy`,
  `product_catalog`, `support_faq`, `technical_kb`, `compliance_policy` — one
  per sub-agent, plus a guardrail-only compliance collection. Each sub-agent's
  RAG tool only ever queries its own collection, by design — precise
  retrieval, low cross-domain hallucination risk, independently testable.

### 4.2 Ingestion

1. **Source → Markdown** (`rag/parsers.py`): URLs via `requests` +
   `BeautifulSoup` + `html2text`; PDFs via `pypdf`, page-by-page; local
   `.md` files skip conversion entirely.
2. **Chunking** (`rag/chunking.py::chunk_markdown`, `chunk_size=1000` chars,
   `chunk_overlap=150` chars):
   - Structure-aware: splits on heading boundaries and blank lines, then
     sentence-tokenizes (NLTK `punkt`, regex fallback), tracking the
     last-seen heading per sentence.
   - Greedily packs sentences up to `chunk_size`, but forces a new chunk on a
     heading change once the current chunk already has substantial content
     (`HEADING_SPLIT_MIN_CHARS = 120`) — this exists specifically to stop two
     unrelated sections/tables from being blended into one diluted
     embedding.
   - Oversized single sentences are hard-split by word count.
   - Overlap is re-applied afterward at sentence boundaries only (never
     mid-sentence), capped at `chunk_overlap`.
3. **TF-IDF descriptions** (`rag/tfidf.py`) — per-chunk keyword summaries
   (sklearn `TfidfVectorizer`, uni+bigrams, `sublinear_tf=True`) stored as
   chunk metadata, used for human-readable chunk previews/debugging.
4. **Categorization** (`rag/categorizer.py`) — either label-guided (cosine
   similarity of chunk embeddings against caller-supplied labels, top-2 kept
   above a 0.05 threshold) or unsupervised (KMeans, k=8, clusters
   auto-labeled by merged TF-IDF top terms). Domain-agnostic — works for any
   KB, not just telecom.
5. **Language detection** (`langdetect`, first ~500 chars) stored per chunk.
6. **Storage**: chunk IDs are SHA-256 content hashes, stored via
   `collection.upsert()` — re-ingesting unchanged content is a no-op
   (idempotent). Metadata per chunk includes source, title, chunk index/total,
   category, description, heading, language, word count, ingested-at
   timestamp, content hash.

### 4.3 Retrieval — real hybrid BM25 + vector search

`rag/hybrid.py::hybrid_query()`, called from `rag/manager_read.py::read()`:

1. **Dense leg**: Chroma vector query over a widened candidate pool
   (`pool_n = min(max(n_results*4, 10), total)`) so BM25-favored matches
   still have a chance to surface after fusion.
2. **Sparse leg**: a real `rank_bm25.BM25Okapi` index built per collection,
   cached and auto-invalidated when the collection's document count changes.
3. **Fusion**: weighted sum, `VECTOR_WEIGHT = 0.5`, `BM25_WEIGHT = 0.5`,
   after min-max normalizing raw BM25 scores against cosine similarity;
   candidates present in only one leg score 0 on the missing leg.
4. Top `n_results` by fused score are returned (fused score is converted back
   to a Chroma-style `distance = 1 - fused_score` for compatibility with
   existing `sim = 1 - distance` callers).
5. Metadata `where` filters (source, language) are supported on both legs.

**How results reach the LLM**: each sub-agent's RAG tool
(`rag/retriever.py`) is a `@tool`-wrapped closure that calls
`content_manager.read(query, n_results=3, collection_name=...)` and formats
hits as

```
[relevance=0.87 | Billing > Refund Policy]
<chunk text>

[relevance=0.81 | Billing > Payment Methods]
<chunk text>
```

returned as the tool's string result. This means retrieved context is
injected as a **`ToolMessage` in the conversation**, not spliced directly
into the system prompt — the model chooses to call the search tool (or not),
the same way it chooses any other tool.

**Confidence tracking**: a per-turn `RetrievalTracker` records the max
similarity seen across every RAG call that turn; this becomes
`GraphState["retrieval_score"]`, which `guardrail_node` compares against
`min_similarity` (live default **0.3**) to decide whether to hand off.

> **⚠️ Discrepancy**: `config.py::Settings.retrieval_confidence_threshold`
> (0.80) and `rag/retriever.py::HybridRetriever.confidence_threshold` (0.75)
> are *not* what governs live behavior — those belong to the unused
> `HybridRetriever` legacy path. The number that actually gates handoffs in
> production is `DEFAULT_MIN_SIMILARITY = 0.3` in `core_utils.py`
> (`app.py` also passes `min_similarity: 0.3` into the initial state). If you
> go tuning retrieval confidence, tune this one.
>
> Also: `rag/bm25.py::BM25SearchEngine` is an older stub that ignored the
> query and returned a fabricated score — it is **not** what powers hybrid
> search today. The real, correct BM25 leg lives inline in `rag/hybrid.py`
> via `rank_bm25`. `bm25.py` survives only as a standalone utility for
> ad-hoc in-memory document lists.

---

## 5. TTS pipeline

- **Engine**: `edge-tts` (Microsoft Edge's neural voices, unofficial free
  API), wrapped in `vay/tts/engine.py`.
- **Voice map** — 18 languages, each pinned to a specific neural voice, e.g.:

  | Lang | Voice |
  |---|---|
  | en | en-IN-NeerjaNeural |
  | hi | hi-IN-SwaraNeural |
  | ta | ta-IN-PallaviNeural |
  | te | te-IN-ShrutiNeural |
  | kn | kn-IN-SapnaNeural |
  | ml | ml-IN-SobhanaNeural |
  | mr | mr-IN-AarohiNeural |
  | gu | gu-IN-DhwaniNeural |
  | ur | ur-IN-GulNeural |
  | fr / de / es / ja / ko / zh / it / ru / ar | one native neural voice each |

  Any unmapped language falls back to `en-IN-NeerjaNeural`.

- **Script-aware safety net**: before picking a voice, `speak()` re-detects
  the actual Unicode script of the text itself (Tamil, Devanagari, Telugu,
  Kannada, Malayalam ranges) and overrides the declared language if it
  disagrees — so a stale/wrong `lang` tag can't make Tamil text get read by
  an English voice as garbled text/numbers.

- **Text cleanup**: `_clean_text_for_speech()` strips any Markdown that
  leaked through (bold markers, headings, bullet dashes) before synthesis —
  backstops the prompt-level rule that agents shouldn't produce Markdown in
  spoken replies in the first place.

- **Chunked, pipelined streaming (latency optimization)**:
  - `_split_into_speech_chunks()` splits the reply on sentence-ending
    punctuation across all 18 supported scripts (Latin `.!?`, Devanagari
    danda `। ॥`, CJK fullwidth `。！？`, Arabic `؟`, Urdu `۔`). Short replies
    (< 120 chars) are left as one chunk — no benefit to chunking a one-liner.
  - `_speak_pipelined()` synthesizes chunk 0, then **while chunk 0 is
    playing**, synthesizes chunk 1 in the background concurrently, and so on.
    Playback itself stays strictly sequential. This bounds time-to-first-audio
    by the synthesis time of just the first sentence, not the whole reply.
  - Documented improvement (`tts-opt.md`): ~1.85s → ~1.08s time-to-first-audio
    on a 3-sentence sample (~40% reduction), with the gap widening for longer
    replies.
  - **This pipelined path only applies to the play-audio-locally call shape**
    (`play=True, output_path=None`) — i.e. the non-Streamlit `tts_node` call.
    `app.py`'s Streamlit path always passes an explicit `output_path`
    (it needs bytes to send to the browser), which takes the older
    single-shot synchronous path: the whole reply is synthesized as one MP3
    before anything is returned. **The Streamlit UI does not currently get
    the pipelined-latency win** — worth knowing if you're chasing perceived
    latency in the web UI specifically.

- **Output format**: MP3 file (`edge_tts.Communicate(...).save(path)`),
  written to a temp file, then in the Streamlit path base64-encoded and sent
  to the browser component for playback.

- **Never fails the call**: all synthesis/playback errors are caught and
  logged, not raised — a missing audio device or an edge-tts outage doesn't
  crash the turn.

- **Prompt-side cooperation**: the sub-agent system prompt itself is written
  to produce TTS-friendly text — no Markdown/tables/pipes, rates spoken as
  words ("2 GB per day", not "2GB/day"), natural spoken dates instead of ISO
  dates, every sentence properly terminated (which the sentence-splitter
  relies on), replies capped at 3–4 sentences. TTS quality here is a
  joint effort between the LLM prompt and the engine's cleanup step, not the
  engine alone.

---

## 6. ASR / audio input (brief — feeds the pipeline above)

- **VAD**: `audio/vad.py::SileroVADStreamer` — Silero VAD over a 16kHz mic
  stream, with a pre-speech ring buffer so utterance onset isn't clipped, and
  a configurable end-of-utterance silence threshold (650ms).
- **ASR routing** (`asr/router.py::ASRRouter`): a single Groq Whisper call
  (`whisper-large-v3-turbo`, `verbose_json`) returns transcript + detected
  language + confidence together (deliberately one call, not detect-then-
  transcribe, for latency). If the detected language is one of 22 Tier-1
  Indic languages, a second local pass through `ai4bharat/indic-conformer-
  600m-multilingual` re-transcribes for accuracy (falls back to the Whisper
  text if empty). Tier-2 languages (including English) stop after the one
  Whisper call.
- **Hallucination filtering**: per-language blacklists of Whisper's known
  filler-phrase hallucinations ("thank you for watching", etc.) plus
  consecutive-duplicate-word dedup.
- **Confidence**: derived from average segment log-probability, discounted
  when `no_speech_prob` is high; near-silence is short-circuited to an empty
  transcript rather than guessed at.
- Normalization of the transcript into a clean, coreference-resolved query
  happens **inline inside the orchestrator's single LLM call** (the
  `normalized_query` field of its JSON output) in the live path — not via the
  separate `normalization/pass_llm.py::LLMTranscriptNormalizer`, which exists
  but is only wired into the unused legacy `nodes.py` graph.

---

## 7. State schema (`graph/state.py::GraphState`)

`TypedDict, total=False` — rebuilt fresh every turn except for `session`:

| Field | Meaning |
|---|---|
| `phone_number` | Caller identity, fixed at session start |
| `language` | Current turn's detected/caller language |
| `transcript` | Raw customer utterance this turn |
| `conversation_history` | LangChain message history for the call |
| `min_similarity` | Guardrail confidence threshold (app.py sets 0.3) |
| `intent`, `entities`, `normalized_query`, `nlu_confidence` | Orchestrator NLU output |
| `sensitive` | New dispute/cancellation/fraud, or the PII-disclosure guardrail (§2.1/§3.1) → forces handoff |
| `route` | billing / plans / complaints / coverage / chitchat / unclear |
| `call_end_requested` | Customer ending call, or 2nd-offence cut |
| `session` | `SessionContext` — the only cross-turn-persistent object |
| `retrieval_score` | Best RAG similarity seen this turn |
| `draft_reply` | Sub-agent's reply before the guardrail |
| `final_reply` | Guardrail-approved (or templated) reply, sent to TTS |
| `handoff`, `handoff_reason` | Escalation flag + logged reason |
| `llm_unavailable` | All failover LLM candidates failed this turn |
| `unclear_escalate` | Repeated-unclear-turn escalation flag |
| `aggressive_count` | Running count of abusive turns this call |
| `previous_route` | Prior turn's route (detects domain switches) |
| `warning_reply` | Pre-built localized warning/call-cut text |
| `identity_mismatch_reply` | Non-empty when `entities.phone_number` differs from the verified `session.phone_number` — routes to `identity_mismatch_node` instead of a sub-agent |

`SessionContext` (`tools/session.py`) carries what actually survives across
turns: `phone_number`, `verified`, `language`/`preferred_language`,
`escalation_requested`/`reason`, `pending_action` (staged sensitive action),
`consecutive_unclear`, `aggressive_count`, `last_route`.

---

## 8. Key config (`vay/config.py::Settings` + `core_utils.py` constants)

| Setting | Value | Notes |
|---|---|---|
| `whisper_asr_model` | `whisper-large-v3-turbo` | Tier-2 ASR (Groq-hosted) |
| `indic_asr_model` | `ai4bharat/indic-conformer-600m-multilingual` | Tier-1 ASR |
| `sample_rate` | 16000 Hz | Throughout ASR/VAD |
| `silence_duration_ms` | 650 | VAD end-of-utterance threshold |
| `tier1_languages` | 22 codes | Indic languages routed through IndicConformer |
| `DEFAULT_MIN_SIMILARITY` (core_utils.py) | **0.3** | **Live** guardrail confidence gate |
| `DEFAULT_NLU_CONFIDENCE` | 0.4 | Orchestrator confidence floor before clarify/handoff |
| `UNCLEAR_ESCALATION_THRESHOLD` | 2 | Consecutive unclear turns before forced handoff |
| `MAX_TOOL_ITERATIONS` | 6 | Tool-calling loop bound |
| `DEFAULT_MAX_HISTORY_TURNS` | 6 | Trimmed conversation history window (12 messages) |
| `PERSIST_DIRECTORY` (vector_store.py) | `chroma_db` | Chroma persistence path |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Chunk/query embeddings |
| `VECTOR_WEIGHT` / `BM25_WEIGHT` (hybrid.py) | 0.5 / 0.5 | Hybrid fusion weights |
| `retrieval_confidence_threshold` (Settings) | 0.80 | Legacy/unused path only |

Env vars: `GROQ_API_KEY` (+ `_2`, `_3`, … for failover), `GROQ_MODEL`,
`HF_TOKEN` (required for the gated IndicConformer model).

---

## 9. Legacy/unused code (for orientation only — not part of the live path)

- `src/vay/graph/nodes.py` — an earlier, simpler prototype graph
  (VAD → ASR → normalization → RAG → handoff-gate → LLM → TTS) using
  `HybridRetriever`, `LLMTranscriptNormalizer`, `HandoffQueueManager`. Not
  imported by `workflow.py` or `app.py`.
- `rag/retriever.py::HybridRetriever` — vector-only retrieval with its own
  0.75 confidence threshold; superseded by `rag/hybrid.py`'s real BM25+vector
  fusion used by the live sub-agent RAG tools.
- `normalization/pass_llm.py::LLMTranscriptNormalizer` — a dedicated
  code-switch (Tanglish/Hinglish) transcript cleanup LLM call; superseded by
  the orchestrator doing normalization inline as part of its single JSON
  call.
- `handoff/queue.py::HandoffQueueManager` — an in-memory ticket list with no
  persistence or consumer; superseded by the flag + `handoff_log.jsonl`
  mechanism in §3.

If any of this legacy code is resurrected or wired back in, this document
should be updated accordingly.
