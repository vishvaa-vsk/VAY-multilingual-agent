# RAG + TTS — Complete Technical Reference (VAY / Nexatel Voice Assistant)

> One-stop doc covering **how RAG works, how it's implemented, what algorithms are used, how the
> vector DB retrieval works, how chunking works, how the mock CRM works, how TTS works**, file
> paths to the real code, and a "what does what" function table. Written for a hackathon
> hiring-round deep-dive — every claim below is traced to a real file/line in this repo, not
> aspirational docs.
>
> Repo root: `VAY-multilingual-agent/` · Package root: [`src/vay/`](src/vay/)

---

## 0. TL;DR Architecture

```
Customer speech ──▶ ASR ──▶ Orchestrator (intent+route) ──▶ Sub-Agent (LLM + tools)
                                                                   │
                                          ┌────────────────────────┴───────────────────────┐
                                          ▼                                                  ▼
                                  Scoped RAG tool call                              SQL tool call (mock CRM)
                          (search_billing_policy, etc.)                     (getBalance, listPlans, createComplaint…)
                                          │                                                  │
                              Hybrid BM25 + Vector search                          SQLite nexatel_customers.db
                              over 1-of-5 ChromaDB collections
                                          │
                                          ▼
                              Grounded answer text ──▶ Guardrail ──▶ TTS (edge-tts) ──▶ spoken reply
```

Five independent knowledge domains, four domain sub-agents, one SQLite "CRM", one TTS engine.
Every sub-agent only ever sees **its own** RAG collection and **its own** SQL tools — this
isolation is deliberate (see §1.2) and is the main anti-hallucination design choice in the system.

---

## 1. RAG — Retrieval-Augmented Generation

### 1.1 Why RAG is used here

The LLM (Groq-hosted `llama-3.1-8b-instant` / `openai/gpt-oss-20b`) has **zero built-in knowledge**
of Nexatel's actual tariffs, plan prices, SLA policy, or TRAI compliance scripts — those are
business facts that change over time and must never be hallucinated on a phone call. RAG grounds
every plan/price/policy claim in a real retrieved document chunk instead of the model's parametric
memory.

### 1.2 Multi-Collection ("Scoped RAG") Architecture

Instead of one big vector index, VAY splits the knowledge base into **5 domain-scoped ChromaDB
collections**, defined in [`src/vay/rag/vector_store.py:32-38`](src/vay/rag/vector_store.py):

| Collection | Owner Agent | Content |
|---|---|---|
| `billing_policy` | Billing sub-agent | Tariffs, billing cycles, late fees, refund rules |
| `product_catalog` | Plans sub-agent | Prepaid/postpaid/broadband plans, add-ons, eligibility |
| `support_faq` | Complaints sub-agent | Troubleshooting guides, SLAs, complaint policy |
| `technical_kb` | Coverage sub-agent | APN/VoLTE setup, SIM/eSIM, 5G, outage FAQs |
| `compliance_policy` | Guardrail layer only | TRAI-mandated consent scripts, identity-check rules |

**Why scoped instead of one shared index:** a sub-agent given the whole KB could retrieve
off-topic chunks (e.g. a billing question pulling in a technical-KB chunk) and the LLM would
happily narrate them as if relevant — this is a classic RAG hallucination vector. Scoping cuts
retrieval search space to only what that agent is authorized to reason about, which is both a
precision win and a safety boundary. `compliance_policy` is never exposed as an LLM-callable tool
at all — it's called directly by the guardrail code path (`compliance_policy_search()` in
[`src/vay/rag/retriever.py:115-123`](src/vay/rag/retriever.py)), so the LLM can't selectively
ignore it.

Source knowledge lives as plain Markdown in [`data/kb/`](data/kb/) (one file per collection,
domain-editable without touching code).

### 1.3 Retrieval Algorithm — Hybrid BM25 + Dense Vector Fusion

**File:** [`src/vay/rag/hybrid.py`](src/vay/rag/hybrid.py) — function `hybrid_query()`.

Pure dense embedding search with a small model (`all-MiniLM-L6-v2`, 384-dim) is weak at exact
alphanumeric anchors — plan codes (`PPD_VALUE`), rupee amounts (`₹299`), data caps (`2 GB/day`).
VAY fuses two retrieval signals per query:

1. **Dense vector leg** — ChromaDB's own `.query()` (cosine similarity under the hood, HNSW
   index). Similarity is derived as `sim = 1 - cosine_distance`.
2. **Sparse keyword leg** — a real `rank_bm25.BM25Okapi` index, built fresh from
   `collection.get()` (all chunks) and **cached per collection**, invalidated automatically when
   `collection.count()` changes (`_get_index()`, [`hybrid.py:63-89`](src/vay/rag/hybrid.py)).
   Tokenizer: lowercase regex `[a-zA-Z0-9஀-௿ऀ-ॿ]+` — keeps Tamil/Devanagari codepoints intact
   too, future-proofing for non-English KB content.

**Fusion algorithm** (`hybrid_query()`, [`hybrid.py:112-184`](src/vay/rag/hybrid.py)):

```
pool_n         = max(n_results * 4, 10)              # wide candidate net
vector_sims    = 1 - chroma_cosine_distance            # dense leg, top pool_n
bm25_raw       = BM25Okapi.get_scores(query_tokens)     # sparse leg, over WHOLE collection
bm25_norm      = min_max_normalize(bm25_raw)            # [0,1] over the candidate pool
fused_score    = 0.5 * vector_sim  +  0.5 * bm25_norm   # VECTOR_WEIGHT / BM25_WEIGHT
rank           = sort candidates by fused_score, desc
return top_k, with distance = 1 - fused_score            # so downstream code (sim = 1-dist) is unchanged
```

Candidates found by *either* leg (vector-only or BM25-only) enter the fused pool with the missing
leg's score defaulting to 0 — so a strong exact-keyword hit the embedding ranked poorly can still
surface after fusion, and vice versa. `where` metadata filters (source/language) are supported on
both legs — BM25's is a manual post-filter (`_matches_where()`) since `BM25Okapi` has no native
metadata filtering.

**Why 0.5/0.5 and not RRF (Reciprocal Rank Fusion):** a straightforward min-max-normalized
weighted sum was chosen for a KB this size (tens–low hundreds of chunks per collection) — simple,
tunable, and cheap to reason about. This is a legitimate improvement area (see §6).

⚠️ **This wasn't always true hybrid search** — see [`hybrid.py:8-25`](src/vay/rag/hybrid.py)'s
own doc comment: the *original* implementation only had a BM25 stub
([`src/vay/rag/bm25.py`](src/vay/rag/bm25.py)) that returned input-order documents with a
**fabricated** `0.80 - i*0.05` score and never looked at the query — nothing in the live path
called it. `manager_read.read()` was doing plain vector-only `collection.query()` the whole time,
despite docs claiming "hybrid." This was caught and fixed during an internal audit
([`rag-tts-evaluvation.md`](rag-tts-evaluvation.md) §2.3) — worth knowing because it's a good
example of "measure your RAG pipeline, don't trust the docstring."

### 1.4 Vector Database — ChromaDB

**File:** [`src/vay/rag/vector_store.py`](src/vay/rag/vector_store.py)

- **Engine:** `chromadb.PersistentClient`, on-disk at [`chroma_db/`](chroma_db/) (SQLite-backed
  storage + per-collection HNSW index segments — visible as the UUID-named folders under
  `chroma_db/`).
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` via ChromaDB's
  `SentenceTransformerEmbeddingFunction` (falls back to Chroma's `DefaultEmbeddingFunction` if
  the package import fails) — 384-dim sentence embeddings, 256-token effective context window.
- **Similarity metric:** cosine (`metadata={"hnsw:space": "cosine"}` at collection creation,
  [`vector_store.py:86`](src/vay/rag/vector_store.py)) — ChromaDB's HNSW (Hierarchical Navigable
  Small World graph) index gives approximate-nearest-neighbor search over the embedding space.
- **Caching:** client, embedding function, and per-`(persist_dir, collection_name)` collection
  handles are cached at module level so switching between the 5 scoped collections mid-session
  (which the orchestrator/sub-agents do constantly) never reloads the embedding model.
- **CLI status/reset:** `python -m vay.rag.vector_store` or `python -m vay.rag.vector_store --reset`.

### 1.5 Chunking Algorithm

**File:** [`src/vay/rag/chunking.py`](src/vay/rag/chunking.py) — function `chunk_markdown()`.

Structure-aware, sentence-boundary chunking (not naive fixed-size character slicing):

1. **Structural split** — markdown is split on heading boundaries (`\n#{1,6}\s`) and blank lines
   into blocks.
2. **Sentence segmentation** — each block is tokenized into sentences via `nltk.sent_tokenize`
   (regex fallback `(?<=[.!?])\s+` if nltk isn't available), and every sentence remembers the
   **last markdown heading seen** (`sentence_headings`) so section context survives chunking.
3. **Greedy packing with a section-boundary guard** — sentences are packed into a chunk up to
   `chunk_size` (default **1000 chars**), BUT: if the heading changes AND the current chunk
   already has ≥120 chars of real content (`HEADING_SPLIT_MIN_CHARS`), the chunk is force-flushed
   rather than silently blending two different sections/tables together. Small heading-only
   fragments still merge forward so the KB doesn't fill up with near-empty chunks.
   - *(This guard was added after a real bug: greedy packing used to glue an unrelated intro
     paragraph to the next section's pricing table, which diluted the embedding enough that the
     exact-match plan chunk dropped out of the top-5 results entirely — see §6/evaluation doc.)*
4. **Oversized-sentence hard split** — any single sentence longer than `chunk_size` is hard-split
   by whitespace-joined words so nothing silently overflows the model's context.
5. **Sentence-boundary overlap** — default **150 chars** overlap (`DEFAULT_CHUNK_OVERLAP`,
   [`chunking.py:49`](src/vay/rag/chunking.py); the doc-comment target used elsewhere is ~100),
   carried over as *whole trailing sentences* from the previous chunk (never a mid-sentence
   character cut) so a chunk boundary never orphans a fact mid-thought.
6. **Language detection** — `detect_language()` uses `langdetect` on the first 500 chars, tagging
   each ingested document's ISO 639-1 code into chunk metadata.

Result: **65 chunks total** across all 5 collections after the last rebuild, averaging
537–668 chars/chunk (`billing_policy` 17 @ 537c avg, `product_catalog` 11 @ 646c avg, `support_faq`
15 @ 595c avg, `technical_kb` 11 @ 609c avg, `compliance_policy` 11 @ 668c avg) — each chunk now
scoped to one section/table, not a blend.

### 1.6 Ingestion Pipeline

**File:** [`src/vay/rag/manager_ingest.py`](src/vay/rag/manager_ingest.py) — `_ingest_markdown()`,
driven by [`scripts/build_kb.py`](scripts/build_kb.py).

```
markdown text
   │
   ├─▶ detect_language(text[:1000])                       chunking.py
   ├─▶ chunk_markdown(text, size=1000, overlap=150)        chunking.py  →  (chunk_text, heading) tuples
   ├─▶ compute_tfidf_descriptions(chunks, top_n=10)        tfidf.py     →  keyword summary per chunk
   ├─▶ categorize_chunks(chunks, labels?)                  categorizer.py → category tags per chunk
   └─▶ for each chunk:
          id        = SHA-256(chunk_text)                  ← content-addressed, idempotent re-ingest
          metadata  = {source, source_type, title, chunk_index, total_chunks,
                       category, description, heading, language, num_words,
                       ingested_at, content_hash}
       collection.upsert(ids, documents, metadatas)          ChromaDB
```

**Content-addressed IDs** (`_chunk_id()` = SHA-256 of chunk text) make re-running `build_kb.py`
fully idempotent — unchanged chunks upsert to the same ID (no duplication), changed chunks get a
new ID and the old one is orphaned only if the source is explicitly `update()`d/`delete()`d.

### 1.7 Categorization — Two Algorithms

**File:** [`src/vay/rag/categorizer.py`](src/vay/rag/categorizer.py)

| Mode | Function | Algorithm |
|---|---|---|
| **Guided** (labels supplied) | `categorize_chunks_guided()` | Embeds each chunk and each label with the same SentenceTransformer, ranks labels by **cosine similarity** to the chunk embedding, keeps the top-`k` labels above a `sim > 0.05` floor. Domain-agnostic — pass any label set (telecom/medical/legal/…). |
| **Unsupervised** (no labels) | `categorize_chunks_unsupervised()` | **KMeans** clustering (`sklearn.cluster.KMeans`, `k=min(8, n_chunks)`) over L2-normalized chunk embeddings, then each cluster is auto-labeled by running **TF-IDF** (`compute_tfidf_descriptions()`, [`src/vay/rag/tfidf.py`](src/vay/rag/tfidf.py)) over the concatenated cluster text and taking its top terms. |

`categorize_chunks()` dispatches between the two based on whether `category_labels` was passed.

### 1.8 Confidence Thresholding & Safety Escalation

Every RAG tool call records the best similarity score of that turn on a `RetrievalTracker`
([`src/vay/rag/retriever.py:25-39`](src/vay/rag/retriever.py)):

```python
def record(self, score: float) -> None:
    self.called = True
    self.last_score = max(self.last_score, score)
```

If the best retrieval similarity for the turn falls below the configured
**`retrieval_confidence_threshold` (τ = 0.80, range 0.75–0.85 by design)** —
[`src/vay/config.py:26`](src/vay/config.py) — the guardrail layer treats the draft answer as
insufficiently grounded and routes to a human-handoff / clarification path instead of letting an
under-grounded answer reach the caller. This is the system's core anti-hallucination safety net —
retrieval confidence gates whether an LLM-generated answer is trusted to go out over voice at all.

### 1.9 Retrieval Query Flow (End-to-End)

**File:** [`src/vay/rag/manager_read.py`](src/vay/rag/manager_read.py) — `read()`, wrapped by
per-agent tool factories in [`src/vay/rag/retriever.py`](src/vay/rag/retriever.py).

```
sub-agent LLM calls search_billing_policy("what's the late payment fee")
        │
        ▼
retriever.py: _make_retriever()._retriever(query)
        │
        ▼
manager_read.read(query, n_results=3, collection_name="billing_policy")
        │  builds `where` filter (source/language, additive AND)
        ▼
hybrid.hybrid_query(collection, query, n_results, where)   ← §1.3 fusion algorithm
        │
        ▼
_format_hits(results, tracker) → "[relevance=0.87 | Late Payment Policy]\n<chunk text>\n\n..."
        │  (RetrievalTracker.record() called per hit — feeds §1.8's confidence gate)
        ▼
returned as the tool's string output → LLM incorporates it into the grounded reply
```

### 1.10 KB Management Commands

```bash
uv run python scripts/build_kb.py               # ingest all data/kb/*.md into ChromaDB
uv run python scripts/build_kb.py --reset        # wipe + rebuild all 5 collections
uv run python scripts/manage_kb.py --status                              # chunk counts per collection
uv run python scripts/manage_kb.py --search "QUERY" --collection X       # test hybrid retrieval, no LLM
uv run python scripts/manage_kb.py --rebuild <collection|all>            # wipe + re-ingest + invalidate BM25 cache
```

### 1.11 How LangChain / LangGraph Is Actually Used Here

A common evaluator trap question: *"you say LangGraph — walk me through where."* Be precise about
what's LangChain/LangGraph vs. hand-rolled, because this codebase is **not** a
`create_react_agent`/`AgentExecutor` black box — the tool loop is written by hand for full control
over the anti-hallucination guards (§1.8, dedup, language conformance).

**LangGraph — the actual state machine.** [`src/vay/graph/workflow.py`](src/vay/graph/workflow.py)
`build_graph()` builds a real `langgraph.graph.StateGraph(GraphState)` — one node per stage
(`orchestrator`, `billing`, `plans`, `complaints`, `coverage`, `guardrail`, `human_handoff`,
`identity_mismatch`, `warning`, `chitchat`, `clarify`, `closing`, `tts`), wired with
`add_conditional_edges()` for the two real branch points:

```
START → orchestrator ─(route_after_orchestrator)─→ {billing|plans|complaints|coverage|
                                                       human_handoff|identity_mismatch|
                                                       warning|chitchat|clarify|closing}
{billing,plans,complaints,coverage} → guardrail ─(route_after_guardrail)─→ {human_handoff|tts}
{human_handoff,identity_mismatch,warning,chitchat,clarify,closing} → tts → END
```

`GraphState` ([`src/vay/graph/state.py`](src/vay/graph/state.py)) is the typed dict threaded
through every node (transcript, language, phone number, route, reply, escalation flags,
`RetrievalTracker`, etc.) — this is LangGraph's core contract: nodes are plain functions
`(state) -> state`, edges decide the next node, `graph.compile()` returns a runnable. One call =
one pass through this compiled graph per customer utterance (looped by the call harness in
[`scripts/run_voice.py`](scripts/run_voice.py) / [`scripts/run_assistant.py`](scripts/run_assistant.py)).

**LangChain — the pieces actually used, not the whole framework:**

| Piece used | Where | Why |
|---|---|---|
| `langchain_core.tools.tool` decorator | [`rag/retriever.py`](src/vay/rag/retriever.py), every file under [`tools/`](src/vay/tools/) | Turns a plain Python function into an LLM-callable tool object with a name/schema the model can invoke — this is what makes `search_billing_policy(query)`, `getBalance()`, `changePlan(new_plan_id)` etc. tool-callable at all. |
| `langchain_groq.ChatGroq` / `langchain_openai.ChatOpenAI` | [`graph/core_utils.py`](src/vay/graph/core_utils.py) | Chat-model wrapper around the Groq API (native) or any OpenAI-compatible endpoint (fallback/failover candidates — see `_FAILOVER_CANDIDATES_SPEC`) — gives a uniform `.invoke()` / `.bind_tools()` interface regardless of provider. |
| `llm.bind_tools(tools)` | [`graph/tool_agent.py:413`](src/vay/graph/tool_agent.py) | LangChain's mechanism for attaching a tool schema list to a chat model so the model's response can include structured `tool_calls` instead of only free text. |
| `HumanMessage` / `SystemMessage` / `AIMessage` / `ToolMessage` | [`graph/tool_agent.py`](src/vay/graph/tool_agent.py), [`graph/nodes/orchestrator.py`](src/vay/graph/nodes/orchestrator.py) | LangChain's typed chat-message objects — the message list passed to `.invoke()` and returned from it. |

**What is *not* LangChain:** the tool-calling loop itself. `run_tool_agent()`
([`graph/tool_agent.py:392-539`](src/vay/graph/tool_agent.py)) is a **hand-written bounded loop**
(`for iteration in range(MAX_TOOL_ITERATIONS)`) around `bound_llm.invoke()` — not
`langchain.agents.AgentExecutor` or `create_tool_calling_agent`. It manually:
1. Invokes the LLM with the running message list.
2. If the response has `tool_calls`, executes each one (`tool_fn.invoke(call["args"])`), appends a
   `ToolMessage`, and loops.
3. If not, treats the content as the final reply — after running it through the repetition-dedup
   (`_detoxify_repetition`), script-conformance (`_enforce_language`) and `STOP_AND_SAY` consent-
   verbatim guards described in §1.8/§3.
4. Exact-signature **and** near-duplicate (Jaccard token overlap ≥0.5) tool-call dedup —
   `seen_calls`/`seen_queries`/`_is_near_duplicate_query()` — stops the model from burning its
   iteration budget re-searching the same RAG query reworded three ways.

**Why hand-rolled instead of `AgentExecutor`:** none of the off-the-shelf LangChain agent loops
give first-class hooks for "detect the model produced a repeated/looping non-English reply and
truncate it," "verify the reply's Unicode script matches the target language and force a
translation retry," or "treat this specific tool's sentinel return string as verbatim output, not
something the LLM gets to see/paraphrase" (`STOP_AND_SAY:` in `changePlan`/`sendPaymentLink`) —
all voice-call-specific safety requirements that needed to sit *inside* the loop, not bolted on
after. This is a legitimate, defensible design answer if an evaluator asks "why not just use
`create_react_agent`."

**Orchestrator vs. sub-agent LLM calls** — two distinct LangChain-wrapped calls per turn:
- **Orchestrator** ([`graph/nodes/orchestrator.py`](src/vay/graph/nodes/orchestrator.py)) — one
  `ChatGroq`/`ChatOpenAI` `.invoke()` call, no tools bound, prompted for structured
  intent/route/entity JSON (NLU) — decides which of the 10 next nodes to route to.
- **Sub-agent** (`_run_subagent()` → `run_tool_agent()`) — a *fresh* `bind_tools()` call scoped to
  that sub-agent's own tool list (its domain SQL tools + its own scoped RAG tool, never another
  domain's) — this is where the scoped-RAG isolation from §1.2 is actually enforced in code: a
  sub-agent physically cannot call another domain's retriever because it was never bound as a tool
  for that LLM invocation.

---

## 2. Mock CRM (SQLite "nexatel_customers.db")

VAY doesn't call a real telecom billing system — it ships a **self-contained, realistic SQLite
mock CRM** that every domain tool actually reads from and writes to (not a stub — mutations
persist across process restarts).

### 2.1 Schema

**File:** [`src/vay/tools/db_schema.py`](src/vay/tools/db_schema.py) · DB file:
[`src/vay/tools/nexatel_customers.db`](src/vay/tools/nexatel_customers.db)

| Table | Key columns | Purpose |
|---|---|---|
| `CUSTOMERS` | `phone_number` (PK), `full_name`, `dob`, `verified`, `city`, `pincode`, `account_type`, `language_pref` | Identity + KYC + language preference |
| `PLANS` | `plan_id` (PK), `name`, `plan_type`, `price`, `validity`, `data_benefit`, `voice_benefit`, `sms_benefit`, `benefits` | Master plan catalog (prepaid/postpaid/broadband) |
| `SUBSCRIPTIONS` | `phone_number` (FK), `plan_id` (FK), `start_date`, `end_date`, `status`, `data_used_gb`, `data_limit_gb` | A customer's active/past plan(s) |
| `BILLS` | `bill_id` (PK), `phone_number` (FK), `billing_period`, `amount`, `tax`, `total_amount`, `due_date`, `status` | Postpaid invoices |
| `PAYMENTS` | `payment_id` (PK), `phone_number` (FK), `amount`, `payment_date`, `payment_method`, `status` | Payment history |
| `TICKETS` | `ticket_id` (PK), `phone_number` (FK), `category`, `issue_description`, `status`, `priority`, `sla_days` | Complaint/service-request tracking |
| `COVERAGE` | `pincode` (PK), `area_name`, `network_2g/4g/5g`, `outage_status`, `expected_resolution` | Network coverage per area |

### 2.2 Seed Data

**File:** [`src/vay/tools/db_seed_data.py`](src/vay/tools/db_seed_data.py)

- **18 realistic plans** across prepaid/postpaid/broadband (`PLANS` list) — e.g. `PPD_VALUE`
  ₹299/28d/2GB-day, `POST_PRO` ₹999/30d/100GB+5G, `FIBER_ULTRA` ₹1999/30d/1Gbps.
- **11 seeded customers**, one per demo scenario/language combo (Tamil, Hindi, English; prepaid,
  postpaid, broadband), plus a fixed demo account `9876543210` used by
  `scripts/run_voice.py --phone 9876543210`.
- **Relative prepaid activation dates** — `_days_ago(n)` computes `activated_on` as `today - n
  days` at seed time rather than a hardcoded calendar date
  ([`db_seed_data.py:7-17`](src/vay/tools/db_seed_data.py)). *Why:* prepaid validity =
  `activated_on + validity_days`; a fixed past date silently drifts into "expired" as real time
  passes it — this bit the demo directly (3 of 5 prepaid accounts showed EXPIRED before the fix,
  see [`rag-tts-evaluvation.md`](rag-tts-evaluvation.md) §2.14).
- **4 sample tickets** across `open`/`in_progress`/`resolved` statuses and
  `network`/`billing`/`technical`/`service_request` categories, so "is my issue fixed" and
  "what's the status of my complaint" are both answerable out of the box.
- **9 coverage rows** keyed by pincode, including one deliberate `fault` outage row (`110002`) and
  one `planned_maintenance` row (`600020`) for demo-ready outage scenarios.

### 2.3 Domain Tool Catalog (LLM-callable functions over the mock CRM)

| File | Tool | What it does |
|---|---|---|
| [`tools/billing.py`](src/vay/tools/billing.py) | `getBalance()` | Balance/data-usage/validity (prepaid) or amount-due (postpaid) |
| | `getBillBreakup()` | Line-item bill: base tariff, taxes, add-ons, roaming fees |
| | `getDueDate()` | Current billing-cycle due date |
| | `sendPaymentLink()` | Generates a mock SMS payment link (Two-Phase Consent tool) |
| | `explainCharge(charge_type)` | Explains a recurring/one-time charge against billing policy |
| [`tools/plans.py`](src/vay/tools/plans.py) | `listPlans(plan_type=None)` | Lists active prepaid/postpaid/broadband plans |
| | `comparePlans(id1, id2)` | Side-by-side plan comparison |
| | `changePlan(new_plan_id)` | Stages a plan switch (Two-Phase Consent — commits on next-turn "yes") |
| | `activateAddOn(addon_id)` | Adds a data booster / roaming pack |
| | `checkEligibility(plan_id)` | Verifies migration eligibility |
| [`tools/complaints.py`](src/vay/tools/complaints.py) | `createComplaint(category, issue)` | Opens a ticket, computes SLA deadline from `SLA_DAYS` |
| | `getTicketStatus(ticket_id=None)` | Ticket status/SLA-remaining/resolution notes |
| | `runTroubleshootFlow(issue_type)` | Deterministic step-by-step diagnostic guide (slow data, call drop, SMS failure, etc.) |
| | `escalateToHuman(reason)` | Sets `session.escalation_requested` — explicit human handoff |
| [`tools/coverage.py`](src/vay/tools/coverage.py) | `checkCoverage(pincode=None)` | 4G/5G signal + tower status for a pincode |
| | `getOutageStatus(pincode=None)` | Active unplanned outages + ETA |
| | `getDeviceSettings(os_type)` | APN/VoLTE/eSIM setup steps (iOS/Android) |
| | `guideSimSwap()` | SIM replacement/activation guidance |
| [`tools/session.py`](src/vay/tools/session.py) | `build_escalate_tool()` | Shared factory that gives every sub-agent a uniform `escalateToHuman` |

**Two-Phase Consent pattern:** state-mutating actions (`changePlan`, `sendPaymentLink`,
`activateAddOn`) *stage* the action and require an explicit affirmative on the customer's next
turn (`confirm_pending_action`) before the SQLite row is actually written — verified live: the
old subscription flips to `status='cancelled'`, the new one inserts `status='active'`,
`conn.commit()` makes it durable.

### 2.4 Management Scripts

```bash
uv run python scripts/setup_app.py         # one-shot: seed DB + build KB + cache ASR weights + launch app.py
uv run python scripts/manage_db.py --seed  # (re)seed customers/plans/tickets/coverage
uv run python scripts/manage_db.py --phone 9876543210   # inspect one customer's record + subscriptions
uv run python scripts/manage_db.py --reset # wipe and reseed cleanly
```

---

## 3. TTS — Text-to-Speech Pipeline

**File:** [`src/vay/tts/engine.py`](src/vay/tts/engine.py) · docs:
[`docs/tts_pipeline.md`](docs/tts_pipeline.md)

### 3.1 Engine

VAY uses **Microsoft Edge Neural Voices** via the `edge-tts` Python package (`edge_tts.Communicate`
— an unofficial client for Microsoft's Edge read-aloud neural TTS service). This was chosen over a
local model because it needs **no local GPU/VRAM**, supports natural-sounding neural voices in 18
languages out of the box, and is free.

### 3.2 Language → Voice Map

`VOICES: dict[str, str]` ([`engine.py:35-54`](src/vay/tts/engine.py)) maps ISO 639-1 codes to
specific Microsoft neural voice IDs:

| Lang | Voice | Lang | Voice |
|---|---|---|---|
| ta | `ta-IN-PallaviNeural` | mr | `mr-IN-AarohiNeural` |
| hi | `hi-IN-SwaraNeural` | gu | `gu-IN-DhwaniNeural` |
| en | `en-IN-NeerjaNeural` (fallback voice) | bn | *(see docs table)* |
| te | `te-IN-ShrutiNeural` | ur | `ur-IN-GulNeural` |
| kn | `kn-IN-SapnaNeural` | fr/de/es/ja/ko/zh/it/ru/ar | locale-native neural voices |
| ml | `ml-IN-SobhanaNeural` | | |

Any language code not in the map falls back to `en-IN-NeerjaNeural` (`FALLBACK_VOICE`).

### 3.3 Script-Aware Voice Realignment

Multilingual generation can code-switch: the turn's detected language label might say `en` while
the actual reply text contains Tamil/Devanagari script. `speak()`
([`engine.py:263-272`](src/vay/tts/engine.py)) re-derives the voice from the **actual Unicode
codepoints in the text**, not the label:

```python
if re.search(r"[஀-௿]", text): effective_lang = "ta"          # Tamil block
elif re.search(r"[ऀ-ॣ०-ॿ]", text): effective_lang = "hi"  # Devanagari (excl. danda)
elif re.search(r"[ఀ-౿]", text): effective_lang = "te"        # Telugu
elif re.search(r"[ಀ-೿]", text): effective_lang = "kn"        # Kannada
elif re.search(r"[ഀ-ൿ]", text): effective_lang = "ml"        # Malayalam
```

Note the Devanagari range deliberately **excludes** U+0964/U+0965 (danda/double-danda) — those
punctuation marks are reused as sentence-enders by several *other* Indic scripts (Bengali,
Gujarati, Odia…), so matching on them alone previously mis-tagged non-Hindi sentences as Hindi
just for ending with a danda. This prevents the English voice from trying to pronounce Indic
Unicode phonetically as gibberish/numbers.

### 3.4 Latency Optimization — Sentence-Level Pipelining

**Problem:** naive TTS blocks until the *entire* multi-sentence reply is synthesized and written
to disk before any audio plays — 1.8–2.5s of dead air on a live call.

**Solution — `_split_into_speech_chunks()` + `_speak_pipelined()`:**

1. **Sentence-boundary splitting** (`_SENTENCE_SPLIT_RE`, [`engine.py:78`](src/vay/tts/engine.py)):
   a single regex covering every supported script's sentence-enders — Latin `. ! ?`, Devanagari
   `। ॥`, CJK fullwidth `。！？`, Arabic `؟`, Urdu `۔`. Text under **120 chars**
   (`_MIN_CHARS_FOR_CHUNKING`) is kept whole — chunking a one-liner only adds network overhead.
2. **Overlapping synthesize/play pipeline** (`_speak_pipelined()`,
   [`engine.py:169-210`](src/vay/tts/engine.py)):
   ```
   synthesize(chunk0) ──▶ play(chunk0) ──┐
                          synthesize(chunk1) [background, runs during play(chunk0)]
                                          ▼
                                     play(chunk1) ──┐
                                     synthesize(chunk2) [background]
                                                    ▼
                                                   ...
   ```
   Chunk *N+1* synthesis is kicked off as an `asyncio` task **before** the coroutine blocks on
   chunk *N*'s playback, so by the time playback of chunk N finishes, chunk N+1's audio is already
   on disk. Time-to-first-audio is now bounded only by chunk-0's synthesis time.
3. **Measured impact:** ~1.85s → ~1.08s time-to-first-audio for a 3-sentence reply (~40%
   reduction), with the gain scaling further on longer replies.

### 3.5 Barge-In (Interruptible Playback)

- **`stop_event: threading.Event`** is threaded through `speak()` → `_speak_pipelined()` →
  `_play_file()`, set by the call loop ([`scripts/run_voice.py`](scripts/run_voice.py)) the moment
  the caller starts talking over the assistant.
- **Non-blocking playback + polling**: audio plays via `playsound3` with `block=False`; a 50ms
  poll loop (`_BARGE_IN_POLL_S`) checks `sound.is_alive()` and `stop_event.is_set()`.
- **Immediate cut**: if the event fires mid-sentence, `sound.stop()` kills the audio subprocess
  immediately, the pending next-chunk synthesis task is cancelled, and the temp MP3 is deleted —
  no further chunks are synthesized or played.

### 3.6 Cleanup & Failure Handling

- Every synthesized chunk is a `tempfile.mkstemp(suffix=".mp3")` file, deleted in a `finally`
  block after playback (or on barge-in cancel) — no orphaned temp audio.
- `speak()` **never raises** — synthesis/playback exceptions are caught and logged so a headless
  environment or missing audio device degrades to text-only output instead of crashing the call.
- `_run_async()` handles the "already inside a running event loop" case (Streamlit/Gradio/Jupyter
  host) by falling back to a dedicated thread pool executor.

### 3.7 Public API

```python
from vay.tts.engine import speak, TTSEngine

speak("Your balance is 250 rupees", lang="ta", stop_event=call_stop_event)  # pipelined, plays immediately

engine = TTSEngine()
engine.synthesize("text", language="hi", output_path="out.mp3")   # writes file only, no playback
engine.speak("text", lang="en")                                    # convenience wrapper around speak()
```

CLI smoke test: `python -m vay.tts.engine` (prompts for text + language code interactively).

---

## 4. Function / Tool Reference Table

| Layer | File | Function/Class | What it does |
|---|---|---|---|
| **Chunking** | [`rag/chunking.py`](src/vay/rag/chunking.py) | `chunk_markdown()` | Sentence-boundary, heading-aware, overlap-carrying chunker |
| | | `detect_language()` | ISO 639-1 language detection via `langdetect` |
| | | `_tokenize_sentences()` | nltk sentence tokenizer with regex fallback |
| **Categorization** | [`rag/categorizer.py`](src/vay/rag/categorizer.py) | `categorize_chunks_guided()` | Cosine-similarity label assignment against supplied labels |
| | | `categorize_chunks_unsupervised()` | KMeans clustering + TF-IDF auto-labeling |
| **TF-IDF** | [`rag/tfidf.py`](src/vay/rag/tfidf.py) | `compute_tfidf_descriptions()` | Per-chunk keyword description via TF-IDF |
| **Vector store** | [`rag/vector_store.py`](src/vay/rag/vector_store.py) | `get_client()` / `get_collection()` | Cached ChromaDB client/collection accessors |
| | | `get_embedding_function()` | Cached `all-MiniLM-L6-v2` SentenceTransformer embedder |
| | | `init_db()` / `reset_db()` | Status print / full collection wipe |
| **Hybrid search** | [`rag/hybrid.py`](src/vay/rag/hybrid.py) | `hybrid_query()` | BM25 + vector fusion, cached BM25 index, min-max normalize, weighted sum, rerank |
| | | `_get_index()` | Builds/caches a `BM25Okapi` index per collection, auto-invalidated on count change |
| | | `invalidate_cache()` | Manual BM25 cache eviction (used after bulk re-ingest) |
| **Standalone BM25** | [`rag/bm25.py`](src/vay/rag/bm25.py) | `BM25SearchEngine` | Real BM25 engine over an in-memory `Document` list (not on the live query path) |
| **Ingestion** | [`rag/manager_ingest.py`](src/vay/rag/manager_ingest.py) | `_ingest_markdown()` | Full pipeline: chunk → TF-IDF describe → categorize → SHA-256 ID → upsert |
| | | `_chunk_id()` | Content-addressed SHA-256 chunk ID (idempotent re-ingest) |
| **Read** | [`rag/manager_read.py`](src/vay/rag/manager_read.py) | `read()` | Builds `where` filter, calls `hybrid_query()`, post-filters category substring |
| **Retriever tools** | [`rag/retriever.py`](src/vay/rag/retriever.py) | `build_billing_rag_tool()` etc. | LangChain `@tool`-wrapped scoped retrievers, one per sub-agent |
| | | `RetrievalTracker` | Records best similarity score per turn, feeds the confidence gate |
| | | `compliance_policy_search()` | Direct (non-LLM-tool) guardrail-only compliance lookup |
| | | `HybridRetriever` | Class wrapper exposing `.retrieve()` → `RetrievalResult` with confidence flag |
| **Parsers** | [`rag/parsers.py`](src/vay/rag/parsers.py) | URL/PDF → Markdown conversion for `create()` ingestion |
| **TTS core** | [`tts/engine.py`](src/vay/tts/engine.py) | `speak()` | Main entry point: script-aware voice pick, clean, chunk, pipeline-synthesize, play |
| | | `_split_into_speech_chunks()` | Multi-script sentence splitter for pipelined synthesis |
| | | `_speak_pipelined()` | Overlapping synthesize-next/play-current async pipeline |
| | | `_play_file()` | Non-blocking playback + barge-in polling + temp-file cleanup |
| | | `_clean_text_for_speech()` | Strips markdown formatting before synthesis |
| | | `TTSEngine` | Class wrapper (`.synthesize()` file-only, `.speak()` play) |
| **Mock CRM schema** | [`tools/db_schema.py`](src/vay/tools/db_schema.py) | table DDL | `CUSTOMERS/PLANS/SUBSCRIPTIONS/BILLS/PAYMENTS/TICKETS/COVERAGE` |
| **Mock CRM seed** | [`tools/db_seed_data.py`](src/vay/tools/db_seed_data.py) | `PLANS/CUSTOMERS/SUBSCRIPTIONS/TICKETS/COVERAGE` | Demo data; `_days_ago()` keeps prepaid validity fresh |
| **Billing tools** | [`tools/billing.py`](src/vay/tools/billing.py) | `getBalance/getBillBreakup/getDueDate/sendPaymentLink/explainCharge` | SQL reads + mock payment-link generation |
| **Plans tools** | [`tools/plans.py`](src/vay/tools/plans.py) | `listPlans/comparePlans/changePlan/activateAddOn/checkEligibility` | Catalog reads + two-phase-consent plan mutation |
| **Complaints tools** | [`tools/complaints.py`](src/vay/tools/complaints.py) | `createComplaint/getTicketStatus/runTroubleshootFlow/escalateToHuman` | Ticket CRUD + deterministic troubleshooting guides |
| **Coverage tools** | [`tools/coverage.py`](src/vay/tools/coverage.py) | `checkCoverage/getOutageStatus/getDeviceSettings/guideSimSwap` | Coverage/outage reads + setup guidance |
| **Session/shared** | [`tools/session.py`](src/vay/tools/session.py) | `build_escalate_tool()`, `SessionContext`, `SLA_DAYS` | Shared escalation tool factory, per-call session state, SLA lookup |
| **KB build CLI** | [`scripts/build_kb.py`](scripts/build_kb.py) | — | Ingest/rebuild all 5 collections from `data/kb/*.md` |
| **KB admin CLI** | [`scripts/manage_kb.py`](scripts/manage_kb.py) | — | `--status` / `--search` / `--rebuild` against live collections |
| **DB admin CLI** | [`scripts/manage_db.py`](scripts/manage_db.py) | — | `--seed` / `--phone` / `--reset` the mock CRM |
| **App bootstrap** | [`scripts/setup_app.py`](scripts/setup_app.py) | — | One-shot: seed DB + build KB + cache ASR weights + launch `app.py` |

---

## 5. Algorithms Used — Quick Reference

| Component | Algorithm | Library |
|---|---|---|
| Dense retrieval | Cosine similarity over sentence embeddings, ANN via HNSW | `chromadb`, `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Sparse retrieval | Okapi BM25 (term-frequency/inverse-doc-frequency ranking) | `rank_bm25.BM25Okapi` |
| Score fusion | Min-max normalization + weighted linear sum (0.5/0.5) | plain Python (`rag/hybrid.py`) |
| Sentence tokenization | NLTK Punkt tokenizer (regex fallback) | `nltk` |
| Chunk keyword description | TF-IDF (term frequency–inverse document frequency) | `sklearn` (`rag/tfidf.py`) |
| Unsupervised topic labeling | K-Means clustering over normalized embeddings | `sklearn.cluster.KMeans` |
| Chunk hashing | SHA-256 content addressing | `hashlib` |
| Language detection | `langdetect` (Naive-Bayes-based, char n-grams) | `langdetect` |
| Text-to-speech | Microsoft Edge neural voices (cloud) | `edge-tts` |
| Sentence-splitting for TTS pipelining | Multi-script punctuation regex | `re` |
| Barge-in detection | Threaded polling on `threading.Event` | stdlib `threading` |

---

## 6. Known Limitations & Future Improvement Ideas

*(Grounded in the internal audit — [`rag-tts-evaluvation.md`](rag-tts-evaluvation.md) §7 — plus
architectural observations. Good talking points for "how would you improve this" in an interview.)*

**RAG:**
- **Fusion weighting is static (0.5/0.5)** — could be learned/tuned per collection, or replaced
  with Reciprocal Rank Fusion (RRF) which is less sensitive to score-scale differences between
  BM25 and cosine similarity.
- **BM25 cache invalidation is count-based**, not content-hash-based — editing a KB file and
  re-ingesting with the *same* total chunk count could theoretically serve a stale cached index
  until an explicit `--rebuild`. A content-hash (or ChromaDB collection version/timestamp) based
  invalidation key would close this edge case.
- **Confidence threshold (τ=0.80) was not re-validated** after the hybrid-search fix changed the
  score distribution (family-plan-fee query went from 0.41→0.768 purely from the retrieval fix,
  not from τ changing) — worth an empirical re-tuning pass with labeled query/relevance pairs.
- **No re-ranker model** (e.g. a cross-encoder) — current pipeline is retrieve-then-fuse, no
  second-stage reranking of the top candidates, which is a common further-precision lever.
  A cross-encoder is more expensive per query but could safely run only on the (small) fused
  candidate set.
- **No chunk-level eval/gold-set regression test** — retrieval quality was validated by hand
  (documented in the audit) rather than a repeatable precision@k / MRR test suite tied to CI.
  Building `tests/test_rag_quality.py` with a fixed query→expected-chunk gold set would catch
  future chunking/fusion regressions automatically.
- **Embedding model is a small general-purpose model** (`all-MiniLM-L6-v2`, 384-dim) — a
  domain-fine-tuned or larger embedding model (e.g. `bge-small`/`e5`) could improve dense-leg
  recall on telecom-specific phrasing without changing the fusion architecture.
- **No query rewriting/expansion** — a query-rewrite step (e.g. LLM-paraphrase or HyDE-style
  hypothetical-document embedding) before retrieval could help the repeated-near-identical-search
  failure mode the audit found (§2.11), on top of the tool-call dedup fix already shipped.

**TTS:**
- **Cloud-dependent** — `edge-tts` needs network access to Microsoft's service; no offline
  fallback voice exists today. A local neural TTS fallback (e.g. Piper/Coqui) for
  degraded-connectivity scenarios would improve reliability.
- **Fixed 0.5/0.5-style heuristics aside, TTS chunk splitting is punctuation-only** — doesn't
  account for SSML prosody, so numbers/currency (₹299) are read using edge-tts's default text
  normalization rather than a Nexatel-tuned pronunciation dictionary; a custom SSML layer for
  domain terms (plan codes, ISO dates) is a natural next step.
- **No streaming synthesis within a single chunk** — pipelining is at sentence granularity;
  true audio-streaming (start playback before the whole sentence's MP3 finishes downloading)
  would shave additional latency off the first chunk specifically.
- **`_detoxify_repetition()` is a truncation safety net**, not a generation-time fix, for the
  small LLM's tendency to loop on repeated phrases when translating numeric/tabular facts into a
  non-English target language — a stronger generation model (or a regenerate-once-on-detection
  retry) for that specific step is the more durable fix.

**Mock CRM:**
- Add a `getRoamingPacks()` billing tool (currently roaming-pack facts are RAG-only text the LLM
  has to summarize from a markdown table on every question — a real tool would ground it and cut
  repeat-question token cost).
- Add a generic `getMyRecentActivity()` tool for cross-turn continuity instead of relying purely
  on conversation-history trimming.
- Add `tests/test_tools_smoke.py` that calls every tool once per sub-agent against the seeded DB —
  the audit found two previously-undetected `NameError` crashes (`getBalance`, `createComplaint`)
  that only surfaced when the tool was actually invoked, not at import time or in the existing
  test suite.

---

## 7. Hackathon / Evaluator Q&A Cheat-Sheet

Grouped by theme, with the one-line answer plus the file that proves it. Use this to rehearse —
every answer traces to a real path in §1–§4.

**RAG fundamentals**
- *"What is RAG and why do you need it here?"* → LLM has zero built-in knowledge of Nexatel's
  actual prices/policies, which change over time and can't be hallucinated on a live call; RAG
  grounds answers in retrieved chunks instead of parametric memory (§1.1).
- *"Walk me through a query end-to-end."* → §1.9 diagram: tool call → `manager_read.read()` →
  `hybrid_query()` fusion → `_format_hits()` formats + records confidence → LLM incorporates it.
- *"Why 5 collections instead of 1?"* → precision + safety boundary: a sub-agent can't retrieve
  (and the LLM can't narrate) another domain's off-topic chunk if that collection was never bound
  as its tool (§1.2, §1.11 last paragraph).
- *"Is `compliance_policy` reachable by the LLM?"* → No — it's called directly by guardrail code
  (`compliance_policy_search()`, [`retriever.py:115`](src/vay/rag/retriever.py)), never registered
  as an LLM tool, so the model can't choose to skip it.

**Chunking & embeddings**
- *"How do you chunk?"* → structure-aware: split on markdown headings → sentence-tokenize (nltk) →
  greedy-pack to 1000 chars with a heading-boundary guard (force-flush at ≥120 chars if section
  changes) → hard-split any oversized single sentence → 150-char sentence-boundary overlap (§1.5).
- *"Why not fixed-size character chunking?"* → it mid-sentence-cuts facts and blends unrelated
  sections/tables together, which measurably hurt retrieval precision on exact-match queries — a
  documented, fixed real bug (§1.5 footnote, `chunking.py:113-120`).
- *"What embedding model, and why that one?"* → `sentence-transformers/all-MiniLM-L6-v2`, 384-dim,
  via ChromaDB's `SentenceTransformerEmbeddingFunction` — small/fast/no-GPU-needed, adequate for a
  KB this size; explicitly named as an improvement lever (bge-small/e5) in §6.
- *"How many chunks total?"* → 65 across all 5 collections, ~537–668 chars/chunk average (§1.5).
- *"Are chunk IDs stable across re-ingests?"* → yes, content-addressed SHA-256 of chunk text
  (`_chunk_id()`), so re-running `build_kb.py` is idempotent — unchanged chunks upsert to the same
  ID (§1.6).

**Vector DB & retrieval algorithm**
- *"What vector DB, what index, what similarity metric?"* → ChromaDB `PersistentClient`, on-disk
  SQLite + HNSW segments under `chroma_db/`, cosine similarity (`metadata={"hnsw:space":"cosine"}`)
  (§1.4).
- *"Is retrieval pure vector search?"* → No — hybrid: dense cosine leg (ChromaDB `.query()`) fused
  with a real sparse BM25 (`rank_bm25.BM25Okapi`) leg via min-max-normalized 0.5/0.5 weighted sum
  (§1.3). Be ready to explain the *prior* bug: this used to be vector-only despite being documented
  as hybrid — a good "how do you validate your own pipeline" story (§1.3 footnote).
- *"Why fuse instead of just using a bigger embedding model?"* → small embeddings are weak on exact
  alphanumeric anchors (plan codes, ₹ amounts, GB caps) that keyword/BM25 matching nails directly;
  fusion recovers the best of both signals cheaply at this KB scale (§1.3).
- *"Why 0.5/0.5 and not RRF?"* → simple, tunable, cheap to reason about for a KB this small; RRF is
  a named future-improvement (§6) because it's less sensitive to score-scale mismatches.
- *"How is the BM25 index kept fresh?"* → cached per collection, invalidated when
  `collection.count()` changes (`_get_index()`), plus an explicit `invalidate_cache()` hook wired
  into `manage_kb.py --rebuild`; known edge case: same-count edits could serve stale cache (§6).

**Confidence / safety**
- *"How do you stop the bot from making things up?"* → `RetrievalTracker` records the best
  similarity per turn; if it falls below `retrieval_confidence_threshold` (τ=0.80,
  [`config.py:26`](src/vay/config.py)), the guardrail routes to human handoff instead of speaking
  an under-grounded answer (§1.8). Note honestly: a separate constant `DEFAULT_MIN_SIMILARITY=0.3`
  also exists in [`tool_agent.py`](src/vay/graph/tool_agent.py) — flag this inconsistency if asked,
  it's a real, undocumented-elsewhere gap worth re-auditing.
- *"Has τ been empirically validated?"* → Not after the hybrid-search fix changed the score
  distribution — a named limitation, good "what would you do next" answer (§6).

**LangChain / LangGraph**
- *"Where's LangGraph actually used?"* → `StateGraph(GraphState)` in
  [`graph/workflow.py`](src/vay/graph/workflow.py), compiled node/edge state machine, one pass per
  utterance (§1.11).
- *"Is the tool loop LangChain's `AgentExecutor`?"* → No — hand-rolled `run_tool_agent()` bounded
  loop; explain why (repetition/script/consent guards needed inside the loop) — §1.11.
- *"What LangChain primitives ARE used?"* → `@tool` decorator, `ChatGroq`/`ChatOpenAI`,
  `bind_tools()`, and the four typed message classes — table in §1.11.
- *"How is scoped RAG enforced at the code level, not just by convention?"* → each sub-agent's
  `bind_tools()` call only ever receives that domain's own tool list — a sub-agent's LLM literally
  has no schema for another domain's retriever, so it can't call it (§1.11 last paragraph).

**Mock CRM**
- *"Is the CRM data real or hardcoded stub responses?"* → Real SQLite DB
  ([`nexatel_customers.db`](src/vay/tools/nexatel_customers.db)), 7 tables, tools do real
  `SELECT`/`UPDATE`/`INSERT` with `conn.commit()` — mutations persist across restarts (§2, §2.3).
- *"How does a plan change actually get committed?"* → Two-Phase Consent: `changePlan()` stages the
  action and returns a `STOP_AND_SAY:` sentinel (verbatim consent script, not LLM-paraphrased); the
  next turn's affirmative triggers `confirm_pending_action()`, which flips the old subscription row
  to `cancelled` and inserts the new one as `active` (§2.3).
- *"Why compute `activated_on` at seed time instead of hardcoding a date?"* → prepaid validity =
  `activated_on + validity_days`; a fixed past date silently drifts into "expired" as real time
  passes — this was a live demo bug that got fixed (§2.2).

**TTS**
- *"What TTS engine, and why not a local model?"* → Microsoft Edge neural voices via `edge-tts` —
  no GPU/VRAM needed, natural voices across 18 languages, free; tradeoff is cloud-dependency, named
  as a limitation (§3.1, §6).
- *"How do you pick the right voice for code-switched text?"* → re-derive the language from actual
  Unicode codepoints in the final text (`re.search` per script block) rather than trusting the
  turn's language label, so an `en`-labeled reply containing Tamil script still gets
  `ta-IN-PallaviNeural` (§3.3).
- *"How did you cut time-to-first-audio?"* → sentence-boundary split + overlapping
  synthesize-next/play-current async pipeline (`_speak_pipelined()`) — ~1.85s → ~1.08s measured for
  a 3-sentence reply (§3.4).
- *"How does barge-in work?"* → `threading.Event` threaded into playback; non-blocking
  `playsound3` + 50ms poll loop checks `stop_event`; on trip, kills the sound process, cancels the
  pending next-chunk synthesis task, deletes the temp file (§3.5).
- *"What happens if TTS fails (no audio device, network drop)?"* → `speak()` never raises —
  exceptions are caught/logged, degrading to text-only rather than crashing the call (§3.6).

**"How would you improve this" (always have 2-3 ready)**
- RAG: add a cross-encoder reranker on the small fused candidate set; move to RRF or a learned
  fusion weight; build a gold-set precision@k/MRR regression test in CI (§6).
- TTS: add a local offline fallback voice (Piper/Coqui) for degraded connectivity; a domain SSML
  pronunciation layer for plan codes/currency (§6).
- CRM: add `getRoamingPacks()`/`getMyRecentActivity()` tools so more facts are tool-grounded instead
  of RAG-summarized every turn (§6).

---

## 8. Related Docs In This Repo

- [`docs/rag_system.md`](docs/rag_system.md) — original RAG architecture doc (diagrams)
- [`docs/tts_pipeline.md`](docs/tts_pipeline.md) — original TTS pipeline doc (diagrams, full voice table)
- [`docs/database_and_tools.md`](docs/database_and_tools.md) — original mock CRM/tools doc (ER diagram)
- [`rag-tts-evaluvation.md`](rag-tts-evaluvation.md) — full bug-hunt/audit session with before/after measurements
- [`docs/agent_graph.md`](docs/agent_graph.md), [`docs/guardrails_and_handoff.md`](docs/guardrails_and_handoff.md), [`docs/asr_stt_pipeline.md`](docs/asr_stt_pipeline.md), [`docs/evaluation_and_benchmarks.md`](docs/evaluation_and_benchmarks.md) — adjacent subsystems (orchestrator, guardrails, ASR, benchmarks)
