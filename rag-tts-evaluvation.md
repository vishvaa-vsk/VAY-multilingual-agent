# RAG + TTS Pipeline Evaluation — Aug 16, 2026 Session

> **Addendum (same day, post-review):** the user ran `run_assistant.py` live after the fixes
> above and surfaced 4 more real bugs by hand-testing an aggressive-caller + troubleshooting
> scenario. All 4 are documented and fixed in **§9** below. §1–§8 are the original audit;
> §9 is the follow-up.

> Scope: everything from `(transcript, language_code, phone_number)` onward — orchestrator NLU,
> the 4 domain sub-agents, their tools, the 5 scoped RAG knowledge bases, the mock CustomerDB,
> and the guardrail/handoff layer. **ASR/VAD/transcription was explicitly out of scope and not
> touched.**
>
> Method: read every module in `src/vay/{rag,graph,tools}`, then validated findings by actually
> running the live LangGraph (`build_graph()`) against real scenarios derived from the two test
> transcripts supplied (`agent result.txt`, `Complaints & Service-Request Agent.txt`), with
> `--show_debug` equivalent tracing. Every bug below was reproduced live before being fixed, and
> re-verified live (or via a targeted unit check, where noted) after the fix. Chunking/retrieval
> quality was measured directly against the live ChromaDB collections, not estimated.

---

## 1. Summary — What Was Actually Wrong

| # | Severity | Finding | Status |
|---|---|---|---|
| 1 | **Critical** | `createComplaint` (core ticket-logging tool) crashed on every call — `NameError: SLA_DAYS` | **Fixed** |
| 2 | **Critical** | `getBalance` (most-used billing tool) crashed on every call — `NameError: _row_to_dict` | **Fixed** |
| 3 | **High** | "Hybrid RAG" was actually **vector-only** search — the BM25 engine was a non-functional stub never wired into the query path | **Fixed** |
| 4 | **High** | KB chunking silently blended unrelated sections together, burying the correct answer outside the top-5 results | **Fixed** |
| 5 | **High** | Guardrail scanned the **assistant's own reply** for "human agent" and silently discarded good answers | **Fixed** |
| 6 | **High** | Orchestrator conflated "checking status of an existing dispute" with "raising a new dispute" — both went straight to human handoff, skipping the ticket lookup entirely | **Fixed** |
| 7 | **Medium** | Only the complaints agent had `escalateToHuman`; billing/plans/coverage had no clean way to signal escalation | **Fixed** |
| 8 | **Medium** | Account context excluded resolved/closed tickets, so "is my issue fixed" was unanswerable from context | **Fixed** |
| 9 | **Medium** | Coverage agent had no ticket-status tool — "is my 5G issue fixed" re-asked for a pincode instead of checking the ticket | **Fixed** |
| 10 | **Medium** | "Chitchat" (thanks/ok/acknowledgement) had no route — got dumped into "unclear," and after 2 turns could wrongly escalate a customer to a human agent for saying "thank you" twice | **Fixed** |
| 11 | **Medium** | Tool-calling loop had no defense against repeating an identical failed search — observed 6 near-identical searches burning 195s/all 6 iterations on one question | **Fixed** |
| 12 | **Medium** | Smaller model (`llama-3.1-8b-instant`) degenerated into a 30x phrase-repetition hallucination loop when translating price data to Tamil | **Fixed** (mitigation) |
| 13 | **Low** | `scripts/manage_kb.py` (documented KB admin CLI) was completely dead code | **Fixed** |
| 14 | **Low** | 3 of 5 demo prepaid accounts showed as **expired** (stale hardcoded seed dates vs. today's date) | **Fixed** |
| 15 | **Low** | Prepaid "balance" answers could hallucinate the plan *price* as an *amount owed* | **Fixed** (mitigation) |
| 16 | **Info** | KB content gap: no explicit "does adding a family-plan line cost extra" fact — root cause of #11's repeated searches | **Fixed** (KB content added) |

Findings **1, 2, 13** were pre-existing bugs unrelated to the two test transcripts, found via live
tool execution + a static undefined-name sweep (`ruff --select F821`) across the whole package
after the first two were found "by luck." **3, 4** are structural RAG-quality issues. **5, 6, 7,
8, 9, 10, 11** map directly to the specific failures described in the user-supplied test
transcripts. **12, 15** were discovered live while re-verifying the other fixes.

---

## 2. Bug-by-Bug: Evidence, Root Cause, Fix

### 2.1 [Critical] `createComplaint` crashed — `NameError: SLA_DAYS`

`src/vay/tools/complaints.py`'s `createComplaint()` referenced `SLA_DAYS` but never imported it
(it lives in `tools/session.py`). **Every single complaint/ticket logging attempt failed.**

```
NameError: name 'SLA_DAYS' is not defined
  File "...\tools\complaints.py", line 60, in createComplaint
```

Fix: import `SLA_DAYS` from `vay.tools.session` (the dict there has the matching category keys —
`network`/`billing`/`service_request`/`technical`/`other`; a second, unused `SLA_DAYS` dict with
different keys also exists in `db_queries.py`, left untouched as dead code, low priority).
Verified live: ticket now logs and the row appears in `tickets` (then reverted the test row).

### 2.2 [Critical] `getBalance` crashed — `NameError: _row_to_dict`

Same bug class in `src/vay/tools/billing.py`: `getBalance()`, the single most commonly needed
billing tool ("what's my balance", "what's my plan"), called `_row_to_dict()` without importing
it. Found via `ruff check src --select F821` after the first NameError bug suggested checking
the whole package for the same mistake — and it was there too.

Fix: import `_row_to_dict` from `vay.tools.session`. Verified live — `getBalance` now returns
correctly for both prepaid and postpaid accounts.

**Takeaway:** neither crash surfaces at import time or in the existing test suite (`test_rag.py`
only checks retriever init, `test_routing.py`/`test_types.py` don't touch tools at all) — they
only appear when the tool is actually *invoked*, e.g. mid-call. A quick smoke test that calls
every tool once against the seeded DB would have caught both before they reached a demo.

### 2.3 [High] "Hybrid RAG" was actually vector-only search

`project_context.md` and `context-rag-tts.md` both document **"Hybrid search: keyword (BM25) +
vector, with top-k reranking."** In reality:

- `rag/bm25.py`'s `BM25SearchEngine.search()` didn't do BM25 at all — it returned whatever
  documents were passed in, in input order, with a **fabricated** score (`0.80 - i*0.05`). It
  never looked at the query.
- Nothing in the live query path (`rag/manager_read.py::read()`, called by every
  `search_*_kb`/`search_*_policy` tool) ever called it — `read()` just did a plain ChromaDB
  `collection.query()`. Pure dense/vector search, the whole time.

This directly hurt precision on exact-keyword queries (prices, plan names, data amounts) that a
small `all-MiniLM-L6-v2` embedding doesn't reliably separate — see §3 for the measured before/after.

Fix: `rag/hybrid.py` (new) builds a real `rank_bm25.BM25Okapi` index per collection (cached,
auto-invalidated on chunk-count change), fuses BM25 + vector cosine similarity via a weighted
sum (0.5/0.5) over min-max-normalized scores, and reranks. Wired into `manager_read.read()` so
every existing caller benefits with no interface change. `rag/bm25.py`'s `BM25SearchEngine` was
also fixed to do real scoring (kept as a standalone utility class; the live path uses
`hybrid.py` directly for caching/efficiency).

### 2.4 [High] Chunking silently blended unrelated sections

`rag/chunking.py::chunk_markdown()` split text into heading-tagged sentences, then **greedily
packed sentences from DIFFERENT sections into the same chunk** whenever they fit under
`chunk_size` (1000 chars) together — defeating the whole point of heading-aware splitting for a
KB this small.

Measured, before any fix, querying `product_catalog` for `"2GB daily data plan 300 rupees
budget"` (the customer's exact ask in the supplied test transcript):

```
chunk 0 (883 chars): "# Nexatel Product Catalog ... This is the authoritative
  Product-Catalog knowledge base ... ## Prepaid Plans ### Nexatel Prepaid — Everyday Range
  | Prepaid Lite | ... | Prepaid Value | ₹299 | 28 days | 2 GB/day | ... |"
```

The chunk containing the EXACT matching plan (Prepaid Value, ₹299, 28 days, 2GB/day) was diluted
by ~500 characters of generic front-matter prose, and **did not even appear in the top 5 search
results** — the retriever instead surfaced the Long-Validity table (₹859 plan), a Broadband
table, and an Add-Ons table, none of which answer the question.

Fix: `chunk_markdown()` now flushes the current chunk when the heading changes AND the chunk
already has ≥120 chars of real content (small heading-only fragments still merge forward, so we
don't flood the KB with near-empty chunks). Rebuilt all 5 collections
(`uv run python scripts/build_kb.py --reset`): chunk counts moved from 8→11 (product_catalog),
and every collection's average chunk size dropped to 537–668 chars (was up to 883+), each one
now scoped to a single section/table.

### 2.5 [High] Guardrail discarded good answers that mentioned "human agent"

`graph/nodes/utils.py::guardrail_node()` did:

```python
if HUMAN_REQUEST_PATTERNS.search(state["transcript"]) or HUMAN_REQUEST_PATTERNS.search(draft):
    return {"handoff": True, "handoff_reason": "Customer requested a human agent."}
```

`draft` is the **sub-agent's own reply**, not the customer's. The sub-agent prompt explicitly
tells agents to say things like *"I can connect you to a human agent"* when offering escalation
— which matches `HUMAN_REQUEST_PATTERNS` (`\bhuman\b`, `\bagent\b`-adjacent phrases) against the
draft, silently swapping a specific, correct, on-topic answer for the generic
`HANDOFF_MESSAGE_TEMPLATES` line. Reproduced live: the SIM-replacement-approval scenario's LLM
reply ("I don't have the ability to approve tickets directly... I can connect you to a human
agent") was correct and specific, but the customer actually heard the generic
"I want to make sure I get this right for you..." line instead.

Fix: only check the customer's own transcript. A sub-agent that legitimately decides to escalate
should do so via `escalateToHuman` (see 2.6), which is a `session.escalation_requested` flag
already checked separately — not by pattern-matching its own prose. Verified with a targeted
unit test (`guardrail_node` given a draft mentioning "human agent" now preserves it verbatim).

### 2.6 [High] "Checking status" vs. "raising a dispute" both treated as sensitive

Reproduced exactly per the supplied test transcript: *"What's the status update on my roaming
charge dispute?"* against a customer with an actual in-progress ticket (`NXT-100235`) —

```
Before fix:
  {"intent":"billing_dispute_status", "route":"billing", "sensitive": true, ...}
  -> straight to human handoff, ticket never looked up.

After fix:
  {"intent":"check_dispute_status", "route":"complaints", "sensitive": false, ...}
  -> getTicketStatus(ticket_id='NXT-100235') ->
     "NXT-100235 [billing] status=in_progress ... notes: Field team dispatched; awaiting confirmation."
  -> "Your roaming charge dispute (ticket NXT-100235) is still in progress. A field team has
      been dispatched and we're waiting for their confirmation..."
```

Fix: rewrote the `sensitive` definition and routing guide in `ORCHESTRATOR_SYSTEM_PROMPT`
(`graph/core_utils.py`) to explicitly distinguish RAISING a new dispute/cancellation/fraud report
(sensitive=true) from CHECKING STATUS of one already raised (sensitive=false, route=complaints,
answerable via `getTicketStatus`). Added a matching FAQ entry to `support_faq.md` for RAG
grounding. **Confirmed live, full end-to-end, with a real grounded answer** (see quote above).

### 2.7 & 2.9 [Medium] "Is my issue fixed" — coverage agent had no way to answer

Reproduced per the supplied test transcript: *"Is my 5G issue fixed now?"* against a customer
whose ticket (`NXT-100236`, category=technical, status=**resolved**) already had the answer.

```
Before fix: routed to coverage -> "Could you please share your pincode so I can check if
  there's any outage or coverage issue in your area?"   (never looked at the ticket at all)

After fix:  routed to complaints -> getTicketStatus(ticket_id='') ->
  "NXT-100236 [technical] status=resolved ... notes: Resolved: VoLTE re-provisioned on account."
  -> "Yes, your 5G issue has been resolved. The ticket NXT-100236 shows as resolved – VoLTE
      was re-provisioned on your account, so 5G should now be working normally..."
```

Two compounding root causes, both fixed:
- Routing guide didn't distinguish a NEW coverage report (needs a pincode) from a STATUS
  follow-up on an already-reported issue (needs ticket lookup, not a pincode) — added explicit
  guidance + examples to `ORCHESTRATOR_SYSTEM_PROMPT`.
- `_fetch_account_context()` (`graph/nodes/orchestrator.py`) explicitly excluded resolved/closed
  tickets (`WHERE status NOT IN ('resolved','closed')`) — precisely the tickets a customer asks
  about when asking "is it fixed." Broadened to the last 3 tickets regardless of status, with
  resolution notes included, so this is answerable from context in **every** sub-agent, not just
  complaints.
- Also added a read-only `getTicketStatus` tool to the coverage agent directly (previously only
  complaints had one), and gave **all four** sub-agents `escalateToHuman` (previously only
  complaints had it — see 2.8), via a new shared `build_escalate_tool()` factory in
  `tools/session.py`.

### 2.8 [Medium] Only complaints had `escalateToHuman`

Billing/plans/coverage agents had no tool-based way to signal "this needs a human" — their only
option was to say so in free text, which is exactly what triggered bug 2.5. Fixed by extracting
`escalateToHuman` into a shared `build_escalate_tool(session)` factory and adding it to all four
tool lists. Verified: `billing`/`plans`/`coverage`/`complaints` all expose it now.

### 2.10 [Medium] Chitchat had no route — could trigger a wrongful human handoff

Reproduced from the supplied test transcript's pattern (bot repeating "I didn't understand, tell
me about your bill/plan/complaint/coverage" after "super"/"teanks"/"ok"/"seringa"/"seri
avlodhan"). Root cause: the orchestrator's route schema was
`{billing, plans, complaints, coverage, unclear}` — an understood-but-non-actionable utterance
("thanks", "ok") had nowhere to go but "unclear," which both (a) repeats the same "please
clarify" script at a customer who said something perfectly clear, and (b) increments the same
unclear-escalation counter as a genuinely garbled utterance — **two "thank you"s in a row could
trigger an unwanted human handoff.**

Fix: added a `chitchat` route (`ORCHESTRATOR_SYSTEM_PROMPT`), a `chitchat_node` (fixed localized
templates, no LLM call needed), and excluded it from the unclear-escalation counter. Verified
live, full end-to-end:

```
>>> USER: thanks
  {"intent":"thank_you","route":"chitchat","confidence":1.0,...}
<<< "You're welcome! Is there anything else I can help you with -- your bill, your plan,
     a complaint, or network coverage?"

>>> USER: ok
  {"intent":"acknowledgement","route":"chitchat","confidence":0.9,...}
<<< "You're welcome! Is there anything else I can help you with..."
    [route=chitchat handoff=None]   <- no escalation, no repeated confusion
```

### 2.11 [Medium] Tool loop wasted 195s on 6 near-identical failed searches

Reproduced exactly per the supplied test transcript: *"How much does it cost to add another
number to my family plan?"* — before any KB content fix, the sub-agent tried:

```
search_product_catalog('add another number to family plan cost')       relevance=0.41
search_product_catalog('family plan add another number cost')          relevance=0.44
search_product_catalog('Postpaid Family add line cost')                relevance=0.38
search_product_catalog('Postpaid Family add line cost activation fee') relevance=0.46
search_product_catalog('Postpaid Family add line cost activation fee') relevance=0.46  <- literal repeat
search_product_catalog('Postpaid Family add line fee')                 relevance=0.39
[ran out of 6 tool iterations -- forcing a grounded wrap-up]
```

195.1 seconds and 6 LLM/tool round-trips for a single turn that ends in a non-answer.

Two fixes, compounding:
- **Code-level dedup** in `graph/tool_agent.py::run_tool_agent()`: tracks `(tool_name, args)`
  signatures seen this turn; an exact repeat is answered locally (no real tool re-invocation, no
  extra KB-context tokens) with a nudge to try something different or wrap up. Verified with a
  mock LLM that stubbornly repeats an identical call 5 times: only 1 real invocation happens.
- **KB content gap closed** (see 2.16) so the answer is now found on the first search.

Verified live, post-fix: **1 search, 19.3 seconds, correct grounded answer** ("no separate
per-line fee — Family plan already includes up to 4 connections at one flat price").

### 2.12 [Medium] LLM generation repetition-loop (mirrors the documented ASR hallucination issue)

`project_context.md` §5.3 documents Whisper's known hallucination/repetition failure mode and a
post-ASR filter as the accepted mitigation. Live-testing this session's fixes with
`llama-3.1-8b-instant` surfaced the **same failure mode on the generation side**: asked to
translate concrete plan/price facts into Tamil, the model locked onto a short phrase and repeated
it ~30 times instead of stopping:

```
"...நிகர வருமானம் 299 ரூபாய் வரையிலான பயனர்களுக்கு நிகர வருமானம் 299 ரூபாய்
 வரையிலான பயனர்களுக்கு நிகர வருமானம் 299 ரூபாய் வரையிலான பயனர்களுக்கு ..." (repeated ~30x)
```

Fix: added `_detoxify_repetition()` to `graph/tool_agent.py` — a regex (`(.{12,80}?)\1{2,}`)
that detects any 12–80 char span repeating 3+ times back-to-back and truncates just before it.
Applied to every reply `run_tool_agent()` returns. Verified: the captured hallucination truncates
from 284 chars to a clean 67-char partial sentence; normal non-repeating replies pass through
byte-for-byte unchanged. This is a safety net (truncation, not a full fix for the underlying
model weakness) — see §5 for the recommendation this motivates.

### 2.13 [Low] `scripts/manage_kb.py` was dead code

Documented in `context-rag-tts.md` as `python scripts/manage_kb.py --status` / `--rebuild`, but
the actual script imported a nonexistent local file (`scripts/chroma_setup (1).py` — a leftover
from before the project was restructured into the `vay` package) and called functions
(`create`/`update`/`delete`/`list_sources` with a `DEFAULT_CHUNK_SIZE` import) that don't match
`vay.rag.manager`'s current exports. **It could not run at all**:

```
FileNotFoundError: [Errno 2] No such file or directory: '...\scripts\chroma_setup.py'
```

Fix: rewritten against the current package. `--status` (chunk counts), `--search QUERY
--collection X` (test hybrid retrieval directly, no LLM needed), `--rebuild
<collection|all>` (wipe + re-ingest one or all 5 KBs, with BM25 cache invalidation). Verified
working (`--status` output in §3).

### 2.14 [Low] Stale demo dates — 3 of 5 prepaid accounts showed as expired

`db_seed_data.py` hardcoded absolute `activated_on` dates (e.g. `"2026-06-20"`) for prepaid
subscriptions. Since prepaid validity = `activated_on + validity_days`, a fixed past date
silently drifts into "expired" as real time passes it. Checked against today (2026-08-16):

```
9876500001 Prepaid Value        expiry=2026-07-18  days_left=-29  EXPIRED
9876500007 Prepaid Basic        expiry=2026-06-29  days_left=-48  EXPIRED
9876500010 Prepaid 84-Day Value expiry=2026-08-02  days_left=-14  EXPIRED
```

Any "what's my balance" call for 3 of the 5 seeded prepaid demo customers would have reported an
expired plan — not a great look mid-demo (the hackathon evaluation is Aug 19). Fixed both the
live seeded DB directly (fresh `activated_on` dates) and `db_seed_data.py` (now computes prepaid
`activated_on` as `_days_ago(N)` relative to whenever the DB is seeded, so a future `--reset`
stays fresh regardless of when it's run — postpaid/broadband dates are untouched since they only
represent "customer since" tenure, not a validity window).

### 2.15 [Low] Prepaid "balance" hallucination

Live-observed: asked "what's my balance?" for a prepaid customer, `getBalance()` correctly
returned plan/validity info with no "amount owed" (prepaid has no such concept in this schema),
but the LLM's reply invented one anyway — *"Your current account balance is Rs 299.0, which is
the active plan amount"* — pulling the plan **price** out of account context and mislabeling it
a balance owed. Fixed with an explicit grounding rule added to the sub-agent GUARDRAILS section
distinguishing plan price from amount-due, and clarifying what "balance" means for prepaid
(remaining validity/data) vs. postpaid (amount due).

### 2.16 KB content gap: family-plan add-line cost

Root cause of 2.11's repeated searches: `product_catalog.md` said family plans support "up to 4
connections" but never stated whether adding one costs extra. Added an explicit fact under
Eligibility & Offer Terms: adding a line to Postpaid Family/Pro has **no separate fee** — the
₹699/₹999 already covers up to 4 shared connections; only KYC is required per added number.
Also added two FAQ entries to `support_faq.md` grounding 2.6 (dispute-status vs. new-dispute)
and the SIM-replacement-approval compliance rule (an already-logged SIM-swap ticket can never be
self-approved — it always needs human identity verification, per `compliance_policy.md`).

---

## 3. RAG Retrieval Quality — Measured Before/After

Same query, same collection, before vs. after the chunking + hybrid-search fix (no LLM involved —
pure retrieval):

**Query:** `"2GB daily data plan 300 rupees budget"` against `product_catalog`

| | Before | After |
|---|---|---|
| Top-5 contains the exact-match plan (Prepaid Value, ₹299, 2GB/day, 28d)? | **No** (not in top 5 at all) | **Yes** — rank #2, sim=0.554 (later re-measured at 0.75–0.81 with the exact customer phrasing) |
| Top-1 result | "Comparing Plans — Guidance" boilerplate (sim=0.540) | Same boilerplate, but now sim=0.729 (a real signal, and the correct chunk is right behind it) |
| #2 result | Wrong table (Long-Validity ₹859 plan, sim=0.498) | **Correct table** (Everyday Range incl. ₹299 plan, sim=0.554–0.81) |

**Query:** `"family plan add another number cost"` against `product_catalog`

| | Before | After |
|---|---|---|
| Top-1 relevance | 0.41 (best of 6 different phrasings tried, none useful) | **0.768**, and it's the exact fact ("no separate per-line addition fee...") |

**Chunk granularity** (`scripts/manage_kb.py --status` after rebuild):

```
billing_policy        17 chunks   avg 537 chars (was up to 882+ per blended chunk)
product_catalog       11 chunks   avg 646 chars
support_faq           15 chunks   avg 595 chars
technical_kb          11 chunks   avg 609 chars
compliance_policy     11 chunks   avg 668 chars
Total: 65 chunks across all 5 collections
```

End-to-end effect measured live: the family-plan-add-line question went from **6 failed
searches / 195.1s / no answer** to **1 search / 19.3s / correct grounded answer**.

---

## 4. Sub-Agent & Tool Audit

| Agent | Tools (after fixes) | DB mutation verified? |
|---|---|---|
| Billing | `getBalance`, `getBillBreakup`, `getDueDate`, `sendPaymentLink`, `explainCharge`, `escalateToHuman` | N/A (read-only + mock SMS) — **`getBalance` was crashing, now fixed and confirmed working live for both prepaid and postpaid.** |
| Plans | `listPlans`, `comparePlans`, `changePlan`, `activateAddOn`, `checkEligibility`, `escalateToHuman` | **Yes.** Live-tested the full two-phase consent flow (`changePlan` stages a pending action → `confirm_pending_action` on the customer's next-turn "yes") end-to-end against the real SQLite DB: old subscription row correctly flips to `status='cancelled'`, new one inserted `status='active'`. Test mutation reverted afterward. |
| Complaints | `createComplaint`, `getTicketStatus`, `runTroubleshootFlow`, `escalateToHuman` | **Yes** (after fixing the `SLA_DAYS` crash). `createComplaint` now correctly inserts a new ticket row and returns a real ticket ID; verified via before/after row count, then reverted the test row. |
| Coverage | `checkCoverage`, `getOutageStatus`, `getDeviceSettings`, `guideSimSwap`, **`getTicketStatus` (new)**, **`escalateToHuman` (new)** | N/A (read-only) — now has ticket visibility it previously lacked entirely. |

**Additional tools worth adding next** (not implemented this session — flagged for the team):
- A `getRoamingPacks()` billing tool (currently roaming-pack details are RAG-only text the
  sub-agent has to summarize from a markdown table every time — a real tool would ground it and
  save tokens on repeat questions).
- A generic `getMyRecentActivity()` tool usable by any agent for "what did I ask about last
  time" type continuity, instead of relying purely on conversation history trimming.

---

## 5. Mock DB — Findings

- **Schema/seed data are structurally sound** — no missing columns, no foreign-key violations
  found. `manage_db.py` (unlike `manage_kb.py`) was already correctly wired to the current
  package and works as documented.
- **Two tools crashed on every call** (2.1, 2.2) — both are now fixed and DB-mutation-verified.
- **Stale seed dates** made 3/5 prepaid demo accounts look expired (2.14) — fixed for both the
  live DB and future reseeds.
- **`db_queries.py` has a second, unused, differently-keyed `SLA_DAYS` dict** (keys:
  `billing/network/technical/service_request/general`, vs. the one actually used in
  `session.py` with `.../other`) — dead code, not wired to anything, left in place to minimize
  risk; worth deleting in a follow-up cleanup pass.
- **`chroma_db/` and `nexatel_customers.db` are both real, working, persistent stores** — nothing
  here is a mock in the sense of "fake/stub data that doesn't actually connect to anything."
  Every tool that's supposed to mutate state (`changePlan`, `createComplaint`,
  `confirm_pending_action`, `activateAddOn`) does correctly call `conn.commit()` and the changes
  are durable across process restarts (verified for the two above; the other two follow the
  identical pattern).

---

## 6. Token / Cost Efficiency

- **System prompt fixed cost, per turn:** `ORCHESTRATOR_SYSTEM_PROMPT` ≈ 1,185 tokens (rough
  4-chars/token estimate), `SUBAGENT_SYSTEM_PROMPT_TEMPLATE` ≈ 1,735 tokens *before* the
  per-call `{account_context}`/`{phone_number}`/`{language}` fill-in. Every turn that reaches a
  sub-agent pays both, plus tool schemas, plus conversation history, plus whatever RAG context
  gets pulled in. The chitchat fix (2.10) means acknowledgement-only turns now skip the entire
  sub-agent prompt + tool-schema cost, not just the reply text.
- **Rate-limit hit live, mid-evaluation:** this session's own testing exhausted the account's
  full **200,000 tokens/day** Groq quota for `openai/gpt-oss-20b` partway through the second
  test run (`Rate limit reached ... Limit 200000, Used 197394 ... on tokens per day (TPD)`).
  This is a **direct, measured signal** — not a projection — that a live/production deployment
  needs one of: a cheaper/faster model for the orchestrator + normalization passes (which run on
  *every* turn regardless of question complexity), explicit per-session token budgeting, or a
  paid-tier quota increase. The team switched the configured model to
  `llama-3.1-8b-instant` mid-session, which resolved the immediate block but introduced the
  repetition-hallucination failure mode in §2.12 — there is a real quality/cost tradeoff between
  the two models currently available to this project, not a free win either way.
- **The repeated-tool-call bug (2.11) was an independent ~6x token multiplier** on affected
  turns: 6 near-identical searches × ~500–800 tokens of injected KB context each, plus 6 full
  LLM round-trips (system prompt + history repeated in every one), for a question that should
  cost one round-trip. Now capped at 1 real tool call per unique query.
- **Chunking fix reduces per-call RAG context size** while improving relevance: collections now
  average 537–668 chars/chunk (was up to 883+ chars when sections got blended), so a `n_results=3`
  RAG tool call now injects less irrelevant filler while surfacing the actually-relevant chunk
  more often — net fewer wasted tokens per answer, not just better recall.
- **Recommendation:** if a stronger model is needed for sub-agent generation quality (§2.12), it
  doesn't have to be used everywhere — the orchestrator's job (strict JSON extraction) and the
  normalization pass are both good candidates to stay on a cheap/fast model, reserving a
  stronger model specifically for the sub-agent's final natural-language generation step, where
  the repetition/hallucination and grounding-precision failures actually showed up.

---

## 7. Remaining Known Gaps (Not Fixed This Session — Flagged for the Team)

- **SIM-replacement-approval handoff message is compliance-correct but not ticket-specific.**
  `sensitive=true` correctly routes straight to human handoff (SIM swap approval must always go
  through identity verification — never self-served), but the customer hears the fully generic
  handoff line rather than something acknowledging their specific ticket ID/status first. Fixing
  this cleanly means teaching the `sensitive` path to do one cheap, safe lookup before handing
  off, which is a bigger architecture change than this session's scope.
- **`_detoxify_repetition()` (2.12) is a safety-net truncation, not a real fix** for the
  underlying model's tendency to lose coherence generating non-English text about numeric/tabular
  facts. A proper fix likely needs either a stronger generation model, an explicit max_tokens cap
  tuned per language, or a regenerate-once-on-detection retry loop.
- **Chunk-level BM25 index rebuild is count-based, not content-hash-based** — if a KB file is
  edited and re-ingested with the SAME total chunk count as before (unlikely but possible), the
  cached BM25 index could serve stale results until the process restarts or
  `invalidate_cache()`/`--rebuild` is called explicitly. Low risk given KB updates go through
  `scripts/manage_kb.py --rebuild`, which now calls `invalidate_cache()` directly.
- **No automated tool-level smoke test exists** — the two crash bugs (2.1, 2.2) both slipped
  past `uv run pytest` because the test suite never actually invokes a tool against the seeded
  DB. Recommend adding a `tests/test_tools_smoke.py` that calls every tool once per sub-agent
  with a valid seeded phone number and asserts no exception — this is the single highest-value
  test gap given what this session found.
- **Retrieval confidence threshold (τ)** is still the pre-existing `0.75` (per
  `project_context.md` §8's deliberate strict-safety rationale) — not retuned against the new
  hybrid-search score distribution. Worth re-validating empirically now that scores read
  differently (e.g. the family-plan-fee query went from 0.41 to 0.768 for the exact same
  underlying fact, purely from the retrieval-quality fix, not from τ changing).

---

## 8. Files Changed This Session

```
src/vay/rag/chunking.py          section-boundary-aware chunking (was blending sections)
src/vay/rag/hybrid.py            NEW — real BM25 + vector hybrid search, cached per collection
src/vay/rag/manager_read.py      wired hybrid_query() into the live retrieval path
src/vay/rag/bm25.py              replaced non-functional stub with a real BM25Okapi-backed engine
src/vay/graph/core_utils.py      orchestrator + subagent prompt fixes (see §2.6/2.10/2.15); CHITCHAT_TEMPLATES
src/vay/graph/utils.py           export CHITCHAT_TEMPLATES
src/vay/graph/nodes/orchestrator.py   chitchat routing; broadened account-context ticket filter
src/vay/graph/nodes/utils.py     guardrail bugfix (2.5); new chitchat_node; routing update
src/vay/graph/workflow.py        wired the new chitchat node into the graph
src/vay/graph/tool_agent.py      tool-call dedup (2.11); repetition-loop filter (2.12)
src/vay/tools/session.py         new shared build_escalate_tool() factory
src/vay/tools/billing.py         fixed _row_to_dict crash (2.2); added escalateToHuman
src/vay/tools/plans.py           added escalateToHuman
src/vay/tools/coverage.py        added getTicketStatus + escalateToHuman
src/vay/tools/complaints.py      fixed SLA_DAYS crash (2.1); refactored to shared escalate tool
src/vay/tools/db_seed_data.py    prepaid activation dates now relative, not stale-hardcoded (2.14)
data/kb/product_catalog.md       added family-plan add-line cost fact
data/kb/support_faq.md           added dispute-status and SIM-approval FAQ entries
scripts/manage_kb.py             rewritten — was completely broken dead code (2.13)
chroma_db/ (all 5 collections)   rebuilt from scratch with the fixed chunker
nexatel_customers.db             fixed 3 stale-expired prepaid subscription dates
```

No changes were made to `src/vay/asr/`, `src/vay/audio/`, or `src/vay/tts/` — out of scope per
the task, and confirmed untouched.

---

## 9. Addendum — 4 More Bugs Found From Live Hand-Testing

The user ran the real CLI (`uv run python scripts/run_assistant.py --show_debug`) with phone
`9876500006`, language `ta`, and this sequence:

```
1. "fuck you nexatel ennaku internet eh olanga varalaaaa !!!"   (actual profanity)
2. "internet olanga varala !!!"                                 (frustrated, no profanity)
3. "nan ippo ethumeh sollala verum net varala than sonen"       (still frustrated, no profanity)
4. "internet very slow va irruku enna nu sollu"
5. "bye"
```

Two questions this surfaced, both were real bugs, plus two more found while fixing them.

### 9.1 [Critical] `aggressive_count` never actually persisted across turns

**Root cause:** `scripts/run_assistant.py`'s main loop builds a **brand-new `GraphState` dict on
every turn** — it only carries `phone_number`, `language`, `transcript`, `conversation_history`,
`show_debug`, `min_similarity`, and `session` forward; it never merges the previous
`graph.invoke()` result back in. `orchestrator_node` was reading the running aggressive-offence
count from `state.get("aggressive_count", 0)` — which is **always 0**, every single turn,
because `state` never carries it. `session` (the one object that genuinely persists across
turns — it already has an `aggressive_count` field) was written to but never read from for this
check. The identical, already-working pattern next to it (`session.consecutive_unclear`) shows
what this was supposed to look like.

**Effect:** a caller could swear on turn 1 (correctly gets the first-offence warning), then swear
again on turn 5, turn 10, turn 20 — and get the SAME first-offence warning every single time.
**The call-cut on a 2nd offence could never actually fire**, no matter how many times the
customer was abusive.

**The same root cause also silently broke `previous_route`** — `_run_subagent`'s Aug-14
"domain-switch history trimming" fix reads `state.get("previous_route", "")`, which
`orchestrator_node` computed as `state.get("route", "")` (also always ""). So switching from
billing to plans mid-call never actually trimmed the stale tool messages, despite that fix being
documented as shipped on Aug 14.

**Fix:** added `last_route: str` to `SessionContext` (mirroring the existing
`aggressive_count`/`consecutive_unclear` fields) and made both counters read from/write to
`session`, not `state`. Verified live, reproducing the user's exact scenario end-to-end: turn 1
(profanity) → warning, `aggressive_count=1`; turn 2 → now correctly evaluated as a 2nd offence.

### 9.2 [High] The call-cut message was being silently discarded

Once 9.1 was fixed, turn 2 correctly triggered `cut_call=True` — but the customer heard a
cheerful, generic **"Thank you for calling Nexatel"**-style line instead of being told their call
was ending due to repeated abusive language. Root cause: `route_after_orchestrator` sends a
forced call-cut to the **`closing`** node ("treat as closing so TTS plays the goodbye"), but
`closing_node` unconditionally asked the LLM to generate *"ONE brief, warm closing line thanking
[the customer] for calling"* — completely ignoring `state["warning_reply"]`, the deterministic,
hand-written `CALL_CUT_TEMPLATES` text that `orchestrator_node` had already built specifically
for this moment (the same text the code comments explicitly say "cannot be side-stepped by a
hallucinating model" — except it was, by a different node). Fixed: `closing_node` now checks
`state.get("warning_reply")` first and speaks it verbatim, skipping the LLM call entirely for a
forced abuse-cut closing. Verified live — the customer now correctly hears *"தொடர்ந்து தகாத
மொழி பயன்படுத்தியதால், இந்த அழைப்பு இப்போது முடிக்கப்படுகிறது..."* ("Due to repeated use of
abusive language, this call is now being ended...").

### 9.3 [Medium] `aggressive` classification over-triggered on mere frustration/punctuation

The user's second question, implicitly: *why did turn 2 (no profanity, just "!!!") get flagged
the same as turn 1?* The orchestrator LLM (`llama-3.1-8b-instant`) classified
`aggressive=true` for "internet olanga varala !!!" — a frustrated repeat of a real complaint,
with zero abusive words. Tightened `ORCHESTRATOR_SYSTEM_PROMPT`'s definition of `aggressive`
with an explicit non-example, but **prompt tightening alone did not fully fix it** — re-tested
live and the same model still flagged the same turn as aggressive even with the tightened
prompt. Because 9.1's fix means this flag now has real teeth (it can end a legitimate customer's
call), a soft prompt-only fix wasn't good enough. Added a second, deterministic gate:
`ABUSIVE_LANGUAGE_PATTERN` (a profanity/threat-term regex) must ALSO match the raw transcript
before `aggressive` is honored — the same "don't trust the LLM alone for a consequential
decision" pattern this codebase already uses for `AFFIRMATION_PATTERN`/`PII_LEAK_PATTERNS`.
Verified live: turn 2 (LLM still said `aggressive=true`) now correctly stays `aggressive=false`
end-to-end (`aggressive_count` stayed at 1, no false-positive escalation), while turn 1's real
profanity still correctly triggers the warning.

**Known limitation, disclosed, not fixed this session:** the regex is deliberately
English-profanity-focused (code-switched callers frequently swear in English mid-sentence, as in
the reproducing example) — a threat expressed purely in Tamil/Hindi script without an English
anchor word won't be caught by this gate. A reviewed, native-speaker-validated multilingual term
list is a real follow-up item, not attempted here to avoid guessing at terms.

### 9.4 [High] "Internet very slow" was misrouted to `coverage`, which has no troubleshooting content

The user's other question: *why didn't the slow-internet reply include any actual steps (restart
phone, toggle airplane mode, etc.)? Is the knowledge base not good enough?*

**The knowledge base was fine — the routing was wrong.** `support_faq.md` already has a full
5-step "Slow or No Mobile Data" guide (confirm plan active → check APN → restart device + toggle
mobile data → check for an outage → check the FUP cap), and `complaints.py` already has a
`runTroubleshootFlow(issue_type='slow_data')` tool that returns exactly those steps. But
"internet very slow" was being routed to **coverage**, whose only tools are `checkCoverage`/
`getOutageStatus`/`getDeviceSettings`/`guideSimSwap` (+ `getTicketStatus`/`escalateToHuman` from
§2.9) — none of which is a generic troubleshooting-steps tool, and whose RAG search is scoped to
`technical_kb` (APN/VoLTE/SIM/coverage FAQs), NOT `support_faq` (where the actual "Slow or No
Mobile Data" guide lives). The orchestrator's routing guide didn't clearly separate "is there
coverage/service at all in this area" (genuinely coverage's job) from "why is my existing service
behaving badly right now" (a troubleshooting question that belongs to complaints, matching
`runTroubleshootFlow`'s issue types: `call_drop`, `slow_data`, `sms_issue`, `cannot_call`,
`recharge_not_reflecting`) — both can reasonably contain the words "internet"/"network"/"signal".

Fixed by rewriting the routing guide to explicitly route "my internet is slow", "calls keep
dropping", "SMS isn't sending", "I can't make calls", "recharge isn't reflecting" to
`complaints`, reserving `coverage` for pincode-based coverage/outage lookups and device/APN/SIM
setup procedures. Verified live: the same "internet very slow" question now routes to
`complaints`, calls `runTroubleshootFlow(issue_type='slow_data')`, and the reply contains the
real 5-step guide (plan/data check, APN check, restart + toggle mobile data, area outage check,
FUP cap check) — verbatim-grounded in the KB content, not generic filler.

### 9.5 Files changed in this addendum

```
src/vay/tools/session.py              added SessionContext.last_route
src/vay/graph/nodes/orchestrator.py   aggressive_count + previous_route now read/write session,
                                       not state; aggressive gated by ABUSIVE_LANGUAGE_PATTERN
src/vay/graph/nodes/utils.py          closing_node speaks warning_reply verbatim on a forced
                                       abuse-cut instead of generating a generic goodbye
src/vay/graph/core_utils.py           tightened 'aggressive' definition + example; new
                                       ABUSIVE_LANGUAGE_PATTERN; routing guide now explicitly
                                       sends troubleshooting-flavored complaints to `complaints`
src/vay/graph/utils.py                export ABUSIVE_LANGUAGE_PATTERN
```
