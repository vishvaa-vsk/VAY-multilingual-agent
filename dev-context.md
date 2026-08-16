# VAY — Dev Context: Aug 16 Voice Pipeline Session

> **Last updated:** 2026-08-16
> **Branch:** `vicky-rag`
> **Purpose:** Quick-start reference for the current state of the VAD → ASR → LangGraph → TTS pipeline after this session's changes. Read this first when re-joining the project. Full historical context: [`context-rag-tts.md`](context-rag-tts.md).

---

## What Was Done This Session

### 1. Bug Fix — ASR Language Never Changed After First Lock (`asr/router.py`)

**Root Cause:**  
`ASRRouter` had a `locked_language` field that was set after 2 utterances with ≥60% confidence and **never cleared**. Once Hindi locked, _every_ subsequent utterance — regardless of what language was spoken — was permanently routed to IndicConformer with `lang="hi"`. Same bug applied when starting in English: Whisper was forced on every turn thereafter.

**Fix (in `src/vay/asr/router.py`):**
- Removed the persistent `locked_language` field entirely.
- Added `_reset_utterance_state()` called inside `route_and_transcribe()` via a `try/finally` — guarantees cleanup even if transcription errors.
- The intra-utterance accumulator window (`min_utterances_for_lock`) still exists as a warmup hint within a single detection pass, but is reset after every utterance.
- `last_detected_language` / `last_detected_confidence` exposed for logging only.

**Before / After:**
```
# Before: speak Hindi → lock → speak English → still gets Hindi transcript
locked_language = "hi"   # set once, never cleared

# After: each utterance detects fresh
_detect_language()  →  route  →  transcribe  →  _reset_utterance_state()
                                                  ↑ cleared before next call
```

---

### 2. STTPipeline Callback Hook (`audio/pipeline.py`)

Added optional `callback: Callable[[ASRResult], None]` parameter to `STTPipeline.__init__`.

- When provided: called on the consumer background thread after every successful transcription.
- When `None` (default): pipeline behaves exactly as before (results printed to stdout).
- The callback is the integration seam between raw audio and LangGraph.

---

### 3. New Voice Entry Point (`scripts/run_voice.py`)

Full real-time voice loop: **Mic → VAD → ASR → LangGraph → edge-TTS**.

**Key behaviors:**
- **Phone number prompt at startup** (required for LangGraph session context).
- **Per-utterance language detection**: the `detected_language` from each `ASRResult` is injected as `language` into `GraphState`. The orchestrator, sub-agents, TTS, and all localized templates respond in the caller's _current_ language — not the language from the start of the call.
- **Language switches handled correctly**: speak Hindi → get Hindi reply; speak English next → get English reply. No restart needed.
- **Termination conditions**: Ctrl+C, customer ends call (`call_end_requested`), or human escalation (`handoff=True`).

**Run:**
```powershell
# Interactive (prompts for phone number)
uv run python scripts/run_voice.py

# Flags
uv run python scripts/run_voice.py --phone 9876543210 --show_debug
uv run python scripts/run_voice.py --min_similarity 0.4
```

---

## Current Architecture: Full Voice Pipeline

```
Microphone
  └─► SileroVADStreamer (vad.py)
        │  ~650ms silence = utterance boundary
        ▼
   utterance_queue  (STTPipeline consumer thread)
        ▼
   ASRRouter.route_and_transcribe(audio_tensor)
        │
        │  [Whisper detect_language()]  ← fresh EVERY utterance (bug fix)
        │
        ├─ tier1_languages (22 Indic) → IndicConformerASR.transcribe(lang)
        └─ other            → WhisperASR.transcribe(lang)
               ▼
         ASRResult { raw_text, detected_language, model_used, ... }
               │
               │  callback(result)   ← STTPipeline hook
               ▼
         VoiceCallSession.on_asr_result(result)   [run_voice.py]
               │
               │  GraphState { phone_number, language, transcript, ... }
               ▼
         LangGraph.invoke(state)   [workflow.py → build_graph()]
               │
               ├─► orchestrator_node  (NLU, routing, intent)
               │
               ├─► billing_node / plans_node / complaints_node / coverage_node
               │       └─► tool-calling loop (max 6 iterations)
               │             ├─ DB tools (getBalance, listPlans, ...)
               │             └─ scoped RAG tool (ChromaDB hybrid search)
               │
               ├─► guardrail_node  (confidence gate, PII, compliance)
               │
               ├─► human_handoff_node / warning_node / clarify_node / closing_node
               │
               └─► tts_node  (edge-tts, 18 neural voices, responds in caller's language)
```

---

## Data Contract: ASR → LangGraph

Each utterance produces an `ASRResult` (from `types.py`):

```python
class ASRResult(BaseModel):
    raw_text: str            # Transcribed text — fed as `transcript` to GraphState
    detected_language: str   # ISO 639-1 code — fed as `language` to GraphState
    language_tier: LanguageTier  # tier_1 (IndicConformer) | tier_2 (Whisper)
    confidence: float
    model_used: str
```

`GraphState` receives:

```python
state: GraphState = {
    "phone_number": phone_number,   # from startup prompt
    "language": result.detected_language,  # ← drives TTS + sub-agent locale
    "transcript": result.raw_text,         # ← drives orchestrator NLU
    "conversation_history": [...],
    "session": SessionContext(...),        # carries cross-turn state
    ...
}
```

The `language` field in `GraphState` flows through to:
- `orchestrator_node` → LLM prompt (reply in this language)
- `tts_node` → `edge-tts` voice selection (`VOICES[lang]`)
- All localized template dicts (`CLARIFY_TEMPLATES`, `HANDOFF_MESSAGE_TEMPLATES`, etc.)

---

## How to Run the Integrated Pipeline

### Prerequisites (one-time)

```powershell
# Build ChromaDB knowledge base
uv run python scripts/build_kb.py

# Seed mock customer database
uv run python scripts/manage_db.py --seed

# Verify env
echo $env:GROQ_API_KEY   # must be set
```

### Run

```powershell
# Real-time voice loop (new — this session)
uv run python scripts/run_voice.py

# Text-input loop (existing — for testing without mic)
uv run python scripts/run_assistant.py

# ASR pipeline standalone test (prints transcripts, no LangGraph)
uv run python -m vay.audio.pipeline

# Full test suite
uv run pytest tests/ -v
```

---

## Files Changed This Session

| File | Change |
|---|---|
| `src/vay/asr/router.py` | **BUG FIX** — removed persistent `locked_language`; per-utterance state reset via `_reset_utterance_state()`. Full rewrite for clarity. |
| `src/vay/audio/pipeline.py` | Added `callback: Callable[[ASRResult], None] \| None` param to `STTPipeline`. Callback invoked after each transcription on the consumer thread. |
| `scripts/run_voice.py` | **[NEW]** Full real-time voice entry point. Phone number prompt → VAD → ASR → LangGraph → TTS loop. Handles language switching, call termination, and human handoff. |
| `dev-context.md` | **[NEW]** This file. |
| `src/vay/audio/pipeline.py` | **BUG FIX (2nd pass)** — added `mute()` / `unmute()` methods + `_MIN_UTTERANCE_SAMPLES` guard. |
| `scripts/run_voice.py` | **BUG FIX (2nd pass)** — mute pipeline before `graph.invoke`, unmute + 0.4 s delay after; punctuation-only transcript filter. |

---

---

## Bug Fixes — 2nd Pass (same session)

### VAD captures TTS speaker output as user speech

**Symptom:** After the assistant spoke, a phantom utterance like `"."` appeared in the transcript immediately after — Whisper hallucinating on the speaker echo or room ring.

**Root cause:** `STTPipeline` continued listening while `tts_node` (inside `graph.invoke`) played audio. The speaker output fed back into the mic, VAD detected it as speech, and it was queued for ASR.

**Fix — `pipeline.py`:**
- Added `mute()` / `unmute()` methods backed by a `threading.Event`.
- `mute()` — VAD producer discards utterances instead of queuing them.
- `unmute()` — drains any residual captures from the queue, then clears the flag.

**Fix — `run_voice.py`:**
- `pipeline.mute()` is called immediately before `graph.invoke(state)` (TTS plays inside this call).
- `pipeline.unmute()` is called in the `finally` block after a `0.4 s` post-TTS silence delay, giving the room time to go quiet.

### Punctuation-only transcripts (Whisper hallucination)

**Symptom:** `"."` or `"..."` transcripts reaching LangGraph, causing spurious chitchat turns.

**Fix — `run_voice.py`:** `_NOISE_TRANSCRIPT_RE` regex (`^[\s.,!?…\-–—\'\"]+$`) rejects any transcript that is entirely punctuation/whitespace before it hits LangGraph.

**Fix — `pipeline.py`:** `_MIN_UTTERANCE_SAMPLES = 8 000` guard in the consumer loop — utterances shorter than ~0.5 s are discarded before even reaching the ASR router.

---

## Bug Fixes — 3rd Pass (latency + language accuracy)

### Double Groq API call for Whisper paths (main latency source)

**Root cause:** `ASRRouter._detect_language()` called `whisper_asr.detect_language()` (1st Groq call), then called `whisper_asr.transcribe()` again with the detected code (2nd Groq call). For all Tier-2 languages (en, fr, de, etc.) this doubled per-turn latency.

**Fix — `whisper.py`:** New `transcribe_auto()` method — a single Groq call with no `language=` hint. Whisper's `verbose_json` response includes both the detected language code (`response.language`) AND the transcribed text in one round-trip.

**Fix — `router.py`:** New single-pass flow:
```
whisper_asr.transcribe_auto(audio)  →  (text, lang)   ← ONE Groq call
  ├─ lang in tier1_languages?  →  IndicConformer.transcribe()  (local, fast)
  │                                  fallback: use whisper text if Indic empty
  └─ lang in tier2?  →  return whisper result directly  (NO 2nd call)
```

**Impact:** Tier-2 paths (English, etc.) cut from ~2× Groq latency to 1×. Language detection now uses the FULL utterance text+audio, not a separate partial-audio pass → more accurate.

### Unrecognised language codes crash (`thai`, etc.)

**Root cause:** Groq's Whisper returns full English language names (`"thai"`, `"japanese"`) in `response.language`. The old `LANGUAGE_MAP` only covered ~23 Indian languages + English, so `"thai"` passed through unmapped → `language="thai"` → Groq 400 error.

**Fix — `whisper.py`:**
- `_LANGUAGE_NAME_TO_CODE`: comprehensive map of 60+ full English names → ISO codes.
- `_GROQ_VALID_LANGUAGE_CODES`: frozenset of all codes the Groq endpoint accepts.
- `_normalise_language(raw)`: validates the raw string; returns `None` for anything unrecognised.
- `transcribe()` now calls `transcribe_auto()` as fallback when `_normalise_language` returns `None` instead of crashing with a 400.

---

## Known Limitations & Next Steps

| Issue | Notes |
|---|---|
| `ABUSIVE_LANGUAGE_PATTERN` regex is English-only | Abusive speech in Tamil/Hindi will not trigger the regex gate — relies on LLM's `aggressive=true` output alone, which `llama-3.1-8b-instant` can miss. Add Tamil/Hindi abuse patterns or upgrade the model. |
| Whisper language detection quality | Very short utterances (<0.5s) can mis-detect. The low-confidence fallback (0.5 threshold) now re-runs with IndicConformer when the customer's registered DB language is Tier-1. |
| No barge-in / interruption | Out of scope for this build. |
| Gradio/Streamlit UI not wired | `ui/app.py` is still a stub. The new `STTPipeline.callback` pattern makes it straightforward to wire — pass `on_asr_result` as the callback from the UI backend. |
| ChromaDB must be pre-built | `scripts/build_kb.py` must run before first use. If the collections are empty, all retrieval scores will be 0 and every query will escalate to human handoff. |

---

## Bug Fixes — 4th Pass (language detection + DB + TTS)

### Bug 1: Tamil speech detected as Indonesian (`id`) at low confidence

**Symptom:** Speaking Tamil into the mic produced `[Router] Detected language 'id' (confidence: 0.43)`. Whisper then returned a garbled Indonesian-sounding transcript (`"yang current plan di Bensina"`) and the TTS played the reply using the English fallback voice, making it unintelligible.

**Root cause:**  
Whisper's single-pass `transcribe_auto()` misidentified Tamil speech as Indonesian with confidence 0.43. Since `id` is a Tier-2 code, the router returned the Whisper result directly — no IndicConformer. The TTS node then used `lang="id"` which isn't in `VOICES`, so it fell back to `en-IN-NeerjaNeural` to speak Tamil text, producing garbled output.

**Fix — `tools/session.py`:**
- Added `preferred_language: str = "en"` field to `SessionContext`.
- Added `load_customer_preferred_language(phone_number)` helper that reads the `language_pref` column from the `customers` table.

**Fix — `scripts/run_voice.py`:**
- At startup, `load_customer_preferred_language(phone_number)` is called and stored on `session.preferred_language` and used as the initial `self.language`.
- In `on_asr_result`: when Whisper confidence < 0.5 AND the result is Tier-2 AND the customer's registered language is a Tier-1 Indic language, the pipeline re-runs `router.route_and_transcribe(last_audio_tensor, override_language=preferred_lang)` using IndicConformer. If IndicConformer produces a non-empty transcript, it replaces the Whisper result.

**Fix — `audio/pipeline.py`:**
- Added `self.last_audio_tensor: torch.Tensor | None = None`.
- Consumer loop now sets `self.last_audio_tensor = audio_tensor` before calling the router, so `on_asr_result` can access the raw audio for the retry.

---

### Bug 2: Demo phone `9876543210` had no DB account

**Symptom:** `_fetch_account_context("9876543210")` returned `""` (customer not found). The plans sub-agent called `listPlans()` on all plan types but could not look up the caller's active plan — because the phone wasn't seeded in the DB. The LLM hallucinated a confused Tamil answer about not being able to find the plan.

**Root cause:** The seed data only had `9876500001`–`9876500010`. The `--phone 9876543210` flag used in the dev/demo `run_voice.py` invocation pointed to a non-existent customer.

**Fix — `tools/db_seed_data.py`:**
- Added `("9876543210", "Vishwa Raj", "1995-06-15", 1, "Chennai", "600001", "prepaid", "ta")` to `CUSTOMERS`.
- Added `"9876543210": ("PPD_VALUE", _days_ago(8), "")` to `SUBSCRIPTIONS`.

**Reseed:** Run `uv run python scripts/manage_db.py --reset` to apply.

Now `_fetch_account_context("9876543210")` returns:
```
Name: Vishwa Raj
Active Plan: Prepaid Value | Rs 299.0/month | 2 GB/day data | Unlimited voice | Valid: 28 days
Outstanding Balance: None (all bills paid)
```

---

## Bug Fixes — 5th Pass (Tamil LLM fragment reply)

### LLM reply truncated to a sentence fragment (e.g. `"உங்கள் தற்போதைய பிரதிநிதித்துவமான தகவல் காரணமாக,"`)

**Symptom:** The sub-agent returned a garbled comma-terminated Tamil fragment instead of a real answer, even though the tool call (`runTroubleshootFlow`) returned correct troubleshooting steps in English.

**Root cause (double failure):**
1. `llama-3.1-8b-instant` generated a highly repetitive Tamil reply where the same 12–350-char phrase repeated immediately at position 0.
2. `_detoxify_repetition` in `tool_agent.py` detected the loop, truncated to the first occurrence of the repeated span — which was `"உங்கள் தற்போதைய பிரதிநிதித்துவமான தகவல் காரணமாக,"` (a comma-terminated fragment). The `len(truncated) >= 40` guard then returned this fragment instead of `""`, so the `or localized(HANDOFF_MESSAGE_TEMPLATES)` fallback never fired.

**Fix 1 — `src/vay/graph/tool_agent.py`:**
- Added `_is_complete_reply(text)` validator: returns False when the text ends with `,;:-–—/` or has no terminal punctuation (`.!?।`) and is < 80 chars.
- Added `_TERMINAL_PUNCT_RE` and `_FRAGMENT_END_RE` regex constants.
- `_detoxify_repetition` now calls `_is_complete_reply(candidate)` after truncation: if the candidate is a fragment, returns `""` so `run_tool_agent`'s `or localized(HANDOFF_MESSAGE_TEMPLATES, language)` fires a proper Tamil fallback.

**Fix 2 — `src/vay/graph/core_utils.py`:**
- Added **Rule 12 (ANTI-REPETITION)** to `SUBAGENT_SYSTEM_PROMPT_TEMPLATE`: caps replies at 3-4 sentences, explicitly bans phrase repetition, and warns the model this is especially critical for Indic languages. This attacks the root cause (prevents loops from being generated) rather than only cleaning them up after the fact.

**Verification:**
```python
# Fragment loop → empty string (fallback fires)
_detoxify_repetition('உங்கள் தற்போதைய பிரதிநிதித்துவமான தகவல் காரணமாக,' * 5)  # -> ''

# Clean Tamil reply → unchanged
_detoxify_repetition('உங்கள் data plan active ஆக உள்ளது. APN settings சரியாக உள்ளதா என்று confirm செய்யுங்கள்.')  # -> original text
```

---

## Bug Fixes — 6th Pass (Plan Recommendations & Tool Schema Typing)

### 1. Model failed to list available plans when asked to change plan
**Symptom:** In Turn 2 (*"I want to change my plan to a new plan. What are the plans available?"*), the model called `listPlans` but then only asked *"Would you like to compare or upgrade to a different plan?"* without stating the options.
**Root Cause:** A rule in `core_utils.py` instructed the model to *"call a lookup tool, then ask a clarifying question"* on non-specific plan change queries. The model followed this too literally and omitted the plan details.
**Fix:** Updated the prompt in `core_utils.py`:
> *"When the customer asks about available plans or wants to change/upgrade plans (e.g. 'what plans are available', 'change my plan'), call `listPlans`, briefly present 2 to 3 main plan options with their price and data (e.g., 'We have Prepaid Basic at Rs 239 with 1.5 GB/day and Prepaid Value at Rs 299 with 2 GB/day'), and ask which one they would like to choose."*

### 2. Tool-calling schema 400 error on `listPlans`
**Symptom:** Groq returned `400 Bad Request: attempted to call tool 'listPlans plan_type="prepaid"' which was not in request.tools` due to failed XML formatting.
**Root Cause:** `listPlans(plan_type: str = "")` had a default empty string, causing Pydantic / ChatGroq schema generation to confuse the tool parser.
**Fix:** Changed signature in `src/vay/tools/plans.py` to `listPlans(plan_type: str | None = None)`.

**Verification:**
```
Input: "I want to change my plan to a new plan. What are the plans available?"
Tool Call: listPlans({'plan_type': 'prepaid'})
Output: "We have Prepaid Basic at Rs 239 with 1.5 GB/day and Prepaid Plus at Rs 399 with 3 GB/day. Which one would you like to choose?"
```

---

## Files Changed This Session (4th + 5th + 6th Pass)

| File | Change |
|---|---|
| `src/vay/tools/db_seed_data.py` | Added `9876543210` (Vishwa Raj, Tamil prepaid, PPD_VALUE) to `CUSTOMERS` and `SUBSCRIPTIONS`. |
| `src/vay/tools/session.py` | Added `preferred_language` field to `SessionContext`; added `load_customer_preferred_language()` DB helper. |
| `src/vay/audio/pipeline.py` | Added `last_audio_tensor` cache on `STTPipeline` — set each turn, read by `run_voice.py` for IndicConformer retry. |
| `scripts/run_voice.py` | Load `preferred_language` from DB at startup; re-run IndicConformer when Whisper confidence < 0.5 on a Tier-2 detection for a Tier-1 customer. Fixed `wait_for_stop` polling for instant Ctrl+C. |
| `src/vay/graph/tool_agent.py` | Added `_is_complete_reply()` fragment guard to `_detoxify_repetition`. |
| `src/vay/graph/core_utils.py` | Added Rule 12 (ANTI-REPETITION), updated plan recommendation rule, and enforced current turn language. |
| `src/vay/tools/plans.py` | Updated `listPlans(plan_type: str | None = None)` to prevent Groq 400 schema error. |
| `src/vay/tts/engine.py` | Added script-aware voice selection (e.g. Tamil Unicode $\rightarrow$ Tamil voice) to prevent English TTS reading Tamil Unicode as numbers. |


---

## Language Tier Reference

| Tier | Languages | ASR Model |
|---|---|---|
| **Tier 1** | as, bn, brx, doi, gu, hi, kn, kok, ks, mai, ml, mni, mr, ne, or, pa, sa, sat, sd, ta, te, ur | `ai4bharat/indic-conformer-600m-multilingual` |
| **Tier 2** | en + 90 other languages | `openai/whisper-large-v3-turbo` (via Groq API) |

TTS (edge-tts) covers 18 languages: `ta, hi, en, fr, de, es, ja, ko, zh, te, kn, ml, mr, gu, ur, ar, it, ru`. Falls back to `en-IN-NeerjaNeural` for unlisted codes.

