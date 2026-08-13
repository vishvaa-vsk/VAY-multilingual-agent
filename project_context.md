# Project Context: Multilingual GenAI Voice Assistant for Customer Care

**Purpose of this file**: This is the locked source of truth for any AI coding agent (Claude Code, Antigravity, or similar) working on this project. Every fact below is either a decision already made, or a number already measured. Do not substitute assumptions, "best practices," or generic patterns for anything stated here — treat conflicts between this file and an agent's prior training as this file being correct. If something genuinely isn't covered here, flag it as a question rather than inventing an answer.

---

## 1. Event Context

- Hackathon: Velammal-AIA Partnership / Cognizant hackathon
- Use case #15 of 18: **"Multilingual GenAI Voice Assistant for Customer Care"**
- Team size: 8 members
- Timeline: build window Aug 12–18, 2026 (7 days), final evaluation/presentation Aug 19, 2026
- Judging criteria include: use-case understanding, solution architecture, innovation, UI/UX, technical implementation & code quality, model performance & evaluation (accuracy/precision/recall/F1), deployment & integration, presentation, teamwork
- Reference dataset named in the original problem statement: Mozilla Common Voice (crowdsourced multilingual speech + transcript dataset)

## 2. Problem Statement (as given)

"Voice-based customer care is dominated by a few high-resource languages, leaving large customer segments underserved. Build a GenAI voice assistant that understands and responds across multiple languages/accents for telecom self-service (bill queries, plan changes, complaints)." Required components per the brief: ASR (fine-tune/evaluate on multilingual accented audio), LLM+RAG grounded on an operator knowledge base, TTS, language auto-detection, intent recognition, graceful hand-off to a human agent on low confidence.

## 3. Language Scope — DO NOT DEVIATE WITHOUT DISCUSSION

**In scope, specialized (Tier 1): Tamil, Hindi**
**In scope, general fallback (Tier 2): English + any other language Whisper supports**

Rationale (do not re-litigate this without cause): the problem statement explicitly frames high-resource languages (English, and others) as already well-served — the underserved segment it calls out is exactly where Tamil/Hindi specialization targets effort. Two-tier design means the system degrades gracefully to a Whisper baseline for any language without a specialized route, rather than breaking on out-of-scope input (e.g. Dutch, Spanish, Japanese).

**Do not attempt to add more Tier-1 languages during the hackathon window** unless the team explicitly decides to — scope is Tamil + Hindi as the specialized pair, by design, given the 7-day constraint.

## 4. Model Choices — EXACT, DO NOT SUBSTITUTE

| Component | Model | Notes |
|---|---|---|
| ASR (Tier 1: Tamil, Hindi) | `ai4bharat/indic-conformer-600m-multilingual` | Does NOT support English. Do not attempt to route English through it. |
| ASR (Tier 2: English + fallback) | `openai/whisper-large-v3-turbo` | Handles ~90+ languages at baseline quality; this is the general-purpose fallback, not just the English model. |
| Language ID | Whisper's own encoder-only language detection pass | No separate LID model — do not add one. This is a deliberate cost/latency decision. |
| TTS (Tamil, Hindi) | AI4Bharat IndicF5 (preferred) or AI4Bharat Indic-TTS (simpler fallback if IndicF5's reference-audio requirement is too much overhead) | IndicF5 requires a reference prompt audio clip + its transcript to guide voice/prosody — it is NOT plain text-in/speech-out. |
| TTS (English + fallback) | Any standard TTS (gTTS is the simplest to wire up; Coqui/Piper if quality needed) | |
| LLM | Not yet pinned to a specific provider/model in this doc — confirm with team before hardcoding | |
| Orchestration | LangGraph (state graph, not LangChain linear chains) | LangChain retriever/vector-store primitives may be used *inside* LangGraph nodes |
| VAD | Silero VAD (or equivalent) | Utterance boundary = ~600–700ms of detected silence after speech |
| Vector DB | ChromaDB or FAISS | Hybrid search: keyword (BM25-style) + vector, with top-k reranking |

## 5. CRITICAL Implementation Gotchas (measured, not theoretical)

These were discovered through actual debugging during evaluation. An agent unaware of these WILL waste time rediscovering them or silently produce wrong results.

### 5.1 IndicConformer cannot be loaded via `transformers.pipeline()`
```python
# WRONG — will fail with "Unrecognized configuration class IndicASRConfig"
pipe = pipeline("automatic-speech-recognition", model="ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True)
```
`pipeline()` for the ASR task only recognizes a fixed set of Auto classes (AutoModelForCTC, AutoModelForTDT, AutoModelForSpeechSeq2Seq), and IndicConformer's custom `IndicASRConfig` isn't among them, even with `trust_remote_code=True`.

**Correct approach:**
```python
model = AutoModel.from_pretrained("ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True)
transcription = model(wav_tensor, lang_code, "ctc")  # or "rnnt" for the other decoding strategy
```
- `wav_tensor`: shape `[1, num_samples]`, float32, 16kHz mono
- `lang_code`: ISO code, e.g. `"ta"` for Tamil, `"hi"` for Hindi
- Return value may be a string or a list depending on version — normalize with `if isinstance(result, list): result = result[0]`
- You will see `onnxruntime` warnings about `CUDAExecutionProvider` not being available in some environments — this is a runtime/provider availability issue, investigate `onnxruntime-gpu` install if GPU acceleration is expected but not occurring.
- You will also see repeated `"Please check FRAME_DURATION_MS"` warnings during load — these are informational from the model's internals, not fatal.

### 5.2 IndicConformer does not support English
Do not attempt to evaluate or route English through it. Two-tier routing exists precisely because of this — English always goes to Whisper.

### 5.3 Whisper hallucinates on Tamil/Hindi, especially on silence/noise/code-switching
Mitigation already decided: VAD-based silence trimming before ASR, plus a post-ASR hallucination/repetition filter applied specifically on the Whisper path (not needed on IndicConformer, which doesn't share this failure mode).

### 5.4 Code-switching (Tanglish/Hinglish) is a known weak spot for BOTH models
Neither Whisper nor IndicConformer was trained primarily on code-switched data. Mitigation: an LLM-based transcript normalization/cleanup pass after ASR, before retrieval — this is NOT optional polish, it's the actual fix for this failure mode. Do not skip it.

### 5.5 Text normalization is required before computing WER/CER
Raw-text WER comparison inflates error rates due to punctuation/whitespace/Unicode-form mismatches. Always normalize (lowercase, strip punctuation, collapse whitespace, Unicode NFC normalize) both reference and hypothesis before scoring.

### 5.6 Mozilla Common Voice Kaggle mirror (`mozillaorg/common-voice`) has a filename collision trap
This specific Kaggle dataset reuses generic filenames (`sample-000000.mp3`, `sample-000001.mp3`, ...) across ALL splits (`cv-valid-test`, `cv-valid-train`, `cv-other-test`, `cv-other-train`, `cv-valid-dev`, `cv-other-dev`, `cv-invalid`). **If you index audio files by bare filename across the whole extracted tree, files from different splits will silently overwrite each other**, pairing the wrong audio with a transcript and producing impossible WER (>100%). Fix: only index mp3s from within the specific split folder you need (e.g. only paths containing `cv-valid-test`), never index by bare filename across the whole tree.

### 5.7 Tamil/Hindi datasets for this project were sourced via Mozilla Data Collective, not Kaggle
English was sourced via the Kaggle `mozillaorg/common-voice` mirror instead (see 5.6 for its specific trap). This is a deliberate mixed-source setup, not an oversight — Mozilla Data Collective did not have a confirmed English dataset ID at the time this was set up.

## 6. Measured ASR Evaluation Results (ground truth — do not re-estimate or guess these)

Evaluated on 200 samples per language, Mozilla Common Voice (Tamil/Hindi) and Kaggle Common Voice mirror (English), same normalization pipeline for both models:

| Language | Whisper WER | Whisper CER | Whisper Avg Time/sample | IndicConformer WER | IndicConformer CER | IndicConformer Avg Time/sample |
|---|---|---|---|---|---|---|
| Tamil | 62.44% | 17.87% | 0.86s | 26.06% | 5.52% | 1.66s |
| Hindi | 35.10% | 17.60% | 0.59s | 12.00% | 6.30% | 1.10s |
| English | ~3.8% (measured in an earlier clean run; a later run with a corrupted Kaggle audio-text pairing produced an invalid >100% result — see §5.6) | — | — | N/A (not supported) | N/A | N/A |

**Conclusions this data supports:**
- IndicConformer meaningfully outperforms Whisper on both Tamil (roughly half the WER) and Hindi (roughly a third the WER) — this is the empirical justification for the two-tier routing architecture.
- Whisper is strong on English (~3.8% WER in a clean run) — validates using it as the Tier 2 fallback.
- These are clean, monolingual, read-aloud speech numbers. Code-switched (Tanglish/Hinglish) performance has NOT been separately measured as of this writing — if an agent needs code-switching numbers, they must be generated fresh, not assumed to match the above.
- Do not present the English WER as a settled number in any deliverable until re-validated after the fix in §5.6 — the last known-good English WER figure is ~3.8%, from a run prior to the Kaggle pairing bug being introduced.

## 7. Full Architecture — Pipeline Order

This is the current locked architecture (matches the team's finalized diagram). Implement in this order; do not reorder or skip stages without discussion.

```
USER SPEAKS
  → Microphone input → audio stream
  → VAD (Voice Activity Detection) — detects speech vs silence, ~600-700ms silence = utterance boundary
  → Language ID (Whisper encoder-only pass, no separate model)
      ├─ Tamil/Hindi → ai4bharat/indic-conformer-600m-multilingual → Transcription (raw)
      └─ Other (incl. English) → openai/whisper-large-v3-turbo → Whisper Output Filter
                                   (hallucination/repetition filtering, Whisper-path only) → Transcription (raw)
  → Transcription Normalization
      (code-switch normalization, ASR error correction, language tagging, entity tagging — LLM-based cleanup)
  → Structured Format output: { Language, Intent, Normalized, Entities, Confidence }
  → Intent + Entity Extraction (analyzes structured format)
      ├─ Sensitive/Restricted intent detected → escalate directly to HUMAN HANDOFF (bypasses retrieval entirely)
      └─ Not sensitive → continue to RAG Module
  → RAG MODULE:
      Intent-Aware Hybrid RAG (keyword + vector search) queries Vector DB (ChromaDB/FAISS)
      → top-k reranked results → Retrieval Score
  → Retrieval Score check (threshold τ, empirically ~0.75-0.85 for "high confidence")
      ├─ Low confidence → escalate to HUMAN HANDOFF
      └─ High confidence → Handoff Gate
  → Handoff Gate (human request detected? LLM uncertain in its draft response?)
      ├─ YES → HUMAN HANDOFF
      └─ NO → LLM Response Generation (grounded, in the user's detected language)
  → Language-Aware TTS (IndicF5/Indic-TTS for Tamil/Hindi, multilingual fallback for other languages)
  → Audio Output → played back to user
  → loop back to VAD for next utterance
```

**HUMAN HANDOFF** is a single terminal node reachable from three separate triggers (sensitive intent, low retrieval confidence, handoff gate) — implement it as one shared escalation path, not three separate ones, to keep logging/queueing consistent.

## 8. Design Decisions and Their Rationale (so agents don't "fix" intentional choices)

- **No dedicated LID model** — Whisper's own encoder does language detection. This was a deliberate cost/latency decision, not an oversight.
- **RAG queries an English-normalized representation, but responses generate directly in the user's language** — do not add a separate "translate response back" step; the LLM is instructed to respond in-language directly using the retrieved English context.
- **Retrieval confidence threshold (~0.75-0.85) is intentionally strict**, prioritizing safe hand-off over confidently wrong answers in a telecom self-service context (billing/account info at stake). Do not loosen this without an explicit decision from the team — it trades off demo "wow factor" for correctness/safety, and that trade was made deliberately.
- **Sensitive/Restricted Intent gate exists for compliance/privacy reasons**, not just accuracy — it catches account/PII-adjacent intents and routes to a human before they ever reach retrieval or generation.
- **LangGraph, not LangChain, for orchestration** — the pipeline has real conditional branching (language routing, confidence checks, handoff gate), which is a state-machine problem. LangChain primitives (retrievers, vector store wrappers) can still be used *inside* LangGraph nodes.
- **No plans to fine-tune ASR, TTS, or LLM models from scratch** during the hackathon window — all models are used pretrained. This is a scope decision given the 7-day timeline, not a technical limitation being worked around.
- **Streaming approach**: sentence-level streaming from LLM generation to TTS (start synthesizing/playing the first sentence while later sentences are still generating), not full-duplex/barge-in interruption handling. Barge-in is explicitly out of scope for this hackathon.

## 9. Explicit Non-Goals / Things NOT to Build

- Do not attempt real-time barge-in / interruption handling (user talking over the assistant mid-response).
- Do not attempt to fine-tune any ASR, TTS, or LLM model — pretrained models only.
- Do not add languages beyond Tamil/Hindi (Tier 1) and Whisper's general coverage (Tier 2) without an explicit team decision.
- Do not build real human-agent telephony/handoff infrastructure — a mock escalation queue/dashboard is sufficient for the demo.
- Do not assume IndicConformer supports English, or any language outside its documented 22 Indian languages.
- Do not invent Mozilla Data Collective or Kaggle dataset IDs — use only the ones already confirmed working (Tamil/Hindi via Mozilla Data Collective, English via the `mozillaorg/common-voice` Kaggle dataset), and if a new one is needed, it must be looked up fresh, not guessed.

## 10. Open Items (not yet decided — do not assume answers)

- Exact LLM provider/model for response generation and the transcript-normalization cleanup pass — not yet pinned in this document.
- Whether IndicF5 (needs reference audio) or plain Indic-TTS (simpler) is the final TTS choice for Tamil/Hindi.
- Code-switched (Tanglish/Hinglish) ASR accuracy has not been separately measured — do not present code-switching performance numbers as validated.
- Exact retrieval threshold τ may still be tuned empirically against the actual built KB, rather than a scoped test set.