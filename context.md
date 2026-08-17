# Project context: Frontend Interface & Integration Blueprint

This file outlines the current implementation of the Multilingual GenAI Voice Assistant Streamlit frontend (v1.0.0) and serves as the integration roadmap for subsequent pipeline and model connection phases.

---

## 1. Current Implementation (v1.0.0)

We have built a responsive, dark-themed Streamlit application that features a custom, voice-reactive WebGL animation, a browser-based audio recorder/player, and a real-time call center agent monitoring dashboard.

### File Structure
```
VAY-multilingual-agent/
├── app.py                     # Main Streamlit application & layout orchestrator
├── audio_handler.py           # Helper for base64 audio decoding/encoding
├── component_strands/         # Streamlit Custom Component
│   ├── __init__.py            # Component Python declaration
│   └── frontend/
│       └── index.html         # WebGL Strands, mic capture, WAV encoding, & overlay
├── project_context.md         # Reference document containing project requirements & constraints
└── context.md                 # [THIS FILE] Changes and future updates guide
```

### Completed Features

1. **Voice-Reactive WebGL Visualizer** (`component_strands/frontend/index.html`):
   - Integrates the `<Strands />` component from React Bits using vanilla WebGL2 (`ogl` library) served via CDN.
   - Reads mic volume in the browser using the Web Audio API `AnalyserNode` and dynamically increases strand amplitude and speed in real-time, creating a premium visualizer.
   - Automatically animates the strands when playing back synthesized voice responses.

2. **In-Browser Audio Capture (WAV PCM)**:
   - Accesses user microphone, configures a 16kHz mono audio context, and processes input samples.
   - Encodes raw audio data into a compliant 16-bit 16kHz mono PCM WAV format entirely inside the browser, eliminating python-side external conversion dependencies (like `ffmpeg`).

3. **Streamlit Component Wrapper** (`component_strands/__init__.py`):
   - Declares the custom component and routes parameters (presets, speed, status, base64 audio payloads) bidirectionally between Python and the HTML iframe.

4. **Speech Synthesis Playback**:
   - Synthesizes audio responses using `gTTS` in Tamil (`ta`), Hindi (`hi`), and English (`en`).
   - Encodes audio into base64 and forwards it to the browser component, which decodes and plays it using `AudioContext.decodeAudioData()`.

5. **Demo Accent Controller**:
   - Provides a sidebar controller to simulate accents (Tamil Speaker Accent, Hindi Speaker Accent, English Speaker Accent).
   - Generates simulated speech transcripts dynamically when testing the voice recorder.

6. **Safety Handoff & RAG Scorer**:
   - Performs a hybrid keyword-matching search against a mock telecom knowledge base, computing confidence scores.
   - Implements a safety gate where sensitive intent (e.g., connection termination requests) or low RAG scores ($\tau < 0.65$) trigger a human agent escalation.
   - Provides a **Live Agent Dashboard** that visualizes real-time call metrics, active transcripts, and the escalated call queue.

---

## 2. Technical Architecture & Communication Flow

```
+------------------------------------------------------------+
|                       Web Browser                          |
|  +--------------------+             +-------------------+  |
|  | WebGL Strands      |             | Mic Audio Capture |  |
|  | canvas (OGL)       |             | (16kHz mono WAV)  |  |
|  +---------^----------+             +---------+---------+  |
|            |                                  |            |
|    FFT Reactivity                       Recorded WAV       |
|            |                                  |            |
|  +---------+----------+             +---------v---------+  |
|  | Audio Playback     |             | postMessage       |  |
|  | (gTTS Response)    |             | (Base64 WAV)      |  |
|  +---------^----------+             +---------+---------+  |
+------------|----------------------------------|------------+
             |                                  |
             | Base64 MP3                       | Base64 WAV
             |                                  |
+------------|----------------------------------v------------+
|            |         Streamlit Backend        |            |
|  +---------+----------+             +---------v---------+  |
|  | TTS Synthesis      |             | Base64 Decoding   |  |
|  | (gTTS in ta/hi/en) |             | (audio_handler)   |  |
|  +---------^----------+             +---------+---------+  |
|            |                                  |            |
|     Response text                         Raw WAV bytes    |
|            |                                  |            |
|  +---------+----------+             +---------v---------+  |
|  | LLM Response       <-------------+ ASR, RAG & Handoff  |  |
|  | Generation         |             | Pipeline          |  |
|  +--------------------+             +-------------------+  |
+------------------------------------------------------------+
```

---

## 3. Future Updates & Integration Milestones (Backlog)

The frontend is prepared to be linked with the production-ready machine learning models outlined in the hackathon plan. Below are the steps required to transition from the current simulated backend to the actual models.

### Milestone 1: Local Model Setup & Dependencies
- Install production framework packages: `torch`, `torchaudio`, `transformers`, `onnxruntime-gpu`, and `langgraph`.
- Save model configurations and check path access.

### Milestone 2: Audio Routing & ASR Integration (IndicConformer & Whisper)
- **Speech Routing**: Replace the simulated transcriber with real audio evaluation:
  1. Capture the raw WAV bytes decoded by `audio_handler.py`.
  2. Perform Language ID (LID) using Whisper's encoder-only first pass.
  3. **Tier 1 Route**: If Hindi or Tamil is detected, format the audio tensor shape to `[1, num_samples]` (16kHz float32 mono) and route to `ai4bharat/indic-conformer-600m-multilingual`.
     - *Gotcha:* Load `IndicConformer` directly (`AutoModel.from_pretrained(..., trust_remote_code=True)`) and execute CTC decoding. Do not use `transformers.pipeline()` as it will fail (Gotcha §5.1).
  4. **Tier 2 Route**: If English or other languages are detected, route to `openai/whisper-large-v3-turbo`. Apply custom repetition filters on the output to prevent silence/noise hallucinations.

### Milestone 3: Normalization & Intent Extraction (LLM Cleanup Node)
- Wire up a local LLM API (or lightweight client) to perform the **Transcription Normalization** pass:
  - Clean up punctuation, spaces, and casing.
  - Correct ASR transcriptions for code-switched queries (e.g., converting mixed Hinglish/Tanglish phrases to normalized English terms for RAG searching).
  - Extract Intent and Entities (e.g., `phone_number`, `pack_name`).
  - **Sensitive Intent Check**: If the intent is connection cancellation/sensitive account updates, bypass retrieval and execute direct human escalation.

### Milestone 4: Vector Store & Hybrid RAG Retrieval
- Replace the mock keyword scorer in `app.py` with a true Vector DB (ChromaDB or FAISS):
  - Load the operator knowledge base.
  - Implement a **hybrid retriever**: Combine BM25 keyword matching with dense embeddings (e.g., HuggingFace sentence-transformers).
  - Apply top-k reranking to compute a final relevance retrieval score.
  - **Confidence Gate**: Compare retrieval score against the strict threshold $\tau \approx 0.75$. If below, route the state machine directly to the human handoff queue.

### Milestone 5: Orchestration (LangGraph State Machine)
- Replace basic Python procedural branches with a **LangGraph state graph**:
  - Define nodes: `VAD_ASR_Node`, `Normalization_Node`, `Intent_Check_Node`, `RAG_Retrieve_Node`, `Handoff_Gate_Node`, `TTS_Node`, and `Human_Escalation_Node`.
  - Configure state parameters (`Language`, `Transcript`, `Retrieved_Docs`, `Escalation_Status`, `Audio_Response`).
  - Define conditional edges to control state transitions (e.g., routing to `Human_Escalation_Node` if RAG score is low or intent is restricted).

### Milestone 6: Language-Aware TTS (IndicF5/Indic-TTS)
- Replace `gTTS` for Hindi and Tamil with the local `IndicF5` or `Indic-TTS` models:
  - Configure reference audio and reference transcript files for `IndicF5` to guide voice prosody.
  - Keep `gTTS` (or another lightweight TTS) as the Tier 2 general fallback for English.

---

## 4. Changes Made — Aug 17 Session (Handoff Call-Cut + Sub-Agent Latency)

### 4.1 Human handoff now cuts the call and returns to the homepage automatically

**File changed:** `app.py`

**Before:** once the graph routed to human handoff, the app spoke the handoff message and then
just sat in a `"handoff"` status ("Call Ended" shown in the WebGL component) — the Streamlit
session (`session_started`, `chat_history`, `agent_history`, `agent_session`) stayed alive
indefinitely with no automatic reset. Returning to the homepage required the customer to
manually click the End Session button.

**Fix:** in the `AUDIO_ENDED` event handler, when `pending_handoff` is set (i.e. the just-played
reply was the handoff message), the app now performs the same reset the Cancel/End Session
button triggers — clears `session_started`, `phone_number`, `chat_history`, `agent_history`, and
drops `agent_session` — breaking the conversational chain and returning straight to the homepage
with no manual click needed. The manual End Session handler (`end_session` / `escalate_click`
events) was updated to clear `agent_history`/`agent_session` too, so both paths (automatic
handoff and manual cancel) leave the session in the same clean state for the next call.

### 4.2 Near-duplicate RAG query guard (sub-agent tool-calling loop latency)

**File changed:** `src/vay/graph/tool_agent.py`

**Symptom:** a single sub-agent turn took 118.75s and cost 4 sequential Groq LLM round-trips —
3 tool calls to `search_product_catalog` with barely-reworded queries (`'travel plan recharge
requirement postpaid'` → `'...postpaid travel add-on'` → `'Travel Pack recharge requirement'`,
relevance 0.71 / 0.73 / 0.57) followed by the final answer generation. The existing
`seen_calls` dedup (`graph/tool_agent.py`) only catches byte-identical `(tool_name, args)`
repeats, so a small model rewording the same failed search slips through every time, each
retry paying full Groq latency plus a growing system-prompt + history + RAG-context payload.

**Fix:** added `_is_near_duplicate_query()` — a Jaccard token-overlap check (threshold 0.5) over
the free-text `query` argument of tool calls, scoped per tool name (`seen_queries` dict). When a
new query overlaps ≥50% of its tokens with any query already tried this turn for the same tool,
it's treated like an exact repeat: the tool is not re-invoked, and the model gets a nudge message
("you already searched for this or something very similar... use a meaningfully different query,
or stop and answer with what you already have") instead of a fresh LLM+tool round-trip. Verified
against the exact queries from the 118.75s trace above (`_is_near_duplicate_query` at threshold
0.5 correctly flags the 3rd query as a near-duplicate of the 1st) and against `uv run pytest`
(all 11 tests still pass — this only changes behavior when a near-duplicate query is detected).

### 4.3 `test.md` corruption cleanup

**File changed:** `test.md` (RAG pipeline test report)

Fixed two classes of formatting corruption that had crept into the file (likely from a
tab-eating paste): stray single-backtick lines that were meant to be triple-backtick code
fences (architecture diagram, debug traces) now render as proper code blocks; and several
words that had their first letter eaten by a stray tab character —
`illing_policy`→`billing_policy`, `echnical_kb`→`technical_kb`, `ll-MiniLM`→`all-MiniLM`,
`ool_use_failed`→`tool_use_failed`, `ool_agent.py`→`tool_agent.py`,
`est_runner.py`/`est_results.jsonl`→`test_runner.py`/`test_results.jsonl`,
`un_tests.ps1`→`run_tests.ps1`.
