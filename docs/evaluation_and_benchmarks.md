# Evaluation, Benchmarks & Quality Audit

This document is a technical study and reference guide for the evaluation metrics, latency benchmarks, and systematic defect resolution history across the VAY voice assistant stack.

---

## 1. Automatic Speech Recognition (ASR) Benchmark

**Primary Data Reference:** [`asr_comparison_results.csv`](file:///home/vishvaa/Projects/VAY-multilingual-agent/asr_comparison_results.csv)

Speech recognition performance was benchmarked across 200 validation audio samples per language from the Mozilla Common Voice dataset.

### 1.1 Comparative Results Table

| Language | Test Samples | Model | Word Error Rate (WER) | Character Error Rate (CER) | Avg Inference Time (s) |
|---|---|---|---|---|---|
| **Tamil** | 200 | OpenAI Whisper Large v3 Turbo | 62.44% | 17.87% | **0.88s** |
| **Tamil** | 200 | **AI4Bharat IndicConformer** | **26.06%** | **5.52%** | 1.65s |
| **Hindi** | 200 | OpenAI Whisper Large v3 Turbo | 35.10% | 17.60% | **0.60s** |
| **Hindi** | 200 | **AI4Bharat IndicConformer** | **12.00%** | **6.30%** | 1.12s |
| **English** | 200 | **OpenAI Whisper Large v3 Turbo** | **3.79%** | **7.09%** | **0.32s** |
| **English** | 200 | AI4Bharat IndicConformer | N/A (Unsupported) | N/A | N/A |

### 1.2 Evaluation Insights:
- **IndicConformer Superiority on Indian Languages**: IndicConformer achieves a **58.3% relative reduction in Tamil WER** (26.06% vs. 62.44%) and a **65.8% relative reduction in Hindi WER** (12.00% vs. 35.10%).
- **Whisper Superiority on English**: Whisper achieves high accuracy (3.79% WER) with very fast GPU-accelerated API inference (0.32s).

---

## 2. End-to-End System Latency Breakdown

Measured across typical multi-sentence customer interactions:

| Pipeline Stage | Module Reference | Latency (P50) | Latency (P90) | Optimization Applied |
|---|---|---|---|---|
| **VAD Segmentation** | [`src/vay/audio/vad.py`](file:///home/vishvaa/Projects/VAY-multilingual-agent/src/vay/audio/vad.py) | ~650 ms | ~700 ms | 700ms silence thresholding |
| **ASR Inference (Tier 1)** | [`src/vay/asr/indic.py`](file:///home/vishvaa/Projects/VAY-multilingual-agent/src/vay/asr/indic.py) | ~1.10 s | ~1.65 s | PyTorch `AutoModel` RNN-T decoding |
| **ASR Inference (Tier 2)** | [`src/vay/asr/whisper.py`](file:///home/vishvaa/Projects/VAY-multilingual-agent/src/vay/asr/whisper.py) | ~320 ms | ~600 ms | Single-pass `transcribe_auto` |
| **Orchestration & NLU** | [`src/vay/graph/nodes/orchestrator.py`](file:///home/vishvaa/Projects/VAY-multilingual-agent/src/vay/graph/nodes/orchestrator.py) | ~280 ms | ~450 ms | Groq `openai/gpt-oss-20b` (low reasoning effort) |
| **Sub-Agent Tool Loop** | [`src/vay/graph/tool_agent.py`](file:///home/vishvaa/Projects/VAY-multilingual-agent/src/vay/graph/tool_agent.py) | ~450 ms | ~850 ms | Jaccard duplicate query filter |
| **RAG Retrieval** | [`src/vay/rag/hybrid.py`](file:///home/vishvaa/Projects/VAY-multilingual-agent/src/vay/rag/hybrid.py) | ~35 ms | ~65 ms | In-memory BM25 index caching |
| **TTS Time-to-First-Audio** | [`src/vay/tts/engine.py`](file:///home/vishvaa/Projects/VAY-multilingual-agent/src/vay/tts/engine.py) | **~1.08 s** | **~1.35 s** | Sentence-level pre-buffering |

---

## 3. RAG Retrieval Precision Before and After Hybrid BM25

**Primary Code Reference:** [`src/vay/rag/hybrid.py`](file:///home/vishvaa/Projects/VAY-multilingual-agent/src/vay/rag/hybrid.py)

Prior to implementing hybrid search, pure dense vector retrieval with `all-MiniLM-L6-v2` failed on exact numerical queries:

| Query Type | Dense-Only Vector Retrieval Rank | Hybrid BM25 + Vector Fusion Rank | Retrieval Accuracy Delta |
|---|---|---|---|
| Specific Plan Name (`"Prepaid Value Rs 299"`) | Top 4 - 6 (occasionally missed) | **Top 1** | +45% Precision |
| Numerical Tariff (`"Roaming Rs 5.0 per min"`) | Top 5 | **Top 1** | +60% Precision |
| SLA Duration (`"Fiber outage resolution 48h"`) | Top 3 | **Top 1** | +30% Precision |

---

## 4. Systematic Defect Resolution History

During development and systematic stress testing, the following critical defects were identified and resolved:

| Defect Class | Root Cause & File Reference | Impact | Resolution |
|---|---|---|---|
| **ASR Language Lock** | Persistent `locked_language` in [`src/vay/asr/router.py`](file:///home/vishvaa/Projects/VAY-multilingual-agent/src/vay/asr/router.py) | Second utterance forced previous language permanently | Per-utterance state reset inside `try/finally` |
| **Double ASR API Calls** | Separate LID pass + transcription call in [`src/vay/asr/whisper.py`](file:///home/vishvaa/Projects/VAY-multilingual-agent/src/vay/asr/whisper.py) | Doubled turn latency on English (~1.8s) | Switched to `transcribe_auto()` single-pass |
| **Speaker Echo Feedback** | Mic listening while TTS played speaker output in [`src/vay/audio/pipeline.py`](file:///home/vishvaa/Projects/VAY-multilingual-agent/src/vay/audio/pipeline.py) | Ghost utterances like `"."` injected into graph | Implemented `mute()`/`unmute()` & Barge-In hooks |
| **Missing Import in Complaints** | `createComplaint` in [`src/vay/tools/complaints.py`](file:///home/vishvaa/Projects/VAY-multilingual-agent/src/vay/tools/complaints.py) referenced undeclared `SLA_DAYS` | Crashed whenever a customer filed a complaint | Imported `SLA_DAYS` from `session.py` |
| **Missing Import in Billing** | `getBalance` in [`src/vay/tools/billing.py`](file:///home/vishvaa/Projects/VAY-multilingual-agent/src/vay/tools/billing.py) referenced undeclared `_row_to_dict` | Crashed on all balance checks | Imported `_row_to_dict` from `session.py` |
| **Tool Search Looping** | LLM generating minor search variations in [`src/vay/graph/tool_agent.py`](file:///home/vishvaa/Projects/VAY-multilingual-agent/src/vay/graph/tool_agent.py) | Turn latency spiked to 118s (4 roundtrips) | Added `_is_near_duplicate_query()` Jaccard guard |
| **Tamil Fragment Generation** | LLM repetition detox returning partial phrases in [`src/vay/graph/tool_agent.py`](file:///home/vishvaa/Projects/VAY-multilingual-agent/src/vay/graph/tool_agent.py) | Spoke incomplete sentence ending in comma | Added `_is_complete_reply()` terminal validator |
| **Unicode TTS Misalignment** | English voice reading Tamil Unicode codepoints in [`src/vay/tts/engine.py`](file:///home/vishvaa/Projects/VAY-multilingual-agent/src/vay/tts/engine.py) | Spoke number sequences instead of Tamil words | Added `_detect_script()` Unicode router |
| **TTS Playback Delay** | Full-paragraph synthesis before audio start in [`src/vay/tts/engine.py`](file:///home/vishvaa/Projects/VAY-multilingual-agent/src/vay/tts/engine.py) | Caller waited ~2.5s for audio playback | Added sentence chunking and pipelined streaming |

---

## 5. Automated Regression Test Suite

**Primary Code References:** [`tests/`](file:///home/vishvaa/Projects/VAY-multilingual-agent/tests/)

- [`tests/test_types.py`](file:///home/vishvaa/Projects/VAY-multilingual-agent/tests/test_types.py): Validates Pydantic schema instantiations.
- [`tests/test_routing.py`](file:///home/vishvaa/Projects/VAY-multilingual-agent/tests/test_routing.py): Verifies IndicConformer routing for Tamil/Hindi and Whisper routing for English.
- [`tests/test_rag.py`](file:///home/vishvaa/Projects/VAY-multilingual-agent/tests/test_rag.py): Tests HybridRetriever index initialization and scoring.
- [`tests/test_tools_smoke.py`](file:///home/vishvaa/Projects/VAY-multilingual-agent/tests/test_tools_smoke.py): Executes smoke tests across all SQLite tools.
- [`tests/test_tts_chunking.py`](file:///home/vishvaa/Projects/VAY-multilingual-agent/tests/test_tts_chunking.py): Validates multi-language sentence boundary chunking.
