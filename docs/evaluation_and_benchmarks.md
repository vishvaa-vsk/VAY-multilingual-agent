# Evaluation, Benchmarks & Quality Audit

This document compiles the quantitative evaluation metrics, latency benchmarks, and systematic defect resolution history across the VAY voice assistant stack.

---

## 1. Automatic Speech Recognition (ASR) Benchmark

VAY evaluates speech transcription performance across 200 validation samples per language using the Mozilla Common Voice benchmark dataset.

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

| Pipeline Stage | Implementation | Latency (P50) | Latency (P90) | Optimization Applied |
|---|---|---|---|---|
| **VAD Segmentation** | Silero VAD Streamer | ~650 ms | ~700 ms | Tail silence thresholding |
| **ASR Inference (Tier 1)** | IndicConformer CTC | ~1.10 s | ~1.65 s | Local PyTorch execution |
| **ASR Inference (Tier 2)** | Whisper Large v3 Turbo | ~320 ms | ~600 ms | Single-pass `transcribe_auto` |
| **Orchestration & NLU** | Groq `llama-3.1-8b-instant` | ~280 ms | ~450 ms | Strict JSON output format |
| **Sub-Agent Tool Loop** | Tool Invocation + Groq LLM | ~450 ms | ~850 ms | Jaccard duplicate query filter |
| **RAG Retrieval** | BM25 + ChromaDB Cosine | ~35 ms | ~65 ms | In-memory BM25 index caching |
| **TTS Time-to-First-Audio** | Edge-TTS Pipelining | **~1.08 s** | **~1.35 s** | Sentence-level pre-buffering |

---

## 3. RAG Retrieval Precision Before and After Hybrid BM25

Prior to implementing hybrid search, pure dense vector retrieval with `all-MiniLM-L6-v2` failed on exact numerical queries:

| Query Type | Dense-Only Vector Retrieval Rank | Hybrid BM25 + Vector Fusion Rank | Retrieval Accuracy Delta |
|---|---|---|---|
| Specific Plan Name (`"Prepaid Value Rs 299"`) | Top 4 - 6 (occasionally missed) | **Top 1** | +45% Precision |
| Numerical Tariff (`"Roaming Rs 5.0 per min"`) | Top 5 | **Top 1** | +60% Precision |
| SLA Duration (`"Fiber outage resolution 48h"`) | Top 3 | **Top 1** | +30% Precision |

---

## 4. Defect Audit Log & Resolution Summary

During development and systematic stress testing, the following critical defects were identified and resolved:

| Defect Class | Root Cause | Impact | Resolution |
|---|---|---|---|
| **ASR Language Lock** | Persistent `locked_language` variable in router | Second utterance forced previous language permanently | Per-utterance state reset inside `try/finally` |
| **Double ASR API Calls** | Separate LID pass + transcription call | Doubled turn latency on English (~1.8s) | Switched to `transcribe_auto()` single-pass |
| **Speaker Echo Hallucination** | Mic listening while TTS played speaker output | Ghost utterances like `"."` injected into graph | Implemented `mute()`/`unmute()` lifecycle |
| **Missing Import in Complaints** | `createComplaint` referenced undeclared `SLA_DAYS` | Crashed whenever a customer filed a complaint | Imported `SLA_DAYS` from `session.py` |
| **Missing Import in Billing** | `getBalance` referenced undeclared `_row_to_dict` | Crashed on all balance checks | Imported `_row_to_dict` from `session.py` |
| **Tool Search Looping** | LLM generating minor search variations in tool loop | Turn latency spiked to 118s (4 roundtrips) | Added `_is_near_duplicate_query()` Jaccard guard |
| **Tamil Fragment Generation** | LLM repetition detox returning partial phrases | Spoke incomplete sentence ending in comma | Added `_is_complete_reply()` terminal validator |
| **Unicode TTS Misalignment** | English voice reading Tamil Unicode codepoints | Spoke number sequences instead of Tamil words | Added `_detect_script()` Unicode router |
| **TTS Playback Delay** | Full-paragraph synthesis before audio start | Caller waited ~2.5s for audio playback | Added sentence chunking and pipelined streaming |
