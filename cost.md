# VAY — Cost & Deployment

Cost breakdown for the multilingual voice-agent pipeline actually implemented in this repo, plus deployment options. Figures are estimates for planning/pitching purposes, not a bill.

## 1. Pipeline & what's actually paid

```
Mic input → ASR (paid API or local model) → LLM orchestrator + sub-agent tool loop (paid API, every turn)
          → RAG retrieval (local, free) → guardrail checks (code, free) → TTS (free) → playback
```

| Stage | What runs | Cost type |
|---|---|---|
| ASR — language ID / English | Groq-hosted `whisper-large-v3-turbo` ([src/vay/asr/whisper.py](src/vay/asr/whisper.py)) | **Paid API**, always called at least once per utterance |
| ASR — 22 Indic languages (Tamil, Hindi, ...) | `ai4bharat/indic-conformer-600m-multilingual`, local via HF `transformers` ([src/vay/asr/indic.py](src/vay/asr/indic.py)) | Local compute (GPU if available, else CPU) — no API fee |
| LLM — orchestrator | Groq `ChatGroq`, model configurable via `GROQ_MODEL` ([src/vay/graph/core_utils.py:48,63](src/vay/graph/core_utils.py)) | **Paid API**, every turn |
| LLM — normalization pass | Same Groq model ([src/vay/normalization/pass_llm.py](src/vay/normalization/pass_llm.py)) | **Paid API**, every turn |
| LLM — sub-agent tool loop | Same Groq model, up to ~4 iterations, resends full system prompt each time | **Paid API**, dominant cost driver |
| RAG retrieval | ChromaDB (local `PersistentClient`) + `sentence-transformers/all-MiniLM-L6-v2` embeddings ([src/vay/rag/vector_store.py](src/vay/rag/vector_store.py)) | Local compute — no API fee |
| Guardrails (PII, identity-mismatch, human handoff) | Plain code ([src/vay/graph/nodes/orchestrator.py:244-269](src/vay/graph/nodes/orchestrator.py)) | Free |
| TTS | `edge-tts` — Microsoft Edge neural voices ([src/vay/tts/engine.py:28](src/vay/tts/engine.py)) | Free (external, not self-hosted) |
| Mock CRM backend | SQLite | Free |

The only two metered dependencies are **Groq LLM calls** and **Groq Whisper ASR calls**. Everything else (RAG, TTS, guardrails, backend) is free or local compute.

## 2. Groq pricing (as of Aug 2026)

The repo supports two LLM choices via `GROQ_MODEL` ([SETUP.md:35](SETUP.md)):

| Model | Input | Output | Notes |
|---|---|---|---|
| `llama-3.1-8b-instant` (default, [core_utils.py:63](src/vay/graph/core_utils.py)) | $0.05 / 1M tok | $0.08 / 1M tok | Batch API: $0.025 / $0.04 |
| `openai/gpt-oss-20b` (recommended in SETUP.md) | $0.075 / 1M tok | $0.30 / 1M tok | Cached input: $0.0375 / 1M tok |
| `whisper-large-v3-turbo` (ASR) | $0.04 / **audio hour** | — | Billed by audio duration, not tokens |

Sources: [cloudzero.com/blog/groq-pricing](https://www.cloudzero.com/blog/groq-pricing/), [helicone.ai — llama-3.1-8b-instant](https://www.helicone.ai/llm-cost/provider/groq/model/llama-3.1-8b-instant), [aipricing.guru/groq-pricing](https://www.aipricing.guru/groq-pricing/).

edge-tts, ChromaDB, MiniLM embeddings, and SQLite carry **no API fee**. If self-hosted at scale, the local ASR model and embeddings do have real GPU/CPU infra cost — the repo has no deployment doc that sizes this, so treat it as a real but unquantified line item, not zero.

## 3. Per-conversation cost model

Unlike a generic "300 in / 200 out" estimate, this project has **fixed system-prompt overhead on every turn**, per the team's own token audit ([rag-tts-evaluvation.md §6](rag-tts-evaluvation.md)):

- Orchestrator system prompt: ~1,185 tokens
- Sub-agent system prompt template: ~1,735 tokens, **resent on every tool-loop iteration** (up to ~4 iterations for a complex query)
- Plus conversation history, tool schemas, RAG context, and the normalization pass — all additional LLM calls

**Worked estimate per conversational turn** (1 orchestrator call + 1 normalization call + 1–2 sub-agent iterations, ~10 sec of audio):

| Scenario | Model | Est. input tok | Est. output tok | LLM cost | Whisper (10s) | Total |
|---|---|---|---|---|---|---|
| Simple turn (1 sub-agent iteration) | `llama-3.1-8b-instant` | ~3,200 | ~300 | $0.00019 | $0.00011 | **≈ $0.0003** (₹0.026) |
| Simple turn (1 sub-agent iteration) | `gpt-oss-20b` | ~3,200 | ~300 | $0.00033 | $0.00011 | **≈ $0.0004** (₹0.038) |
| Complex turn (4 sub-agent iterations) | `llama-3.1-8b-instant` | ~11,000 | ~900 | $0.00062 | $0.00011 | **≈ $0.0007** (₹0.065) |
| Complex turn (4 sub-agent iterations) | `gpt-oss-20b` | ~11,000 | ~900 | $0.00110 | $0.00011 | **≈ $0.0012** (₹0.112) |

(INR at ₹87.9/USD — see note below on FX sensitivity.) These are meaningfully higher than the flat "300/200 token" estimate because of the two large fixed system prompts, but still sub-paisa-to-few-paise per turn.

## 4. Cost at scale

Using the "simple turn" figures as a representative floor and "complex turn" as a ceiling, both on `llama-3.1-8b-instant` (the repo's default):

| Conversations | Low (simple turns) | High (complex turns) |
|---|---|---|
| 1 | $0.0003 (₹0.03) | $0.0007 (₹0.07) |
| 100 | $0.03 (₹2.6) | $0.07 (₹6.5) |
| 1,000 | $0.30 (₹26) | $0.70 (₹65) |
| 10,000 | $3.00 (₹263) | $7.00 (₹650) |

On `gpt-oss-20b`, roughly 1.5–2x these figures.

**FX sensitivity note:** the original back-of-envelope math swung from ₹0.017 to ₹0.114 per conversation purely from changing the USD/INR rate (₹87 → ₹95.78) and the token-count assumption. At this scale (sub-cent per turn) the exchange rate and system-prompt overhead matter more than the headline per-token price — state the FX rate used and keep it easy to update rather than presenting one precise number.

## 5. Known real-world cost risk (not hypothetical)

Two issues actually occurred during development, per [rag-tts-evaluvation.md:422-450](rag-tts-evaluvation.md) and [context-rag-tts.md:806](context-rag-tts.md):

- **Daily quota exhaustion**: testing hit Groq's 200,000 tokens/day limit on `gpt-oss-20b` mid-evaluation (`Rate limit reached ... Limit 200000, Used 197394 ... tokens per day`). The team switched to `llama-3.1-8b-instant` to keep working — a real quality/cost tradeoff (more repetition/hallucination risk on the cheaper model), not a free win.
- **Token multiplier bug**: a repeated-tool-call bug measured ~6x the expected token usage on affected turns before it was fixed.

Takeaway for judges: per-call cost is genuinely low, but **rate limits and bugs are the real operational cost risk** at demo/production scale — worth a paid tier or multi-key failover (the repo already has failover logic across `GROQ_API_KEY_2`, `_3`, ... — [core_utils.py:455-547](src/vay/graph/core_utils.py)).

## 6. Deployment options

**Nothing is deployed yet** — there is no Dockerfile, docker-compose, or cloud config in the repo. It currently runs as a local Streamlit app (`app.py`), launched via `uv run python scripts/setup_app.py` on Python 3.11 + `uv`. GPU is optional and only speeds up the local IndicConformer ASR model; the Groq API path (Whisper + LLM) needs no GPU.

Realistic options for a hackathon/demo deployment:

1. **Streamlit Community Cloud (free tier) + Groq API keys** — fastest path, zero infra cost, CPU-only (Indic ASR falls back to CPU, fine for a demo). Good for judging/demo day.
2. **Small cloud VM (e.g. 2 vCPU / 4GB, ~$5–10/mo)**, containerized with a Dockerfile (not yet present — would need to be added), running Streamlit behind a reverse proxy. Adds control over uptime/secrets but has a small fixed hosting cost on top of the per-call Groq usage above.
3. **GPU VM**, only justified if the local Indic ASR tier needs to serve many concurrent users with low latency — otherwise the CPU/API-only path (option 1 or 2) is cheaper and sufficient, since the paid Groq Whisper tier already handles English and can be the fallback for all languages.

## Sources

- [Groq Pricing 2026 — CloudZero](https://www.cloudzero.com/blog/groq-pricing/)
- [Groq llama-3.1-8b-instant Pricing — Helicone](https://www.helicone.ai/llm-cost/provider/groq/model/llama-3.1-8b-instant)
- [Groq API Pricing — AI Pricing Guru](https://www.aipricing.guru/groq-pricing/)
- In-repo: [rag-tts-evaluvation.md](rag-tts-evaluvation.md), [context-rag-tts.md](context-rag-tts.md), [project_context.md](project_context.md), [SETUP.md](SETUP.md)
