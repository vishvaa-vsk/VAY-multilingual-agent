# VAY — Multilingual GenAI Voice Assistant for Customer Care

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![UV](https://img.shields.io/badge/package%20manager-uv-purple.svg)](https://github.com/astral-sh/uv)
[![Type Checked](https://img.shields.io/badge/mypy-strict-brightgreen.svg)](https://mypy.readthedocs.io/)

VAY is a GenAI voice assistant tailored for telecom self-service (bill queries, plan changes, complaints) that understands and responds across multiple languages and accents.

## Architecture Highlights

- **Tier 1 ASR (Tamil, Hindi)**: `ai4bharat/indic-conformer-600m-multilingual` loaded via `AutoModel`
- **Tier 2 ASR (English & Fallback)**: `openai/whisper-large-v3-turbo` with hallucination filtering
- **Language Detection (LID)**: Built-in Whisper encoder pass (zero extra model overhead)
- **VAD**: Utterance boundary detection via silence thresholding (~600–700ms)
- **Transcript Normalization**: LLM cleanup pass for code-switched audio (Tanglish / Hinglish)
- **Hybrid RAG**: BM25 keyword search + Vector embeddings (ChromaDB / FAISS) with confidence threshold $\tau \approx 0.75\text{--}0.85$
- **Orchestration**: LangGraph state machine with conditional branching & human handoff gate
- **TTS**: AI4Bharat IndicF5 / Indic-TTS (Tamil, Hindi) & multilingual fallback

---

## 1. Installing `uv` (Package Manager)

This project strictly uses **`uv`** as its package and virtual environment manager.

### Linux / macOS
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows (PowerShell)
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Alternative Installation Methods
```bash
# Using pip
pip install uv

# Using Homebrew (macOS)
brew install uv
```

Verify installation:
```bash
uv --version
```

---

## 2. Project Setup & Installation

Clone the repository and run `uv sync` to automatically set up the virtual environment (with Python 3.11) and install all dependencies:

```bash
# 1. Clone the repository
git clone https://github.com/vishvaa-vsk/VAY-multilingual-agent.git
cd VAY-multilingual-agent

# 2. Sync dependencies and create .venv automatically
uv sync
```

`uv` will automatically download Python 3.11 if it is not already installed on your system and create a isolated `.venv`.

---

## 3. Development Workflow & Commands

### Activate Virtual Environment (Optional)
```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```
*(Note: You do not need to manually activate the venv if you prefix commands with `uv run`.)*

### Run Type Checking (Strict `mypy`)
```bash
uv run mypy src
```

### Run Code Linting & Formatting (`ruff`)
```bash
# Check code for lint errors
uv run ruff check src tests

# Format code automatically
uv run ruff format src tests
```

### Run Unit Tests (`pytest`)
```bash
uv run pytest
```

### Run the Demo Interface
```bash
uv run python -m vay.ui.app
```

---

## 4. Project Directory Structure

```
VAY-multilingual-agent/
├── pyproject.toml            # UV project configuration and dependency specifications
├── .python-version           # Pinned Python version (3.11)
├── README.md                 # Project documentation and setup guide
├── project_context.md        # Locked source of truth context & architecture details
├── AGENTS.md                 # Repository rules (UV usage, Type safety)
├── src/                      # Source code directory
│   └── vay/                  # Core package
│       ├── py.typed          # PEP 561 type marker file
│       ├── config.py         # Application settings & models configuration
│       ├── types.py          # Strict Pydantic models & state schemas
│       ├── audio/            # Silence VAD & audio preprocessing
│       │   ├── vad.py
│       │   └── utils.py
│       ├── asr/              # ASR models (IndicConformer & Whisper) & language router
│       │   ├── base.py
│       │   ├── indic.py
│       │   ├── whisper.py
│       │   └── router.py
│       ├── normalization/    # LLM transcript cleanup & code-switch normalization
│       │   └── pass_llm.py
│       ├── rag/              # Hybrid retrieval (BM25 + ChromaDB) & confidence scorer
│       │   ├── vector_store.py
│       │   ├── bm25.py
│       │   └── retriever.py
│       ├── graph/            # LangGraph workflow orchestration & nodes
│       │   ├── state.py
│       │   ├── nodes.py
│       │   └── workflow.py
│       ├── tts/              # Text-to-speech synthesis (IndicF5 / Indic-TTS / Fallback)
│       │   └── engine.py
│       ├── handoff/          # Human escalation queue & dashboard
│       │   └── queue.py
│       └── ui/               # Gradio / Web demonstration app
│           └── app.py
└── tests/                    # Test suite
    ├── test_types.py
    ├── test_routing.py
    └── test_rag.py
```

---

## 5. License & Credits

Built for the **Velammal-AIA Partnership / Cognizant Hackathon 2026** (Use Case #15: Multilingual GenAI Voice Assistant for Customer Care).