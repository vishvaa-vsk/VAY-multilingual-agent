# Multilingual GenAI Voice Assistant

A multilingual GenAI voice assistant for customer care with RAG-based knowledge retrieval, customer database integration, text-to-speech, and intelligent agent handoff.

## Setup

### 1. Install uv

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify:

```powershell
uv --version
```

### 2. Clone & Install

```powershell
git clone <YOUR_REPOSITORY_URL>
cd <REPOSITORY_FOLDER>
uv sync
```

### 3. Configure API Key

Create `.env` in the project root:

```env
GROQ_API_KEY="your_groq_api_key"
GROQ_MODEL=openai/gpt-oss-20b
or llama-3.1-8b-instant
```

### 4. Setup and Run

Run the master setup script, which automatically initializes the database, builds the knowledge base, caches the required ASR models, and launches the app!

```powershell
uv run python scripts/setup_app.py
```

Debug mode:

```powershell
uv run python scripts/run_assistant.py --show_debug
```

## Testing

```powershell
uv run pytest tests/ -v
```

## Code Quality

```powershell
uv run ruff check src tests
uv run mypy src
```

## Generated Local Files

The following are local and should **not be committed to Git**:

```text
.env
.venv/
chroma_db/
src/vay/tools/nexatel_customers.db
handoff_log.jsonl
*.mp3
*.wav
*.ogg
```

## Quick Start

```powershell
uv sync
# Create .env and add GROQ_API_KEY
uv run python scripts/manage_db.py
uv run python scripts/build_kb.py
uv run python scripts/run_assistant.py --phone 9876543210 --language en
```
