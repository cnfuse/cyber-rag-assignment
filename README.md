# Cybersecurity RAG Agent

A RAG (Retrieval-Augmented Generation) system for cybersecurity Q&A that answers questions strictly from the provided dataset. Uses Ollama for local LLM inference.

## Features

- Answers only from dataset documents (no hallucination)
- Three-agent architecture: Indexing, QA, and Grounding
- Refuses to answer when information isn't in the dataset
- Full source citations with every answer
- Thai PDF support via Typhoon OCR API
- FastAPI server with health monitoring

## Prerequisites

- Python 3.10 or higher
- Ollama with Gemma model (https://ollama.ai)
  - Windows users: `winget install Ollama`
- Typhoon API Key (optional, only for Thai PDF support)
  - Get one at: https://opentyphoon.ai

## Quick Start

### 1. Install Ollama & Model

```bash
ollama pull llama3.1:8b    # Requires ~8GB VRAM
```

### 2. Install Dependencies

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure (Optional)

```powershell
copy .env.example .env
# Edit .env if using different model or Ollama URL
```

### 4. Verify Setup

```powershell
python verify_setup.py
```

### 5. Run Evaluation Tests (Required for Assignment)

```powershell
# Install the new Gemini package
pip install google-genai

# Set Gemini API key for LLM-as-a-judge evaluation (optional but recommended)
$env:GEMINI_API_KEY="your-gemini-api-key"

# Run evaluation
python run_evaluation.py
```

This will:
- Run all required test cases (covers all 3 dataset documents)
- Evaluate answers using Google Gemini 2.0 Flash as an independent judge
- Save results to `evaluation_results.json`

**Note:** LLM-as-a-judge is optional. Tests will run without it, but judgment scores won't be available.
**Get API key:** https://aistudio.google.com/apikey

### 6. Start the Server

```powershell
python -m src.api
```

### 7. Ask Questions

```powershell
curl -X POST http://localhost:8000/ask `
  -H "Content-Type: application/json" `
  -d '{"query": "What is Broken Access Control?"}'
```

Or open http://localhost:8000/docs for interactive API docs.

---

## Project Structure

```
cyber-rag-assignment/
├── dataset/                          # Source documents (DO NOT MODIFY)
│   ├── owasp-top-10.pdf
│   ├── thailand-web-security-standard-2025.pdf
│   └── mitre-attack-philosophy-2020.pdf
├── src/
│   ├── __init__.py
│   ├── config.py                     # Configuration settings
│   ├── tools.py                      # Explicit tool interface
│   ├── agent.py                      # Three specialized agents
│   ├── api.py                        # FastAPI production server
│   ├── typhoon_ocr_integration.py    # Thai PDF OCR integration
│   └── inspect_db.py                 # Database inspection utility
├── chroma_db/                        # Vector database (auto-created)
├── requirements.txt                  # Python dependencies
├── verify_setup.py                   # Setup verification script
├── .env.example                      # Configuration template
├── QUICKSTART.md                     # Quick start guide
├── ARCHITECTURE_EXPLANATION.md       # Detailed architecture
├── EVALUATION_EXAMPLES.md            # Test results
├── SYSTEM_ARCHITECTURE_C4.puml       # C4 diagram
└── README.md                         # This file
```

---

## Architecture

![System Architecture](architecture.png)

### Three Agents

| Agent | What it does |
|-------|-------------|
| Indexing | Loads PDFs, chunks them, generates embeddings, stores in ChromaDB |
| QA | Takes queries, retrieves relevant chunks, generates answers |
| Grounding | Verifies answer claims against evidence, refuses if insufficient |

### Tool Interface

All agents communicate through explicit tools:

| Tool | Description |
|------|-------------|
| `list_documents()` | List PDF files in dataset/ |
| `build_index()` | Create/rebuild vector index |
| `vector_search(query, top_k)` | Semantic similarity search |
| `get_chunk_text(chunk_id)` | Fetch specific chunk |
| `verify_grounding(answer, chunks)` | Validate answer claims |

### Technology Stack

| Component | Technology |
|-----------|------------|
| Vector DB | ChromaDB (SQLite-backed) |
| Embeddings | Qwen3-embedding via Ollama |
| LLM | Gemma 3 via Ollama |
| OCR | Typhoon OCR API (Thai PDFs) |
| API | FastAPI + Uvicorn |
| Logging | Structlog (JSON) |
| Metrics | Prometheus |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ask` | POST | Ask a cybersecurity question |
| `/health` | GET | Health check endpoint |
| `/docs` | GET | Interactive Swagger UI |

---

## Configuration

Environment variables (`.env`):

```bash
# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# Embeddings (Ollama)
EMBEDDING_MODEL=qwen3-embedding:0.6b

# Typhoon OCR (Optional - for Thai PDFs)
TYPHOON_API_KEY=your-typhoon-api-key
TYPHOON_BASE_URL=https://api.opentyphoon.ai/v1
ENABLE_TYPHOON_OCR=true

# RAG Parameters
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.3

# Server
API_HOST=0.0.0.0
API_PORT=8000
```

---

## Refusal Logic

The system refuses to answer when:

1. No documents match the query
2. Similarity score too low (< 0.3)
3. Answer confidence below 50%
4. Too many unsupported claims

Example:
```json
{
  "status": "refused",
  "reason": "Cannot answer from dataset: The dataset does not contain information about ransomware.",
  "sources": []
}
```

---

## Usage Examples

### Python Client

```python
import requests

# Ask a question
response = requests.post(
    "http://localhost:8000/ask",
    json={"query": "What is the difference between Tactics and Techniques in MITRE ATT&CK?"}
)

result = response.json()
if result["status"] == "answered":
    print(result["answer"])
    for source in result["sources"]:
        print(f"  - {source['file']}, Page {source['page']}")
else:
    print(f"Refused: {result['reason']}")
```

### Direct Agent Usage

```python
from src.agent import CybersecurityRAGAgent

agent = CybersecurityRAGAgent()
agent.initialize()

response = agent.query("What security controls does Thailand require?")
print(response.answer)
```

---

## Assumptions & Limitations

**Assumptions:**
- PDFs in dataset have extractable text (not scanned images without OCR)
- Ollama is running locally with required models
- Questions are in English (or Thai for documents with Typhoon OCR)
- Dataset documents remain static during runtime

**Limitations:**
- Fixed chunk size (1000 chars) may not be optimal for all content types
- Semantic search only (no keyword/hybrid search)
- No query history or conversation context
- Single-turn Q&A only (no follow-up questions)
- Similarity threshold (0.3) is fixed, not adaptive

---

## Documentation

- [QUICKSTART.md](QUICKSTART.md) - Step-by-step setup guide
- [ARCHITECTURE_EXPLANATION.md](ARCHITECTURE_EXPLANATION.md) - Detailed system design
- [EVALUATION_EXAMPLES.md](EVALUATION_EXAMPLES.md) - Test questions and results
- [SYSTEM_ARCHITECTURE_C4.puml](SYSTEM_ARCHITECTURE_C4.puml) - C4 architecture diagram
- [run_evaluation.py](run_evaluation.py) - Automated evaluation script

---

## Assignment Deliverables Checklist

✅ **1. Working Agent Prototype**
- FastAPI server: [src/api.py](src/api.py)
- Agent implementation: [src/agent.py](src/agent.py)
- Tool interface: [src/tools.py](src/tools.py)

✅ **2. Architecture Explanation**
- [ARCHITECTURE_EXPLANATION.md](ARCHITECTURE_EXPLANATION.md)
- Covers agent roles, tool usage, and grounding logic

✅ **3. System Diagram**
- [architecture.png](architecture.png)
- [SYSTEM_ARCHITECTURE_C4.puml](SYSTEM_ARCHITECTURE_C4.puml)

✅ **4. Evaluation Examples**
- [EVALUATION_EXAMPLES.md](EVALUATION_EXAMPLES.md)
- 5 answerable questions with citations
- 2 refusal examples (insufficient data)
- Run tests: `python run_evaluation.py`

✅ **5. Source Code**
- Clean, modular structure
- Clear separation of concerns
- All required tools implemented

✅ **6. Required Tools (Assignment Specification)**
- `list_documents()` - List dataset files
- `build_index()` - Create vector index
- `refresh_index()` - Check index status
- `vector_search(query, top_k)` - Semantic search
- `get_chunk_text(chunk_id)` - Fetch chunk text
- `verify_grounding(answer, chunks)` - Validate claims

---

## Troubleshooting

### Ollama not running
```powershell
ollama serve
```

### Model not found
```powershell
ollama pull llama3.1:8b
```

### Index is empty
```powershell
curl -X POST http://localhost:8000/index -H "Content-Type: application/json" -d '{"force_rebuild": true}'
```

### Port already in use
Edit `.env` and change `API_PORT=8001`

---

## License

Educational project for Cybersecurity RAG Assignment.
