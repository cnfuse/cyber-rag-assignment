# Quick Start Guide

Get the Cybersecurity RAG Agent running in 5 minutes.

## Prerequisites

- **Python 3.10+** installed
- **Ollama** running with Gemma model
- The `dataset/` folder with PDF documents

---

## Step 1: Verify Ollama is Running

Make sure your Ollama server is running and has the required model:

```powershell
# Check Ollama status
ollama list

# If you need the model, pull it:
ollama pull gemma3:12b

```

If using a different model, update `.env`:
```bash
OLLAMA_MODEL=gemma3:2b
```

---

## Step 2: Install Python Dependencies

```powershell
# Create virtual environment
python -m venv venv

# Activate it (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

## Step 3: Create Configuration

Copy the example configuration:

```powershell
copy .env.example .env
```

Edit `.env` if needed (defaults should work):
```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:12b
```

---

## Step 4: Verify Setup

Run the verification script:

```powershell
python verify_setup.py
```

Expected output:
```
============================================================
  Cybersecurity RAG Agent - Setup Verification
============================================================

🔍 Checking Dependencies...
  ✅ FastAPI
  ✅ Uvicorn
  ✅ ChromaDB
  ...

🔍 Checking Dataset...
  ✅ Found 3 documents:
     • owasp-top-10.pdf (XXX KB)
     • thailand-web-security-standard-2025.pdf (XXX KB)
     • mitre-attack-philosophy-2020.pdf (XXX KB)

🔍 Checking Ollama...
  ✅ Ollama is running
  ✅ Model 'gemma3:12b' is available

============================================================
🎉 All checks passed! Ready to start the server.
```

---

## Step 5: Start the Server

```powershell
python -m src.api
```

You should see:
```
╔══════════════════════════════════════════════════════════════╗
║        Cybersecurity RAG Agent - Production Server           ║
╠══════════════════════════════════════════════════════════════╣
║  • API Docs:    http://0.0.0.0:8000/docs                     ║
║  • Health:      http://0.0.0.0:8000/health                   ║
║  • Metrics:     http://0.0.0.0:8000/metrics                  ║
╚══════════════════════════════════════════════════════════════╝

INFO:     Building index from 3 documents...
INFO:     Index built with 156 chunks
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## Step 6: Test the API

### Option A: Use the Interactive Docs

Open http://localhost:8000/docs in your browser.

### Option B: Use cURL

```powershell
# Health check
curl http://localhost:8000/health

# Ask a question
curl -X POST http://localhost:8000/ask `
  -H "Content-Type: application/json" `
  -d '{"query": "What is Broken Access Control according to OWASP?"}'
```

### Option C: Use Python

```python
import requests

response = requests.post(
    "http://localhost:8000/ask",
    json={"query": "What is Broken Access Control?"}
)

data = response.json()
print(f"Status: {data['status']}")
print(f"Answer: {data['answer']}")
print(f"Sources: {data['sources']}")
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ask` | POST | Ask a question |
| `/index` | POST | Build/rebuild index |
| `/index/status` | GET | Get index status |
| `/documents` | GET | List dataset documents |
| `/health` | GET | Health check |
| `/ready` | GET | Readiness probe |
| `/metrics` | GET | Prometheus metrics |
| `/docs` | GET | Interactive API docs |

---

## Troubleshooting

### "Ollama connection refused"
```powershell
# Start Ollama
ollama serve
```

### "Model not found"
```powershell
ollama pull gemma3:12b
```

### "No documents found"
Make sure the `dataset/` folder exists and contains the PDF files.

### "Index is empty"
```powershell
# Force rebuild the index
curl -X POST http://localhost:8000/index `
  -H "Content-Type: application/json" `
  -d '{"force_rebuild": true}'
```

---

## Next Steps

- Read [ARCHITECTURE_EXPLANATION.md](ARCHITECTURE_EXPLANATION.md) for system design details
- See [EVALUATION_EXAMPLES.md](EVALUATION_EXAMPLES.md) for test questions and expected results
- Check the [README.md](README.md) for comprehensive documentation
