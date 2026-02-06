# Architecture

## Overview

This system answers cybersecurity questions using only the documents in the dataset. No external knowledge, no hallucinations.

## Key Design Choices

### Dataset-Only Answers
- Everything is retrieved from the dataset documents
- If it's not in the docs, the system refuses to answer
- Every answer includes source citations

### Three-Agent Design
The system has three agents that work together through a shared tool interface:

```
┌─────────────────────────────────────────────────────────────────┐
│                    CybersecurityRAGAgent                        │
│                    (Main Orchestrator)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐        │
│  │  Indexing   │  │     QA      │  │    Grounding     │        │
│  │   Agent     │  │   Agent     │  │     Agent        │        │
│  └──────┬──────┘  └──────┬──────┘  └────────┬─────────┘        │
│         │                │                   │                  │
│         └────────────────┼───────────────────┘                  │
│                          │                                      │
│                    ┌─────┴─────┐                                │
│                    │   Tools   │                                │
│                    │ Interface │                                │
│                    └─────┬─────┘                                │
│                          │                                      │
└──────────────────────────┼──────────────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
        ┌─────┴─────┐ ┌────┴────┐ ┌─────┴─────┐
        │  ChromaDB │ │ Ollama  │ │  Dataset  │
        │  (Vector) │ │  (LLM)  │ │  (PDFs)   │
        └───────────┘ └─────────┘ └───────────┘
```

## Agent Roles

### 1. Indexing Agent
**Responsibility:** Prepare and maintain the knowledge base

**Operations:**
- Load PDF documents from `dataset/`
- Split documents into semantic chunks (1000 chars, 200 overlap)
- Generate embeddings using Ollama `qwen3-embedding`
- Store in ChromaDB with metadata (file, page, chunk index)
- Track index freshness via file hashing

**Tools Used:**
- `list_documents()` - Enumerate dataset files
- `build_index()` - Create/rebuild vector index
- `refresh_index()` - Check index status

### 2. QA Agent
**Responsibility:** Answer user questions using retrieved evidence

**Operations:**
- Accept and normalize user queries
- Retrieve top-k relevant chunks via semantic search
- Generate answers using Ollama (Gemma 3:12b)
- Attach source citations to responses

**Tools Used:**
- `vector_search(query, top_k)` - Semantic retrieval
- `get_chunk_text(chunk_id)` - Fetch specific chunks

**LLM Prompting Strategy:**
```
You are a cybersecurity expert. Answer ONLY using the provided context.
If the context doesn't contain the answer, say "I cannot answer this."
Always cite sources using [Source N] format.
```

### 3. Grounding Agent
**Responsibility:** Verify answers are grounded in evidence

**Operations:**
- Check evidence sufficiency before generation
- Verify claims in generated answers
- Force refusal when grounding is insufficient

**Tools Used:**
- `verify_grounding(answer, chunks)` - Validate claims
- `check_sufficient_evidence(chunks)` - Pre-generation check

**Refusal Criteria:**
- No relevant chunks retrieved
- Average similarity below threshold (0.3)
- Grounding confidence below 50%
- More unsupported claims than supported

## Tool Interface

All agents interact with the system exclusively through the `ToolInterface` class:

| Tool | Description | Used By |
|------|-------------|---------|
| `list_documents()` | List PDF files in dataset | Indexing |
| `build_index()` | Create/rebuild vector index | Indexing |
| `refresh_index()` | Check index status | Indexing |
| `vector_search()` | Semantic similarity search | QA |
| `get_chunk_text()` | Fetch chunk by ID | QA |
| `verify_grounding()` | Validate answer claims | Grounding |
| `check_sufficient_evidence()` | Pre-check evidence | Grounding |

## Query Flow

```
User Query
    │
    ▼
┌─────────────────────┐
│  Normalize Query    │  (QA Agent)
│  - Remove filler    │
│  - Expand abbrevs   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Vector Search      │  (Tool: vector_search)
│  - Embed query      │
│  - Find top-k       │
│  - Filter by score  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Check Sufficiency  │  (Grounding Agent)
│  - Min chunks?      │
│  - Avg similarity?  │
└──────────┬──────────┘
           │
     ┌─────┴─────┐
     │           │
  REFUSE      PROCEED
     │           │
     ▼           ▼
┌─────────┐ ┌─────────────────────┐
│ Return  │ │  Generate Answer    │  (QA Agent + Ollama)
│ Refusal │ │  - Build context    │
└─────────┘ │  - Strict prompt    │
            │  - Call LLM         │
            └──────────┬──────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │  Verify Grounding   │  (Grounding Agent)
            │  - Extract claims   │
            │  - Check overlap    │
            │  - Calculate conf   │
            └──────────┬──────────┘
                       │
                 ┌─────┴─────┐
                 │           │
              REFUSE      ACCEPT
                 │           │
                 ▼           ▼
            ┌─────────┐ ┌─────────────┐
            │ Return  │ │ Return      │
            │ Refusal │ │ Answer +    │
            └─────────┘ │ Citations   │
                        └─────────────┘
```

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Vector DB | ChromaDB | Persistent vector storage |
| Embeddings | Qwen3-embedding (Ollama) | Semantic embeddings |
| LLM | Gemma 3:12b (Ollama) | Answer generation |
| API | FastAPI + Uvicorn | Production HTTP server |
| Logging | Structlog | Structured JSON logging |
| Metrics | Prometheus | Observability |

## Configuration

Key settings in `.env`:

```bash
# LLM
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:12b

# Embeddings (Ollama)
EMBEDDING_MODEL=qwen3-embedding:0.6b

# RAG Parameters
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.3
```

## Grounding Logic

The grounding verification uses a multi-stage approach:

1. **Pre-Generation Check:**
   - Minimum 1 chunk retrieved
   - Average similarity ≥ 0.4

2. **Post-Generation Verification:**
   - Extract sentences from answer
   - Check word overlap with evidence (≥30%)
   - Calculate confidence ratio

3. **Refusal Decision:**
   - Confidence < 50% → refuse
   - More unsupported than supported claims → refuse

## Limitations & Assumptions

1. **PDF Only:** Dataset must contain PDF files
2. **English:** Optimized for English text
3. **Ollama Required:** Local LLM via Ollama
4. **No OCR:** PDFs must have extractable text
5. **Chunk Size:** Fixed 1000-char chunks may not be optimal for all content

## Future Improvements

- [ ] Hybrid search (keyword + semantic)
- [ ] Re-ranking with cross-encoder
- [ ] Multi-query expansion
- [ ] Streaming responses
- [ ] Document update detection
- [ ] Query caching
