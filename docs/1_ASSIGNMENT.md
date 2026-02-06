██████╗  █████╗ ████████╗ █████╗ ███████╗ █████╗ ██████╗ ███╗   ███╗
██╔══ █╗██╔══██╗╚══██╔══╝██╔══██╗██╔════╝██╔══██╗██╔══██╗████╗ ████║
██╔═══█╔███████║   ██║   ███████║███████╗███████║██████╔╝██╔████╔██║
██╔═══█╗██╔══██║   ██║   ██╔══██║██╔════╝██╔══██║██══██═╗██║╚██╔╝██║
██████╔╝██║  ██║   ██║   ██║  ██║██║     ██║  ██║██║  ██║██║ ╚═╝ ██║
╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝


====================================================================
# Cybersecurity RAG Assistant – Assignment

- You are required to build a **Retrieval-Augmented Generation (RAG) Agent** that can answer cybersecurity-related questions **using only the documents provided in the `dataset/` folder**.
- This requirement is strict and non-negotiable.
- Your system must not rely on external knowledge, internet data, or LLM hallucinations.
- All answers must be grounded strictly in the dataset.
- The system must behave as a **closed-book agent** with explicit retrieval and grounding steps.

---

## Project Objective
Develop a **RAG-based Agent** that:
1. Loads and indexes documents from `dataset/`
2. Splits documents into chunks with traceable metadata
3. Creates embeddings and stores them in a vector database
4. Retrieves the most relevant chunks for a user query
5. Uses an LLM to generate answers **only from retrieved content**
6. Produces answers with **clear citations referencing dataset files**
7. Detects and handles cases where the dataset is insufficient to answer the query

---

## Agent-Based Design Requirement
Your system must be implemented as an **agent**, not just a linear pipeline.

The agent should:
- Maintain internal state (e.g. index status, document manifest)
- Use explicit tools to perform actions
- Follow strict policies that prevent answering beyond retrieved evidence
- Decide when it can answer and when it must refuse due to missing information

---

## Agent Roles and Responsibilities

### 1. Indexing Agent
Responsible for preparing and maintaining the knowledge base:
- Load all files from `dataset/`
- Chunk documents and attach metadata (file name, chunk ID, page/line if available)
- Generate embeddings and build/update the vector database
- Track index version and freshness

### 2. Question Answering (QA) Agent
Responsible for answering user questions:
- Accept user queries
- Normalize or rewrite queries for retrieval (without adding new knowledge)
- Retrieve top-k relevant chunks from the vector database
- Generate answers strictly from retrieved chunks
- Attach citations mapping answers to dataset files

### 3. Grounding & Verification Agent
Responsible for enforcing dataset-only answers:
- Verify that each claim in the answer is supported by retrieved chunks
- Ensure citations directly correspond to the supporting text
- Prevent hallucinated or general cybersecurity knowledge
- If evidence is insufficient, force a “cannot answer from dataset” response

---

## Tooling Interface (Required)
The agent must interact with the system exclusively through tools, such as:
- `list_documents()` – list files in `dataset/`
- `build_index()` / `refresh_index()` – create or update the vector index
- `vector_search(query, top_k)` – retrieve relevant chunks with metadata
- `get_chunk_text(chunk_id)` – fetch original text for citation verification
- `verify_grounding(answer, chunks)` – ensure all claims are supported

---

## Answer Policy (Strict)
- Answers must be generated **only** from retrieved dataset content
- External knowledge and general cybersecurity concepts are forbidden
- Every answer must include citations to dataset files
- If the dataset does not contain sufficient information:
  - The agent must explicitly state that it cannot answer
  - The agent must not guess or extrapolate

---

## System Flow (Agent Perspective)
Indexing phase:
load dataset → chunk → embed → store in vector DB

Runtime question answering:
user query → QA agent → vector retrieval → grounding verification → answer with citations  
or  
user query → insufficient evidence → refusal with explanation

---

## Deliverables
You must submit:
1. **A working agent prototype**
   (Notebook, script, or simple API)

2. **A brief architecture explanation (0.5–1 page)**
   - Agent roles
   - Tool usage
   - Grounding and refusal logic

3. **A simple system diagram**
   - Indexing Agent → Vector DB
   - QA Agent → Retrieval → Verification → Answer

4. **Evaluation examples**
   - 3–5 test questions
   - Retrieved evidence
   - Final answers with citations
   - At least one example where the agent refuses due to missing data

5. **Source code**
   - Clean, readable, well organized
   - Clear separation between agent logic, tools, and data handling

---

## Core Requirement (Important)
Your agent must:
- Use **only** the documents in `dataset/`
- Generate answers strictly based on retrieved content
- Avoid introducing information not present in the dataset
- Clearly cite which dataset file(s) support each answer
- Correctly handle unanswerable questions

---

## Bonus Consideration (Optional)
You may experiment with **fine-tuned or specialized models**, such as:
- Qwen fine-tuned variants
- Gemma or Gemma-based fine-tunes
- Llama fine-tuned models (LoRA / QLoRA / PEFT)
- Optimized or custom inference pipelines

Bonus credit will be awarded if you demonstrate:
- Clear justification for model choice
- Improved grounding or factual accuracy
- Clean integration into the agent-based RAG design

This is optional. Do not over-engineer.

---

## Submission Guidelines
- Submit your work as a ZIP file or GitHub repository
- Include a README with setup and execution instructions
- You have **7 days** to complete the assignment

If anything is unclear, make reasonable assumptions and document them.
