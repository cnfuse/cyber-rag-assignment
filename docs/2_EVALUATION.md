██████╗  █████╗ ████████╗ █████╗ ███████╗ █████╗ ██████╗ ███╗   ███╗
██╔══ █╗██╔══██╗╚══██╔══╝██╔══██╗██╔════╝██╔══██╗██╔══██╗████╗ ████║
██╔═══█╔███████║   ██║   ███████║███████╗███████║██████╔╝██╔████╔██║
██╔═══█╗██╔══██║   ██║   ██╔══██║██╔════╝██╔══██║██══██═╗██║╚██╔╝██║
██████╔╝██║  ██║   ██║   ██║  ██║██║     ██║  ██║██║  ██║██║ ╚═╝ ██║
╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝

====================================================================
# Evaluation Criteria

Your submission will be evaluated using the criteria below.  
The scoring reflects clarity, correctness, **agent behavior**, system design quality, and strict adherence to the dataset-only requirement.

---

## 1. Answer Grounding & Dataset Compliance (35%)
- All answers are strictly grounded in retrieved content from the provided dataset.  
- Citations clearly reference specific dataset files, pages, or chunks.  
- The agent demonstrates **closed-book behavior** (no external knowledge or assumptions).  
- Hallucinations and general cybersecurity knowledge not present in the dataset are avoided.  
- When the dataset is insufficient, the agent correctly refuses to answer instead of guessing.

This is the **most important criterion**.

---

## 2. Agent Design & System Architecture (25%)
- The system is implemented as an **agent**, not just a linear RAG pipeline.  
- Agent roles (e.g. indexing, question answering, grounding/verification) are clearly defined.  
- The agent uses explicit tools (retrieval, indexing, verification) to perform actions.  
- The overall architecture is logical, technically sound, and well justified.  
- Reasoning behind design decisions and guardrails is clearly articulated.

---

## 3. Code Quality & Maintainability (20%)
- Code is clean, readable, and modular.  
- Clear separation between agent logic, tools, and data handling.  
- Functions and components have clear responsibilities.  
- No unnecessary complexity or over-engineering.

---

## 4. Communication & Documentation (15%)
- Architecture and agent behavior are clearly explained and easy to understand.  
- System diagram accurately reflects the agent-based design and data flow.  
- README provides clear instructions for setup, execution, and evaluation.  
- Assumptions and limitations are explicitly documented.

---

## 5. Bonus Points (Up to +10%)
Bonus points will be awarded for enhancements that demonstrate initiative and deeper technical capability, such as:
- Using a **fine-tuned or specialized LLM** (Qwen, Gemma, Llama, etc.)  
- Implementing **PEFT techniques** (LoRA / QLoRA)  
- Applying **advanced retrieval methods** (e.g. hybrid search, re-ranking)  
- Running models using **self-hosted inference frameworks** (vLLM, llama.cpp)

All bonus enhancements must:
- Preserve strict dataset grounding  
- Integrate cleanly with the agent-based RAG design  
- Not compromise answer correctness or verification

---

## Total Score
**100 points + 10 bonus points**
