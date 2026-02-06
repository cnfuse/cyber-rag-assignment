██████╗  █████╗ ████████╗ █████╗ ███████╗ █████╗ ██████╗ ███╗   ███╗
██╔══ █╗██╔══██╗╚══██╔══╝██╔══██╗██╔════╝██╔══██╗██╔══██╗████╗ ████║
██╔═══█╔███████║   ██║   ███████║███████╗███████║██████╔╝██╔████╔██║
██╔═══█╗██╔══██║   ██║   ██╔══██║██╔════╝██╔══██║██══██═╗██║╚██╔╝██║
██████╔╝██║  ██║   ██║   ██║  ██║██═════║██║  ██║██║  ██║██║ ╚═╝ ██║
╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝


====================================================================
# Reference Notes for This Assignment

This document summarizes the three files included in the `dataset/` folder and provides simple definitions that will help you understand their purpose in the RAG project.  
Your system must use these documents as its only knowledge source.

---

## 1. OWASP Top 10 (2021)
**Type:** Web Application Security Standard  
**Purpose:** Defines the most critical security risks found in modern web applications.  
**What it contains:**
- Ten major categories of vulnerabilities (e.g., Injection, Broken Access Control)
- Real-world examples of how these vulnerabilities occur
- Recommended prevention and mitigation practices

**How it will be used in RAG:**
- Explaining web vulnerabilities  
- Identifying impacts and risks  
- Providing mitigation guidance based on OWASP recommendations  

---

## 2. Thailand Web Security Standard (2025)
**Type:** National Security Standard (Thailand)  
**Purpose:** Provides security requirements, controls, and best practices for securing public-facing websites in Thailand.  
**What it contains:**
- Security controls for website configuration, authentication, logging, monitoring
- Requirements for preventing common attacks
- Compliance-oriented guidance and operational security expectations

**How it will be used in RAG:**
- Explaining official security requirements for websites  
- Answering compliance- and control-related questions  
- Clarifying expected configurations and risk-reduction measures  

---

## 3. MITRE ATT&CK – Design & Philosophy (2020)
**Type:** Threat Behavior & Attack Model  
**Purpose:** Explains the philosophy behind MITRE ATT&CK and how adversary behaviors are structured into Tactics and Techniques.  
**What it contains:**
- Definitions of Tactics (“why attackers act”) and Techniques (“how they act”)
- How adversary behaviors are modeled in ATT&CK Matrix
- Examples of how ATT&CK supports threat intelligence, detection engineering, and incident response

**How it will be used in RAG:**
- Understanding attacker behavior  
- Distinguishing Tactics vs Techniques  
- Mapping defense or detection recommendations to ATT&CK categories  

---

## Glossary (Simple Definitions)

**RAG (Retrieval-Augmented Generation)**  
A system that retrieves relevant text before generating an answer using an LLM.

**Embedding**  
A numerical representation of text used for similarity search.

**Vector Database**  
A storage engine for embeddings that supports fast nearest-neighbor search.

**Tactic (MITRE)**  
The attacker’s goal (e.g., Persistence, Privilege Escalation).

**Technique (MITRE)**  
The specific method used to achieve a Tactic.

**Vulnerability (OWASP)**  
A weakness that can be exploited to compromise a system.

**Ransomware / Malware**  
Not included as a dataset document, but referenced in multiple standards as a risk category.

---

## Sample Questions You May Use to Test Your System

1. What is Broken Access Control according to OWASP?  
2. What website security controls are required by the Thailand Web Security Standard?  
3. What is the difference between a Tactic and a Technique in MITRE ATT&CK?  
4. What mitigation steps does OWASP recommend for Injection vulnerabilities?  
5. How does MITRE describe the purpose of Persistence techniques?  

These questions are provided only as guidance.  
You may create your own questions for evaluation.

---

## Important Note
Your RAG system must use **only** the documents in `dataset/`.  
External content (e.g., Google search, unsupported LLM knowledge) should not be included in the final answers unless explicitly retrieved from these files.

