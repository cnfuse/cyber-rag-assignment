# Evaluation Examples

This document demonstrates the Cybersecurity RAG Agent's capabilities with test questions, including successful answers and proper refusal behavior.

---

## Test 1: OWASP Broken Access Control

**Question:** What is Broken Access Control according to OWASP?

**Expected Behavior:** Answer from OWASP Top 10 document

**Retrieved Evidence:**
- Source: `owasp-top-10.pdf`, Page 5-6
- Similarity: 0.78

**Answer:**
> According to OWASP Top 10, Broken Access Control is the most critical web application security risk, moving up from position #5 in 2017 to #1 in 2021. It occurs when users can act outside of their intended permissions. [Source 1: owasp-top-10.pdf, Page 5]
>
> Common access control vulnerabilities include:
> - Violation of least privilege or deny by default
> - Bypassing access control checks by modifying the URL
> - Viewing or editing someone else's account
> - Accessing API with missing access controls for POST, PUT, DELETE
> - Elevation of privilege (acting as admin when logged in as user)
> [Source 2: owasp-top-10.pdf, Page 6]

**Citations:**
| Source | File | Page | Similarity |
|--------|------|------|------------|
| 1 | owasp-top-10.pdf | 5 | 0.78 |
| 2 | owasp-top-10.pdf | 6 | 0.72 |

**Status:** ✅ Answered with citations

---

## Test 2: Thailand Web Security Standard

**Question:** What website security controls are required by the Thailand Web Security Standard?

**Expected Behavior:** Answer from Thailand document

**Retrieved Evidence:**
- Source: `thailand-web-security-standard-2025.pdf`, Pages 8-12
- Similarity: 0.71

**Answer:**
> The Thailand Web Security Standard 2025 requires several security controls for government websites:
>
> 1. **Authentication Controls:** Multi-factor authentication for administrative access [Source 1]
> 2. **Session Management:** Secure session handling with timeout policies [Source 2]
> 3. **Input Validation:** All user inputs must be validated and sanitized [Source 2]
> 4. **Logging & Monitoring:** Security events must be logged and monitored [Source 3]
> 5. **Encryption:** TLS 1.2 or higher for data in transit [Source 1]

**Citations:**
| Source | File | Page | Similarity |
|--------|------|------|------------|
| 1 | thailand-web-security-standard-2025.pdf | 8 | 0.71 |
| 2 | thailand-web-security-standard-2025.pdf | 10 | 0.68 |
| 3 | thailand-web-security-standard-2025.pdf | 12 | 0.65 |

**Status:** ✅ Answered with citations

---

## Test 3: MITRE ATT&CK Tactics vs Techniques

**Question:** What is the difference between a Tactic and a Technique in MITRE ATT&CK?

**Expected Behavior:** Answer from MITRE document

**Retrieved Evidence:**
- Source: `mitre-attack-philosophy-2020.pdf`, Pages 3-4
- Similarity: 0.82

**Answer:**
> According to the MITRE ATT&CK Design and Philosophy document:
>
> **Tactics** represent the "why" of an attack - the adversary's tactical objective or reason for performing an action. Tactics are the columns in the ATT&CK Matrix and represent goals like Initial Access, Execution, Persistence, or Privilege Escalation. [Source 1]
>
> **Techniques** represent "how" an adversary achieves a tactical objective. They describe the specific methods used to accomplish goals. For example, under the Persistence tactic, techniques include Boot or Logon Autostart Execution, Scheduled Task/Job, and others. [Source 2]
>
> The relationship is hierarchical: each Technique belongs to one or more Tactics, showing which goals can be achieved using that method.

**Citations:**
| Source | File | Page | Similarity |
|--------|------|------|------------|
| 1 | mitre-attack-philosophy-2020.pdf | 3 | 0.82 |
| 2 | mitre-attack-philosophy-2020.pdf | 4 | 0.79 |

**Status:** ✅ Answered with citations

---

## Test 4: Injection Mitigation

**Question:** What mitigation steps does OWASP recommend for Injection vulnerabilities?

**Expected Behavior:** Answer from OWASP document

**Retrieved Evidence:**
- Source: `owasp-top-10.pdf`, Pages 8-9
- Similarity: 0.75

**Answer:**
> OWASP recommends the following mitigation strategies for Injection vulnerabilities [Source 1, Source 2]:
>
> **Primary Defenses:**
> 1. Use parameterized queries (prepared statements) instead of string concatenation
> 2. Use stored procedures with parameterized inputs
> 3. Validate and sanitize all user-supplied input
> 4. Escape special characters in user input
>
> **Additional Measures:**
> - Implement least privilege for database accounts
> - Use LIMIT and other SQL controls to prevent mass disclosure
> - Apply positive server-side input validation
> - Use ORMs (Object Relational Mappers) which naturally parameterize queries

**Citations:**
| Source | File | Page | Similarity |
|--------|------|------|------------|
| 1 | owasp-top-10.pdf | 8 | 0.75 |
| 2 | owasp-top-10.pdf | 9 | 0.70 |

**Status:** ✅ Answered with citations

---

## Test 5: Persistence in MITRE ATT&CK

**Question:** How does MITRE describe the purpose of Persistence techniques?

**Expected Behavior:** Answer from MITRE document

**Retrieved Evidence:**
- Source: `mitre-attack-philosophy-2020.pdf`, Pages 5-6
- Similarity: 0.73

**Answer:**
> According to MITRE ATT&CK, Persistence techniques represent methods by which adversaries maintain their foothold on systems across restarts, changed credentials, and other interruptions. [Source 1]
>
> The purpose of Persistence is to ensure continued access to the target environment. Adversaries need to maintain access because:
> - Systems reboot
> - Credentials expire or get changed
> - Security tools may terminate malicious processes
>
> Common Persistence techniques include registry run keys, scheduled tasks, account creation, and boot/logon scripts. [Source 2]

**Citations:**
| Source | File | Page | Similarity |
|--------|------|------|------------|
| 1 | mitre-attack-philosophy-2020.pdf | 5 | 0.73 |
| 2 | mitre-attack-philosophy-2020.pdf | 6 | 0.68 |

**Status:** ✅ Answered with citations

---

## Test 6: Refusal Example - Ransomware

**Question:** What is ransomware and how does it spread?

**Expected Behavior:** REFUSE - topic not covered in dataset

**Retrieved Evidence:**
- Chunks retrieved: 2
- Average similarity: 0.28 (below threshold)
- Best match: Generic reference to "malware" in OWASP context

**Response:**
> I cannot answer this question based on the provided documents.
>
> **Reason:** The dataset does not contain sufficient information about ransomware. While the documents mention "malware" as a general risk category, they do not provide detailed information about ransomware specifically, including its definition, mechanisms, or propagation methods.
>
> **Documents searched:**
> - owasp-top-10.pdf
> - thailand-web-security-standard-2025.pdf  
> - mitre-attack-philosophy-2020.pdf

**Grounding Details:**
- Evidence sufficiency: FAILED (similarity 0.28 < 0.3 threshold)
- Chunks relevant: 0 of 2 met threshold

**Status:** ✅ Correctly refused (insufficient evidence)

---

## Test 7: Refusal Example - Zero-Day Exploits

**Question:** What are the latest zero-day vulnerabilities discovered in 2024?

**Expected Behavior:** REFUSE - dataset is static (2020-2025), no current events

**Retrieved Evidence:**
- Chunks retrieved: 1
- Average similarity: 0.22
- Content: General vulnerability discussion, no specific CVEs

**Response:**
> I cannot answer this question based on the provided documents.
>
> **Reason:** The dataset documents do not contain information about specific zero-day vulnerabilities or CVEs from 2024. The documents are reference standards and frameworks (OWASP Top 10, Thailand Web Security Standard, MITRE ATT&CK Philosophy) rather than vulnerability databases or current threat intelligence feeds.

**Status:** ✅ Correctly refused (out of scope)

---

## Summary

| Test | Question Topic | Expected | Actual | Status |
|------|----------------|----------|--------|--------|
| 1 | Broken Access Control | Answer | Answer | ✅ |
| 2 | Thailand Security Controls | Answer | Answer | ✅ |
| 3 | Tactics vs Techniques | Answer | Answer | ✅ |
| 4 | Injection Mitigation | Answer | Answer | ✅ |
| 5 | Persistence Purpose | Answer | Answer | ✅ |
| 6 | Ransomware | Refuse | Refuse | ✅ |
| 7 | Zero-Day 2024 | Refuse | Refuse | ✅ |

**Success Rate:** 7/7 (100%)

---

## Running Your Own Tests

```bash
# Start the API server
python -m src.api

# Test via curl
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Your question here"}'

# Or use the interactive docs
open http://localhost:8000/docs
```
