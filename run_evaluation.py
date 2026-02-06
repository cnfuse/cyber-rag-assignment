"""
Run evaluation tests and capture actual results.

This script runs the test questions from the assignment requirements
and generates real outputs to verify the system works correctly.
Includes LLM-as-a-judge evaluation using OpenAI.
"""

import json
import time
import os
from src.agent import CybersecurityRAGAgent

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: Google Gemini not available. Install with: pip install google-genai")

# Test questions from the assignment - covering all 3 dataset documents
TEST_QUESTIONS = [
    # OWASP Top 10 questions
    "What is Broken Access Control according to OWASP?",
    "What mitigation steps does OWASP recommend for Injection vulnerabilities?",
    
    # Thailand Web Security Standard questions
    "What website security controls are required by the Thailand Web Security Standard?",
    "What authentication requirements does the Thailand Web Security Standard mandate?",
    
    # MITRE ATT&CK questions
    "What is the difference between a Tactic and a Technique in MITRE ATT&CK?",
    "How does MITRE describe the purpose of Persistence techniques?",
    
    # Refusal cases - information not in dataset
    "What does the dataset say about ransomware attacks?",  # Should refuse
    "What are the latest zero-day vulnerabilities discovered in 2024?",  # Should refuse
]


JUDGE_PROMPT = """You are an expert evaluator for a RAG (Retrieval-Augmented Generation) system.

Evaluate the following answer based on these criteria:

1. **Grounding (0-10)**: Are all claims supported by the provided sources? No hallucinations?
2. **Completeness (0-10)**: Does the answer fully address the question?
3. **Citation Quality (0-10)**: Are sources properly cited and traceable?
4. **Clarity (0-10)**: Is the answer clear and well-structured?
5. **Correctness (0-10)**: Is the information factually accurate based on the sources?

For refused answers, evaluate if the refusal was appropriate given the sources.

Question: {question}

Answer Status: {status}

Answer: {answer}

Sources: {sources}

Provide your evaluation in JSON format:
{{
    "grounding_score": <0-10>,
    "completeness_score": <0-10>,
    "citation_quality_score": <0-10>,
    "clarity_score": <0-10>,
    "correctness_score": <0-10>,
    "total_score": <0-50>,
    "feedback": "<brief explanation>",
    "issues": ["<list any issues>"]
}}
"""


def judge_answer(question: str, status: str, answer: str, sources: list) -> dict:
    """Use Google Gemini 2.0 Flash to evaluate the answer quality."""
    
    if not GEMINI_AVAILABLE:
        return {"error": "Google Gemini not available"}
    
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return {"error": "GEMINI_API_KEY or GOOGLE_API_KEY not set"}
    
    try:
        client = genai.Client(api_key=api_key)
        
        # Format sources for judge
        sources_str = "\n".join([
            f"- {s['file']}, Page {s.get('page', 'N/A')}"
            for s in sources
        ]) if sources else "No sources provided"
        
        prompt = JUDGE_PROMPT.format(
            question=question,
            status=status,
            answer=answer or "N/A",
            sources=sources_str
        )
        
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json"
            )
        )
        
        judgment = json.loads(response.text)
        return judgment
        
    except Exception as e:
        return {"error": f"Judge failed: {str(e)}"}



def run_tests():
    """Run all test questions and display results."""
    
    print("=" * 70)
    print("Cybersecurity RAG Agent - Evaluation Tests")
    print("=" * 70)
    print()
    
    # Initialize agent
    print("Initializing agent...")
    agent = CybersecurityRAGAgent()
    agent.initialize()
    print("✓ Agent initialized\n")
    
    results = []
    
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n{'='*70}")
        print(f"Test {i}: {question}")
        print(f"{'='*70}")
        
        start_time = time.time()
        response = agent.query(question)
        elapsed = time.time() - start_time
        
        print(f"\nStatus: {response.status.upper()}")
        print(f"Processing Time: {elapsed:.2f}s")
        
        if response.status == "answered":
            print(f"\nAnswer:\n{response.answer}")
            print(f"\nSources ({len(response.sources)}):")
            for source in response.sources:
                print(f"  - {source['file']}, Page {source.get('page', 'N/A')}")
            print(f"\nConfidence: {response.confidence:.2%}")
            if response.grounding_details:
                details = response.grounding_details
                print(f"Grounding: is_grounded={details.get('is_grounded', 'N/A')}, "
                      f"confidence={details.get('confidence', 0):.2%}, "
                      f"evidence_chunks={len(details.get('evidence_chunks', []))}")
        
        elif response.status == "refused":
            print(f"\nReason: {response.reason}")
        
        else:  # error
            print(f"\nError: {response.reason}")
        
        # LLM-as-a-judge evaluation
        print("\n🤖 LLM Judge Evaluation...")
        judgment = judge_answer(
            question=question,
            status=response.status,
            answer=response.answer or response.reason or "",
            sources=response.sources
        )
        
        if "error" not in judgment:
            print(f"  Overall Score: {judgment.get('total_score', 0)}/50")
            print(f"  - Grounding: {judgment.get('grounding_score', 0)}/10")
            print(f"  - Completeness: {judgment.get('completeness_score', 0)}/10")
            print(f"  - Citation Quality: {judgment.get('citation_quality_score', 0)}/10")
            print(f"  Feedback: {judgment.get('feedback', 'N/A')}")
        else:
            print(f"  {judgment.get('error', 'Unknown error')}")
            judgment = None
        
        # Store result
        results.append({
            "question": question,
            "status": response.status,
            "answer": response.answer,
            "sources": response.sources,
            "confidence": response.confidence,
            "reason": response.reason,
            "elapsed_seconds": elapsed,
            "judge_evaluation": judgment
        })
    
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    
    answered = sum(1 for r in results if r["status"] == "answered")
    refused = sum(1 for r in results if r["status"] == "refused")
    errors = sum(1 for r in results if r["status"] == "error")
    
    print(f"Total Questions: {len(results)}")
    print(f"Answered: {answered}")
    print(f"Refused: {refused}")
    print(f"Errors: {errors}")
    
    # Calculate average judge scores
    judge_scores = [
        r["judge_evaluation"]["total_score"] 
        for r in results 
        if r.get("judge_evaluation") and "error" not in r["judge_evaluation"]
    ]
    
    if judge_scores:
        avg_score = sum(judge_scores) / len(judge_scores)
        print(f"\n📊 LLM Judge Average Score: {avg_score:.1f}/50 ({avg_score*2:.1f}%)")
        print(f"   Evaluated: {len(judge_scores)}/{len(results)} questions")
    
    # Save results
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Detailed results saved to: evaluation_results.json")


if __name__ == "__main__":
    try:
        run_tests()
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
