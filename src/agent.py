"""
RAG Agent for cybersecurity Q&A.

Three agents:
1. Indexing - Loads PDFs, creates embeddings, stores in vector DB
2. QA - Retrieves context and generates answers
3. Grounding - Checks if answers are actually supported by the docs

All work through a shared tool interface.
"""

import logging
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

import ollama
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import settings
from .tools import get_tools, ToolInterface, RetrievedChunk, GroundingResult

# Configure logging
logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent operational states."""
    IDLE = "idle"
    INDEXING = "indexing"
    QUERYING = "querying"
    VERIFYING = "verifying"
    ERROR = "error"


@dataclass
class QueryResponse:
    """Response from the agent."""
    status: str  # "answered", "refused", "error"
    answer: Optional[str] = None
    sources: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    reason: Optional[str] = None
    retrieval_stats: Dict[str, Any] = field(default_factory=dict)
    grounding_details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "answer": self.answer,
            "sources": self.sources,
            "confidence": self.confidence,
            "reason": self.reason,
            "retrieval_stats": self.retrieval_stats,
            "grounding_details": self.grounding_details
        }


class IndexingAgent:
    """
    Agent responsible for preparing and maintaining the knowledge base.
    
    Responsibilities:
    - Load all files from dataset/
    - Chunk documents and attach metadata
    - Generate embeddings and build/update vector database
    - Track index version and freshness
    """
    
    def __init__(self, tools: ToolInterface):
        self.tools = tools
        self.state = AgentState.IDLE
        self._index_version: Optional[str] = None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current indexing status."""
        return self.tools.refresh_index()
    
    def build_index(self, force: bool = False) -> Dict[str, Any]:
        """Build or rebuild the document index."""
        self.state = AgentState.INDEXING
        try:
            result = self.tools.build_index(force_rebuild=force)
            if result["status"] in ["rebuilt", "current"]:
                self._index_version = datetime.now().isoformat()
            return result
        finally:
            self.state = AgentState.IDLE
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all available documents."""
        return self.tools.list_documents()


class QAAgent:
    """
    Agent responsible for answering user questions.
    
    Responsibilities:
    - Accept and normalize user queries
    - Retrieve relevant chunks from vector database
    - Generate answers strictly from retrieved content
    - Attach citations to answers
    """
    
    def __init__(self, tools: ToolInterface):
        self.tools = tools
        self.state = AgentState.IDLE
        self._ollama_available: Optional[bool] = None
    
    def _check_ollama(self) -> bool:
        """Check if Ollama is available."""
        if self._ollama_available is not None:
            return self._ollama_available
        
        try:
            # Create client with custom host
            client = ollama.Client(host=settings.OLLAMA_BASE_URL)
            client.list()
            self._ollama_available = True
            logger.info(f"Ollama is available at {settings.OLLAMA_BASE_URL}")
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            self._ollama_available = False
        
        return self._ollama_available
    
    def normalize_query(self, query: str) -> str:
        """
        Normalize query for better retrieval.
        
        Important: This only cleans the query - it does NOT add
        any external knowledge or context.
        """
        # Remove excessive whitespace
        query = " ".join(query.split())
        
        # Remove question marks and common filler words
        query = query.rstrip("?").strip()
        
        # Expand common abbreviations (dataset-neutral)
        abbreviations = {
            "auth": "authentication",
            "authz": "authorization",
            "vuln": "vulnerability",
            "sec": "security",
        }
        for abbrev, full in abbreviations.items():
            pattern = rf'\b{abbrev}\b'
            query = re.sub(pattern, full, query, flags=re.IGNORECASE)
        
        return query
    
    def retrieve(self, query: str, top_k: int = None) -> List[RetrievedChunk]:
        """Retrieve relevant chunks for a query."""
        normalized_query = self.normalize_query(query)
        return self.tools.vector_search(normalized_query, top_k)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with retry logic."""
        if not self._check_ollama():
            raise RuntimeError("Ollama is not available. Please start Ollama server.")
        
        try:
            # Create client with custom host
            client = ollama.Client(host=settings.OLLAMA_BASE_URL)
            response = client.generate(
                model=settings.OLLAMA_MODEL,
                prompt=prompt,
                options={
                    "temperature": 0.1,  # Low temperature for factual answers
                    "num_predict": 1024,
                    "top_p": 0.9
                }
            )
            return response["response"]
        except Exception as e:
            logger.error(f"Ollama LLM error: {type(e).__name__}: {str(e)}")
            raise RuntimeError(f"Failed to call LLM: {str(e)}")
    
    def generate_answer(
        self, 
        query: str, 
        chunks: List[RetrievedChunk]
    ) -> str:
        """
        Generate an answer from retrieved chunks.
        
        The LLM is constrained to use ONLY the provided context.
        """
        if not chunks:
            return ""
        
        # Build context from chunks
        context_parts = []
        for i, rc in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}: {rc.chunk.source_file}, Page {rc.chunk.page_number}]\n"
                f"{rc.chunk.text}\n"
            )
        context = "\n".join(context_parts)
        
        # Strict grounding prompt
        prompt = f"""You are a cybersecurity expert assistant. Your task is to answer questions using ONLY the provided context.

STRICT RULES:
1. Answer ONLY using information from the context below
2. Do NOT use any external knowledge or assumptions
3. If the context doesn't contain the answer, say "I cannot answer this based on the provided documents"
4. Always cite sources using [Source N] format
5. Be precise and factual

CONTEXT:
{context}

QUESTION: {query}

ANSWER (using only the context above):"""
        
        return self._call_llm(prompt)
    
    def format_sources(self, chunks: List[RetrievedChunk]) -> List[Dict[str, Any]]:
        """Format chunks as citation sources."""
        sources = []
        for rc in chunks:
            sources.append({
                "file": rc.chunk.source_file,
                "page": rc.chunk.page_number,
                "chunk_id": rc.chunk.chunk_id,
                "similarity": round(rc.similarity_score, 3),
                "excerpt": rc.chunk.text[:200] + "..." if len(rc.chunk.text) > 200 else rc.chunk.text
            })
        return sources


class GroundingAgent:
    """
    Agent responsible for verifying answer grounding.
    
    Responsibilities:
    - Verify each claim is supported by retrieved chunks
    - Ensure citations directly correspond to supporting text
    - Prevent hallucinated or general knowledge
    - Force refusal when evidence is insufficient
    """
    
    def __init__(self, tools: ToolInterface):
        self.tools = tools
        self.state = AgentState.IDLE
    
    def check_evidence_sufficiency(
        self, 
        chunks: List[RetrievedChunk]
    ) -> tuple[bool, str]:
        """Check if retrieved evidence is sufficient."""
        return self.tools.check_sufficient_evidence(
            chunks,
            min_chunks=1,
            min_avg_similarity=settings.SIMILARITY_THRESHOLD
        )
    
    def verify_answer(
        self, 
        answer: str, 
        chunks: List[RetrievedChunk]
    ) -> GroundingResult:
        """Verify answer is grounded in evidence."""
        self.state = AgentState.VERIFYING
        try:
            return self.tools.verify_grounding(answer, chunks)
        finally:
            self.state = AgentState.IDLE
    
    def should_refuse(
        self, 
        grounding_result: GroundingResult,
        chunks: List[RetrievedChunk]
    ) -> tuple[bool, str]:
        """Determine if agent should refuse to answer."""
        # Refuse if no evidence
        if not chunks:
            return True, "No relevant documents found in the dataset"
        
        # Refuse if grounding confidence is too low
        if grounding_result.confidence < 0.5:
            return True, "Answer cannot be sufficiently grounded in dataset evidence"
        
        # Refuse if too many unsupported claims
        if len(grounding_result.unsupported_claims) > len(grounding_result.supported_claims):
            return True, "Too many claims cannot be verified against the dataset"
        
        return False, ""

class CybersecurityRAGAgent:
    """
    Main agent that coordinates everything.
    
    Handles indexing, retrieving docs, generating answers, and verifying
    that answers are actually from the dataset.
    """
    
    def __init__(self):
        self.tools = get_tools()
        self.indexing_agent = IndexingAgent(self.tools)
        self.qa_agent = QAAgent(self.tools)
        self.grounding_agent = GroundingAgent(self.tools)
        
        self._initialized = False
        logger.info("CybersecurityRAGAgent initialized")
    
    def initialize(self, force_reindex: bool = False) -> Dict[str, Any]:
        """
        Initialize the agent and ensure index is ready.
        
        Args:
            force_reindex: Force rebuild of the index.
            
        Returns:
            Initialization status.
        """
        logger.info("Initializing RAG agent...")
        
        # Check current index status
        status = self.indexing_agent.get_status()
        
        # Build index if needed
        if status["status"] == "empty" or force_reindex:
            build_result = self.indexing_agent.build_index(force=force_reindex)
            status.update(build_result)
        
        self._initialized = True
        return {
            "status": "ready",
            "index_status": status,
            "documents": self.indexing_agent.list_documents()
        }
    
    def build_index(self, force: bool = False) -> Dict[str, Any]:
        """Build or rebuild the document index."""
        return self.indexing_agent.build_index(force=force)
    
    def query(self, question: str) -> QueryResponse:
        """
        Process a user query and return a grounded answer.
        
        Args:
            question: The user's question.
            
        Returns:
            QueryResponse with answer, sources, and metadata.
        """
        if not self._initialized:
            # Auto-initialize if needed
            init_result = self.initialize()
            if init_result.get("index_status", {}).get("total_chunks", 0) == 0:
                return QueryResponse(
                    status="error",
                    reason="Index is empty. No documents have been indexed."
                )
        
        logger.info(f"Processing query: {question}")
        
        # Step 1: Retrieve relevant chunks (using TOP_K_RESULTS from config)
        chunks = self.qa_agent.retrieve(question)
        
        retrieval_stats = {
            "chunks_retrieved": len(chunks),
            "top_similarity": chunks[0].similarity_score if chunks else 0,
            "avg_similarity": sum(c.similarity_score for c in chunks) / len(chunks) if chunks else 0
        }
        
        # Step 2: Check evidence sufficiency
        sufficient, reason = self.grounding_agent.check_evidence_sufficiency(chunks)
        
        if not sufficient:
            logger.info(f"Refusing to answer: {reason}")
            return QueryResponse(
                status="refused",
                reason=f"Cannot answer from dataset: {reason}",
                sources=self.qa_agent.format_sources(chunks),
                retrieval_stats=retrieval_stats
            )
        
        # Step 3: Generate answer
        try:
            answer = self.qa_agent.generate_answer(question, chunks)
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return QueryResponse(
                status="error",
                reason=f"Failed to generate answer: {str(e)}",
                retrieval_stats=retrieval_stats
            )
        
        # Step 4: Verify grounding
        grounding_result = self.grounding_agent.verify_answer(answer, chunks)
        
        # Step 5: Check if we should refuse based on grounding
        should_refuse, refuse_reason = self.grounding_agent.should_refuse(
            grounding_result, chunks
        )
        
        if should_refuse:
            logger.info(f"Refusing after grounding check: {refuse_reason}")
            return QueryResponse(
                status="refused",
                reason=refuse_reason,
                sources=self.qa_agent.format_sources(chunks),
                retrieval_stats=retrieval_stats,
                grounding_details={
                    "confidence": grounding_result.confidence,
                    "supported_claims": len(grounding_result.supported_claims),
                    "unsupported_claims": len(grounding_result.unsupported_claims)
                }
            )
        
        # Step 6: Return successful answer
        return QueryResponse(
            status="answered",
            answer=answer,
            sources=self.qa_agent.format_sources(chunks),
            confidence=grounding_result.confidence,
            retrieval_stats=retrieval_stats,
            grounding_details={
                "is_grounded": grounding_result.is_grounded,
                "confidence": grounding_result.confidence,
                "evidence_chunks": grounding_result.evidence_chunks
            }
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "initialized": self._initialized,
            "index": self.indexing_agent.get_status(),
            "documents": self.indexing_agent.list_documents(),
            "ollama_available": self.qa_agent._check_ollama()
        }


# Convenience function for direct usage
def create_agent() -> CybersecurityRAGAgent:
    """Create and return a configured RAG agent."""
    agent = CybersecurityRAGAgent()
    return agent
