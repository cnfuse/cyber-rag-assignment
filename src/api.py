"""
FastAPI server for the RAG agent.

Includes:
- Health check endpoint
- JSON logging
- Prometheus metrics
- CORS
"""

import logging
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
from datetime import datetime

import structlog
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import uvicorn

from .config import settings
from .agent import CybersecurityRAGAgent, QueryResponse

# ==================== Logging Setup ====================

def setup_logging():
    """Setup JSON logging."""
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer() if settings.LOG_FORMAT == "json" 
            else structlog.dev.ConsoleRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.LOG_LEVEL)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Also configure standard logging
    logging.basicConfig(
        level=settings.LOG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Suppress noisy PDF parsing warnings
    logging.getLogger("pypdf._reader").setLevel(logging.ERROR)

setup_logging()
logger = structlog.get_logger()

# ==================== Prometheus Metrics ====================

# Use try/except to avoid duplicate registration errors when module is reloaded
try:
    REQUEST_COUNT = Counter(
        "rag_requests_total",
        "Total number of RAG requests",
        ["endpoint", "status"]
    )

    REQUEST_LATENCY = Histogram(
        "rag_request_latency_seconds",
        "Request latency in seconds",
        ["endpoint"]
    )

    QUERY_RESULTS = Counter(
        "rag_query_results_total",
        "Query results by status",
        ["status"]
    )
except ValueError:
    # Metrics already registered, retrieve them
    from prometheus_client import REGISTRY
    REQUEST_COUNT = REGISTRY._names_to_collectors.get("rag_requests_total")
    REQUEST_LATENCY = REGISTRY._names_to_collectors.get("rag_request_latency_seconds")
    QUERY_RESULTS = REGISTRY._names_to_collectors.get("rag_query_results_total")
except ValueError:
    # Metrics already registered, retrieve them
    from prometheus_client import REGISTRY
    REQUEST_COUNT = REGISTRY._names_to_collectors.get("rag_requests_total")
    REQUEST_LATENCY = REGISTRY._names_to_collectors.get("rag_request_latency_seconds")
    QUERY_RESULTS = REGISTRY._names_to_collectors.get("rag_query_results_total")
    "Query results by status",
    ["status"]


# ==================== Pydantic Models ====================

class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str = Field(..., min_length=1, max_length=2000, description="The question to ask")


class QueryResponseModel(BaseModel):
    """Response model for query endpoint."""
    status: str = Field(..., description="Response status: answered, refused, or error")
    answer: Optional[str] = Field(None, description="The generated answer")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Source citations")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Answer confidence score")
    reason: Optional[str] = Field(None, description="Reason for refusal or error")
    retrieval_stats: Dict[str, Any] = Field(default_factory=dict, description="Retrieval statistics")
    grounding_details: Optional[Dict[str, Any]] = Field(None, description="Grounding verification details")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str = "1.0.0"
    checks: Dict[str, bool]


class IndexRequest(BaseModel):
    """Request model for index operations."""
    force_rebuild: bool = Field(default=False, description="Force rebuild the index")


class IndexResponse(BaseModel):
    """Response model for index operations."""
    status: str
    message: str
    documents_processed: int = 0
    total_chunks: int = 0
    details: Optional[Dict[str, Any]] = None


class DocumentsResponse(BaseModel):
    """Response model for document listing."""
    documents: List[Dict[str, Any]]
    count: int


# ==================== Application Lifespan ====================

# Global agent instance
_agent: Optional[CybersecurityRAGAgent] = None


def get_agent() -> CybersecurityRAGAgent:
    """Get the global agent instance."""
    global _agent
    if _agent is None:
        raise RuntimeError("Agent not initialized")
    return _agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global _agent
    
    logger.info("Starting Cybersecurity RAG Agent API...")
    
    # Initialize agent
    _agent = CybersecurityRAGAgent()
    
    # Auto-initialize index if not empty dataset
    try:
        init_result = _agent.initialize()
        logger.info(
            "Agent initialized",
            index_status=init_result.get("index_status", {}).get("status"),
            total_chunks=init_result.get("index_status", {}).get("total_chunks", 0)
        )
    except Exception as e:
        logger.error("Failed to initialize agent", error=str(e))
    
    yield
    
    # Cleanup
    logger.info("Shutting down RAG Agent API...")
    _agent = None


# ==================== FastAPI Application ====================

app = FastAPI(
    title="Cybersecurity RAG Agent API",
    description="A strict dataset-only RAG agent for cybersecurity Q&A",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Middleware ====================

@app.middleware("http")
async def add_request_timing(request: Request, call_next):
    """Add request timing header."""
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    response.headers["X-Process-Time-Ms"] = str(round(process_time, 2))
    return response


# ==================== Health Endpoints ====================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint for load balancers and orchestrators.
    """
    agent = get_agent()
    status = agent.get_status()
    
    checks = {
        "agent_initialized": status.get("initialized", False),
        "index_ready": status.get("index", {}).get("total_chunks", 0) > 0,
        "ollama_available": status.get("ollama_available", False)
    }
    
    overall_status = "healthy" if all(checks.values()) else "degraded"
    
    REQUEST_COUNT.labels(endpoint="/health", status=overall_status).inc()
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now().isoformat(),
        checks=checks
    )

# ==================== Core API Endpoints ====================

@app.post("/ask", response_model=QueryResponseModel, tags=["Query"])
async def ask_question(request: QueryRequest):
    """
    Ask a cybersecurity question.
    
    The answer will be generated strictly from the dataset documents.
    If the dataset doesn't contain sufficient information, the agent
    will refuse to answer rather than hallucinate.
    """
    start_time = time.time()
    
    try:
        agent = get_agent()
        
        with REQUEST_LATENCY.labels(endpoint="/ask").time():
            response: QueryResponse = agent.query(
                question=request.query
            )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Record metrics
        REQUEST_COUNT.labels(endpoint="/ask", status=response.status).inc()
        QUERY_RESULTS.labels(status=response.status).inc()
        
        logger.info(
            "Query processed",
            query=request.query[:100],
            status=response.status,
            confidence=response.confidence,
            sources_count=len(response.sources),
            processing_time_ms=processing_time
        )
        
        return QueryResponseModel(
            status=response.status,
            answer=response.answer,
            sources=response.sources,
            confidence=response.confidence,
            reason=response.reason,
            retrieval_stats=response.retrieval_stats,
            grounding_details=response.grounding_details,
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/ask", status="error").inc()
        logger.error("Query failed", error=str(e), query=request.query[:100])
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Error Handlers ====================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(
        "Unhandled exception",
        path=request.url.path,
        method=request.method,
        error=str(exc)
    )
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "detail": "Internal server error",
            "message": str(exc) if settings.LOG_LEVEL == "DEBUG" else "An error occurred"
        }
    )


# ==================== Main Entry Point ====================

def main():
    """Run the production server."""
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║        Cybersecurity RAG Agent - Production Server           ║
╠══════════════════════════════════════════════════════════════╣
║  • API Docs:    http://{settings.API_HOST}:{settings.API_PORT}/docs              ║
║  • Health:      http://{settings.API_HOST}:{settings.API_PORT}/health            ║
╠══════════════════════════════════════════════════════════════╣
║  • Ollama URL:  {settings.OLLAMA_BASE_URL:<35}       ║
║  • LLM Model:   {settings.OLLAMA_MODEL:<35}       ║
║  • Embeddings:  {settings.EMBEDDING_MODEL[:35]:<35}       ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Use single worker for development (workers > 1 can cause issues with ChromaDB)
    uvicorn.run(
        "src.api:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=1,  # Single worker to avoid ChromaDB locking issues
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True,
        reload=False
    )


if __name__ == "__main__":
    main()
