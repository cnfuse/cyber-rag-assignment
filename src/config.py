"""
Configuration settings for the Cybersecurity RAG Agent.

Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load .env file before reading environment variables
load_dotenv()


class Settings:
    """Application configuration loaded from environment variables."""
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATASET_DIR: Path = BASE_DIR / "dataset"
    CHROMA_PERSIST_DIR: Path = Path(os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))
    
    # Ollama Configuration
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://192.168.4.12:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gemma3:12b")
    OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "120"))
    
    # Embedding Model (Ollama)
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
    
    # Vector Database
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "cybersecurity_docs")
    
    # RAG Settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))
    MAX_TOP_K: int = int(os.getenv("MAX_TOP_K", "20"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
    
    # API Server
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_WORKERS: int = int(os.getenv("API_WORKERS", "4"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "json")
    
    # Optional: OpenAI Fallback
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    # Typhoon OCR Configuration
    TYPHOON_API_KEY: Optional[str] = os.getenv("TYPHOON_API_KEY") or os.getenv("TYPHOON_OCR_API_KEY")
    TYPHOON_BASE_URL: str = os.getenv("TYPHOON_BASE_URL", "https://api.opentyphoon.ai/v1")
    TYPHOON_OCR_MODEL: str = os.getenv("TYPHOON_OCR_MODEL", "typhoon-ocr")
    ENABLE_TYPHOON_OCR: bool = os.getenv("ENABLE_TYPHOON_OCR", "true").lower() == "true"
    
    def __init__(self):
        # Resolve relative paths
        if not self.CHROMA_PERSIST_DIR.is_absolute():
            self.CHROMA_PERSIST_DIR = self.BASE_DIR / self.CHROMA_PERSIST_DIR


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings
