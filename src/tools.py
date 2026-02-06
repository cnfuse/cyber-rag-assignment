"""
Tools Interface for the Cybersecurity RAG Agent.

This module defines explicit tools that agents use to interact with the system.
All operations are performed through these tools, ensuring clear separation
of concerns and auditability.
"""

import hashlib
import logging
import re
import unicodedata
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO

import chromadb
from chromadb.config import Settings as ChromaSettings
import ollama
from langchain_community.document_loaders import PyPDFLoader
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from pdf2image import convert_from_path
    from PIL import Image
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logger.warning("pdf2image not available. OCR for scanned PDFs will be limited.")

from .config import settings
from .typhoon_ocr_integration import get_typhoon_ocr_extractor

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a document chunk with metadata."""
    chunk_id: str
    text: str
    source_file: str
    page_number: int
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_file": self.source_file,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            **self.metadata
        }


@dataclass
class RetrievedChunk:
    """Represents a chunk retrieved from vector search."""
    chunk: Chunk
    similarity_score: float
    rank: int


@dataclass
class GroundingResult:
    """Result of grounding verification."""
    is_grounded: bool
    supported_claims: List[str]
    unsupported_claims: List[str]
    confidence: float
    evidence_chunks: List[str]


class ToolInterface:
    """
    Explicit tool interface for RAG agents.
    
    All agent operations go through these tools, ensuring:
    - Clear separation of concerns
    - Audit trail of all operations
    - Consistent behavior across agents
    """
    
    def __init__(self):
        self._chroma_client: Optional[chromadb.PersistentClient] = None
        self._collection: Optional[chromadb.Collection] = None
        self._index_manifest: Dict[str, Any] = {}
        self._initialized = False
        # Create Ollama client with custom host
        self._ollama_client = ollama.Client(host=settings.OLLAMA_BASE_URL)
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Ollama."""
        embeddings = []
        for text in texts:
            response = self._ollama_client.embed(
                model=settings.EMBEDDING_MODEL,
                input=text
            )
            embeddings.append(response["embeddings"][0])
        return embeddings
    
    def _ensure_initialized(self):
        """Lazy initialization of resources."""
        if self._initialized:
            return
        
        logger.info("Initializing tool interface...")
        
        # Verify Ollama embedding model is available
        logger.info(f"Using Ollama embedding model: {settings.EMBEDDING_MODEL}")
        
        # Initialize ChromaDB
        settings.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        self._chroma_client = chromadb.PersistentClient(
            path=str(settings.CHROMA_PERSIST_DIR),
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self._collection = self._chroma_client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        
        self._initialized = True
        logger.info("Tool interface initialized successfully")
    
    # ========== Document Management Tools ==========
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents in the dataset folder.
        
        Returns:
            List of document metadata dictionaries.
        """
        documents = []
        dataset_path = settings.DATASET_DIR
        
        if not dataset_path.exists():
            logger.warning(f"Dataset directory not found: {dataset_path}")
            return documents
        
        for file_path in dataset_path.glob("*.pdf"):
            stat = file_path.stat()
            documents.append({
                "filename": file_path.name,
                "path": str(file_path),
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "file_hash": self._compute_file_hash(file_path)
            })
        
        logger.info(f"Found {len(documents)} documents in dataset")
        return documents
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of a file for change detection."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:16]
    
    def _is_text_quality_acceptable(self, text: str, min_length: int = 50) -> bool:
        """Check if text quality is acceptable for indexing.
        
        Args:
            text: Text to validate.
            min_length: Minimum length for meaningful text.
            
        Returns:
            True if text quality is acceptable, False otherwise.
        """
        if not text or len(text.strip()) < min_length:
            return False
        
        # Count printable vs non-printable characters
        printable_count = sum(1 for c in text if c.isprintable() or c.isspace())
        total_count = len(text)
        
        if total_count == 0:
            return False
        
        printable_ratio = printable_count / total_count
        
        # Reject if more than 30% non-printable characters
        if printable_ratio < 0.7:
            return False
        
        # Check for reasonable alphanumeric content (at least 20%)
        alnum_count = sum(1 for c in text if c.isalnum())
        alnum_ratio = alnum_count / total_count
        
        if alnum_ratio < 0.2:
            return False
        
        # Check for excessive repeated special characters
        special_char_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
        special_ratio = special_char_count / total_count
        
        if special_ratio > 0.5:
            return False
        
        return True
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text.
        
        Args:
            text: Raw text from PDF extraction.
            
        Returns:
            Cleaned and normalized text.
        """
        if not text:
            return ""
        
        # Remove control characters (except common whitespace)
        # Keep: tab (\t), newline (\n), carriage return (\r)
        cleaned = ''.join(
            char for char in text 
            if unicodedata.category(char)[0] != 'C' or char in '\t\n\r'
        )
        
        # Normalize Unicode (NFC form for consistent Thai character representation)
        cleaned = unicodedata.normalize('NFC', cleaned)
        
        # Remove excessive whitespace while preserving structure
        cleaned = re.sub(r' +', ' ', cleaned)  # Multiple spaces to single
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)  # Max 2 consecutive newlines
        cleaned = re.sub(r'\t+', '\t', cleaned)  # Multiple tabs to single
        
        # Remove lines that are mostly special characters or formatting artifacts
        lines = cleaned.split('\n')
        filtered_lines = []
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                filtered_lines.append(line)
                continue
            
            # Count meaningful characters (alphanumeric + space)
            meaningful_chars = sum(1 for c in line_stripped if c.isalnum() or c.isspace())
            if len(line_stripped) > 0 and meaningful_chars / len(line_stripped) > 0.3:
                filtered_lines.append(line)
        
        cleaned = '\n'.join(filtered_lines)
        
        # Final cleanup: strip excessive leading/trailing whitespace
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _extract_text_with_vision(self, pdf_path: Path, page_num: int) -> str:
        """Extract text from PDF page using vision model OCR.
        
        Args:
            pdf_path: Path to PDF file.
            page_num: Page number (0-indexed).
            
        Returns:
            Extracted text from the page.
        """
        if not PDF2IMAGE_AVAILABLE:
            logger.warning("pdf2image not available, cannot perform OCR")
            return ""
        
        try:
            # Convert PDF page to image
            images = convert_from_path(
                pdf_path,
                first_page=page_num + 1,
                last_page=page_num + 1,
                dpi=200
            )
            
            if not images:
                return ""
            
            # Convert image to base64
            img = images[0]
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Use Ollama vision model to extract text
            prompt = """Extract ALL text from this image. Include:
- All headers, titles, and section names
- All body text and paragraphs
- All table content with structure
- Any bullet points or lists
- Maintain original language (Thai/English)

Provide only the extracted text, preserving formatting and structure."""
            
            response = self._ollama_client.generate(
                model=settings.OLLAMA_MODEL,  # Vision-capable model
                prompt=prompt,
                images=[img_base64],
                options={"temperature": 0.1}  # Low temperature for accurate extraction
            )
            
            extracted_text = response.get('response', '').strip()
            logger.debug(f"OCR extracted {len(extracted_text)} characters from page {page_num + 1}")
            return extracted_text
            
        except Exception as e:
            logger.error(f"Vision OCR failed for page {page_num + 1}: {e}", exc_info=True)
            return ""
    
    # ========== Indexing Tools ==========
    
    def build_index(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """
        Build or rebuild the vector index from dataset documents.
        
        Args:
            force_rebuild: If True, rebuild even if index exists.
            
        Returns:
            Index build status and statistics.
        """
        self._ensure_initialized()
        
        documents = self.list_documents()
        if not documents:
            return {
                "status": "error",
                "message": "No documents found in dataset",
                "documents_processed": 0
            }
        
        # Check if rebuild is needed
        current_hashes = {d["filename"]: d["file_hash"] for d in documents}
        if not force_rebuild and self._is_index_current(current_hashes):
            return {
                "status": "current",
                "message": "Index is up to date",
                "documents_indexed": len(documents),
                "total_chunks": self._collection.count()
            }
        
        # Clear existing collection
        logger.info("Rebuilding index...")
        self._chroma_client.delete_collection(settings.CHROMA_COLLECTION_NAME)
        self._collection = self._chroma_client.create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Process each document
        total_chunks = 0
        chunk_details = []
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        for doc_info in documents:
            logger.info(f"Processing: {doc_info['filename']}")
            chunks = self._process_document(doc_info, text_splitter)
            
            if chunks:
                # Generate embeddings using Ollama
                texts = [c.text for c in chunks]
                embeddings = self._get_embeddings(texts)
                
                # Store in ChromaDB
                self._collection.add(
                    ids=[c.chunk_id for c in chunks],
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=[{
                        "source_file": c.source_file,
                        "page_number": c.page_number,
                        "chunk_index": c.chunk_index
                    } for c in chunks]
                )
                
                total_chunks += len(chunks)
                chunk_details.append({
                    "filename": doc_info["filename"],
                    "chunks": len(chunks)
                })
        
        # Update manifest
        self._index_manifest = {
            "built_at": datetime.now().isoformat(),
            "document_hashes": current_hashes,
            "total_chunks": total_chunks
        }
        
        return {
            "status": "rebuilt",
            "message": f"Index rebuilt with {total_chunks} chunks",
            "documents_processed": len(documents),
            "total_chunks": total_chunks,
            "chunk_details": chunk_details
        }
    
    def _detect_thai_text(self, text: str) -> bool:
        """Detect if text contains Thai characters."""
        if not text:
            return False
        thai_char_count = sum(1 for char in text if '\u0E00' <= char <= '\u0E7F')
        return thai_char_count > 10  # More than 10 Thai characters
    
    def _process_document(
        self, 
        doc_info: Dict[str, Any], 
        text_splitter: RecursiveCharacterTextSplitter
    ) -> List[Chunk]:
        """Process a single document into chunks with text cleaning and OCR fallback."""
        chunks = []
        skipped_chunks = 0
        ocr_pages = 0
        typhoon_pages = 0
        pdf_path = Path(doc_info["path"])
        
        # Check if this is a Thai PDF by filename or initial content check
        is_thai_pdf = 'thailand' in doc_info["filename"].lower() or 'thai' in doc_info["filename"].lower()
        
        # Get Typhoon OCR extractor if available and this is a Thai PDF
        typhoon_extractor = None
        if is_thai_pdf:
            typhoon_extractor = get_typhoon_ocr_extractor()
            if typhoon_extractor:
                logger.info(f"Using Typhoon OCR for Thai PDF: {doc_info['filename']}")
        
        try:
            loader = PyPDFLoader(doc_info["path"])
            pages = loader.load()
            
            for page_idx, page in enumerate(pages):
                page_num = page.metadata.get("page", 0) + 1
                
                # Clean the page content before splitting
                page_content_cleaned = self._clean_text(page.page_content)
                
                # Detect if page contains Thai text
                has_thai = self._detect_thai_text(page_content_cleaned)
                
                # For Thai PDFs with Typhoon available, always use Typhoon OCR for better quality
                if is_thai_pdf and typhoon_extractor and typhoon_extractor.is_available():
                    logger.info(f"Using Typhoon OCR for {doc_info['filename']} page {page_num} (Thai PDF)")
                    typhoon_text = typhoon_extractor.extract_text(pdf_path, page_idx)
                    if typhoon_text:
                        page_content_cleaned = self._clean_text(typhoon_text)
                        typhoon_pages += 1
                    # If Typhoon fails, keep the original extracted text
                    elif not page_content_cleaned:
                        logger.warning(f"Typhoon OCR failed for page {page_num}, no fallback text available")
                        continue
                else:
                    # Check if page is likely scanned (very little or poor quality text)
                    use_ocr = False
                    if not page_content_cleaned or len(page_content_cleaned) < 100:
                        use_ocr = True
                    elif not self._is_text_quality_acceptable(page_content_cleaned, min_length=50):
                        use_ocr = True
                    
                    # Use OCR for non-Thai PDFs or when Typhoon unavailable
                    if use_ocr and has_thai and typhoon_extractor:
                        logger.info(f"Using Typhoon OCR for {doc_info['filename']} page {page_num}")
                        typhoon_text = typhoon_extractor.extract_text(pdf_path, page_idx)
                        if typhoon_text:
                            page_content_cleaned = self._clean_text(typhoon_text)
                            typhoon_pages += 1
                    # Fallback to vision OCR for non-Thai or if Typhoon fails
                    elif use_ocr and PDF2IMAGE_AVAILABLE:
                        logger.info(f"Using vision OCR for {doc_info['filename']} page {page_num}")
                        ocr_text = self._extract_text_with_vision(pdf_path, page_idx)
                        if ocr_text:
                            page_content_cleaned = self._clean_text(ocr_text)
                            ocr_pages += 1
                
                if not page_content_cleaned:
                    logger.debug(f"Skipped empty page {page_num} from {doc_info['filename']}")
                    continue
                
                page_chunks = text_splitter.split_text(page_content_cleaned)
                
                for idx, chunk_text in enumerate(page_chunks):
                    chunk_text_stripped = chunk_text.strip()
                    
                    # Validate text quality before adding to index
                    if not self._is_text_quality_acceptable(chunk_text_stripped):
                        skipped_chunks += 1
                        logger.debug(
                            f"Skipped low-quality chunk from {doc_info['filename']} "
                            f"page {page_num} chunk {idx} (length: {len(chunk_text_stripped)})"
                        )
                        continue
                    
                    chunk_id = f"{doc_info['filename']}_p{page_num}_c{idx}"
                    chunks.append(Chunk(
                        chunk_id=chunk_id,
                        text=chunk_text_stripped,
                        source_file=doc_info["filename"],
                        page_number=page_num,
                        chunk_index=idx
                    ))
            
            # Log processing summary
            log_msg = f"Processed {doc_info['filename']}: {len(chunks)} chunks"
            if typhoon_pages > 0:
                log_msg += f", {typhoon_pages} pages via Typhoon OCR"
            if ocr_pages > 0:
                log_msg += f", {ocr_pages} pages via Vision OCR"
            if skipped_chunks > 0:
                log_msg += f", {skipped_chunks} skipped"
            logger.info(log_msg)
                
        except Exception as e:
            logger.error(f"Error processing {doc_info['filename']}: {e}", exc_info=True)
        
        return chunks
    
    def _is_index_current(self, current_hashes: Dict[str, str]) -> bool:
        """Check if index matches current document state."""
        if not self._index_manifest:
            return False
        stored_hashes = self._index_manifest.get("document_hashes", {})
        return stored_hashes == current_hashes
    
    def refresh_index(self) -> Dict[str, Any]:
        """
        Check index status and refresh if needed.
        
        Returns:
            Index status information.
        """
        self._ensure_initialized()
        
        count = self._collection.count()
        documents = self.list_documents()
        
        return {
            "status": "active" if count > 0 else "empty",
            "total_chunks": count,
            "documents_available": len(documents),
            "last_built": self._index_manifest.get("built_at", "unknown")
        }
    
    # ========== Retrieval Tools ==========
    
    def vector_search(
        self, 
        query: str, 
        top_k: int = None
    ) -> List[RetrievedChunk]:
        """
        Perform semantic search for relevant chunks.
        
        Args:
            query: The search query.
            top_k: Number of results to return (default from settings).
            
        Returns:
            List of retrieved chunks with similarity scores.
        """
        self._ensure_initialized()
        
        if top_k is None:
            top_k = settings.TOP_K_RESULTS
        
        # Check if index has data
        if self._collection.count() == 0:
            logger.warning("Index is empty. Build index first.")
            return []
        
        # Generate query embedding using Ollama
        query_embedding = self._get_embeddings([query])[0]
        
        # Search ChromaDB
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        retrieved_chunks = []
        for i in range(len(results["ids"][0])):
            chunk_id = results["ids"][0][i]
            text = results["documents"][0][i]
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            
            # Convert distance to similarity (cosine distance -> similarity)
            similarity = 1 - distance
            
            # Filter by threshold
            if similarity < settings.SIMILARITY_THRESHOLD:
                continue
            
            chunk = Chunk(
                chunk_id=chunk_id,
                text=text,
                source_file=metadata.get("source_file", "unknown"),
                page_number=metadata.get("page_number", 0),
                chunk_index=metadata.get("chunk_index", 0)
            )
            
            retrieved_chunks.append(RetrievedChunk(
                chunk=chunk,
                similarity_score=similarity,
                rank=i + 1
            ))
        
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks for query")
        return retrieved_chunks
    
    def get_chunk_text(self, chunk_id: str) -> Optional[Chunk]:
        """
        Fetch specific chunk text for citation verification.
        
        Args:
            chunk_id: The unique identifier of the chunk.
            
        Returns:
            The chunk if found, None otherwise.
        """
        self._ensure_initialized()
        
        try:
            result = self._collection.get(
                ids=[chunk_id],
                include=["documents", "metadatas"]
            )
            
            if result["ids"]:
                metadata = result["metadatas"][0]
                return Chunk(
                    chunk_id=chunk_id,
                    text=result["documents"][0],
                    source_file=metadata.get("source_file", "unknown"),
                    page_number=metadata.get("page_number", 0),
                    chunk_index=metadata.get("chunk_index", 0)
                )
        except Exception as e:
            logger.error(f"Error fetching chunk {chunk_id}: {e}")
        
        return None
    
    # ========== Grounding & Verification Tools ==========
    
    def verify_grounding(
        self, 
        answer: str, 
        chunks: List[RetrievedChunk]
    ) -> GroundingResult:
        """
        Verify that answer claims are supported by retrieved chunks.
        
        Args:
            answer: The generated answer text.
            chunks: The chunks used to generate the answer.
            
        Returns:
            GroundingResult with verification details.
        """
        if not chunks:
            return GroundingResult(
                is_grounded=False,
                supported_claims=[],
                unsupported_claims=["No evidence chunks provided"],
                confidence=0.0,
                evidence_chunks=[]
            )
        
        # Combine all chunk texts for verification
        evidence_text = " ".join([c.chunk.text.lower() for c in chunks])
        evidence_chunks = [c.chunk.chunk_id for c in chunks]
        
        # Simple claim extraction (sentences from answer)
        import re
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        supported = []
        unsupported = []
        
        for sentence in sentences:
            # Check if key terms from sentence appear in evidence
            words = set(re.findall(r'\b\w{4,}\b', sentence.lower()))
            evidence_words = set(re.findall(r'\b\w{4,}\b', evidence_text))
            
            overlap = len(words & evidence_words) / max(len(words), 1)
            
            if overlap >= 0.3:  # At least 30% word overlap
                supported.append(sentence)
            else:
                unsupported.append(sentence)
        
        # Calculate confidence
        total = len(supported) + len(unsupported)
        confidence = len(supported) / max(total, 1)
        
        # Consider grounded if most claims are supported
        is_grounded = confidence >= 0.7 and len(chunks) > 0
        
        return GroundingResult(
            is_grounded=is_grounded,
            supported_claims=supported,
            unsupported_claims=unsupported,
            confidence=confidence,
            evidence_chunks=evidence_chunks
        )
    
    def check_sufficient_evidence(
        self, 
        chunks: List[RetrievedChunk],
        min_chunks: int = 1,
        min_avg_similarity: float = 0.4
    ) -> Tuple[bool, str]:
        """
        Check if retrieved evidence is sufficient to answer.
        
        Args:
            chunks: Retrieved chunks.
            min_chunks: Minimum number of chunks required.
            min_avg_similarity: Minimum average similarity score.
            
        Returns:
            Tuple of (is_sufficient, reason).
        """
        if len(chunks) < min_chunks:
            return False, f"Insufficient evidence: only {len(chunks)} chunks found"
        
        avg_similarity = sum(c.similarity_score for c in chunks) / len(chunks)
        if avg_similarity < min_avg_similarity:
            return False, f"Low relevance: average similarity {avg_similarity:.2f} below threshold"
        
        return True, "Sufficient evidence available"


# Global tool instance
_tools: Optional[ToolInterface] = None


def get_tools() -> ToolInterface:
    """Get the global tool interface instance."""
    global _tools
    if _tools is None:
        _tools = ToolInterface()
    return _tools
