"""
Database inspection utility for debugging.

Use this to inspect the ChromaDB vector store contents.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.tools import get_tools


def main():
    """Inspect the vector database."""
    print("=" * 60)
    print("  ChromaDB Inspection Utility")
    print("=" * 60)
    
    tools = get_tools()
    
    # Get index status
    status = tools.refresh_index()
    print(f"\n📊 Index Status:")
    print(f"   Status: {status['status']}")
    print(f"   Total Chunks: {status['total_chunks']}")
    print(f"   Documents Available: {status['documents_available']}")
    
    # List documents
    docs = tools.list_documents()
    print(f"\n📁 Documents in Dataset:")
    for doc in docs:
        print(f"   • {doc['filename']}")
        print(f"     Size: {doc['size_bytes'] / 1024:.1f} KB")
        print(f"     Hash: {doc['file_hash']}")
    
    # Sample some chunks
    if status['total_chunks'] > 0:
        print(f"\n📝 Sample Chunks (first 3):")
        
        # Do a generic search to get some chunks
        chunks = tools.vector_search("security vulnerability", top_k=3)
        
        for i, rc in enumerate(chunks, 1):
            print(f"\n   [{i}] {rc.chunk.chunk_id}")
            print(f"       Source: {rc.chunk.source_file}, Page {rc.chunk.page_number}")
            print(f"       Similarity: {rc.similarity_score:.3f}")
            print(f"       Text: {rc.chunk.text[:150]}...")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
