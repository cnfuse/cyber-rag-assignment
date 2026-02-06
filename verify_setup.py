"""
Verify that all required models and services are available.

Run this script to check your setup before starting the RAG agent.
"""

import sys
import os

def check_ollama():
    """Check if Ollama is running and has the required model."""
    print("\n🔍 Checking Ollama...")
    
    try:
        import ollama
        
        # Check if Ollama is running
        try:
            models = ollama.list()
            print("  ✅ Ollama is running")
        except Exception as e:
            print(f"  ❌ Ollama is not running: {e}")
            print("     → Start with: ollama serve")
            return False
        
        # Check for the model
        from src.config import settings
        model_name = settings.OLLAMA_MODEL.split(":")[0]
        
        available_models = [m.get("name", "").split(":")[0] for m in models.get("models", [])]
        
        if model_name in available_models or settings.OLLAMA_MODEL in [m.get("name", "") for m in models.get("models", [])]:
            print(f"  ✅ Model '{settings.OLLAMA_MODEL}' is available")
            return True
        else:
            print(f"  ❌ Model '{settings.OLLAMA_MODEL}' not found")
            print(f"     → Install with: ollama pull {settings.OLLAMA_MODEL}")
            print(f"     Available models: {available_models}")
            return False
            
    except ImportError:
        print("  ❌ ollama package not installed")
        print("     → Install with: pip install ollama")
        return False


def check_embedding_model():
    """Check if the Ollama embedding model is available."""
    print("\n🔍 Checking Embedding Model...")
    
    try:
        import ollama
        from src.config import settings
        
        print(f"  📥 Testing {settings.EMBEDDING_MODEL}...")
        
        # Test embedding with Ollama
        response = ollama.embed(
            model=settings.EMBEDDING_MODEL,
            input="test query"
        )
        
        embedding_dim = len(response["embeddings"][0])
        print(f"  ✅ Ollama embedding model loaded (dimension: {embedding_dim})")
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to load embedding model: {e}")
        print(f"     → Pull the model: ollama pull {settings.EMBEDDING_MODEL}")
        return False


def check_chromadb():
    """Check if ChromaDB can be initialized."""
    print("\n🔍 Checking ChromaDB...")
    
    try:
        import chromadb
        from src.config import settings
        
        # Try to initialize
        settings.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(settings.CHROMA_PERSIST_DIR))
        
        # Get collection info
        try:
            collection = client.get_collection(settings.CHROMA_COLLECTION_NAME)
            count = collection.count()
            print(f"  ✅ ChromaDB initialized (collection: {count} chunks)")
        except:
            print("  ✅ ChromaDB ready (no existing collection)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ ChromaDB error: {e}")
        return False


def check_dataset():
    """Check if dataset files exist."""
    print("\n🔍 Checking Dataset...")
    
    from src.config import settings
    
    if not settings.DATASET_DIR.exists():
        print(f"  ❌ Dataset directory not found: {settings.DATASET_DIR}")
        return False
    
    pdf_files = list(settings.DATASET_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print("  ❌ No PDF files found in dataset directory")
        return False
    
    print(f"  ✅ Found {len(pdf_files)} documents:")
    for pdf in pdf_files:
        size_kb = pdf.stat().st_size / 1024
        print(f"     • {pdf.name} ({size_kb:.1f} KB)")
    
    return True


def check_dependencies():
    """Check if all required packages are installed."""
    print("\n🔍 Checking Dependencies...")
    
    required = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("chromadb", "ChromaDB"),
        ("langchain", "LangChain"),
        ("pypdf", "PyPDF"),
        ("ollama", "Ollama Client"),
        ("pydantic", "Pydantic"),
        ("structlog", "Structlog"),
    ]
    
    all_ok = True
    for package, name in required:
        try:
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} ({package})")
            all_ok = False
    
    if not all_ok:
        print("\n  → Install missing packages: pip install -r requirements.txt")
    
    return all_ok


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("  Cybersecurity RAG Agent - Setup Verification")
    print("=" * 60)
    
    results = {
        "Dependencies": check_dependencies(),
        "Dataset": check_dataset(),
        "ChromaDB": check_chromadb(),
        "Embedding Model": check_embedding_model(),
        "Ollama": check_ollama(),
    }
    
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    
    all_passed = True
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check}: {status}")
        if not passed:
            all_passed = False
    
    print()
    
    if all_passed:
        print("🎉 All checks passed! Ready to start the server:")
        print("   python -m src.api")
        return 0
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
