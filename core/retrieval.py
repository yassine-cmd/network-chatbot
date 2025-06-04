import os
import sys
import torch
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List, Optional

# Adjust path to load embedding helper if needed (assuming it's in sibling ingestion folder)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    # Attempt to import the embedding function if available (for consistency)
    # We redefine get_embedding_model here for simplicity if direct import fails
    # or to avoid dependency loops if core becomes more complex
    from ingestion.embedding import get_embedding_model as get_embedding_model_from_ingestion
except ImportError:
    # Redefine locally if import fails or for decoupling
    def get_embedding_model_from_ingestion(model_name: str, device: str = None) -> HuggingFaceEmbeddings:
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading embedding model: {model_name} (using device: {device}) locally in retrieval.py")
        model_kwargs = {'device': device}
        encode_kwargs = {'normalize_embeddings': True}
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        print("Embedding model loaded.")
        return embeddings

# Load environment variables from .env file
load_dotenv()

VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "vector_store/faiss_index")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-large-en-v1.5")
# Automatically use CUDA if available
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", 5))

# --- Global Variables --- 
# Cache the loaded vector store and embeddings to avoid reloading on every query
_vector_store: Optional[FAISS] = None
_embeddings: Optional[HuggingFaceEmbeddings] = None

def _load_vector_store() -> Optional[FAISS]:
    """Load the FAISS vector store from disk, or return the cached instance."""
    global _vector_store, _embeddings

    if _vector_store is not None and _embeddings is not None:
        print("Vector store already loaded. Returning cached instance.")
        return _vector_store

    try:
        if not os.path.exists(VECTOR_STORE_PATH):
            print(f"Vector store not found at {VECTOR_STORE_PATH}")
            return None
        
        print(f"Loading vector store from {VECTOR_STORE_PATH}")
        print(f"Using device: {DEVICE}")
        _embeddings = get_embedding_model_from_ingestion(EMBEDDING_MODEL_NAME, DEVICE)
        _vector_store = FAISS.load_local(
            VECTOR_STORE_PATH, 
            _embeddings,
            allow_dangerous_deserialization=True  # Safe since we created this vector store
        )
        print("Vector store loaded successfully")
        return _vector_store
    except Exception as e:
        print(f"Error loading vector store: {str(e)}")
        _vector_store = None  # Reset on failure
        _embeddings = None
        return None

def get_retriever(k: int = RETRIEVAL_K) -> Optional[FAISS.as_retriever]:
    """Loads the vector store and returns a retriever object."""
    vector_store = _load_vector_store()
    if vector_store:
        return vector_store.as_retriever(search_kwargs={"k": k})
    return None

def search_documents(query: str, k: int = RETRIEVAL_K) -> List[Document]:
    """Search for similar documents using the query."""
    vector_store = _load_vector_store()
    if vector_store is None:
        return []
    
    try:
        results = vector_store.similarity_search(query, k=k)
        return results
    except Exception as e:
        print(f"Error during similarity search: {str(e)}")
        return []

# Example Usage (if run directly)
if __name__ == '__main__':
    print("Running retrieval script directly (example mode)...")
    
    # Ensure the store is loaded (or attempt to load)
    retriever = get_retriever()
    
    if retriever:
        test_query = "What is the main topic of the document?" # Replace with a relevant query for your PDFs
        print(f"\nTesting search with query: '{test_query}'")
        documents = search_documents(test_query)
        
        if documents:
            print("\n--- Top Retrieved Document Snippet ---")
            top_doc = documents[0]
            print(f"Source: {top_doc.metadata.get('source')}, Page: {top_doc.metadata.get('page')}")
            print(f"Content: {top_doc.page_content[:300]}...")
        else:
            print("Search returned no documents. Ensure the vector store exists and the query is relevant.")
    else:
        print("Failed to initialize retriever. Cannot perform search.")