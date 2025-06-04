from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List
import os
import torch
import logging
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# Recommended BGE model - requires internet connection to download on first run
# Consider specifying cache_folder for persistence
DEFAULT_EMBED_MODEL = "BAAI/bge-large-en-v1.5"
# Lighter alternative for faster local dev
# DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"

def get_embedding_model(model_name: str, device: str = None) -> HuggingFaceEmbeddings:
    """Initialize and return the embedding model."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading embedding model: {model_name} (using device: {device})")
    logger.info(f"Embedding model `get_embedding_model` function received device: {device}") # Added logging
    
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': True}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    logger.info(f"HuggingFaceEmbeddings initialized with model_kwargs: {model_kwargs}") # Added logging
    
    return embeddings

def create_and_save_vector_store(
    chunks: List[Document],
    embeddings: HuggingFaceEmbeddings,
    index_path: str
) -> None:
    """
    Creates a FAISS vector store from document chunks and saves it to disk.

    Args:
        chunks: List of LangChain Document objects (the chunks).
        embeddings: The initialized HuggingFace embedding model.
        index_path: The path where the FAISS index should be saved.
    """
    if not chunks:
        print("No chunks provided to create vector store. Skipping.")
        return

    print(f"Creating FAISS vector store from {len(chunks)} chunks...")
    # This can take some time depending on the number of chunks and embedding model
    vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
    print(f"Vector store created. Saving index to: {index_path}")

    # Ensure the directory exists
    index_dir = os.path.dirname(index_path)
    if index_dir:
        os.makedirs(index_dir, exist_ok=True)

    vector_store.save_local(index_path)
    print("FAISS index saved successfully.")

# Example usage (optional, can be triggered if run directly)
if __name__ == '__main__':
    print("Running embedding script directly (example mode)...")
    # This requires creating some dummy chunks for demonstration
    dummy_chunks = [
        Document(page_content="This is the first test document chunk.", metadata={"source": "doc1.pdf", "page": 1}),
        Document(page_content="This is the second chunk from another document.", metadata={"source": "doc2.pdf", "page": 5})
    ]
    print(f"Created {len(dummy_chunks)} dummy chunks for testing.")

    # 1. Get embeddings
    embed_model = get_embedding_model(DEFAULT_EMBED_MODEL) # Uses default model

    # 2. Define where to save the index
    save_path = "../vector_store/faiss_test_index" # Relative path assumes execution context
    print(f"Test index will be saved to: {save_path}")

    # 3. Create and save vector store
    create_and_save_vector_store(dummy_chunks, embed_model, save_path)

    print("Embedding script example finished.")