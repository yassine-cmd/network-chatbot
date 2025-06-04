import os
import sys
import time
import torch
import logging # Added for logging
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

# Ensure the ingestion directory and the project root are potentially findable
# This helps if the script is run from different locations
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(sys.stdout) # Log to console
                        # You can add logging.FileHandler("ingestion.log") here to log to a file
                    ])
logger = logging.getLogger(__name__)
# ---------------------

try:
    from load_pdf import load_pdfs_from_directory
    from chunking import chunk_extracted_data
    from embedding import get_embedding_model
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    logger.error("Please ensure the ingestion modules are in the correct directory structure.")
    sys.exit(1)

# Load environment variables
load_dotenv()

# --- Configuration --- 
# You might want to move these to a .env file or config management system
PDF_DIRECTORY = "data/" # Directory containing the PDFs
VECTOR_STORE_PATH = "vector_store/faiss_index" # Path to save the FAISS index
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5" # Or "all-MiniLM-L6-v2" for faster testing
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
# Automatically use CUDA if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Determined device for ingestion: {DEVICE}") # Added logging
def save_vector_store(vector_store: FAISS, path: str) -> None:
    """Save the FAISS vector store to disk."""
    try:
        logger.info(f"Saving vector store to {path}...")
        vector_store.save_local(path)
        logger.info("Vector store saved successfully.")
    except Exception as e:
        logger.error(f"Error saving vector store: {e}")
        raise

def main():
    """Runs the complete ingestion pipeline."""
    try:
        logger.info("--- Starting Ingestion Pipeline ---")

        # 1. Load PDFs
        logger.info("Stage 1: Loading PDFs...")
        start_time = time.time()
        documents = load_pdfs_from_directory(PDF_DIRECTORY)
        if not documents:
            logger.warning("No documents found to process. Exiting pipeline.")
            return
        logger.info(f"Loaded {len(documents)} documents in {time.time() - start_time:.2f} seconds.")

        # 2. Chunk documents
        logger.info("Stage 2: Chunking documents...")
        start_time = time.time()
        chunks = chunk_extracted_data(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        logger.info(f"Created {len(chunks)} chunks in {time.time() - start_time:.2f} seconds.")

        # 3. Load embedding model
        logger.info("Stage 3: Loading embedding model...")
        start_time = time.time()
        logger.info(f"Initializing embedding model with device: {DEVICE}") # Added logging
        embeddings = get_embedding_model(EMBEDDING_MODEL_NAME, DEVICE)
        logger.info(f"Embedding model '{EMBEDDING_MODEL_NAME}' loaded to {DEVICE} in {time.time() - start_time:.2f} seconds.")

        # 4. Create and save vector store
        logger.info("Stage 4: Creating and saving vector store...")
        start_time = time.time()
        vector_store = FAISS.from_documents(chunks, embeddings)
        save_vector_store(vector_store, VECTOR_STORE_PATH)
        logger.info(f"Vector store created and saved in {time.time() - start_time:.2f} seconds.")

        logger.info("--- Ingestion Pipeline Completed Successfully ---")
    except Exception as e:
        logger.error(f"Error in ingestion pipeline: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    # Ensure the target directory for the vector store exists
    vector_store_dir = os.path.dirname(VECTOR_STORE_PATH)
    if vector_store_dir and not os.path.exists(vector_store_dir):
        logger.info(f"Creating directory for vector store: {vector_store_dir}")
        os.makedirs(vector_store_dir)
    
    main()