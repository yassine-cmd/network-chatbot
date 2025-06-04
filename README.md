# Chatbot Application

This project implements a chatbot application capable of answering questions based on ingested PDF documents. It leverages Large Language Models (LLMs), vector stores, and a robust ingestion pipeline to provide a comprehensive Q&A experience.

## Features

*   **PDF Ingestion:** Process and chunk PDF documents for retrieval.
*   **Local LLM Inference:** Support for running LLMs locally using `llama-cpp-python`.
*   **Vector Store:** Utilizes FAISS for efficient similarity search and retrieval of relevant document chunks.
*   **Retrieval-Augmented Generation (RAG):** Combines document retrieval with LLM capabilities for accurate and context-aware answers.
*   **Web Interface:** A user-friendly interface built with Streamlit for interacting with the chatbot.
*   **API Endpoint:** A FastAPI backend for programmatic access to the chatbot functionalities.

## Setup

Follow these steps to set up and run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/chatbotBackup.git
cd chatbotBackup
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: `venv\Scripts\activate`
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Variables

Create a `.env` file in the root directory of the project and add any necessary API keys or configurations. For example, if you are using Google GenAI, you might need:

```
GOOGLE_API_KEY="your_google_api_key_here"
```

### 5. Ingest Data

Place your PDF documents in the `data/` directory. Then, run the ingestion script to process them and create the FAISS vector store:

```bash
python ingestion/run_ingestion.py
```

This will create `index.faiss` and `index.pkl` files in the `vector_store/faiss_index/` directory.

## Running the Application

You can run the application either as a web interface (Streamlit) or as a FastAPI backend.

### 1. Web Interface (Streamlit)

To launch the Streamlit web application:

```bash
streamlit run app.py
```

This will open the chatbot interface in your web browser.

### 2. API Backend (FastAPI)

To start the FastAPI server:

```bash
uvicorn api.main:app --reload
```

The API will be accessible at `http://127.0.0.1:8000`. You can interact with the API using tools like Postman or `curl`, or by navigating to `http://127.0.0.1:8000/docs` for the interactive OpenAPI documentation (Swagger UI).

## Project Structure

```
.
├── .env                      # Environment variables
├── .gitignore                # Git ignore file
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
├── test_cuda.py              # Script to test CUDA
├── api/
│   └── main.py               # FastAPI application
├── core/
│   ├── llm_handler.py        # Handles LLM interactions
│   ├── qa_chain.py           # Defines the Q&A chain
│   └── retrieval.py          # Document retrieval logic
├── data/
│   └── Ebook_Architecture_des_reseaux_pdf.pdf # Example PDF document
├── ingestion/
│   ├── chunking.py           # Document chunking logic
│   ├── embedding.py          # Handles text embeddings
│   ├── load_pdf.py           # PDF loading utility
│   └── run_ingestion.py      # Script to run data ingestion
└── vector_store/
    └── faiss_index/
        ├── index.faiss       # FAISS vector store index
        └── index.pkl         # FAISS index metadata
```

## Dependencies

The project's dependencies are listed in `requirements.txt`:

*   `langchain`
*   `langchain-community`
*   `langchain-core`
*   `langchain-google-genai`
*   `llama-cpp-python`
*   `langchain-text-splitters`
*   `faiss-cpu`
*   `torch`
*   `sentence-transformers`
*   `langchain-huggingface`
*   `pypdf`
*   `pdf2image`
*   `pytesseract`
*   `Pillow`
*   `fastapi`
*   `uvicorn`
*   `streamlit`
*   `python-dotenv`
*   `pydantic`
*   `tqdm`
*   `numpy`
*   `pandas`