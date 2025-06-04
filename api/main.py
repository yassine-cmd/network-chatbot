import sys
import os
import json
import time
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

# Adjust path to import from core
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from core.qa_chain import get_answer_stream
    from core.retrieval import _load_vector_store  # Import for preloading
except ImportError:
    print("Error importing qa_chain. Ensure core modules are accessible.")
    # Define dummy or exit
    async def get_answer_stream(query: str):
        yield {"type": "error", "data": "QA Chain not loaded."}
    # sys.exit(1)

# --- FastAPI App Setup ---
app = FastAPI(
    title="PDF Chatbot API",
    description="API for querying PDF documents using RAG."
)

# Configure CORS
origins = [
    "http://localhost",       # Default Streamlit
    "http://localhost:8501",  # Default Streamlit port
    # Add any other origins if your frontend is hosted elsewhere
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Startup Event to Preload Models ---
@app.on_event("startup")
async def startup_event():
    """Preload embedding model and vector store on application startup."""
    print("Application starting up. Preloading models...")
    try:
        # Run the synchronous function in a thread pool to avoid blocking the event loop
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        # Create a thread pool and run the synchronous function in it
        with ThreadPoolExecutor() as executor:
            await asyncio.get_event_loop().run_in_executor(executor, _load_vector_store)
            
        print("Embedding model and vector store preloaded successfully.")
    except Exception as e:
        print(f"Error during model preloading: {e}")
        # Depending on the criticality, you might want to raise the error
        # or allow the app to start with a warning.
        # For now, we'll just print the error.

# --- Performance Metrics ---
# Store recent query performance metrics
_performance_metrics: List[Dict[str, Any]] = []
MAX_METRICS_HISTORY = 50

# --- Request Models ---
class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    query: str
    chat_history: Optional[List[Message]] = None

class PerformanceMetrics(BaseModel):
    total_queries: int
    avg_response_time: float
    recent_queries: List[Dict[str, Any]]

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "PDF Chatbot API is running."}

@app.get("/metrics")
def get_metrics() -> PerformanceMetrics:
    """Returns performance metrics for the API."""
    if not _performance_metrics:
        return PerformanceMetrics(
            total_queries=0,
            avg_response_time=0.0,
            recent_queries=[]
        )

    total_queries = len(_performance_metrics)
    avg_response_time = sum(m.get("total_time", 0) for m in _performance_metrics) / total_queries

    return PerformanceMetrics(
        total_queries=total_queries,
        avg_response_time=avg_response_time,
        recent_queries=_performance_metrics[-10:]  # Return the 10 most recent queries
    )

async def stream_response_generator(query: str, chat_history: Optional[List[Dict[str, str]]] = None):
    """
    Generator function to yield formatted Server-Sent Events (SSE).

    Args:
        query: The user's question
        chat_history: Optional list of previous messages in the conversation
    """
    start_time = time.time()
    first_token_time = None
    metrics = {
        "query": query[:100] + "..." if len(query) > 100 else query,
        "timestamp": time.time(),
        "tokens_streamed": 0,
        "has_history": chat_history is not None and len(chat_history) > 0
    }

    # Convert chat history to the format expected by the QA chain
    formatted_history = None
    if chat_history:
        formatted_history = [{"role": msg.role, "content": msg.content} for msg in chat_history]
        metrics["history_length"] = len(formatted_history)

    try:
        async for event in get_answer_stream(query, formatted_history):
            # Track metrics
            if event["type"] == "token":
                metrics["tokens_streamed"] += 1
                if first_token_time is None:
                    first_token_time = time.time()
                    metrics["time_to_first_token"] = first_token_time - start_time
            elif event["type"] == "sources":
                metrics["num_sources"] = len(event["data"])
                metrics["sources_time"] = time.time() - start_time

            # Format as SSE: data: {json_payload}\n\n
            yield f"data: {json.dumps(event)}\n\n"
    finally:
        # Record final metrics
        end_time = time.time()
        metrics["total_time"] = end_time - start_time

        if first_token_time is None:
            metrics["time_to_first_token"] = 0

        # Add to metrics history
        global _performance_metrics
        _performance_metrics.append(metrics)

        # Limit the size of metrics history
        if len(_performance_metrics) > MAX_METRICS_HISTORY:
            _performance_metrics = _performance_metrics[-MAX_METRICS_HISTORY:]

        print(f"Query completed in {metrics['total_time']:.2f}s with {metrics['tokens_streamed']} tokens")

@app.post("/stream_query/")
async def stream_query(request: QueryRequest):
    """
    Accepts a query and optional chat history, and streams back the sources and answer tokens
    using Server-Sent Events (SSE).
    """
    # Log the request details
    history_info = f" with {len(request.chat_history)} previous messages" if request.chat_history else " (new conversation)"
    print(f"Received query: {request.query[:50]}...{history_info}")

    return StreamingResponse(
        stream_response_generator(request.query, request.chat_history),
        media_type="text/event-stream"
    )

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting FastAPI server...")
    # Models are now preloaded via the startup_event
    # Run using uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # To run from command line: uvicorn api.main:app --reload --port 8000