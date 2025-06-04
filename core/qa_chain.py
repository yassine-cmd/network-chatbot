import os
import sys
import time
import hashlib
import re
from operator import itemgetter
:start_line:7
-------
from typing import List, Dict, Any, AsyncIterator, Optional, Tuple
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import BaseMessage # Import BaseMessage for type hinting

# Adjust path for sibling imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from core.retrieval import get_retriever, search_documents # Use search_documents for explicit retrieval
    from core.llm_handler import get_llm, get_rag_prompt, format_chat_history as format_lc_history # Import format_chat_history
except ImportError as e:
    print(f"Error importing core modules: {e}")
    print("Ensure retrieval.py and llm_handler.py are in the core directory.")
    # Define dummies or raise error if essential components missing
    def get_retriever(k=None): return None
    def search_documents(q=None, k=None): return []
    def get_llm(): return None
    def get_rag_prompt(): return None
    def format_lc_history(history): return history # Dummy function
    # sys.exit(1)

def page_info_str_from_ui_item(ui_item: Dict[str, Any]) -> str:
    """
    Helper function to reconstruct a descriptive page info string from a UI source item.
    """
    page_val = ui_item.get('page', 'N/A')
    # Check if it looks like a simple number or comma-separated numbers
    if isinstance(page_val, str) and (page_val.isdigit() or ("," in page_val and all(p.strip().isdigit() for p in page_val.split(',')))):
        prefix = "Pages: " if "," in page_val else "Page: "
        return prefix + page_val
    # If it's already descriptive (like "Section: X" or "multiple"), return as is
    return page_val


def format_docs(docs: List[Document]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Formats retrieved documents into a single string for the prompt context,
    using numbered citations, and returns a list of unique source metadata for UI display.
    """
    if not docs:
        return "No relevant context found.", []

    formatted_docs_list = []
    unique_sources_for_ui = [] # To store unique sources for the UI list
    source_map = {} # To map (filename, page_info_str) to citation number
    citation_counter = 1

    for doc in docs:
        source_path = doc.metadata.get('source', 'Unknown Source')
        filename = os.path.basename(source_path)
        
        page_meta_val = doc.metadata.get('page')
        section_meta_val = doc.metadata.get('section')

        # Create a consistent string representation for page/section info for mapping
        page_info_str_for_map = "N/A"
        if section_meta_val:
            page_info_str_for_map = f"Section: {section_meta_val}"
            if isinstance(page_meta_val, int):
                 page_info_str_for_map += f" (from Page {page_meta_val})"
        elif isinstance(page_meta_val, int):
            page_info_str_for_map = f"Page: {page_meta_val}"
        elif isinstance(page_meta_val, (list, tuple)):
            page_info_str_for_map = f"Pages: {', '.join(map(str, page_meta_val))}"
        elif isinstance(page_meta_val, str) and ',' in page_meta_val:
            page_info_str_for_map = f"Pages: {', '.join(page_meta_val.split(','))}"
        elif isinstance(page_meta_val, str) and page_meta_val.lower() == "multiple":
            page_info_str_for_map = "Pages: multiple"
        
        # Create a cleaner string for UI display
        page_info_str_for_ui = page_info_str_for_map.replace("Page: ", "").replace("Pages: ", "").replace("Section: ", "")

        # Check if this exact source (filename + page_info_str_for_map) has been seen
        source_key = (filename, page_info_str_for_map)
        if source_key not in source_map:
            source_map[source_key] = citation_counter
            ui_source_entry = {
                "citation_number": citation_counter,
                "source": filename,
                "page": page_info_str_for_ui # Use the cleaner string for UI
            }
            # Add other relevant metadata if needed for UI
            if 'page_number_original' in doc.metadata:
                 ui_source_entry['original_page'] = doc.metadata['page_number_original']
            unique_sources_for_ui.append(ui_source_entry)
            citation_number_for_context = citation_counter
            citation_counter += 1
        else:
            citation_number_for_context = source_map[source_key]
        
        content = doc.page_content
        # Format for LLM context with numbered citation
        formatted_docs_list.append(f"Content [{citation_number_for_context}]: {content}")

    # Create the context string for the LLM
    context_str = "\n\n---\n\n".join(formatted_docs_list)
    
    # Add a reference list at the end of the context for the LLM to see the format
    # This helps the LLM understand how to map its generated [N] to the source details.
    # This part is for the LLM's consumption. The UI will get unique_sources_for_ui.
    if unique_sources_for_ui:
        context_str += "\n\n---Source References---\n"
        # Use the original page_info_str_for_map for the LLM's reference list to be more precise
        # Need to retrieve the original page_info_str_for_map based on the source_key
        # This requires iterating through the docs again or storing page_info_str_for_map in unique_sources_for_ui
        # Let's store page_info_str_for_map in unique_sources_for_ui for simplicity
        # Modify ui_source_entry creation above to include 'page_info_for_llm_ref'

        # Re-iterate unique_sources_for_ui to build the LLM reference list
        # (Assuming ui_source_entry now includes 'page_info_for_llm_ref')
        # Let's adjust the ui_source_entry creation to store both UI and LLM versions of page info.
        # For now, let's use the page_info_str_for_map directly from the source_map lookup
        # This requires a reverse lookup or storing it in unique_sources_for_ui

        # Let's simplify and use the page_info_str_for_map from the original loop,
        # which means we need to map citation number back to the original page_info_str_for_map.
        # A better approach is to store the LLM reference string directly in unique_sources_for_ui.

        # Let's modify the ui_source_entry to store the detailed page info for the LLM reference list
        # and the cleaner one for the UI.

        # --- Revised ui_source_entry creation ---
        # Inside the loop, when source_key not in source_map:
        # ui_source_entry = {
        #     "citation_number": citation_counter,
        #     "source": filename,
        #     "page": page_info_str_for_ui, # Cleaner for UI
        #     "page_info_for_llm_ref": page_info_str_for_map # Detailed for LLM reference list
        # }
        # unique_sources_for_ui.append(ui_source_entry)
        # citation_number_for_context = citation_counter
        # citation_counter += 1
        # --- End Revised ui_source_entry creation ---

        # Now, build the LLM reference list using 'page_info_for_llm_ref'
        # This requires re-running the format_docs logic or adapting the loop.
        # Let's adapt the loop to build both lists simultaneously.

        # --- Re-Revised format_docs logic ---
        # (Replace the entire format_docs function with the one below)
        # --- End Re-Revised format_docs logic ---

        # Since I cannot replace the entire function in one go easily with apply_diff,
        # I will apply the changes incrementally based on the provided code block.
        # The provided code block for format_docs seems to have a mix of logic and comments.
        # I will use the provided code block as the source for the replacement.

        # The provided code block for format_docs includes the helper function page_info_str_from_ui_item.
        # I will add this helper function before format_docs.

        # The provided format_docs logic seems to use the 'page' field from ui_source_entry
        # for the LLM reference list, which is the cleaner version. Let's stick to that for now
        # as per the provided code.

        for src_item in unique_sources_for_ui:
            # Use the 'page' field from ui_source_entry which is already formatted for UI
            # The provided code had a helper call here, but then commented it out and used the 'page' field.
            # Let's use the 'page' field as in the final provided code snippet.
            context_str += f"[{src_item['citation_number']}] Source: {src_item['source']}, Page Info: {src_item['page']}\n"


    return context_str, unique_sources_for_ui


# Helper function (can be inside qa_chain.py or a utils module)
# This helper was included in the format_docs section in the prompt,
# but it makes more sense as a separate function.
# I will add it before format_docs.
# def page_info_str_from_ui_item(ui_item: Dict[str, Any]) -> str:
#     # This function would reconstruct the 'Page: X' or 'Section: Y' string
#     # based on how 'page' is stored in ui_item if it was simplified too much.
#     # For now, let's assume ui_item['page'] is descriptive enough.
#     # Example: if ui_item['page'] is "5" it becomes "Page: 5"
#     # if ui_item['page'] is "Section X (from Page 5)" it's already good.
#     page_val = ui_item['page']
#     if isinstance(page_val, str) and (page_val.isdigit() or ("," in page_val and all(p.strip().isdigit() for p in page_val.split(',')))):
#         prefix = "Pages: " if "," in page_val else "Page: "
#         return prefix + page_val
#     return page_val # If it's already like "Section: ..." or "multiple"


# Modify _format_inputs in create_rag_chain
# ... inside create_rag_chain() ...
    def _format_inputs(input_dict):
        query = input_dict.get("query", "")
        chat_history = input_dict.get("chat_history", []) # This is already formatted Langchain messages
        
        # Get context from retriever - this is where the change happens
        context_docs = retriever.invoke(query) # This remains the same
        # format_docs now returns context_string and source_metadata_for_ui
        context_string, _ = format_docs(context_docs) # We only need context_string for the prompt

        return {
            "context": context_string,
            "question": query,
            "chat_history": chat_history
        }
# ...

# Modify get_answer_stream to handle the new source format
# ... inside get_answer_stream() ...
    # 1. Retrieve sources
    retrieved_docs_for_llm_context = []
    ui_source_list = [] # This will hold the [{citation_number: N, source: S, page: P}, ...]

    try:
        if is_conversational or is_non_networking:
            print(f"Skipping document retrieval for {'conversational' if is_conversational else 'non-networking'} query.")
            yield {"type": "sources", "data": []} # Send empty UI sources
        elif cached_docs: # cached_docs would be [(original_query_text, Document_object), ...]
            # We need to re-process these cached Documents with the new format_docs
            # to get consistent citation numbers and UI source list for *this specific query*.
            # The cache stores raw Documents.
            cached_document_objects = [doc_obj for _, doc_obj in cached_docs]
            context_string_from_cache, ui_source_list_from_cache = format_docs(cached_document_objects) # This re-numbers based on current set
            
            retrieved_docs_for_llm_context = cached_document_objects # These are the docs for LLM RAG chain
            ui_source_list = ui_source_list_from_cache # Update ui_source_list
            print(f"Using {len(retrieved_docs_for_llm_context)} documents from cache.")
            yield {"type": "sources", "data": ui_source_list} # Yield the new UI source list
        else:
            print(f"Retrieving sources for query: {query[:50]}...")
            start_time = time.time()
            # source_chain.invoke(query) should return List[Document]
            freshly_retrieved_docs = source_chain.invoke(query)
            retrieval_time = time.time() - start_time
            print(f"Retrieved {len(freshly_retrieved_docs)} documents in {retrieval_time:.2f} seconds.")

            # Cache the raw Document objects
            _doc_cache[cache_key] = (time.time(), [(query, doc) for doc in freshly_retrieved_docs])
            
            # format_docs will generate the UI source list with new citation numbers
            context_string_from_fresh, ui_source_list_from_fresh = format_docs(freshly_retrieved_docs)
            retrieved_docs_for_llm_context = freshly_retrieved_docs # Docs for LLM RAG chain
            ui_source_list = ui_source_list_from_fresh # Update ui_source_list
            yield {"type": "sources", "data": ui_source_list} # Yield the new UI source list

    except Exception as e:
        print(f"Error retrieving/formatting sources: {e}")
        yield {"type": "error", "data": f"Failed to process sources: {e}"}
        # Ensure downstream RAG chain gets an empty context or appropriate handling
        # The RAG chain's _format_inputs will call format_docs with an empty list,
        # which will return "No relevant context found."
        retrieved_docs_for_llm_context = [] # Ensure empty list on error
        ui_source_list = [] # Ensure empty list on error


    # 2. Stream the LLM answer
    try:
        print("Streaming answer from LLM...")
        start_time = time.time()

        # For conversational or non-networking queries, we can use direct LLM streaming
        if is_conversational or is_non_networking:
            # Import the direct LLM streaming function
            from core.llm_handler import stream_llm_response

            print(f"Using direct LLM streaming for {'conversational' if is_conversational else 'non-networking'} query...")
            
            # For non-networking queries, add a note to the query to ensure the LLM declines to answer
            if is_non_networking:
                query = f"{query}\n\nNote: This query appears to be outside the networking domain. Please politely decline to answer as per your instructions."
                
            async for chunk in stream_llm_response(query, chat_history):
                yield {"type": "token", "data": chunk}
        else:
            # For network-related queries, use the RAG chain
            # Prepare input with query and chat history
            chain_input = {
                "query": query,
                # Format chat_history for the RAG chain if it's not already Langchain messages
                # Assuming llm_handler.format_chat_history is available and takes List[Dict]
                "chat_history": format_lc_history(chat_history if chat_history else [])
            }

            # OPTIMIZATION NOTE: The current RAG chain structure (_format_inputs)
            # will call retriever.invoke(query) again, potentially re-retrieving documents.
            # To avoid this, the RAG chain's _format_inputs would need to accept
            # the pre-fetched context string directly.
            # For now, we proceed with the current chain structure, relying on the
            # retriever's internal caching or the _doc_cache in this file to mitigate.

            async for chunk in rag_chain.astream(chain_input):
                yield {"type": "token", "data": chunk}

        generation_time = time.time() - start_time
        print(f"Finished streaming answer in {generation_time:.2f} seconds.")
    except Exception as e:
        print(f"Error streaming LLM response: {e}")
        yield {"type": "error", "data": f"Error generating answer: {e}"}

# Helper function (can be inside qa_chain.py or a utils module)
# This helper was included in the format_docs section in the prompt,
# but it makes more sense as a separate function.
# I will add it before format_docs.
# def page_info_str_from_ui_item(ui_item: Dict[str, Any]) -> str:
#     # This function would reconstruct the 'Page: X' or 'Section: Y' string
#     # based on how 'page' is stored in ui_item if it was simplified too much.
#     # For now, let's assume ui_item['page'] is descriptive enough.
#     # Example: if ui_item['page'] is "5" it becomes "Page: 5"
#     # if ui_item['page'] is "Section X (from Page 5)" it's already good.
#     page_val = ui_item['page']
#     if isinstance(page_val, str) and (page_val.isdigit() or ("," in page_val and all(p.strip().isdigit() for p in page_val.split(',')))):
#         prefix = "Pages: " if "," in page_val else "Page: "
#         return prefix + page_val
#     return page_val # If it's already like "Section: ..." or "multiple"


# Example Usage
if __name__ == "__main__":
    import asyncio

    async def run_example():
        print("Running QA chain script directly (example mode)...")
        # Ensure vector store exists (run ingestion/run_ingestion.py first)

        # Test without conversation history
        test_query = "What is the role of the embedding model in RAG?" # Change to a query relevant to your docs
        print(f"\nTesting QA stream with query: '{test_query}'")

        final_answer = ""
        sources_found = []
        async for event in get_answer_stream(test_query):
            if event['type'] == 'sources':
                sources_found = event['data']
                print(f"\nSources Found:")
                # Now sources_found is a list of dicts with citation_number
                for source_meta in sources_found:
                    print(f"  - [{source_meta.get('citation_number', 'N/A')}] {source_meta.get('source', 'Unknown')} (Page Info: {source_meta.get('page', 'N/A')})")
                print("\nStreaming Answer:")
            elif event['type'] == 'token':
                final_answer += event['data']
                print(event['data'], end="", flush=True)
            elif event['type'] == 'error':
                print(f"\nError: {event['data']}")
                break # Stop processing on error

        print("\n\n--- Stream Complete ---")
        print(f"Final Answer: {final_answer}")
        print(f"Sources: {sources_found}")

        # Test with conversation history
        print("\n\n--- Testing with conversation history ---")
        test_history = [
            {"role": "user", "content": "What is RAG?"},
            {"role": "assistant", "content": "RAG (Retrieval-Augmented Generation) is a technique that combines retrieval of relevant documents with text generation."}
        ]

        follow_up_query = "How does it improve AI responses?"
        print(f"\nTesting follow-up query: '{follow_up_query}'")
        print("With conversation history:")
        for msg in test_history:
            print(f"{msg['role'].upper()}: {msg['content']}")

        final_answer = ""
        sources_found = []
        async for event in get_answer_stream(follow_up_query, test_history):
            if event['type'] == 'sources':
                sources_found = event['data']
                print(f"\nSources Found:")
                # Now sources_found is a list of dicts with citation_number
                for source_meta in sources_found:
                    print(f"  - [{source_meta.get('citation_number', 'N/A')}] {source_meta.get('source', 'Unknown')} (Page Info: {source_meta.get('page', 'N/A')})")
                print("\nStreaming Answer:")
            elif event['type'] == 'token':
                final_answer += event['data']
                print(event['data'], end="", flush=True)
            elif event['type'] == 'error':
                print(f"\nError: {event['data']}")
                break

        print("\n\n--- Stream with History Complete ---")
        print(f"Final Answer: {final_answer}")
        print(f"Sources: {sources_found}")

        # Test conversational query
        print("\n\n--- Testing conversational query ---")
        conversational_query = "Hello, what's your name?"
        print(f"\nTesting conversational query: '{conversational_query}'")

        final_answer = ""
        sources_found = []
        async for event in get_answer_stream(conversational_query):
            if event['type'] == 'sources':
                sources_found = event['data']
                if sources_found:
                    print(f"\nSources Found:")
                    # Now sources_found is a list of dicts with citation_number
                    for source_meta in sources_found:
                        print(f"  - [{source_meta.get('citation_number', 'N/A')}] {source_meta.get('source', 'Unknown')} (Page Info: {source_meta.get('page', 'N/A')})")
                else:
                    print("\nNo sources needed for conversational query.")
                print("\nStreaming Answer:")
            elif event['type'] == 'token':
                final_answer += event['data']
                print(event['data'], end="", flush=True)
            elif event['type'] == 'error':
                print(f"\nError: {event['data']}")
                break

        print("\n\n--- Conversational Query Complete ---")
        print(f"Final Answer: {final_answer}")

        # Test follow-up to conversational query
        print("\n\n--- Testing follow-up to conversational query ---")
        conv_history = [
            {"role": "user", "content": conversational_query},
            {"role": "assistant", "content": final_answer}
        ]

        follow_up_conv = "What can you help me with?"
        print(f"\nTesting follow-up query: '{follow_up_conv}'")

        final_answer = ""
        sources_found = []
        async for event in get_answer_stream(follow_up_conv, conv_history):
            if event['type'] == 'sources':
                sources_found = event['data']
                if sources_found:
                    print(f"\nSources Found:")
                    # Now sources_found is a list of dicts with citation_number
                    for source_meta in sources_found:
                        print(f"  - [{source_meta.get('citation_number', 'N/A')}] {source_meta.get('source', 'Unknown')} (Page Info: {source_meta.get('page', 'N/A')})")
                else:
                    print("\nNo sources needed for conversational query.")
                print("\nStreaming Answer:")
            elif event['type'] == 'token':
                final_answer += event['data']
                print(event['data'], end="", flush=True)
            elif event['type'] == 'error':
                print(f"\nError: {event['data']}")
                break

        print("\n\n--- Conversational Follow-up Complete ---")
        print(f"Final Answer: {final_answer}")

    asyncio.run(run_example())


# Example Usage
if __name__ == "__main__":
    import asyncio

    async def run_example():
        print("Running QA chain script directly (example mode)...")
        # Ensure vector store exists (run ingestion/run_ingestion.py first)

        # Test without conversation history
        test_query = "What is the role of the embedding model in RAG?" # Change to a query relevant to your docs
        print(f"\nTesting QA stream with query: '{test_query}'")

        final_answer = ""
        sources_found = []
        async for event in get_answer_stream(test_query):
            if event['type'] == 'sources':
                sources_found = event['data']
                print(f"\nSources Found:")
                for source_meta in sources_found:
                    print(f"  - {source_meta.get('source', 'Unknown')} (Page {source_meta.get('page', 'N/A')})")
                print("\nStreaming Answer:")
            elif event['type'] == 'token':
                final_answer += event['data']
                print(event['data'], end="", flush=True)
            elif event['type'] == 'error':
                print(f"\nError: {event['data']}")
                break # Stop processing on error

        print("\n\n--- Stream Complete ---")
        print(f"Final Answer: {final_answer}")
        print(f"Sources: {sources_found}")

        # Test with conversation history
        print("\n\n--- Testing with conversation history ---")
        test_history = [
            {"role": "user", "content": "What is RAG?"},
            {"role": "assistant", "content": "RAG (Retrieval-Augmented Generation) is a technique that combines retrieval of relevant documents with text generation."}
        ]

        follow_up_query = "How does it improve AI responses?"
        print(f"\nTesting follow-up query: '{follow_up_query}'")
        print("With conversation history:")
        for msg in test_history:
            print(f"{msg['role'].upper()}: {msg['content']}")

        final_answer = ""
        sources_found = []
        async for event in get_answer_stream(follow_up_query, test_history):
            if event['type'] == 'sources':
                sources_found = event['data']
                print(f"\nSources Found:")
                for source_meta in sources_found:
                    print(f"  - {source_meta.get('source', 'Unknown')} (Page {source_meta.get('page', 'N/A')})")
                print("\nStreaming Answer:")
            elif event['type'] == 'token':
                final_answer += event['data']
                print(event['data'], end="", flush=True)
            elif event['type'] == 'error':
                print(f"\nError: {event['data']}")
                break

        print("\n\n--- Stream with History Complete ---")
        print(f"Final Answer: {final_answer}")
        print(f"Sources: {sources_found}")

        # Test conversational query
        print("\n\n--- Testing conversational query ---")
        conversational_query = "Hello, what's your name?"
        print(f"\nTesting conversational query: '{conversational_query}'")

        final_answer = ""
        sources_found = []
        async for event in get_answer_stream(conversational_query):
            if event['type'] == 'sources':
                sources_found = event['data']
                if sources_found:
                    print(f"\nSources Found:")
                    for source_meta in sources_found:
                        print(f"  - {source_meta.get('source', 'Unknown')} (Page {source_meta.get('page', 'N/A')})")
                else:
                    print("\nNo sources needed for conversational query.")
                print("\nStreaming Answer:")
            elif event['type'] == 'token':
                final_answer += event['data']
                print(event['data'], end="", flush=True)
            elif event['type'] == 'error':
                print(f"\nError: {event['data']}")
                break

        print("\n\n--- Conversational Query Complete ---")
        print(f"Final Answer: {final_answer}")

        # Test follow-up to conversational query
        print("\n\n--- Testing follow-up to conversational query ---")
        conv_history = [
            {"role": "user", "content": conversational_query},
            {"role": "assistant", "content": final_answer}
        ]

        follow_up_conv = "What can you help me with?"
        print(f"\nTesting follow-up query: '{follow_up_conv}'")

        final_answer = ""
        sources_found = []
        async for event in get_answer_stream(follow_up_conv, conv_history):
            if event['type'] == 'sources':
                sources_found = event['data']
                if sources_found:
                    print(f"\nSources Found:")
                    for source_meta in sources_found:
                        print(f"  - {source_meta.get('source', 'Unknown')} (Page {source_meta.get('page', 'N/A')})")
                else:
                    print("\nNo sources needed for conversational query.")
                print("\nStreaming Answer:")
            elif event['type'] == 'token':
                final_answer += event['data']
                print(event['data'], end="", flush=True)
            elif event['type'] == 'error':
                print(f"\nError: {event['data']}")
                break

        print("\n\n--- Conversational Follow-up Complete ---")
        print(f"Final Answer: {final_answer}")

    # Need to run this in an async context
    asyncio.run(run_example())
