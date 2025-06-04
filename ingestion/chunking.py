from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from typing import List, Dict, Tuple, Any
import sys
import os
import re
import hashlib

# Ensure the parent directory is in the Python path
# to allow importing 'load_pdf' from the sibling 'load_pdf.py' file
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Conditional import for example usage
if __name__ == '__main__':
    try:
        from ingestion.load_pdf import load_pdfs_from_directory
    except ImportError:
        print("Error: Could not import 'load_pdfs_from_directory'. Ensure 'load_pdf.py' is in the same directory or accessible via PYTHONPATH.")
        # Define a dummy function or exit if essential for standalone execution test
        def load_pdfs_from_directory(path=None): return [] # Dummy for static analysis
        # sys.exit(1) # Uncomment to make standalone execution fail if import fails


def detect_headers(text: str) -> bool:
    """
    Detect if the text contains potential headers that could be used for semantic chunking.

    Args:
        text: The text to analyze

    Returns:
        True if headers are detected, False otherwise
    """
    # Look for patterns like "Chapter X", "Section X", numbered lists, etc.
    header_patterns = [
        r'\n\s*#+\s+',  # Markdown headers
        r'\n\s*[A-Z][A-Za-z\s]+:',  # Title followed by colon
        r'\n\s*[IVX]+\.\s+',  # Roman numerals
        r'\n\s*\d+\.\s+[A-Z]',  # Numbered sections starting with capital
        r'\n\s*Chapter\s+\d+',  # Chapter headings
        r'\n\s*Section\s+\d+',  # Section headings
        r'\n\s*[A-Z][A-Z\s]+\n'  # All caps headings
    ]

    for pattern in header_patterns:
        if re.search(pattern, text):
            return True

    return False

def chunk_extracted_data(
    extracted_data: List[Tuple[str, List[Dict[str, Any]]]],
    chunk_size: int = 1000,
    chunk_overlap: int = 100
) -> List[Document]:
    """
    Splits the text content from extracted PDF data into chunks using intelligent chunking strategies.
    Uses header-based chunking when possible, falling back to recursive character splitting.

    Args:
        extracted_data: The list of tuples [(filename, pages_content), ...] from load_pdfs_from_directory.
        chunk_size: The target size for each chunk (in characters).
        chunk_overlap: The overlap between consecutive chunks (in characters).

    Returns:
        A list of LangChain Document objects, where each Document represents a chunk
        and contains the text and metadata (source filename, page number).
    """
    # Create the recursive splitter as a fallback
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    # Create a header-based splitter for semantic chunking
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5"),
        ("######", "Header 6"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )

    all_chunks = []
    print(f"Starting enhanced chunking process with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}...")

    for filename, pages_content in extracted_data:
        print(f"Chunking document: {filename}")
        num_pages_processed = 0

        # First, try to process the document as a whole for better semantic chunking
        full_document_text = ""
        for page_info in pages_content:
            page_text = page_info['text']
            if page_text.strip():
                full_document_text += f"\n\n{page_text}"

        # Check if the document has headers for semantic chunking
        has_headers = detect_headers(full_document_text)

        if has_headers and len(full_document_text) > 0:
            print(f"  Using semantic header-based chunking for {filename}")

            # Convert potential headers to markdown format for the splitter
            # This is a simple heuristic - could be improved with more sophisticated header detection
            processed_text = full_document_text
            # Convert "Chapter X" or "Section X" to markdown headers
            processed_text = re.sub(r'\n\s*(Chapter|Section)\s+(\d+)[:\s]*([^\n]+)', r'\n# \1 \2: \3', processed_text)
            # Convert "X.Y Title" patterns to headers
            processed_text = re.sub(r'\n\s*(\d+)\.(\d+)\s+([A-Z][^\n]+)', r'\n## \1.\2 \3', processed_text)
            # Convert "X. Title" patterns to headers
            processed_text = re.sub(r'\n\s*(\d+)\.\s+([A-Z][^\n]+)', r'\n# \1. \2', processed_text)

            try:
                # Try to split by headers
                md_header_splits = markdown_splitter.split_text(processed_text)

                # Convert splits to Documents with metadata
                for split in md_header_splits:
                    # Extract the most specific header level as a section identifier
                    section = ""
                    for header_level in ["Header 6", "Header 5", "Header 4", "Header 3", "Header 2", "Header 1"]:
                        if header_level in split.metadata and split.metadata[header_level]:
                            section = split.metadata[header_level]
                            break
                    
                    # Collect all unique page numbers that contribute to this chunk
                    unique_pages = set()
                    for page_info in pages_content:
                        # A simple check: if the page's text is part of the split's content
                        # This might not be perfect for overlapping chunks or very small pages
                        if page_info['text'] in split.page_content:
                            unique_pages.add(page_info['page_num'])

                    # Create document with metadata
                    doc = Document(
                        page_content=split.page_content,
                        metadata={
                            "source": filename,
                            "page": sorted(list(unique_pages)),  # Store as a sorted list of unique page numbers
                            "section": section
                        }
                    )

                    # Further split if the chunk is too large
                    if len(split.page_content) > chunk_size * 1.5:
                        smaller_chunks = recursive_splitter.split_documents([doc])
                        all_chunks.extend(smaller_chunks)
                    else:
                        all_chunks.append(doc)

                num_pages_processed = len(pages_content)
                print(f"  Created {len(all_chunks)} semantic chunks from {filename}")
                continue  # Skip the page-by-page processing below
            except Exception as e:
                print(f"  Error in semantic chunking: {e}. Falling back to page-by-page chunking.")
                # Fall back to page-by-page chunking

        # Process page by page if semantic chunking failed or wasn't applicable
        for page_info in pages_content:
            page_num = page_info['page_num']
            page_text = page_info['text']

            if not page_text.strip():  # Skip empty pages
                continue

            # Create LangChain Documents for each page before splitting
            page_doc = Document(
                page_content=page_text,
                metadata={"source": filename, "page": page_num}
            )

            # Split the document (representing a single page)
            page_chunks = recursive_splitter.split_documents([page_doc])

            all_chunks.extend(page_chunks)
            num_pages_processed += 1

        print(f"Finished chunking {num_pages_processed} pages from {filename}. Generated {len(all_chunks)} chunks so far.")

    # Deduplicate chunks with very similar content
    unique_chunks = []
    content_hashes = set()

    for chunk in all_chunks:
        # Create a simple hash of the content (ignoring whitespace)
        content = re.sub(r'\s+', ' ', chunk.page_content.strip().lower())
        content_hash = hashlib.md5(content.encode()).hexdigest()

        # Only add if we haven't seen very similar content
        if content_hash not in content_hashes:
            content_hashes.add(content_hash)
            unique_chunks.append(chunk)

    print(f"Total chunks generated: {len(all_chunks)}")
    print(f"After deduplication: {len(unique_chunks)} unique chunks")
    return unique_chunks


if __name__ == '__main__':
    print("Running chunking script as main...")
    # 1. Load data (requires PDFs in ../data directory relative to this script)
    # Adjust path if needed
    pdf_data = load_pdfs_from_directory(pdf_directory="../data/") # Assuming execution from project root or data dir relative to chunking.py

    if not pdf_data:
        print("No PDF data loaded. Exiting.")
    else:
        # 2. Chunk the loaded data
        chunks = chunk_extracted_data(pdf_data)

        if chunks:
            print("--- Example Chunk ---")
            # Print info about the first chunk
            first_chunk = chunks[0]
            print(f"Source: {first_chunk.metadata.get('source')}, Page: {first_chunk.metadata.get('page')}")
            print(f"Content (first 200 chars):\n{first_chunk.page_content[:200]}...")

            # You would typically pass 'chunks' to the next step (embedding and vector store indexing)
        else:
            print("No chunks were generated.")