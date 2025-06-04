from pypdf import PdfReader
import os
from pathlib import Path
from typing import List, Dict, Tuple
import io
import os
from PIL import Image # Import Pillow
import pytesseract # Import pytesseract
import cv2 # OpenCV for image preprocessing
import numpy as np # For OpenCV
from tqdm import tqdm # For progress bars

# --- Tesseract Configuration --- 
# Set the TESSERACT_CMD environment variable to your Tesseract executable path
# OR uncomment and set the path directly below.
# Example for Windows:
# TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Example for Linux (usually not needed if installed via package manager):
# TESSERACT_PATH = r'/usr/bin/tesseract'
TESSERACT_PATH = os.getenv('TESSERACT_CMD')
if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
else:
    # Fallback or default path if environment variable is not set
    # You might want to log a warning here if Tesseract is expected to be found
    # For example, on Windows, a common default install path:
    default_tesseract_path_windows = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if os.path.exists(default_tesseract_path_windows):
        pytesseract.pytesseract.tesseract_cmd = default_tesseract_path_windows
    # Add similar checks for other OS if needed, or let pytesseract try to find it.
# ------------------------------------------

def preprocess_image_for_ocr(image_bytes: bytes):
    """Converts image bytes to a preprocessed image suitable for OCR."""
    try:
        # Convert image bytes to OpenCV format
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 1. Convert to grayscale
        gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # 2. Apply thresholding (Adaptive Gaussian Thresholding can be good for varying illumination)
        # preprocessed_img = cv2.adaptiveThreshold(
        #     gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #     cv2.THRESH_BINARY, 11, 2
        # )
        # Alternatively, simple binary thresholding (Otsu's method is good for bimodal images)
        _, preprocessed_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 3. Optional: Denoising (can sometimes help, sometimes hurt)
        # preprocessed_img = cv2.medianBlur(preprocessed_img, 3) # Kernel size 3 or 5

        # Convert back to PIL Image format for Pytesseract
        pil_image = Image.fromarray(preprocessed_img)
        return pil_image
    except Exception as e:
        print(f"[Warning] Image preprocessing failed: {e}")
        # Fallback to original image if preprocessing fails
        return Image.open(io.BytesIO(image_bytes))

def perform_ocr_on_image(image_bytes: bytes) -> str:
    """Performs OCR on image bytes using pytesseract after preprocessing."""
    try:
        processed_image = preprocess_image_for_ocr(image_bytes)
        ocr_text = pytesseract.image_to_string(processed_image)
        return ocr_text
    except pytesseract.TesseractNotFoundError:
        print("\n--- TESSERACT NOT FOUND ERROR ---")
        print("pytesseract could not find the Tesseract executable.")
        print("Please ensure Tesseract is installed and in your system's PATH,")
        print("or set the path manually in load_pdf.py (see comments).")
        print("OCR will be skipped for subsequent images.")
        # Optionally, raise the error or disable OCR globally after first failure
        # raise # Uncomment to stop execution
        return "" # Return empty string if Tesseract is not found
    except Exception as e:
        # Log other potential errors (e.g., invalid image format)
        print(f"[Warning] OCR failed for an image: {e}")
        return "" # Return empty string on other errors

def load_pdfs_from_directory(pdf_directory: str = "data/") -> List[Tuple[str, List[Dict[str, str]]]]:
    """
    Loads all PDF files from a specified directory, extracts text page by page,
    performs OCR on embedded images, and returns a list of tuples,
    where each tuple contains the filename and a list of page contents.

    Args:
        pdf_directory: The relative path to the directory containing PDF files.

    Returns:
        A list of tuples: [(filename1, pages1), (filename2, pages2), ...],
        where pages is a list: [{'page_num': 1, 'text': '...'}, ...]
    """
    pdf_docs_content = []
    data_path = Path(pdf_directory)

    if not data_path.is_dir():
        print(f"Error: Directory not found: {data_path.resolve()}")
        return []

    print(f"Loading PDFs from: {data_path.resolve()}")
    pdf_files = list(data_path.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {data_path.resolve()}.")
        return []

    for pdf_path in tqdm(pdf_files, desc="Processing PDFs", unit="file"):
        try:
            pages_content = []
            num_images_ocred = 0
            with open(pdf_path, "rb") as f: # Open in binary mode for pypdf
                reader = PdfReader(f)
                
                for page_num in tqdm(range(len(reader.pages)), desc=f"Pages in {pdf_path.name}", unit="page", leave=False):
                    page = reader.pages[page_num]
                    # 1. Extract standard text
                    page_text = page.extract_text() if page.extract_text() else "" # Ensure text is not None
                    ocr_text_combined = ""

                    # 2. Extract images and perform OCR
                    if hasattr(page, 'images') and page.images: # page.images is a list of ImageFile objects
                        # print(f"  Page {page_num + 1}: Found {len(page.images)} images. Performing OCR...")
                        for img_index, image_file in enumerate(page.images):
                            try:
                                image_bytes = image_file.data # This is the raw image data
                                
                                # Perform OCR
                                ocr_result = perform_ocr_on_image(image_bytes)
                                if ocr_result:
                                   image_name_info = f" ({image_file.name})" if hasattr(image_file, 'name') and image_file.name else ""
                                   ocr_text_combined += f"\n[OCR Image {img_index + 1}{image_name_info}]:\n{ocr_result}\n"
                                   num_images_ocred += 1
                            except Exception as img_e:
                                print(f"[Warning] Failed to process image {img_index + 1} on page {page_num + 1} of {pdf_path.name}: {img_e}")
                    
                    # 3. Combine standard text and OCR text
                    full_page_text = page_text + ocr_text_combined

                    pages_content.append({"page_num": page_num + 1, "text": full_page_text}) # 1-indexed page number
            
            pdf_docs_content.append((pdf_path.name, pages_content))
            # No doc.close() needed here as 'with open' handles file closure
            # print(f"Successfully processed {len(pages_content)} pages from {pdf_path.name}. (Performed OCR on {num_images_ocred} images)")
        except Exception as e:
            print(f"Error processing file {pdf_path.name}: {e}")

    if not pdf_docs_content:
        print(f"No PDF files found or processed in {data_path.resolve()}")
    else:
        print(f"Successfully loaded {len(pdf_docs_content)} PDF documents.")

    return pdf_docs_content

if __name__ == '__main__':
    # Example usage: Place your PDFs in a 'data' subdirectory relative to this script
    # Or change the path in the function call
    loaded_data = load_pdfs_from_directory()

    if loaded_data:
        print("\n--- Example Extracted Content (including OCR if any) ---")
        # Print info about the first page of the first loaded PDF
        first_filename, first_doc_pages = loaded_data[0]
        if first_doc_pages:
            print(f"Document: {first_filename}")
            print(f"Page {first_doc_pages[0]['page_num']} Text (first 500 chars):")
            print(first_doc_pages[0]['text'][:500] + "...")
        else:
            print(f"Document {first_filename} seems to have no extractable text content.")

    # You would typically pass loaded_data to the next step (cleaning, chunking, embedding)