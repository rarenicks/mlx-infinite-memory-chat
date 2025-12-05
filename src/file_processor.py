import os
from pypdf import PdfReader
from src.logging_config import setup_logger

logger = setup_logger(__name__)

def process_file(file_path):
    """
    Extracts text from a file based on its extension.
    Supports .pdf and .py
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    logger.info(f"Processing file: {file_path}")
    
    try:
        if ext == '.pdf':
            return _read_pdf(file_path)
        elif ext == '.py':
            return _read_code(file_path)
        else:
            return f"Unsupported file type: {ext}"
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return f"Error processing file {file_path}: {str(e)}"

def _read_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- Page {i+1} ---\n{page_text}\n"
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

def _read_code(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()
