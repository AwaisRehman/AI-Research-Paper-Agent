# ============================================================
# parser.py — PDF Text Extraction & Chunking Module
# ============================================================
# Responsibilities:
#   - Extract raw text from uploaded PDF files using PyMuPDF
#   - Clean and normalize extracted text
#   - Split text into overlapping chunks for embedding
#
# Author: Research Paper Agent
# ============================================================

import fitz  # PyMuPDF — fast PDF processing library
import re


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract all text content from a PDF file.

    Opens the PDF from raw bytes (as uploaded via Streamlit),
    iterates over every page, and concatenates the text.

    Args:
        file_bytes (bytes): Raw PDF file content from st.file_uploader.

    Returns:
        str: Full extracted text from the PDF. Returns empty string
             if the PDF has no extractable text (e.g. scanned image PDFs).
    """
    text = ""

    # Open PDF directly from memory bytes — no temp file needed
    doc = fitz.open(stream=file_bytes, filetype="pdf")

    for page_num, page in enumerate(doc):
        # get_text() returns plain text from the page
        page_text = page.get_text()
        text += page_text

    doc.close()

    # Clean up common PDF artifacts
    text = _clean_text(text)

    return text


def _clean_text(text: str) -> str:
    """
    Clean raw PDF text by removing common artifacts.

    PDF extraction often introduces extra whitespace, 
    broken line endings, and special characters.

    Args:
        text (str): Raw text from PDF extraction.

    Returns:
        str: Cleaned text ready for chunking.
    """
    # Replace multiple newlines with single newline
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Replace multiple spaces with single space
    text = re.sub(r' {2,}', ' ', text)

    # Remove non-printable characters (except newlines and tabs)
    text = re.sub(r'[^\x09\x0A\x0D\x20-\x7E]', ' ', text)

    return text.strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split a long text into overlapping chunks for embedding.

    Overlapping chunks ensure that sentences crossing chunk boundaries
    are still semantically retrievable. This is a sliding window approach
    operating at the word level.

    Args:
        text (str): Full text to be chunked.
        chunk_size (int): Number of words per chunk. Default 500.
                          Larger = more context per chunk, fewer total chunks.
        overlap (int): Number of words to overlap between consecutive chunks.
                       Default 50. Prevents information loss at boundaries.

    Returns:
        list[str]: List of text chunks, each approximately chunk_size words.

    Example:
        chunks = chunk_text(long_paper_text, chunk_size=400, overlap=40)
        # Returns: ["Introduction The study of...", "study of LLMs has...", ...]
    """
    # Tokenize by whitespace — simple but effective for English academic text
    words = text.split()

    # Guard: return empty list for empty text
    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        # Calculate end position for this chunk
        end = start + chunk_size

        # Join words back into a string chunk
        chunk = " ".join(words[start:end])

        # Only add non-empty chunks
        if chunk.strip():
            chunks.append(chunk)

        # Advance start with overlap — ensures continuity between chunks
        # If chunk_size == overlap, this would be an infinite loop, so guard:
        step = chunk_size - overlap
        if step <= 0:
            step = chunk_size  # fallback: no overlap if misconfigured
        start += step

    return chunks


def get_paper_metadata(file_bytes: bytes, filename: str) -> dict:
    """
    Extract basic metadata from a PDF file.

    Attempts to read PDF document properties such as title,
    author, and page count.

    Args:
        file_bytes (bytes): Raw PDF content.
        filename (str): Original filename from upload.

    Returns:
        dict: Metadata dictionary with keys:
              - filename, title, author, page_count, word_count
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")

    # PDF metadata is stored in the document's info dictionary
    meta = doc.metadata or {}

    page_count = doc.page_count

    # Extract full text just to count words
    full_text = ""
    for page in doc:
        full_text += page.get_text()

    doc.close()

    word_count = len(full_text.split())

    return {
        "filename": filename,
        "title": meta.get("title", filename.replace(".pdf", "")),
        "author": meta.get("author", "Unknown"),
        "page_count": page_count,
        "word_count": word_count,
    }
