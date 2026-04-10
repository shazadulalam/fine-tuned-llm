import os
from typing import List, Dict
from pathlib import Path

from PyPDF2 import PdfReader


def extract_text_from_pdf(pdf_path: str) -> str:

    """extract all text from a single PDF file"""

    reader = PdfReader(pdf_path)
    return "\n".join(
        page.extract_text() or "" for page in reader.pages
    )


def extract_pages_from_pdf(pdf_path: str) -> List[Dict[str, str]]:

    """extract text from each page"""

    reader = PdfReader(pdf_path)
    filename = Path(pdf_path).name
    return [
        {
            "text": page.extract_text() or "",
            "source": filename,
            "page": i + 1,
        }
        for i, page in enumerate(reader.pages)
        if page.extract_text()
    ]


def load_pdfs_from_directory(pdf_dir: str) -> List[Dict[str, str]]:

    """all PDFs load and extract in a directory."""

    pdf_files = sorted(
        f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")
    )
    documents = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        documents.extend(extract_pages_from_pdf(pdf_path))
    return documents
