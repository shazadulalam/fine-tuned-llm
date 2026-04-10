from typing import List, Dict


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:

    """Split text into chunks by character count"""

    if not text.strip():
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


def chunk_documents(
    documents: List[Dict[str, str]], chunk_size: int = 512, overlap: int = 50
) -> List[Dict[str, str]]:

    """split a list of page-level documents into smaller chunks with metadata"""

    chunked = []
    for doc in documents:
        text_chunks = chunk_text(doc["text"], chunk_size, overlap)
        chunked.extend(
            {
                "text": chunk,
                "source": doc.get("source", "unknown"),
                "page": doc.get("page", 0),
                "chunk_index": i,
            }
            for i, chunk in enumerate(text_chunks)
        )
    return chunked