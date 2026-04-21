import logging
import os

from configs.default import RAGConfig, ModelConfig
from src.rag.pdf_extractor import load_pdfs_from_directory
from src.rag.chunker import chunk_documents
from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore
from src.rag.retriever import RAGRetriever
from src.model.inference import load_finetuned_model

logger = logging.getLogger(__name__)


def ingest_pdfs(rag_config: RAGConfig) -> VectorStore:

    """Extracting text from PDFs, chunk, embed, and build a vector store"""

    logger.info("Loading PDFs from %s ...", rag_config.pdf_dir)
    documents = load_pdfs_from_directory(rag_config.pdf_dir)
    logger.info("Extracted %d pages from PDFs.", len(documents))

    logger.info("Chunking documents (size=%d, overlap=%d) ...",
                rag_config.chunk_size, rag_config.chunk_overlap)
    chunks = chunk_documents(documents, rag_config.chunk_size, rag_config.chunk_overlap)
    logger.info("Created %d chunks.", len(chunks))

    logger.info("Generating embeddings with %s ...", rag_config.embedding_model)
    embed_model = EmbeddingModel(rag_config.embedding_model)
    texts = [c["text"] for c in chunks]
    embeddings = embed_model.encode(texts)

    store = VectorStore(embed_model.dimension)
    store.add(embeddings, chunks)
    store.save(rag_config.vector_store_path)
    logger.info("Vector store saved to %s (%d vectors).",
                rag_config.vector_store_path, store.total_documents)
    return store

def build_rag_retriever(
    rag_config: RAGConfig,
    model_config: ModelConfig,
    model_path: str,
) -> RAGRetriever:

    """model loading, embeddings, vector store, and return a RAGRetriever."""

    logger.info("Loading fine-tuned model from %s ...", model_path)
    model, tokenizer = load_finetuned_model(model_config.model_id, model_path)

    logger.info("Loading embedding model ...")
    embed_model = EmbeddingModel(rag_config.embedding_model)

    if os.path.exists(os.path.join(rag_config.vector_store_path, "index.faiss")):
        logger.info("Loading existing vector store from %s ...", rag_config.vector_store_path)
        store = VectorStore.load(rag_config.vector_store_path, embed_model.dimension)
    else:
        logger.info("No existing vector store found. Ingesting PDFs ...")
        store = ingest_pdfs(rag_config)

    return RAGRetriever(
        model=model,
        tokenizer=tokenizer,
        embedding_model=embed_model,
        vector_store=store,
        top_k=rag_config.top_k,
        max_new_tokens=rag_config.max_new_tokens,
    )


def run_rag_pipeline(
    rag_config: RAGConfig,
    model_config: ModelConfig,
    model_path: str,
    queries: list[str],
) -> list[dict]:

    retriever = build_rag_retriever(rag_config, model_config, model_path)

    results = []
    for query in queries:
        logger.info("Query: %s", query)
        result = retriever.query(query)
        logger.info("Answer: %s", result["answer"][:200])
        results.append(result)

    return results