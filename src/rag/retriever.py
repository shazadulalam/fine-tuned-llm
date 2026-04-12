from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore


class RAGRetriever:

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        top_k: int = 5,
        max_new_tokens: int = 256,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens

    def retrieve(self, query: str) -> List[Dict]:

        """retrieve chunks for a query."""

        query_embedding = self.embedding_model.encode_query(query)
        results = self.vector_store.search(query_embedding, self.top_k)
        return [{"text": doc["text"], "source": doc.get("source", ""), "score": score}
                for doc, score in results]

    
    def build_prompt(self, query: str, context_chunks: List[Dict]) -> str:
        
        context = "\n\n".join(
            f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
        )
        return (
            f"### Instruction:\n"
            f"Answer the question based on the provided context. "
            f"If the context doesn't contain enough information, say so.\n\n"
            f"### Context:\n{context}\n\n"
            f"### Question:\n{query}\n\n"
            f"### Response:\n"
        )