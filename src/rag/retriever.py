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
            f"Answer the question based on the given context. "
            f"If the context doesn't contain enough information, say so.\n\n"
            f"### Context:\n{context}\n\n"
            f"### Question:\n{query}\n\n"
            f"### Response:\n"
        )
    
    def generate(self, prompt: str) -> str:
        
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def query(self, question: str) -> Dict:

        """retrieve context, build prompt, generate answer"""

        context_chunks = self.retrieve(question)
        prompt = self.build_prompt(question, context_chunks)
        answer = self.generate(prompt)
        return {
            "question": question,
            "answer": answer,
            "sources": [
                {"source": c["source"], "score": c["score"], "text": c["text"][:200]}
                for c in context_chunks
            ],
        }