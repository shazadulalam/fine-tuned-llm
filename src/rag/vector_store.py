import os
import json
from typing import List, Dict, Tuple

import numpy as np
import faiss


class VectorStore:

    """FAISS vector store for storing and retrieving document embedding"""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents: List[Dict[str, str]] = []

    def add(self, embeddings: np.ndarray, documents: List[Dict[str, str]]) -> None:

        """document embeddings and metadata to the store"""

        self.index.add(embeddings.astype(np.float32))
        self.documents.extend(documents)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Dict, float]]:

        """top_k most similar documents search"""

        distances, indices = self.index.search(
            query_embedding.astype(np.float32), min(top_k, self.index.ntotal)
        )
        return [
            (self.documents[idx], float(dist))
            for idx, dist in zip(indices[0], distances[0])
            if idx < len(self.documents)
    
    @property
    def total_documents(self) -> int:

        return self.index.ntotal

    def save(self, path: str) -> None:

        """saving the vector index and metadata"""

        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "documents.json"), "w") as f:
            json.dump(self.documents, f)

    @classmethod
    def load(cls, path: str, dimension: int) -> "VectorStore":

        """vector store from diskloading"""

        store = cls(dimension)
        store.index = faiss.read_index(os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "documents.json")) as f:
            store.documents = json.load(f)
        return store