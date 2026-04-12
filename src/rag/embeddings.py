
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    """sentence-transformers for generating text embeddings"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):

        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:

        return self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True
        )

    def encode_query(self, query: str) -> np.ndarray:
        
        return self.model.encode([query], convert_to_numpy=True)