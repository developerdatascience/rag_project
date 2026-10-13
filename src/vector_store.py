from pathlib import Path
import faiss
import numpy as np
from typing import List
from langchain_core.documents import Document


class FAISSVectorStore:
    """Small abstraction around FAISS
    Stores:
        -FAISS Index
        - original documents
    """
    def __init__(self):
        self.index = None
        self.documents: List[Document] = []

    def add(self, documents: List[Document], embeddings: List[List[float]]):
        if not documents:
            raise ValueError("Cannot add empty documents")

        if not embeddings:
            raise ValueError("Cannot add empty embeddings")

        if len(documents) != len(embeddings):
            raise ValueError("Documents and embeddings must have same length")

        vectors = np.array(embeddings, dtype='float32')

        dimension = vectors.shape[1]

        if self.index is None:
            self.index = faiss.IndexFlatL2(dimension)

        self.index.add(vectors)

        self.documents.extend(documents)


    def search(self, 
               query_embeddings: List[float], 
               top_k: int = 5,
               distance_threshold: float | None = None,
               metadata_filter: dict | None = None ) -> List[tuple[Document, float]]:
        if self.index is None:
            raise ValueError(
                "top_k must be greater than 0"
                )
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query_vector = np.array([query_embeddings], dtype="float32")

        distances, indices = self.index.search(
            query_vector,
            len(self.documents),
        )

        results = []

        for distance, index in zip(distances[0], indices[0]):
            if index < 0:
                continue
            document = self.documents[index]

            if metadata_filter:
                matches = all(
                    document.metadata.get(key) == value
                    for key, value in metadata_filter.items()
                )

                if not matches:
                    continue

            if (distance_threshold is not None and distance > distance_threshold) :
                    continue
            results.append((document, float(distance)))

            if len(results) >= top_k:
                break

        return results
