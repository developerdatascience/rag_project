
import json
import os
from pathlib import Path

import faiss
import numpy as np
from langchain_core.documents import Document


class FAISSVectorStore:
    def __init__(self):
        self.index = None
        self.documents = []

    def add(self, documents, embeddings):
        if not documents:
            return

        vectors = np.asarray(embeddings, dtype=np.float32)

        if vectors.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")

        if len(documents) != len(vectors):
            raise ValueError(
                "The number of documents must match the embeddings."
            )

        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings contain invalid numeric values.")

        if self.index is None:
            self.index = faiss.IndexFlatL2(vectors.shape[1])

        if vectors.shape[1] != self.index.d:
            raise ValueError("Embedding dimension does not match the index.")

        self.index.add(vectors)
        self.documents.extend(documents)

    def search(
        self,
        query_embedding,
        top_k=3,
        distance_threshold=None,
        metadata_filter=None,
    ):
        if self.index is None or self.index.ntotal == 0:
            return []

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        query = np.asarray(
            query_embedding, dtype=np.float32
        ).reshape(1, -1)

        if query.shape[1] != self.index.d:
            raise ValueError("Query embedding dimension mismatch.")

        # Metadata filtering happens after FAISS search.
        # Search all vectors when a filter is supplied so relevant
        # matching documents are not missed by filtering only top_k.
        search_count = (
            self.index.ntotal
            if metadata_filter
            else min(top_k, self.index.ntotal)
        )

        distances, indices = self.index.search(query, search_count)

        results = []

        for distance, index_position in zip(
            distances[0], indices[0]
        ):
            if index_position < 0:
                continue

            document = self.documents[int(index_position)]

            if (
                distance_threshold is not None
                and float(distance) > distance_threshold
            ):
                continue

            if metadata_filter and any(
                document.metadata.get(key) != value
                for key, value in metadata_filter.items()
            ):
                continue

            results.append((document, float(distance)))

            if len(results) == top_k:
                break

        return results

    def save(self, directory):
        if self.index is None:
            raise ValueError("Cannot save an empty vector store.")

        if self.index.ntotal != len(self.documents):
            raise RuntimeError("Index and document counts do not match.")

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        index_path = directory / "index.faiss"
        documents_path = directory / "documents.json"

        temp_index_path = directory / "index.faiss.tmp"
        temp_documents_path = directory / "documents.json.tmp"

        try:
            faiss.write_index(self.index, str(temp_index_path))

            payload = [
                {
                    "page_content": document.page_content,
                    "metadata": document.metadata,
                }
                for document in self.documents
            ]

            with temp_documents_path.open(
                "w", encoding="utf-8"
            ) as file:
                json.dump(payload, file, ensure_ascii=False)

            os.replace(temp_index_path, index_path)
            os.replace(temp_documents_path, documents_path)

        finally:
            temp_index_path.unlink(missing_ok=True)
            temp_documents_path.unlink(missing_ok=True)

    def load(self, directory):
        directory = Path(directory)
        index_path = directory / "index.faiss"
        documents_path = directory / "documents.json"

        if not index_path.exists() or not documents_path.exists():
            raise FileNotFoundError(
                "Saved FAISS index or document metadata is missing."
            )

        loaded_index = faiss.read_index(str(index_path))

        with documents_path.open(
            "r", encoding="utf-8"
        ) as file:
            payload = json.load(file)

        loaded_documents = [
            Document(
                page_content=item["page_content"],
                metadata=item["metadata"],
            )
            for item in payload
        ]

        if loaded_index.ntotal != len(loaded_documents):
            raise RuntimeError(
                "Saved index and document metadata are inconsistent."
            )

        self.index = loaded_index
        self.documents = loaded_documents

    def rebuild(self, documents, embeddings):
        """Replace the entire in-memory index with a fresh index."""
        self.index = None
        self.documents = []
        self.add(documents, embeddings)
