from langchain_core.documents import Document
from typing import List

from src.embeddings import EmbeddingService
from src.vector_store import FAISSVectorStore

class DocumentRetriever:
    def __init__(self, embedding_service: EmbeddingService, vector_store: FAISSVectorStore):
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def retrieve(self, 
                 question: str, 
                 top_k: int = 3,
                 distance_threshold: float | None = None,
                 metadata_filter: dict | None = None
                 ) -> List[tuple[Document, float]]:
        query_embedding = (
            self.embedding_service.embed_query(question)
            )
        return self.vector_store.search(query_embeddings=query_embedding, 
                                        top_k=top_k, 
                                        distance_threshold=distance_threshold,
                                        metadata_filter=metadata_filter
                                        )


    def build_context(self, results: List[tuple[Document, float]]) -> str:
        context_parts = []

        for rank, (document, distance) in enumerate(results, start=1):
            source = document.metadata.get("source", "unknowm")
            page = document.metadata.get("page", "unknown")

            context_parts.append(f"""
            [Source {rank}]
            File: {source}
            Page: {page}
            Distance: {distance}

            Content:
            {document.page_content}
            """)

            return "\n".join(context_parts)

    def format_sources(self, results):
        sources= []

        for rank, (document, distance) in enumerate(results, start=1):
            source = document.metadata.get("source", 'unknown')
            page = document.metadata.get("page", "unknown")

            # PyPDFLoader typically uses zero-based page numbers.
            page_label = (
                page+ 1 if isinstance(page, int) else page 
            )

            sources.append(
                f"[{rank}] {source}, page {page_label}"
            )

        return "\n".join(sources)