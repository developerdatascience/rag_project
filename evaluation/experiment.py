from typing import List

from src.loader import PDFLoader
from src.embeddings import EmbeddingService
from src.chunker import  DocumentChunker
from src.vector_store import FAISSVectorStore
from src.retriever import DocumentRetriever


from evaluation.results import RetrievalResult
from evaluation.metrics import (
    calculate_f1, 
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_page_hit_at_k,
    get_relevant_chunk_ids
    )

from evaluation.retrieval_dataset import EVALUATION_DATASET

PDF_PATH = "data/documents/understanding-the-dom.pdf"

class RetrievalExperiment:
    def __init__(self, pdf_path=PDF_PATH):
        self.pdf_path = pdf_path
        self.loader =PDFLoader()
        self.embedding_service = EmbeddingService()


    def run_chunking_configuration(
            self,
            chunk_size: int,
            chunk_overlap: int,
            top_k_values: List[int]
            ) -> List[RetrievalResult]:
        if not top_k_values:
            raise ValueError("At least one of top_k value is required")

        for item in EVALUATION_DATASET:
            if not item["relevant_pages"]:
                raise ValueError(f"Add verified relevant pages for: {item['question']}")


        documents = self.loader.load(self.pdf_path)

        chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        chunks = chunker.split(documents=documents)

        # Generate document embeddings once for this chunking step
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embedding_service.embed_documents(texts)

        vector_store = FAISSVectorStore()
        vector_store.add(documents=chunks, embeddings=embeddings)

        retriever = DocumentRetriever(embedding_service=self.embedding_service, 
                                      vector_store=vector_store)

        results_for_configuration = []

        # Reuse the same vector index for different top_k values.
        for top_k in top_k_values:
            recalls = []
            precisions = []
            hit_rates = []

            for item in EVALUATION_DATASET:
                relevant_ids = get_relevant_chunk_ids(chunks=chunks,
                                                      relevant_pages=item["relevant_pages"]
                                                    )
                retrieved = retriever.retrieve(question=item["question"], top_k=top_k)
                recalls.append(
                    calculate_recall_at_k(retrieved_chunks=retrieved, relevant_chunk_ids=relevant_ids)
                )

                precisions.append(
                    calculate_precision_at_k(retrieved, relevant_ids)
                )

                hit_rates.append(
                    calculate_page_hit_at_k(
                        retrieved_chunks=retrieved,
                        relevant_pages=item["relevant_pages"],
                    )
                )

            avg_recall = sum(recalls) / len(recalls)
            avg_precision = sum(precisions) / len(precisions)
            hit_rate=sum(hit_rates) / len(hit_rates)

            result = RetrievalResult(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                top_k=top_k,
                total_chunks=len(chunks),
                recall=avg_recall,
                precision=avg_precision,
                f1=calculate_f1(avg_precision, avg_recall),
                hit_rate=hit_rate
            )
            results_for_configuration.append(result)

        return results_for_configuration