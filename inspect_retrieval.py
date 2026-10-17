from tkinter.messagebox import QUESTION

from src.ingestion_service import IngestionService
from src.loader import PDFLoader
from src.chunker import DocumentChunker
from src.embeddings import EmbeddingService
from src.vector_store import FAISSVectorStore
from src.retriever import DocumentRetriever

PDF_PATH = "data/documents/understanding-the-dom.pdf"

QUESTIONS = [
    "How to use Alert using javascript in HTML",
    "How to extract a data from an api using js?",
    "What is the capital of India?",
]



from src.ingestion_service import IngestionService


QUESTIONS = [
    "How to use Alert using javascript in HTML",
    "How to extract a data from an api using js?",
    "What is the capital of India?",
]


def main():
    service = IngestionService()

    # Load the existing index, without forcing a new ingestion.
    if not service._load_active_snapshot():
        raise RuntimeError(
            "No active index found. Run run_ingestion.py first."
        )

    for question in QUESTIONS:
        print("\n" + "=" * 80)
        print(f"QUESTION: {question}")
        print("=" * 80)

        results = service.search(question, top_k=3)

        if not results:
            print("No chunks retrieved.")
            continue

        for rank, (document, distance) in enumerate(results, start=1):
            print(f"\n--- Result {rank} ---")
            print(f"Distance: {distance:.4f}")
            print(f"Source: {document.metadata.get('source')}")
            print(f"Page metadata: {document.metadata.get('page')}")
            print(f"Chunk ID: {document.metadata.get('chunk_id')}")
            print("\nRetrieved text:\n")
            print(document.page_content[:2500])


if __name__ == "__main__":
    main()
    
    # loader = PDFLoader()
    # chunker = DocumentChunker(
    #     chunk_size=1200,
    #     chunk_overlap=150,
    # )
    # embedding_service = EmbeddingService()

    # documents = loader.load(PDF_PATH)
    # chunks = chunker.split(documents)

    # embeddings = embedding_service.embed_documents(
    #     [chunk.page_content for chunk in chunks]
    # )

    # vector_store = FAISSVectorStore()
    # vector_store.add(chunks, embeddings)

    # retriever = DocumentRetriever(
    #     embedding_service=embedding_service,
    #     vector_store=vector_store,
    # )

    # for question in QUESTIONS:
    #     print(f"\n{'=' * 80}")
    #     print(f"Question: {question}")

    #     results = retriever.retrieve(question, top_k=5)

    #     for rank, (doc, distance) in enumerate(results, start=1):
    #         page = doc.metadata.get("page", "unknown")
    #         page_display = page + 1 if isinstance(page, int) else page

    #         print(
    #             f"\nRank={rank} | "
    #             f"Page={page_display} | "
    #             f"Distance={distance:.4f}"
    #         )
    #         print(doc.page_content[:350].replace("\n", " "))


if __name__ == "__main__":
    main()