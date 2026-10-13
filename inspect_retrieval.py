from src.loader import PDFLoader
from src.chunker import DocumentChunker
from src.embeddings import EmbeddingService
from src.vector_store import FAISSVectorStore
from src.retriever import DocumentRetriever

PDF_PATH = "data/documents/understanding-the-dom.pdf"

QUESTIONS = [
    "What is the DOM?",
    "What is the difference between the DOM and HTML source code?",
    "Why can the DOM differ from the original HTML source?",
    "What is the capital of France?",
]


def main():
    loader = PDFLoader()
    chunker = DocumentChunker(
        chunk_size=1200,
        chunk_overlap=150,
    )
    embedding_service = EmbeddingService()

    documents = loader.load(PDF_PATH)
    chunks = chunker.split(documents)

    embeddings = embedding_service.embed_documents(
        [chunk.page_content for chunk in chunks]
    )

    vector_store = FAISSVectorStore()
    vector_store.add(chunks, embeddings)

    retriever = DocumentRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    for question in QUESTIONS:
        print(f"\n{'=' * 80}")
        print(f"Question: {question}")

        results = retriever.retrieve(question, top_k=5)

        for rank, (doc, distance) in enumerate(results, start=1):
            page = doc.metadata.get("page", "unknown")
            page_display = page + 1 if isinstance(page, int) else page

            print(
                f"\nRank={rank} | "
                f"Page={page_display} | "
                f"Distance={distance:.4f}"
            )
            print(doc.page_content[:350].replace("\n", " "))


if __name__ == "__main__":
    main()