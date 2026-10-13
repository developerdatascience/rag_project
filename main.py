from src.loader import PDFLoader
from src.chunker import DocumentChunker
from src.embeddings import EmbeddingService
from src.vector_store import FAISSVectorStore
from src.retriever import DocumentRetriever
from src.generator import AnswerGenerator

from src.rag import RAGPipeline

from dotenv import load_dotenv

load_dotenv()


PDF_PATH = "data/documents/understanding-the-dom.pdf"

def main():
    loader = PDFLoader()

    documents = loader.load(
        "data/documents/understanding-the-dom.pdf"
        )

    chunker = DocumentChunker(chunk_size=800, chunk_overlap=100)

    chunks = chunker.split(documents)

    print(f"Pages loaded: {len(documents)}")
    print(f"Chunks: {len(chunks)}")

    # -----------------------------
    # 3. Embeddings
    # -----------------------------

    embedding_service = EmbeddingService()

    texts = [
        chunk.page_content for chunk in chunks
        ]

    embeddings = embedding_service.embed_documents(texts)
    print(f"\nNumber of embeddings: {len(embeddings)}")

    print(f"Embedding dimenesion: {len(embeddings[0])}")

    print(f"\nFirst 10 values: {embeddings[0][:10]}")

    # -----------------------------------
    # 4. Create vector store
    # -----------------------------------

    vector_store = FAISSVectorStore()

    vector_store.add(
        documents=chunks,
        embeddings=embeddings
        )

    print("Vectors stored successfully")

    # -----------------------------
    # 5. Retriever
    # -----------------------------

    retriever = DocumentRetriever(embedding_service=embedding_service, vector_store=vector_store)

    # -----------------------------
    # 6. Generator
    # -----------------------------

    generator= AnswerGenerator()

    # -------------------------
    # Create RAG pipeline
    # -------------------------

    rag = RAGPipeline(loader=loader,
                      chunker=chunker,
                      embedding_service=embedding_service,
                      vector_store=vector_store,
                      retriever=retriever,
                      generator=generator)

    # -------------------------
    # Ingest document
    # -------------------------
    rag.ingest(PDF_PATH)

    # -------------------------
    # Ask questions
    # -------------------------

    while True:
        question = input("\nAsk a question (type 'exit' to quit): ")

        if question.lower() == 'exit':
            break

        answer, results = rag.ask(question=question, top_k=5, distance_threshold=None)

        print("\n==============================")
        print("ANSWER")
        print("==============================")

        print(answer)

        print("\n==============================")
        print("SOURCES")
        print("==============================")

        for rank, (document, distance) in enumerate(results, start=1):
            print(
                f"{rank}. "
                f"{document.metadata.get('source')} "
                f"(page {document.metadata.get('page')}) "
                f"distance={distance:4f}"
            )
            
        
if __name__ == "__main__":
    main()
