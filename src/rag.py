from src.loader import PDFLoader
from src.chunker import DocumentChunker
from src.embeddings import EmbeddingService
from src.vector_store import FAISSVectorStore
from src.retriever import DocumentRetriever
from src.generator import AnswerGenerator


class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline.

    Responsible for:
        1. Loading documents
        2. Chunking documents
        3. Generating embeddings
        4. Building vector store
        5. Retrieving relevant chunks
        6. Generating answers
    """
    def __init__(self, 
                 loader: PDFLoader,
                 chunker: DocumentChunker,
                 embedding_service: EmbeddingService,
                 vector_store: FAISSVectorStore,
                 retriever: DocumentRetriever,
                 generator: AnswerGenerator
                 ) -> None:
        self.loader = loader
        self.chunker = chunker
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.retriever = retriever
        self.generator = generator

    def ingest(self, file_path: str) -> None:
        """
        Load, chunk, embed and index a document.
        """

        documents = self.loader.load(file_path=file_path)

        print(f"Page loaded: {len(documents)}")

        chunks = self.chunker.split(documents)

        print(f"Chunks created: {len(chunks)}")

        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embedding_service.embed_documents(texts)

        print(f"Embeddings generated: {len(embeddings)}")

        self.vector_store.add(documents=chunks, embeddings=embeddings)

        print("Vector store ready.")


    def ask(self, 
            question: str,
            top_k: int = 3,
            distance_threshold: float | None = None
            ) -> tuple[str, list]:
        """
        Retrieve relevant context and generate an answer.
        """
        results = self.retriever.retrieve(question=question, top_k=top_k, distance_threshold=distance_threshold)

        print("\n==============================")
        print("RETRIEVED CHUNKS")
        print("==============================")

        for rank, (document, distance) in enumerate(results, start=1):
            print(f"\n--- Chunk {rank} ---")
            print(f"Page: {document.metadata.get('page')}")
            print(f"Distance: {distance:.4f}")
            print(document.page_content[:500])

        context = self.retriever.build_context(results=results)

        answer = self.generator.generate(question=question, context=context)

        return answer, results
        
