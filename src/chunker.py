from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib
from collections import defaultdict

class DocumentChunker:
    """Splits the docunments into smaller, overlapping chunks"""

    def __init__(self, chunk_size: int = 800, chunk_overlap : int = 100) -> None:
        if chunk_size <= 0:
            raise ValueError(f" Chunk Size must be greater than 0")

        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )


    def split(self, documents: list[Document]) -> list[Document]:
        """Split documents and attach traceable metadata to each chunk."""
        if not documents:
            return []

        for document in documents:
            if not document.metadata.get("document_id"):
                raise ValueError("Each document must have a document_id in its metadata")

        chunks = self.splitter.split_documents(documents)

        if not chunks:
            raise ValueError("Chunking produced no chunks")

        config_id = hashlib.sha256(
            f"{self.chunk_size}:{self.chunk_overlap}".encode("utf-8")
        ).hexdigest()[:12]
        chunk_indexes: defaultdict[str, int] = defaultdict(int)

        for chunk in chunks:
            document_id = chunk.metadata["document_id"]
            index = chunk_indexes[document_id]
            chunk_indexes[document_id] += 1
            chunk.metadata["chunk_index"] = index
            chunk.metadata["content_hash"] = hashlib.sha256(
                chunk.page_content.encode("utf-8")
            ).hexdigest()
            chunk.metadata["chunk_id"] = (
                f"{document_id}_{config_id}_chunk_{index}"
            )

        return chunks



            