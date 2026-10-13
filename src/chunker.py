from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

from pathlib import Path

class DocumentChunker:
    """Splits the docunments into smaller, overlapping chunks"""

    def __init__(self, chunk_size: int = 800, chunk_overlap : int = 100) -> None:
        if chunk_size <= 0:
            raise ValueError(f" Chunk Size must be greater than 0")

        if chunk_overlap < 0:
            raise ValueError(f"chunk overlap can be negative")

        if chunk_overlap > chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )


    def split(self, documents: list[Document]) -> list[Document]:
        """Split documents and attach traceable metadata to each chunk."""
        if not documents:
            raise ValueError("Cannot chunk an empty document list")

        chunks = self.splitter.split_documents(documents)

        if not chunks:
            raise ValueError("Chunking produced no chunks")

        for index, chunk in enumerate(chunks):
            source = chunk.metadata.get("source", "unknown")
            chunk.metadata["document_id"] = Path(source).stem if source != "unknown" else "unknown"
            chunk.metadata["chunk_index"] = index
            chunk.metadata["chunk_id"] = (
                f"{chunk.metadata['document_id']}_chunk_{index}"
            )

        return chunks



            