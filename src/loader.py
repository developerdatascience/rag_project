import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
)


class BaseDocumentLoader(ABC):
    """Common interface and metadata handling for document loaders."""

    supported_extensions: set[str] = set()

    def load(self, file_path: str | Path) -> list[Document]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        if not path.is_file():
            raise ValueError(f"Expected a file, received: {path}")

        if path.suffix.lower() not in self.supported_extensions:
            raise ValueError(
                f"Unsupported file type: {path.suffix}. "
                f"Supported types: {sorted(self.supported_extensions)}"
            )

        documents = self._load(path)

        if not documents:
            raise ValueError(f"No text extracted from document: {path}")

        document_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        identity = f"{path.resolve()}:{document_hash}"
        document_id = hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()

        for document in documents:
            document.metadata["source"] = str(path)
            document.metadata["document_id"] = document_hash
            document.metadata["file_type"] = path.suffix.lower().lstrip(".")

        return documents

    @abstractmethod
    def _load(self, path: Path) -> list[Document]:
        """Extract documents using a format-specific implementation."""
        raise NotImplementedError


class PDFLoader(BaseDocumentLoader):
    """Load PDF files, preserving page metadata."""

    supported_extensions = {".pdf"}

    def _load(self, path: Path) -> list[Document]:
        return PyPDFLoader(str(path)).load()


class DOCXLoader(BaseDocumentLoader):
    """Load Microsoft Word DOCX files."""

    supported_extensions = {".docx"}

    def _load(self, path: Path) -> list[Document]:
        return Docx2txtLoader(str(path)).load()


def load_document(file_path: str | Path) -> list[Document]:
    """Select the correct loader based on the file extension."""

    path = Path(file_path)
    extension = path.suffix.lower()

    loaders = {
        ".pdf": PDFLoader,
        ".docx": DOCXLoader,
    }

    loader_class = loaders.get(extension)

    if loader_class is None:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            "Supported formats: .pdf, .docx"
        )

    return loader_class().load(path)



def load_directory(directory_path: str | Path, recursive: bool = True) -> List[Document]:
    """Load every supported document in a directory"""

    directory = Path(directory_path)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if not directory.is_dir():
        raise ValueError(f"Expected a directory: {directory}")
    
    iterator = directory.rglob("*") if recursive else directory.iterdir()

    supported_extensions = {".pdf", ".docx"}
    file_paths = sorted(
        path 
        for path in iterator
        if path.is_file() 
        and path.suffix.lower() in supported_extensions
    )

    documents = []

    for path in file_paths:
        documents.extend(load_document(path))
    
    return documents