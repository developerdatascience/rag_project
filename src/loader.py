
from abc import ABC, abstractmethod
from pathlib import Path

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

        for document in documents:
            document.metadata["source"] = str(path)
            document.metadata["document_id"] = path.stem
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