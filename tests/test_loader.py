import pytest
from docx import Document as WordDocument

from src.loader import PDFLoader, DOCXLoader, load_document



def create_docx(path):
    document = WordDocument()
    document.add_heading("Javascript DOM", level=1)
    document.add_paragraph(
        "The Document Object Model represents a document as tree"
        )
    document.save(path)


def test_docx_loader_extracts_text(tmp_path):
    path = tmp_path / "sample.docx"
    create_docx(path)

    documents = DOCXLoader().load(path)
    assert documents
    assert "Document Object Model" in documents[0].page_content
    assert documents[0].metadata["document_id"] == "sample"
    assert documents[0].metadata["file_type"] == "docx"


def test_factory_select_docx_loader(tmp_path):
    path = tmp_path / "sample.docx"
    create_docx(path)

    documents =DOCXLoader().load(path)
    assert documents
    assert documents[0].metadata["file_type"] == "docx"


def test_missing_file_raises_error():
    with pytest.raises(FileNotFoundError):
        load_document("does_not_exists.docx")



def test_unsupported_extension_raises_error(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("Some text", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(path)

def test_pdf_loader_rejects_docx(tmp_path):
    path = tmp_path / "sample.docx"
    create_docx(path)

    with pytest.raises(ValueError, match="Unsupported file type"):
        PDFLoader().load(path)