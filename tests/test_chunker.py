from langchain_core.documents import Document
import pytest

from src.chunker import DocumentChunker


def make_document(document_id, content, page=0):
    return Document(
        page_content=content,
        metadata={
            "document_id": document_id,
            "source": f"{document_id}.pdf",
            "page": page,
        },
    )


def test_chunker_assigns_required_metadata():
    document = make_document(
        "doc_a",
        "JavaScript can modify the DOM. " * 50,
    )

    chunks = DocumentChunker(
        chunk_size=100,
        chunk_overlap=10,
    ).split([document])

    assert chunks

    for chunk in chunks:
        assert chunk.metadata["document_id"] == "doc_a"
        assert chunk.metadata["chunk_id"]
        assert chunk.metadata["content_hash"]
        assert isinstance(chunk.metadata["chunk_index"], int)
        assert chunk.metadata["page"] == 0


def test_chunk_ids_are_deterministic():
    document = make_document(
        "doc_a",
        "The DOM represents an HTML document. " * 20,
    )

    chunker = DocumentChunker(
        chunk_size=100,
        chunk_overlap=10,
    )

    first = chunker.split([document])
    second = chunker.split([document])

    assert [c.metadata["chunk_id"] for c in first] == [
        c.metadata["chunk_id"] for c in second
    ]


def test_chunk_ids_differ_between_configurations():
    document = make_document(
        "doc_a",
        "JavaScript and the DOM. " * 50,
    )

    chunks_a = DocumentChunker(100, 10).split([document])
    chunks_b = DocumentChunker(150, 20).split([document])

    ids_a = {c.metadata["chunk_id"] for c in chunks_a}
    ids_b = {c.metadata["chunk_id"] for c in chunks_b}

    assert ids_a.isdisjoint(ids_b)


def test_chunk_ids_do_not_collide_across_documents():
    document_a = make_document("doc_a", "Content about HTML. " * 20)
    document_b = make_document("doc_b", "Content about CSS. " * 20)

    chunks = DocumentChunker(100, 10).split(
        [document_a, document_b]
    )

    ids = [chunk.metadata["chunk_id"] for chunk in chunks]

    assert len(ids) == len(set(ids))


def test_empty_input_returns_empty_list():
    assert DocumentChunker().split([]) == []


def test_missing_document_id_raises_error():
    document = Document(
        page_content="Some content",
        metadata={"source": "unknown.pdf"},
    )

    with pytest.raises(ValueError, match="document_id"):
        DocumentChunker().split([document])