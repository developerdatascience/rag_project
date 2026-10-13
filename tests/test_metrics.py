import pytest
from langchain_core.documents import Document


from evaluation.metrics import (
    calculate_recall_at_k,
    calculate_page_hit_at_k,
    calculate_f1,
    calculate_precision_at_k,
    get_relevant_chunk_ids,
    )



def make_chunk(chunk_id, page):
    return Document(
        page_content=f"Test content for {chunk_id}",
        metadata={
            "chunk_id": chunk_id,
            "page": page,
        },
    )


def test_recall_at_k():
    retrieved = [
        (make_chunk("chunk_1", 0), 0.1),
        (make_chunk("chunk_2", 1), 0.2),
    ]

    recall = calculate_recall_at_k(
        retrieved,
        {"chunk_1", "chunk_3"},
    )

    assert recall == pytest.approx(0.5)


def test_precision_at_k():
    retrieved = [
        (make_chunk("chunk_1", 0), 0.1),
        (make_chunk("chunk_2", 1), 0.2),
        (make_chunk("chunk_3", 2), 0.3),
    ]

    precision = calculate_precision_at_k(
        retrieved,
        {"chunk_1", "chunk_3"},
    )

    assert precision == pytest.approx(2 / 3)


def test_f1():
    assert calculate_f1(0.5, 0.5) == pytest.approx(0.5)
    assert calculate_f1(0.0, 0.0) == 0.0


def test_page_hit_at_k():
    retrieved = [
        (make_chunk("chunk_1", 2), 0.1),
        (make_chunk("chunk_2", 4), 0.2),
    ]

    assert calculate_page_hit_at_k(retrieved, [4]) == 1.0
    assert calculate_page_hit_at_k(retrieved, [7]) == 0.0


def test_page_hit_requires_relevant_pages():
    with pytest.raises(ValueError):
        calculate_page_hit_at_k([], [])


def test_get_relevant_chunk_ids():
    chunks = [
        make_chunk("chunk_1", 0),
        make_chunk("chunk_2", 1),
        make_chunk("chunk_3", 1),
    ]

    result = get_relevant_chunk_ids(chunks, [1])

    assert result == {"chunk_2", "chunk_3"}