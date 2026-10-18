
import os

import pytest

from evaluation.citation_validator import validate_citations
from src.ingestion_service import IngestionService


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_RAG_INTEGRATION") != "1",
    reason="Set RUN_RAG_INTEGRATION=1 to run live RAG tests",
)


@pytest.fixture(scope="module")
def service():
    # Uses the project's existing service and persisted index.
    # Adapt this constructor if your implementation requires arguments.
    return IngestionService()


def test_alert_question_retrieves_relevant_evidence(service):
    results = service.search(
        "How to use Alert using javascript in HTML",
        top_k=10,
    )

    assert results, "Retriever returned no chunks"

    retrieved_text = "\n".join(
        document.page_content
        for document, _distance in results
    ).casefold()

    assert "alert(" in retrieved_text, (
        "No alert() example found in retrieved chunks. "
        "Inspect the index, chunking, and retrieval ranking."
    )


def test_answer_citations_reference_returned_sources(service):
    result = service.answer_question(
        "How to use Alert using javascript in HTML",
        top_k=5,
        answer_mode="strict",
    )

    assert isinstance(result, dict), (
        "Expected answer_question() to return a dictionary"
    )

    answer = result.get("answer", "")
    sources = result.get("sources", [])

    assert answer.strip(), "Generator returned an empty answer"
    assert sources, "Answer result contains no source records"

    # Source labels in the prompt are expected to be numbered
    # sequentially from 1 to the number of retrieved sources.
    available_source_ids = set(range(1, len(sources) + 1))

    validation = validate_citations(
        answer,
        available_source_ids,
    )

    assert validation["citation_present"], (
        f"Answer has no [Source N] citation: {answer}"
    )

    assert validation["all_citations_valid"], (
        f"Invalid citations: {validation['invalid_citation_ids']}"
    )
