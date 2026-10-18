
from evaluation.citation_validator import (
    extract_citation_ids,
    validate_citations,
)


def test_extracts_unique_citation_ids_in_order():
    answer = "The alert works. [Source 2] Another detail. [Source 1] [Source 2]"

    assert extract_citation_ids(answer) == [2, 1]


def test_all_citations_exist():
    result = validate_citations(
        "The answer is supported. [Source 1] [Source 2]",
        available_source_ids={1, 2},
    )

    assert result["all_citations_valid"] is True
    assert result["invalid_citation_ids"] == []


def test_flags_unknown_source():
    result = validate_citations(
        "The answer is supported. [Source 3]",
        available_source_ids={1, 2},
    )

    assert result["all_citations_valid"] is False
    assert result["invalid_citation_ids"] == [3]


def test_detects_missing_citations():
    result = validate_citations(
        "The answer contains no source references.",
        available_source_ids={1, 2},
    )

    assert result["citation_present"] is False
    assert result["all_citations_valid"] is False


def test_rejects_citation_when_no_sources_are_available():
    result = validate_citations(
        "The answer says something. [Source 1]",
        available_source_ids=set(),
    )

    assert result["all_citations_valid"] is False
    assert result["invalid_citation_ids"] == [1]
