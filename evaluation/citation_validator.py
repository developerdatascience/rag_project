
import re


CITATION_PATTERN = re.compile(
    r"\[Source\s+(\d+)\]",
    flags=re.IGNORECASE,
)


def extract_citation_ids(answer: str) -> list[int]:
    """Return unique citation IDs in the order they appear."""
    return list(dict.fromkeys(
        int(match)
        for match in CITATION_PATTERN.findall(answer or "")
    ))


def validate_citations(
    answer: str,
    available_source_ids: set[int],
) -> dict:
    """
    Check whether citation IDs refer to available sources.

    This validates reference integrity, not whether the cited source
    semantically supports the claim.
    """
    citation_ids = extract_citation_ids(answer)

    valid_ids = [
        citation_id
        for citation_id in citation_ids
        if citation_id in available_source_ids
    ]

    invalid_ids = [
        citation_id
        for citation_id in citation_ids
        if citation_id not in available_source_ids
    ]

    return {
        "citation_present": bool(citation_ids),
        "citation_ids": citation_ids,
        "valid_citation_ids": valid_ids,
        "invalid_citation_ids": invalid_ids,
        "all_citations_valid": (
            bool(citation_ids) and not invalid_ids
        ),
    }
