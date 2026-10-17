
import re


CITATION_PATTERN = re.compile(
    r"\[Source\s+\d+\]",
    flags=re.IGNORECASE,
)

ABSTENTION_PATTERNS = [
    r"not enough information",
    r"does not provide enough information",
    r"cannot answer from (?:the )?(?:provided )?context",
    r"not supported by (?:the )?(?:provided )?context",
    r"(?:the )?(?:provided )?context does not contain (?:the )?answer",
    r"the (?:provided )?documents? do not (?:contain|provide)",
    r"unable to answer from the (?:provided )?documents?",
]


def has_citation(answer: str) -> bool:
    """Return True if the answer includes a source citation marker."""
    return bool(CITATION_PATTERN.search(answer or ""))


def detects_abstention(answer: str) -> bool:
    """
    Heuristic only: detect common phrases used when the model abstains.

    This does not establish that the answer truly abstains or that
    an apparently confident answer is unsupported.
    """
    text = answer or ""

    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in ABSTENTION_PATTERNS
    )


def evaluate_answer(answer: str, answerability: str) -> dict:
    """
    Calculate applicable answer-level metrics.

    answerability must be:
      - supported
      - unsupported
      - pending_review
    """
    valid_states = {"supported", "unsupported", "pending_review"}

    if answerability not in valid_states:
        raise ValueError(
            f"Invalid answerability state: {answerability!r}"
        )

    citation_present = has_citation(answer)
    abstention_detected = detects_abstention(answer)

    if answerability == "supported":
        return {
            "citation_present": citation_present,
            "abstention_detected": abstention_detected,
            "abstention_correct": None,
            "answer_review_required": True,
        }

    if answerability == "unsupported":
        return {
            "citation_present": citation_present,
            "abstention_detected": abstention_detected,
            "abstention_correct": abstention_detected,
            "answer_review_required": True,
        }

    return {
        "citation_present": citation_present,
        "abstention_detected": abstention_detected,
        "abstention_correct": None,
        "answer_review_required": True,
    }
