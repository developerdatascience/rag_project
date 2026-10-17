
from evaluation.answer_metrics import (
    detects_abstention,
    evaluate_answer,
    has_citation,
)


def test_detects_source_citation():
    assert has_citation("The button triggers an alert. [Source 1]")


def test_missing_source_citation():
    assert not has_citation("The button triggers an alert.")


def test_detects_abstention():
    answer = "The provided documents do not provide enough information."
    assert detects_abstention(answer)


def test_does_not_detect_abstention_in_normal_answer():
    answer = "Use alert() to display a browser alert."
    assert not detects_abstention(answer)


def test_supported_answer_requires_manual_review():
    metrics = evaluate_answer(
        "Use alert() to display a browser alert. [Source 1]",
        "supported",
    )

    assert metrics["citation_present"] is True
    assert metrics["answer_review_required"] is True
    assert metrics["abstention_correct"] is None


def test_unsupported_answer_abstention_heuristic():
    metrics = evaluate_answer(
        r"The provided context does not contain the answer.",
        "unsupported",
    )

    assert metrics["abstention_detected"] is True
    assert metrics["abstention_correct"] is True


def test_pending_review_is_not_scored_for_abstention():
    metrics = evaluate_answer(
        "I cannot determine the answer.",
        "pending_review",
    )

    assert metrics["abstention_correct"] is None


def test_invalid_answerability_is_rejected():
    try:
        evaluate_answer("Some answer", "unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for invalid answerability")
