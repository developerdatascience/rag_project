
from evaluation.qa_dataset import EVALUATION_CASES


VALID_STATES = {"supported", "unsupported", "pending_review"}


def test_evaluation_case_ids_are_unique():
    ids = [case["id"] for case in EVALUATION_CASES]
    assert len(ids) == len(set(ids))


def test_every_case_has_required_fields():
    required_fields = {
        "id",
        "question",
        "answerability",
        "required_evidence",
    }

    for case in EVALUATION_CASES:
        assert required_fields.issubset(case.keys())
        assert case["id"]
        assert case["question"]
        assert case["answerability"] in VALID_STATES


def test_supported_cases_have_evidence_annotations():
    for case in EVALUATION_CASES:
        if case["answerability"] == "supported":
            assert case["required_evidence"], (
                f"{case['id']} is supported but has no evidence labels"
            )


def test_unreviewed_cases_are_not_marked_supported_by_default():
    for case in EVALUATION_CASES:
        if case["id"] in {
            "javascript_api_fetch",
            "india_capital",
        }:
            assert case["answerability"] == "pending_review"
