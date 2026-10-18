
import pytest

from src.guardrails import (
    GuardrailConfig,
    GuardrailError,
    InputGuardrails,
)


@pytest.fixture
def guardrails():
    return InputGuardrails(
        GuardrailConfig(
            max_question_length=100,
            min_question_length=1,
            max_top_k=10,
        )
    )


def test_question_is_trimmed(guardrails):
    assert guardrails.validate_question("  What is the DOM?  ") == (
        "What is the DOM?"
    )


def test_empty_question_is_rejected(guardrails):
    with pytest.raises(GuardrailError):
        guardrails.validate_question("   ")


def test_non_string_question_is_rejected(guardrails):
    with pytest.raises(GuardrailError):
        guardrails.validate_question(None)


def test_question_over_limit_is_rejected(guardrails):
    with pytest.raises(GuardrailError):
        guardrails.validate_question("x" * 101)


@pytest.mark.parametrize("top_k", [1, 3, 10])
def test_valid_top_k(guardrails, top_k):
    assert guardrails.validate_top_k(top_k) == top_k


@pytest.mark.parametrize("top_k", [0, -1, 11, True, 2.5, "3"])
def test_invalid_top_k_is_rejected(guardrails, top_k):
    with pytest.raises(GuardrailError):
        guardrails.validate_top_k(top_k)
