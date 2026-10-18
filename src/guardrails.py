
from dataclasses import dataclass


@dataclass(frozen=True)
class GuardrailConfig:
    max_question_length: int = 2_000
    min_question_length: int = 1
    max_top_k: int = 20


class GuardrailError(ValueError):
    """Raised when user input violates RAG request constraints."""


class InputGuardrails:
    def __init__(self, config: GuardrailConfig | None = None):
        self.config = config or GuardrailConfig()

    def validate_question(self, question: str) -> str:
        if not isinstance(question, str):
            raise GuardrailError("Question must be a string.")

        question = question.strip()

        if len(question) < self.config.min_question_length:
            raise GuardrailError("Question cannot be empty.")

        if len(question) > self.config.max_question_length:
            raise GuardrailError(
                f"Question cannot exceed "
                f"{self.config.max_question_length} characters."
            )

        return question

    def validate_top_k(self, top_k: int) -> int:
        # bool is a subclass of int in Python, so reject it explicitly.
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise GuardrailError("top_k must be an integer.")

        if not 1 <= top_k <= self.config.max_top_k:
            raise GuardrailError(
                f"top_k must be between 1 and "
                f"{self.config.max_top_k}."
            )

        return top_k
