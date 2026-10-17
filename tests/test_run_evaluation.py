from types import SimpleNamespace

from evaluation.run_evaluation import evaluate_case


class StubIngestionService:
    def search(self, question, top_k):
        document = SimpleNamespace(
            page_content="The feature is documented here.",
            metadata={"source": "guide.pdf", "page": 0},
        )
        return [(document, 0.25)]

    def answer_question(self, question, top_k, answer_mode):
        return {"answer": "The feature is documented. [Source 1]"}


def test_evaluate_case_computes_metrics_after_generating_answer():
    case = {
        "id": "documented_feature",
        "question": "What is documented?",
        "answerability": "supported",
        "required_evidence": ["documented here"],
    }

    result = evaluate_case(StubIngestionService(), case)

    assert result["answer"] == "The feature is documented. [Source 1]"
    assert result["answer_metrics"]["citation_present"] is True
    assert result["answer_metrics"]["abstention_detected"] is False
