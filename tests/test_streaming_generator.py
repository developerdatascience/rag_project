
from types import SimpleNamespace

from langchain_core.documents import Document

from src.streaming_generator import StreamingAnswerGenerator


class FakeStreamingLLM:
    def stream(self, messages):
        yield SimpleNamespace(content="The DOM ")
        yield SimpleNamespace(content="represents a document.")
        yield SimpleNamespace(content=" [Source 1]")


def test_streaming_yields_incremental_text():
    generator = StreamingAnswerGenerator.__new__(
        StreamingAnswerGenerator
    )
    generator.llm = FakeStreamingLLM()

    results = [
        (
            Document(
                page_content="The DOM represents a document as objects.",
                metadata={"source": "understanding-the-dom.pdf", "page": 6},
            ),
            0.2,
        )
    ]

    fragments = list(
        generator.stream_answer("What is the DOM?", results)
    )

    assert fragments == [
        "The DOM ",
        "represents a document.",
        " [Source 1]",
    ]


def test_empty_results_abstain_without_model_call():
    generator = StreamingAnswerGenerator.__new__(
        StreamingAnswerGenerator
    )

    def unexpected_stream(_messages):
        raise AssertionError("Model should not be called.")

    generator.llm = SimpleNamespace(stream=unexpected_stream)

    output = "".join(
        generator.stream_answer("What is the DOM?", [])
    )

    assert "not provide enough information" in output


def test_empty_question_is_rejected():
    generator = StreamingAnswerGenerator.__new__(
        StreamingAnswerGenerator
    )

    output = generator.stream_answer("   ", [])

    try:
        next(output)
    except ValueError as exc:
        assert "Question cannot be empty" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
