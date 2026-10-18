
from src.conversation_memory import ConversationMemory
from src.question_contextualizer import QuestionContextualizer


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def invoke(self, prompt):
        assert "How is it different from HTML?" in prompt
        return FakeResponse("How does the DOM differ from HTML?")


def test_follow_up_question_uses_conversation_history(tmp_path):
    memory = ConversationMemory(
        db_path=str(tmp_path / "conversations.db")
    )

    memory.add_turn(
        session_id="test-session",
        question="What is the DOM?",
        answer="The DOM represents a document as a tree of objects.",
    )

    contextualizer = QuestionContextualizer.__new__(
        QuestionContextualizer
    )
    contextualizer.llm = FakeLLM()

    history = memory.get_history("test-session")

    rewritten = contextualizer.contextualize(
        "How is it different from HTML?",
        history,
    )

    assert rewritten == "How does the DOM differ from HTML?"
