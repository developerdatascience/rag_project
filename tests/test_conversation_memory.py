
import pytest

from src.conversation_memory import ConversationMemory


@pytest.fixture
def memory(tmp_path):
    return ConversationMemory(
        db_path=str(tmp_path / "conversations.db"),
        max_history_turns=2,
    )


def test_history_is_persisted(memory, tmp_path):
    memory.add_turn("session-a", "What is the DOM?", "A document model.")

    restarted = ConversationMemory(
        db_path=str(tmp_path / "conversations.db"),
        max_history_turns=2,
    )

    history = restarted.get_history("session-a")

    assert len(history) == 1
    assert history[0]["question"] == "What is the DOM?"


def test_history_is_isolated_by_session(memory):
    memory.add_turn("session-a", "Question A", "Answer A")
    memory.add_turn("session-b", "Question B", "Answer B")

    assert len(memory.get_history("session-a")) == 1
    assert memory.get_history("session-a")[0]["question"] == "Question A"


def test_history_is_chronological_and_limited(memory):
    memory.add_turn("session-a", "Q1", "A1")
    memory.add_turn("session-a", "Q2", "A2")
    memory.add_turn("session-a", "Q3", "A3")

    history = memory.get_history("session-a")

    assert [turn["question"] for turn in history] == ["Q2", "Q3"]


def test_clear_history_only_clears_requested_session(memory):
    memory.add_turn("session-a", "Q1", "A1")
    memory.add_turn("session-b", "Q2", "A2")

    memory.clear_history("session-a")

    assert memory.get_history("session-a") == []
    assert len(memory.get_history("session-b")) == 1


@pytest.mark.parametrize("session_id", ["", "   ", None])
def test_invalid_session_id_is_rejected(memory, session_id):
    with pytest.raises(ValueError):
        memory.get_history(session_id)
