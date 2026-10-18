
from fastapi.testclient import TestClient

from src.api.main import create_app


class FakeMemory:
    def __init__(self):
        self.turns = []

    def get_history(self, session_id):
        return [
            turn for turn in self.turns
            if turn["session_id"] == session_id
        ]

    def add_turn(self, session_id, question, answer):
        self.turns.append({
            "session_id": session_id,
            "question": question,
            "answer": answer,
        })


class FakeContextualizer:
    def contextualize(self, question, history):
        return question


class FakeRAGService:
    def answer_question(self, question, top_k, answer_mode):
        return {
            "answer": "The DOM represents a document as objects. [Source 1]",
            "sources": [{"source": "understanding-the-dom.pdf", "page": 7}],
        }

    def search(self, question, top_k):
        return [("fake document", 0.2)]


class FakeStreamingGenerator:
    def stream_answer(self, question, results):
        yield "The DOM "
        yield "represents a document. [Source 1]"


def make_client():
    memory = FakeMemory()

    app = create_app(
        service=FakeRAGService(),
        memory=memory,
        contextualizer=FakeContextualizer(),
        streaming_generator=FakeStreamingGenerator(),
    )

    return TestClient(app), memory


def test_health_endpoint():
    client, _ = make_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_returns_answer_and_saves_history():
    client, memory = make_client()

    response = client.post("/chat", json={
        "question": "What is the DOM?",
        "session_id": "test-session",
        "top_k": 3,
    })

    assert response.status_code == 200
    assert "[Source 1]" in response.json()["answer"]
    assert len(memory.turns) == 1


def test_invalid_top_k_is_rejected():
    client, _ = make_client()

    response = client.post("/chat", json={
        "question": "What is the DOM?",
        "session_id": "test-session",
        "top_k": 100,
    })

    assert response.status_code == 422


def test_stream_endpoint_emits_completion_event():
    client, memory = make_client()

    response = client.post("/chat/stream", json={
        "question": "What is the DOM?",
        "session_id": "test-session",
        "top_k": 3,
    })

    assert response.status_code == 200
    assert "event: token" in response.text
    assert "event: done" in response.text
    assert len(memory.turns) == 1
