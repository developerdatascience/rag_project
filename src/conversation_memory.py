
import sqlite3
from pathlib import Path
from typing import Any


class ConversationMemory:
    """Persist conversation turns independently of the RAG index."""

    def __init__(
        self,
        db_path: str = "data/conversations.db",
        max_history_turns: int = 10,
    ):
        if max_history_turns < 1:
            raise ValueError("max_history_turns must be at least 1.")

        self.db_path = db_path
        self.max_history_turns = max_history_turns

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT
                        (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                )
            """)

            connection.execute("""
                CREATE INDEX IF NOT EXISTS
                idx_conversation_session_id
                ON conversation_turns(session_id, id)
            """)

    def add_turn(
        self,
        session_id: str,
        question: str,
        answer: str,
    ) -> None:
        self._validate_session_id(session_id)

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Answer cannot be empty.")

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversation_turns
                    (session_id, question, answer)
                VALUES (?, ?, ?)
                """,
                (session_id, question.strip(), answer.strip()),
            )

    def get_history(self, session_id: str) -> list[dict[str, Any]]:
        self._validate_session_id(session_id)

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT question, answer, created_at
                FROM conversation_turns
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, self.max_history_turns),
            ).fetchall()

        # The database returns newest first; the LLM needs chronological order.
        return [
            dict(row)
            for row in reversed(rows)
        ]

    def clear_history(self, session_id: str) -> None:
        self._validate_session_id(session_id)

        with self._connect() as connection:
            connection.execute(
                "DELETE FROM conversation_turns WHERE session_id = ?",
                (session_id,),
            )

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("A valid session_id is required.")
