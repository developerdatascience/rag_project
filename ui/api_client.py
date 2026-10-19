
import json
import os
from collections.abc import Iterator

import requests


class RAGAPIClient:
    def __init__(
        self,
        base_url: str | None = None,
        timeout: int = 120,
    ):
        self.base_url = (
            base_url or os.getenv(
                "RAG_API_URL",
                "http://127.0.0.1:8000",
            )
        ).rstrip("/")
        self.timeout = timeout
        self.last_sources: list[dict] = []

    def health_check(self) -> bool:
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=5,
            )
            response.raise_for_status()
            return response.json().get("status") == "ok"
        except requests.RequestException:
            return False

    def chat(
        self,
        question: str,
        session_id: str,
        top_k: int = 3,
    ) -> dict:
        response = requests.post(
            f"{self.base_url}/chat",
            json={
                "question": question,
                "session_id": session_id,
                "top_k": top_k,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def stream_chat(
        self,
        question: str,
        session_id: str,
        top_k: int = 3,
    ) -> Iterator[str]:
        self.last_sources = []
        current_event = None

        with requests.post(
            f"{self.base_url}/chat/stream",
            json={
                "question": question,
                "session_id": session_id,
                "top_k": top_k,
            },
            stream=True,
            timeout=(5, self.timeout),
            headers={"Accept": "text/event-stream"},
        ) as response:
            response.raise_for_status()

            for raw_line in response.iter_lines(decode_unicode=True):
                if raw_line is None:
                    continue

                line = raw_line.strip()

                if not line:
                    current_event = None
                    continue

                if line.startswith("event:"):
                    current_event = line[len("event:"):].strip()
                    continue

                if not line.startswith("data:"):
                    continue

                payload = line[len("data:"):].strip()

                try:
                    data = json.loads(payload)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        "The API returned malformed SSE data."
                    ) from exc

                if current_event == "token":
                    text = data.get("text", "")
                    if text:
                        yield text

                elif current_event == "sources":
                    self.last_sources = data.get("sources", [])

                elif current_event == "error":
                    raise RuntimeError(
                        data.get(
                            "message",
                            "The streamed response failed.",
                        )
                    )

                elif current_event == "done":
                    return
