
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2000,
        description="Question to answer using indexed documents.",
    )
    session_id: str = Field(
        min_length=1,
        max_length=128,
        description="Conversation session identifier.",
    )
    top_k: int = Field(default=3, ge=1, le=20)


class SourceReference(BaseModel):
    source: str
    page: int | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[dict] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
