
import json
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from src.api.schemas import ChatRequest, ChatResponse, HealthResponse
from src.conversation_memory import ConversationMemory
from src.guardrails import GuardrailError, InputGuardrails
from src.ingestion_service import IngestionService
from src.question_contextualizer import QuestionContextualizer
from src.streaming_generator import StreamingAnswerGenerator

logger = logging.getLogger(__name__)


def create_app(
    service=None,
    memory=None,
    contextualizer=None,
    streaming_generator=None,
):
    app = FastAPI(
        title="Document RAG API",
        version="1.0.0",
        description="Grounded question answering over indexed documents.",
    )

    # Dependency injection makes testing easier and avoids duplicating
    # the RAG implementation in the API layer.
    app.state.rag_service = service or IngestionService()
    app.state.memory = memory or ConversationMemory()
    app.state.contextualizer = (
        contextualizer or QuestionContextualizer()
    )
    app.state.streaming_generator = (
        streaming_generator or StreamingAnswerGenerator()
    )
    app.state.guardrails = InputGuardrails()

    @app.get("/health", response_model=HealthResponse)
    def health():
        # This checks process responsiveness, not LLM or database health.
        return {"status": "ok"}

    def prepare_question(request: ChatRequest):
        try:
            question = app.state.guardrails.validate_question(
                request.question
            )
            top_k = app.state.guardrails.validate_top_k(
                request.top_k
            )
        except GuardrailError as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc

        history = app.state.memory.get_history(request.session_id)

        search_question = app.state.contextualizer.contextualize(
            question=question,
            history=history,
        )

        return question, search_question, top_k

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest):
        question, search_question, top_k = prepare_question(request)

        try:
            result = app.state.rag_service.answer_question(
                question=search_question,
                top_k=top_k,
                answer_mode="strict",
            )

            answer = result["answer"]
            sources = result.get("sources", [])

            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError("The answer generator returned no answer.")

            app.state.memory.add_turn(
                session_id=request.session_id,
                question=question,
                answer=answer,
            )

            return ChatResponse(
                session_id=request.session_id,
                answer=answer,
                sources=sources,
            )

        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("RAG chat request failed")
            raise HTTPException(
                status_code=502,
                detail="The RAG service could not complete the request.",
            ) from exc

    @app.post("/chat/stream")
    def chat_stream(request: ChatRequest):
        question, search_question, top_k = prepare_question(request)

        def event_stream():
            answer_parts = []

            try:
                results = app.state.rag_service.search(
                    question=search_question,
                    top_k=top_k,
                )

                for fragment in (
                    app.state.streaming_generator.stream_answer(
                        search_question,
                        results,
                    )
                ):
                    answer_parts.append(fragment)
                    yield (
                        "event: token\n"
                        f"data: {json.dumps({'text': fragment})}\n\n"
                    )

                answer = "".join(answer_parts)

                if not answer.strip():
                    raise RuntimeError("The generated answer was empty.")

                app.state.memory.add_turn(
                    session_id=request.session_id,
                    question=question,
                    answer=answer,
                )

                source_references = []

                for index, (document, _) in enumerate(
                    results,
                    start=1,
                ):
                    metadata = document.metadata or {}
                    page = metadata.get("page")

                    source_references.append({
                        "source": metadata.get(
                            "source",
                            "Unknown document",
                        ),
                        "page": (
                            page + 1
                            if isinstance(page, int) and page >= 0
                            else None
                        ),
                        "excerpt": document.page_content[:500],
                        "citation_id": index,
                    })

                yield (
                    "event: sources\n"
                    f"data: {json.dumps({'sources': source_references})}\n\n"
                )

                yield (
                    "event: done\n"
                    f"data: {json.dumps({'session_id': request.session_id})}\n\n"
                )

            except Exception:
                logger.exception("RAG streaming request failed")
                yield (
                    "event: error\n"
                    + "data: "
                    + json.dumps({
                        "message": (
                            "Answer generation failed. "
                            "The response may be incomplete."
                        )
                    })
                    + "\n\n"
                )

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return app


app = create_app()
