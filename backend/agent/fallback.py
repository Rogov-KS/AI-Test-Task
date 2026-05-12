from __future__ import annotations

from pydantic import BaseModel, Field

from backend.clients import LLMProvider
from backend.core import get_logger
from backend.schemas.request import ChatMessage
from backend.schemas.response import AnswerResponse, SelfCheck, TraceItem

logger = get_logger(__name__)


class DialogFallbackPayload(BaseModel):
    answer: str = Field(min_length=1)
    confidence: str = Field(pattern="^(low|medium|high)$")


def _stringify_dialog(messages: list[ChatMessage]) -> str:
    return "\n".join(f"{message.role}: {message.content.strip()}" for message in messages if message.content.strip())


def _build_fallback_messages(messages: list[ChatMessage]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You answer the user's question directly using your own knowledge.\n"
                "Return JSON only with keys: answer, confidence.\n"
                "Keep the answer concise and useful.\n"
                "If a concrete fact, number, size, or other specific detail is relevant, state it plainly.\n"
                "Do not mention memory, training data, internal knowledge, or that the answer was produced without external research.\n"
                "Do not invent facts. If certainty is limited, keep the wording cautious.\n"
                "confidence must be one of: low, medium, high."
            ),
        },
        {
            "role": "user",
            "content": (
                "Conversation:\n"
                f"{_stringify_dialog(messages)}\n\n"
                "Write the next assistant reply.\n"
                "Requirements:\n"
                "- answer the latest user request;\n"
                "- keep it concise;\n"
                "- no meta-commentary about data sources or memory;\n"
                "- return JSON only."
            ),
        },
    ]


async def answer_dialog_with_fallback(
    messages: list[ChatMessage],
    llm_provider: LLMProvider | None,
    request_id: str | None = None,
) -> AnswerResponse:
    if llm_provider is None:
        raise ValueError("llm_provider is required for dialog fallback")

    logger.info(
        "dialog fallback started",
        extra={"request_id": request_id or "unknown", "messages_count": len(messages)},
    )
    payload = await llm_provider.generate_json(
        messages=_build_fallback_messages(messages),
        request_id=request_id,
        node="dialog_fallback",
        stage="direct_answer",
    )
    result = DialogFallbackPayload.model_validate(payload)
    logger.info(
        "dialog fallback completed",
        extra={
            "request_id": request_id or "unknown",
            "confidence": result.confidence,
        },
    )
    return AnswerResponse(
        answer=result.answer,
        message=ChatMessage(role="assistant", content=result.answer),
        confidence=result.confidence,
        self_check=SelfCheck(
            passed=True,
            notes=["Ответ подготовлен через прямой LLM-фоллбек без графового исследования."],
        ),
        assumptions=[],
        findings=[],
        warnings=[],
        trace=[
            TraceItem.model_validate(
                {
                    "step": "dialog_fallback",
                    "summary": "Подготовлен прямой ответ через LLM-фоллбек.",
                    "details": {
                        "mode": "direct_llm",
                        "messages_count": len(messages),
                        "confidence": result.confidence,
                    },
                }
            )
        ],
    )
