from __future__ import annotations

from collections.abc import AsyncIterator
from time import perf_counter
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status

from backend.clients import OpenAICompatibleLLMProvider, WikipediaKnowledgeProvider
from backend.core import get_logger
from backend.schemas import AnswerResponse, ChatRequest
from backend.services import AgentService
from backend.services.agent import AgentExecutionError

router = APIRouter()
logger = get_logger(__name__)


async def get_agent_service() -> AsyncIterator[AgentService]:
    provider = WikipediaKnowledgeProvider()
    llm_provider = OpenAICompatibleLLMProvider()
    try:
        yield AgentService(provider, llm_provider=llm_provider)
    finally:
        await provider.aclose()


@router.get("/health")
async def healthcheck() -> dict[str, str]:
    logger.info("healthcheck ok")
    return {"status": "ok"}


@router.post(
    "/api/v1/answer",
    response_model=AnswerResponse,
    status_code=status.HTTP_200_OK,
)
async def answer_question(
    payload: ChatRequest,
    service: Annotated[AgentService, Depends(get_agent_service)],
) -> AnswerResponse:
    request_id = str(uuid4())
    started_at = perf_counter()
    logger.info("http request received", extra={"request_id": request_id})
    try:
        response = await service.answer_dialog(payload.messages, request_id=request_id)
        logger.info(
            "http request completed",
            extra={
                "request_id": request_id,
                "duration_ms": round((perf_counter() - started_at) * 1000, 2),
            },
        )
        return response
    except AgentExecutionError as exc:
        logger.warning(
            "http request failed",
            extra={
                "request_id": request_id,
                "duration_ms": round((perf_counter() - started_at) * 1000, 2),
            },
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
