from __future__ import annotations

from collections.abc import Awaitable, Callable
from time import perf_counter
from typing import cast

from pydantic import BaseModel, Field

from backend.agent.state import AgentState, Confidence
from backend.clients import LLMProvider

from .shared import MissingLLMProviderError, append_trace, log_extra, logger, make_trace, summarize_for_log


class AnswerDraftPayload(BaseModel):
    answer: str = Field(min_length=1)
    confidence: str = Field(pattern="^(low|medium|high)$")


def _draft_answer_messages(state: AgentState) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You write the final user-facing answer for an agent.\n"
                "Return JSON only with keys: answer, confidence.\n"
                "Keep the answer faithful to provided context.\n"
                "confidence must be one of: low, medium, high."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{state['question']}\n\n"
                "Structured context:\n"
                f"- topic: {state.get('topic', '')}\n"
                f"- intent: {state.get('intent', '')}\n"
                f"- plan: {state.get('plan', [])}\n"
                f"- assumptions: {state.get('assumptions', [])}\n"
                f"- findings: {state.get('findings', [])}\n"
                f"- self_check_notes: {state.get('self_check_notes', [])}\n"
                f"- warnings: {state.get('warnings', [])}\n\n"
                "Answer requirements:\n"
                "- concise response;\n"
                "- do not invent missing facts;\n"
                "- include key assumption if relevant;\n"
                "- return JSON only."
            ),
        },
    ]


def make_finalize_answer(
    llm_provider: LLMProvider | None,
) -> Callable[[AgentState], Awaitable[AgentState]]:
    if llm_provider is None:
        raise MissingLLMProviderError("llm_provider is required for finalize_answer")

    async def finalize_answer(state: AgentState) -> AgentState:
        node_started_at = perf_counter()
        logger.info("node started", extra=log_extra(state, node="finalize_answer"))
        warnings = state.get("warnings", [])
        llm_started_at = perf_counter()
        logger.info("llm call started", extra=log_extra(state, node="finalize_answer"))
        messages = _draft_answer_messages(state)
        draft_payload = await llm_provider.generate_json(
            messages=messages,
            request_id=state.get("request_id", "unknown"),
            node="finalize_answer",
            stage="draft_answer",
        )
        draft = AnswerDraftPayload.model_validate(draft_payload)
        logger.info(
            "llm call completed",
            extra=log_extra(
                state,
                node="finalize_answer",
                duration_ms=round((perf_counter() - llm_started_at) * 1000, 2),
                confidence=draft.confidence,
            ),
        )
        confidence = cast(Confidence, draft.confidence)
        result: AgentState = {
            "answer": draft.answer,
            "confidence": confidence,
            "warnings": warnings,
            "trace": append_trace(
                state,
                make_trace(
                    "finalize_answer",
                    "Собран итоговый ответ через LLM.",
                    {
                        "confidence": confidence,
                        "mode": "llm",
                    },
                ),
            ),
        }
        logger.info(
            "node result",
            extra=log_extra(state, node="finalize_answer", result=summarize_for_log(result)),
        )
        logger.info(
            "node completed",
            extra=log_extra(
                state,
                node="finalize_answer",
                duration_ms=round((perf_counter() - node_started_at) * 1000, 2),
            ),
        )
        return result

    return finalize_answer
