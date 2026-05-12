from __future__ import annotations

from time import perf_counter

from backend.agent import answer_dialog_with_fallback, build_agent_graph
from backend.agent.state import AgentState
from backend.clients import KnowledgeProvider, LLMProvider, WikipediaLookupError
from backend.core import get_logger, settings
from backend.schemas.request import ChatMessage
from backend.schemas.response import AnswerResponse, SelfCheck, TraceItem

logger = get_logger(__name__)


class AgentExecutionError(RuntimeError):
    pass


class AgentService:
    def __init__(
        self,
        provider: KnowledgeProvider,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        self._provider = provider
        self._llm_provider = llm_provider
        self._graph = build_agent_graph(provider, llm_provider)

    @staticmethod
    def _extract_question(messages: list[ChatMessage]) -> str:
        for message in reversed(messages):
            if message.role == "user" and message.content.strip():
                return message.content.strip()
        raise AgentExecutionError("dialog must contain at least one user message")

    async def answer(self, question: str, request_id: str | None = None) -> AnswerResponse:
        request_id = request_id or "unknown"
        logger.info("agent request started", extra={"request_id": request_id})
        started_at = perf_counter()
        initial_state: AgentState = {
            "request_id": request_id,
            "question": question,
            "max_research_iterations": settings.max_research_iterations,
            "warnings": [],
            "trace": [],
        }

        try:
            final_state = await self._graph.ainvoke(initial_state)
        except (WikipediaLookupError, ValueError) as exc:
            logger.exception("agent request failed", extra={"request_id": request_id})
            raise AgentExecutionError(str(exc)) from exc

        logger.info(
            "agent request completed",
            extra={
                "request_id": request_id,
                "duration_ms": round((perf_counter() - started_at) * 1000, 2),
            },
        )

        return AnswerResponse(
            answer=final_state["answer"],
            message=ChatMessage(role="assistant", content=final_state["answer"]),
            confidence=final_state["confidence"],
            self_check=SelfCheck(
                passed=final_state["self_check_passed"],
                notes=final_state["self_check_notes"],
            ),
            assumptions=final_state["assumptions"],
            findings=final_state.get("findings", []),
            warnings=final_state.get("warnings", []),
            trace=[TraceItem.model_validate(item) for item in final_state["trace"]],
        )

    async def answer_dialog(
        self,
        messages: list[ChatMessage],
        request_id: str | None = None,
    ) -> AnswerResponse:
        self._extract_question(messages)
        if settings.agent_dialog_fallback_enabled:
            return await answer_dialog_with_fallback(
                messages=messages,
                llm_provider=self._llm_provider,
                request_id=request_id,
            )
        question = self._extract_question(messages)
        return await self.answer(question, request_id=request_id)
