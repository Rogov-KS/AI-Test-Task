from __future__ import annotations

from collections.abc import Awaitable, Callable
from time import perf_counter

from pydantic import BaseModel, Field

from backend.agent.state import AgentState
from backend.clients import LLMProvider
from backend.core import settings

from .shared import MissingLLMProviderError, append_trace, log_extra, logger, make_trace, summarize_for_log


class BuildPlanPayload(BaseModel):
    topic: str = Field(min_length=1)
    intent: str = Field(min_length=1)
    assumptions: list[str] = Field(min_length=1)
    plan: list[str] = Field(min_length=1)
    wikipedia_queries: list[str] = Field(min_length=1)


def _build_plan_messages(question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You create a short research plan for a general-purpose Q&A agent.\n"
                "Return JSON only with keys: topic, intent, assumptions, plan, wikipedia_queries.\n"
                "Rules:\n"
                "- topic must be a concise noun phrase for the subject.\n"
                "- intent must describe what user wants in one short phrase.\n"
                "- assumptions must contain 1-3 short factual assumptions.\n"
                "- plan must contain 3-6 ordered short steps.\n"
                "- plan must include fact collection/check and final consistency self-check.\n"
                "- wikipedia_queries must contain 1-5 concrete entity names.\n"
                "- wikipedia_queries should be plain entity titles, without prefixes/suffixes.\n"
                "- wikipedia_queries must be in English only.\n"
                "- Prefer concrete entities from the user question."
            ),
        },
        {"role": "user", "content": f"Question:\n{question}\n\nReturn JSON only."},
    ]


def make_build_plan(
    llm_provider: LLMProvider | None,
) -> Callable[[AgentState], Awaitable[AgentState]]:
    if llm_provider is None:
        raise MissingLLMProviderError("llm_provider is required for build_plan")

    async def build_plan(state: AgentState) -> AgentState:
        node_started_at = perf_counter()
        logger.info("node started", extra=log_extra(state, node="build_plan"))
        warnings = state.get("warnings", [])
        request_id = state.get("request_id", "unknown")

        llm_started_at = perf_counter()
        messages = _build_plan_messages(state["question"])
        logger.info(
            "llm call started",
            extra=log_extra(state, node="build_plan", stage="build_plan"),
        )
        build_plan_payload = await llm_provider.generate_json(
            messages=messages,
            request_id=request_id,
            node="build_plan",
            stage="build_plan",
        )
        parsed = BuildPlanPayload.model_validate(build_plan_payload)
        logger.info(
            "llm call completed",
            extra=log_extra(
                state,
                node="build_plan",
                stage="build_plan",
                duration_ms=round((perf_counter() - llm_started_at) * 1000, 2),
                topic=parsed.topic,
                intent=parsed.intent,
                plan_steps=len(parsed.plan),
            ),
        )

        result: AgentState = {
            "topic": parsed.topic,
            "intent": parsed.intent,
            "assumptions": parsed.assumptions,
            "plan": parsed.plan,
            "pending_wikipedia_queries": parsed.wikipedia_queries,
            "wikipedia_query_history": [],
            "findings": [],
            "enough_information": False,
            "max_research_iterations": state.get(
                "max_research_iterations",
                settings.max_research_iterations,
            ),
            "warnings": warnings,
            "trace": append_trace(
                state,
                make_trace(
                    "build_plan",
                    "Собран план ответа и гипотезы через LLM.",
                    {
                        "plan": parsed.plan,
                        "topic": parsed.topic,
                        "intent": parsed.intent,
                        "assumptions": parsed.assumptions,
                        "wikipedia_queries": parsed.wikipedia_queries,
                        "mode": "llm",
                    },
                ),
            ),
        }
        logger.info(
            "node result",
            extra=log_extra(state, node="build_plan", result=summarize_for_log(result)),
        )
        logger.info(
            "node completed",
            extra=log_extra(
                state,
                node="build_plan",
                duration_ms=round((perf_counter() - node_started_at) * 1000, 2),
            ),
        )
        return result

    return build_plan
