from __future__ import annotations

from collections.abc import Awaitable, Callable
import json
from time import perf_counter

from pydantic import BaseModel, Field

from backend.agent.state import AgentState
from backend.clients import KnowledgeProvider, LLMProvider, WikipediaLookupError
from backend.core import settings

from .shared import (
    MissingLLMProviderError,
    append_trace,
    append_warnings,
    log_extra,
    logger,
    make_trace,
    merge_unique_preserve_order,
    summarize_for_log,
)


class ResearchDecisionPayload(BaseModel):
    enough_information: bool
    findings: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    next_wikipedia_queries: list[str] = Field(default_factory=list)


class DocumentFindingsPayload(BaseModel):
    findings: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)


def _tokens_to_chars(token_budget: int) -> int:
    return max(240, token_budget * 4)


def _truncate_text(value: str, limit: int = 750) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]}...<truncated {len(value) - limit} chars>"


def _limit_items(items: list[str], *, max_items: int, max_chars: int) -> list[str]:
    limited = [_truncate_text(str(item), limit=max_chars) for item in items[:max_items]]
    if len(items) > max_items:
        limited.append(f"...<truncated {len(items) - max_items} items>")
    return limited


def _extract_findings_messages(
    *,
    question: str,
    query: str,
    title: str,
    extract: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You extract grounded findings from one Wikipedia document.\n"
                "Return JSON only with keys: findings, missing_information.\n"
                "Rules:\n"
                "- findings: 0-5 concise factual bullets from provided excerpt only.\n"
                "- Do not infer facts absent from excerpt.\n"
                "- missing_information: 0-4 short bullets about what this document still does not provide for the user question.\n"
                "- If excerpt says document is missing/unavailable, keep findings empty."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Wikipedia query:\n{query}\n\n"
                f"Resolved title:\n{title}\n\n"
                f"Document excerpt:\n{json.dumps(extract, ensure_ascii=False)}\n\n"
                "Return JSON only."
            ),
        },
    ]


def _research_decision_messages(
    *,
    question: str,
    plan: list[str],
    assumptions: list[str],
    previous_findings: list[str],
    iteration_findings: list[str],
    executed_queries: list[str],
) -> list[dict[str, str]]:
    compact_plan = _limit_items(plan, max_items=6, max_chars=220)
    compact_assumptions = _limit_items(assumptions, max_items=4, max_chars=220)
    compact_executed_queries = _limit_items(executed_queries, max_items=8, max_chars=120)
    compact_findings = _limit_items(previous_findings, max_items=8, max_chars=220)
    compact_iteration_findings = _limit_items(iteration_findings, max_items=10, max_chars=220)

    return [
        {
            "role": "system",
            "content": (
                "You decide whether the collected findings are enough to answer the question.\n"
                "Return JSON only with keys: enough_information, findings, missing_information, next_wikipedia_queries.\n"
                "Rules:\n"
                "- enough_information: true only if facts are sufficient for a grounded answer.\n"
                "- findings: 0-8 concise factual bullets that improve/clean the collected findings.\n"
                "- missing_information: short bullet-like strings describing what is still missing.\n"
                "- next_wikipedia_queries must contain plain entity titles only.\n"
                "- next_wikipedia_queries must be in English only.\n"
                "- Return empty next_wikipedia_queries if enough_information is true.\n"
                "- If enough_information is false, you MUST return at least one next_wikipedia_queries item.\n"
                "- Avoid repeating already executed queries unless absolutely necessary."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Plan:\n{compact_plan}\n\n"
                f"Assumptions:\n{compact_assumptions}\n\n"
                f"Executed wikipedia queries:\n{compact_executed_queries}\n\n"
                f"Previous findings:\n{compact_findings}\n\n"
                f"Current iteration findings:\n{compact_iteration_findings}\n\n"
                "Return JSON only."
            ),
        },
    ]


def _normalize_query(query: str) -> str:
    return query.strip().lower()


async def _resolve_wikipedia_query(provider: KnowledgeProvider, query: str) -> dict[str, str]:
    entity = query.strip()
    if not entity:
        raise ValueError("wikipedia query entity must not be empty")
    return await provider.get_entity_page(entity)


def make_research_loop(
    provider: KnowledgeProvider,
    llm_provider: LLMProvider | None,
) -> Callable[[AgentState], Awaitable[AgentState]]:
    if llm_provider is None:
        raise MissingLLMProviderError("llm_provider is required for research_loop")

    async def research_loop(state: AgentState) -> AgentState:
        node_started_at = perf_counter()
        logger.info("node started", extra=log_extra(state, node="research_loop"))
        research_iterations = state.get("research_iterations", 0) + 1
        warnings = state.get("warnings", [])

        pending_queries = state.get("pending_wikipedia_queries", [])
        query_history = state.get("wikipedia_query_history", [])
        findings = [*state.get("findings", [])]
        newly_executed_queries: list[str] = []
        fetched_documents: list[dict[str, str]] = []
        iteration_findings: list[str] = []
        extraction_missing_information: list[str] = []

        for query in pending_queries:
            query_token = _normalize_query(query)
            if query_token in query_history or query_token in newly_executed_queries:
                continue
            try:
                page = await _resolve_wikipedia_query(provider, query)
            except WikipediaLookupError:
                page = {
                    "query": query,
                    "title": query,
                    "extract": "такого документа не существует",
                }
            fetched_documents.append(page)
            newly_executed_queries.append(query_token)

        all_executed_queries = [*query_history, *newly_executed_queries]
        if not fetched_documents and not pending_queries:
            warnings = append_warnings(
                state,
                [
                    "Исследовательский цикл завершен раньше: нет pending wikipedia-запросов.",
                ],
            )
            result: AgentState = {
                "findings": findings,
                "pending_wikipedia_queries": [],
                "wikipedia_query_history": all_executed_queries,
                "enough_information": False,
                "research_iterations": state.get("max_research_iterations", research_iterations),
                "warnings": warnings,
                "trace": append_trace(
                    state,
                    make_trace(
                        "research_loop",
                        "Цикл исследования завершен: новые запросы для Wikipedia отсутствуют.",
                        {
                            "iteration": research_iterations,
                            "pending_queries_in": pending_queries,
                            "executed_queries": newly_executed_queries,
                            "next_queries": [],
                            "documents_count": 0,
                            "findings_count": len(findings),
                            "enough_information": False,
                            "missing_information": [
                                "Невозможно продолжить поиск: отсутствуют pending wikipedia-запросы.",
                            ],
                        },
                    ),
                ),
            }
            logger.info(
                "node result",
                extra=log_extra(state, node="research_loop", result=summarize_for_log(result)),
            )
            logger.info(
                "node completed",
                extra=log_extra(
                    state,
                    node="research_loop",
                    duration_ms=round((perf_counter() - node_started_at) * 1000, 2),
                    enough_information=False,
                    iteration=research_iterations,
                    early_stop=True,
                ),
            )
            return result

        # Per-document extraction: one LLM call per query-document pair.
        extract_budget_chars = settings.llm_max_tokens
        for document in fetched_documents:
            extract_excerpt = document["extract"][:extract_budget_chars]
            extract_payload = await llm_provider.generate_json(
                messages=_extract_findings_messages(
                    question=state["question"],
                    query=document["query"],
                    title=document["title"],
                    extract=extract_excerpt,
                ),
                request_id=state.get("request_id", "unknown"),
                node="research_loop",
                stage="extract_document_findings",
            )
            parsed_extract = DocumentFindingsPayload.model_validate(extract_payload)
            iteration_findings = merge_unique_preserve_order(iteration_findings, parsed_extract.findings)
            extraction_missing_information = merge_unique_preserve_order(
                extraction_missing_information,
                parsed_extract.missing_information,
            )

        llm_started_at = perf_counter()
        logger.info(
            "llm call started",
            extra=log_extra(state, node="research_loop", stage="research_decision"),
        )
        decision_payload = await llm_provider.generate_json(
            messages=_research_decision_messages(
                question=state["question"],
                plan=state.get("plan", []),
                assumptions=state.get("assumptions", []),
                previous_findings=findings,
                iteration_findings=iteration_findings,
                executed_queries=all_executed_queries,
            ),
            request_id=state.get("request_id", "unknown"),
            node="research_loop",
            stage="research_decision",
        )
        logger.info(
            "llm raw payload",
            extra=log_extra(
                state,
                node="research_loop",
                stage="research_decision",
                llm_response_payload=summarize_for_log(decision_payload),
            ),
        )
        decision = ResearchDecisionPayload.model_validate(decision_payload)
        logger.info(
            "llm call completed",
            extra=log_extra(
                state,
                node="research_loop",
                stage="research_decision",
                duration_ms=round((perf_counter() - llm_started_at) * 1000, 2),
                enough_information=decision.enough_information,
                findings_count=len(decision.findings),
                next_queries_count=len(decision.next_wikipedia_queries),
                docs_count=len(fetched_documents),
            ),
        )

        merged_findings = merge_unique_preserve_order(findings, iteration_findings)
        merged_findings = merge_unique_preserve_order(merged_findings, decision.findings)
        next_queries = [
            query
            for query in decision.next_wikipedia_queries
            if _normalize_query(query) not in all_executed_queries
        ]
        warnings_to_add = []
        if not decision.enough_information and not next_queries:
            warnings_to_add.append(
                "LLM не предложил новые wikipedia-запросы, хотя данных недостаточно."
            )
        warnings = append_warnings(
            state,
            [*warnings_to_add, *extraction_missing_information, *decision.missing_information],
        )

        result: AgentState = {
            "findings": merged_findings,
            "pending_wikipedia_queries": next_queries,
            "wikipedia_query_history": all_executed_queries,
            "enough_information": decision.enough_information,
            "research_iterations": research_iterations,
            "warnings": warnings,
            "trace": append_trace(
                state,
                make_trace(
                    "research_loop",
                    "Собраны факты из Wikipedia и выполнена LLM-проверка достаточности данных.",
                    {
                        "iteration": research_iterations,
                        "pending_queries_in": pending_queries,
                        "executed_queries": newly_executed_queries,
                        "next_queries": next_queries,
                        "documents_count": len(fetched_documents),
                        "findings_count": len(merged_findings),
                        "enough_information": decision.enough_information,
                        "missing_information": decision.missing_information,
                    },
                ),
            ),
        }
        logger.info(
            "node result",
            extra=log_extra(state, node="research_loop", result=summarize_for_log(result)),
        )
        logger.info(
            "node completed",
            extra=log_extra(
                state,
                node="research_loop",
                duration_ms=round((perf_counter() - node_started_at) * 1000, 2),
                enough_information=decision.enough_information,
                iteration=research_iterations,
            ),
        )
        return result

    return research_loop
