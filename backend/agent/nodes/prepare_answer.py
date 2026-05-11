from __future__ import annotations

from time import perf_counter

from backend.agent.state import AgentState

from .shared import (
    append_trace,
    append_warnings,
    log_extra,
    logger,
    make_trace,
    run_self_check,
    summarize_for_log,
)


async def prepare_answer(state: AgentState) -> AgentState:
    node_started_at = perf_counter()
    logger.info("node started", extra=log_extra(state, node="prepare_answer"))
    passed, notes, new_warnings = run_self_check(state)
    if not state.get("enough_information"):
        new_warnings = [
            "Недостаточно данных для полностью подтвержденного ответа; ответ будет сформирован с пониженной уверенностью.",
            *new_warnings,
        ]
    warnings = append_warnings(state, new_warnings)

    result: AgentState = {
        "self_check_passed": passed,
        "self_check_notes": notes,
        "warnings": warnings,
        "trace": append_trace(
            state,
            make_trace(
                "prepare_answer",
                "Подготовлена финальная self-check перед выдачей ответа.",
                {
                    "self_check_passed": passed,
                    "warnings_count": len(new_warnings),
                },
            ),
        ),
    }
    logger.info(
        "node result",
        extra=log_extra(state, node="prepare_answer", result=summarize_for_log(result)),
    )
    logger.info(
        "node completed",
        extra=log_extra(
            state,
            node="prepare_answer",
            duration_ms=round((perf_counter() - node_started_at) * 1000, 2),
            self_check_passed=passed,
        ),
    )
    return result


async def self_check(state: AgentState) -> AgentState:
    passed, notes, new_warnings = run_self_check(state)
    return {
        "self_check_passed": passed,
        "self_check_notes": notes,
        "warnings": append_warnings(state, new_warnings),
        "trace": append_trace(
            state,
            make_trace(
                "self_check",
                "Выполнена проверка логики и единиц измерения.",
                {
                    "passed": passed,
                    "notes": notes,
                    "warnings": new_warnings,
                },
            ),
        ),
    }
