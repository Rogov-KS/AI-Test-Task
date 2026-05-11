from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from backend.agent.state import AgentState, TraceDetails, TraceItemState
from backend.core import get_logger

logger = get_logger(__name__)


class MissingLLMProviderError(ValueError):
    pass


def make_trace(step: str, summary: str, details: TraceDetails) -> TraceItemState:
    return {"step": step, "summary": summary, "details": details}


def append_trace(state: AgentState, item: TraceItemState) -> list[TraceItemState]:
    return [*state.get("trace", []), item]


def merge_unique_preserve_order(existing: Iterable[str], incoming: Iterable[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in [*existing, *incoming]:
        normalized = str(value).strip()
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        merged.append(normalized)
    return merged


def append_warnings(state: AgentState, notes: list[str]) -> list[str]:
    return merge_unique_preserve_order(state.get("warnings", []), notes)


def log_extra(state: AgentState, **fields: object) -> dict[str, object]:
    return {"request_id": state.get("request_id", "unknown"), **fields}


def summarize_for_log(value: Any, *, max_items: int = 5, max_str: int = 300) -> Any:
    if isinstance(value, str):
        if len(value) <= max_str:
            return value
        return f"{value[:max_str]}...<truncated {len(value) - max_str} chars>"
    if isinstance(value, list):
        trimmed = [summarize_for_log(item, max_items=max_items, max_str=max_str) for item in value[:max_items]]
        if len(value) > max_items:
            trimmed.append(f"...<truncated {len(value) - max_items} items>")
        return trimmed
    if isinstance(value, dict):
        return {
            key: summarize_for_log(item, max_items=max_items, max_str=max_str)
            for key, item in value.items()
        }
    return value


def run_self_check(state: AgentState) -> tuple[bool, list[str], list[str]]:
    notes: list[str] = []
    warnings: list[str] = []
    passed = True

    if not state.get("topic", "").strip():
        passed = False
        notes.append("Тема вопроса не извлечена.")
    else:
        notes.append("Тема вопроса извлечена корректно.")

    if not state.get("plan"):
        passed = False
        notes.append("План ответа не был собран.")
    else:
        notes.append("План ответа присутствует.")

    if not state.get("findings"):
        warnings.append("Факты не были явно собраны, ответ может быть менее точным.")
        notes.append("Нет явного списка найденных фактов.")
    else:
        notes.append("Собран список релевантных фактов.")

    return passed, notes, warnings
