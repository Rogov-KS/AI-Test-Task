from __future__ import annotations

from typing import Any

__all__ = ["answer_dialog_with_fallback", "build_agent_graph"]


async def answer_dialog_with_fallback(*args: Any, **kwargs: Any) -> Any:
    from backend.agent.fallback import answer_dialog_with_fallback as _answer_dialog_with_fallback

    return await _answer_dialog_with_fallback(*args, **kwargs)


def build_agent_graph(*args: Any, **kwargs: Any) -> Any:
    from backend.agent.graph import build_agent_graph as _build_agent_graph

    return _build_agent_graph(*args, **kwargs)
