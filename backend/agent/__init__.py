from __future__ import annotations

from typing import Any

__all__ = ["build_agent_graph"]


def build_agent_graph(*args: Any, **kwargs: Any) -> Any:
    from backend.agent.graph import build_agent_graph as _build_agent_graph

    return _build_agent_graph(*args, **kwargs)
