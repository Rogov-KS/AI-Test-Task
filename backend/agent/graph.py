from typing import Any, cast

from langgraph.graph import END, StateGraph

from backend.agent.nodes import (
    make_build_plan,
    make_finalize_answer,
    make_research_loop,
    prepare_answer,
)
from backend.agent.state import AgentState
from backend.clients import KnowledgeProvider, LLMProvider


def _route_after_research(state: AgentState) -> str:
    if state.get("enough_information"):
        return "prepare_answer"

    if state.get("research_iterations", 0) >= state.get("max_research_iterations", 2):
        return "prepare_answer"

    return "research_loop"


def _route_after_prepare(state: AgentState) -> str:
    if state.get("self_check_passed"):
        return "finalize_answer"

    if state.get("research_iterations", 0) >= state.get("max_research_iterations", 2):
        return "finalize_answer"

    return "build_plan"


def build_agent_graph(
    provider: KnowledgeProvider,
    llm_provider: LLMProvider | None = None,
) -> Any:
    workflow = StateGraph(AgentState)

    workflow.add_node("build_plan", cast(Any, make_build_plan(llm_provider)))
    workflow.add_node("research_loop", cast(Any, make_research_loop(provider, llm_provider)))
    workflow.add_node("prepare_answer", prepare_answer)
    workflow.add_node("finalize_answer", cast(Any, make_finalize_answer(llm_provider)))

    workflow.set_entry_point("build_plan")
    workflow.add_edge("build_plan", "research_loop")
    workflow.add_conditional_edges(
        "research_loop",
        _route_after_research,
        {
            "research_loop": "research_loop",
            "prepare_answer": "prepare_answer",
        },
    )
    workflow.add_conditional_edges(
        "prepare_answer",
        _route_after_prepare,
        {
            "build_plan": "build_plan",
            "finalize_answer": "finalize_answer",
        },
    )
    workflow.add_edge("finalize_answer", END)

    return workflow.compile()
