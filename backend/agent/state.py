from typing import Literal, TypedDict

TraceScalar = str | float | int | bool
TraceDetails = dict[str, TraceScalar | list[str] | dict[str, TraceScalar]]


class WikipediaResultState(TypedDict):
    query: str
    source: str
    fact_name: str
    value: float

Confidence = Literal["low", "medium", "high"]


class TraceItemState(TypedDict):
    step: str
    summary: str
    details: TraceDetails


class AgentState(TypedDict, total=False):
    request_id: str
    question: str
    topic: str
    intent: str
    plan: list[str]
    pending_wikipedia_queries: list[str]
    wikipedia_query_history: list[str]
    assumptions: list[str]
    findings: list[str]
    warnings: list[str]
    trace: list[TraceItemState]
    enough_information: bool
    research_iterations: int
    max_research_iterations: int
    self_check_passed: bool
    self_check_notes: list[str]
    confidence: Confidence
    answer: str
