from typing import Literal

from pydantic import BaseModel

from backend.schemas.request import ChatMessage

Confidence = Literal["low", "medium", "high"]
TraceScalar = str | float | int | bool
TraceDetails = dict[str, TraceScalar | list[str] | dict[str, TraceScalar]]


class SelfCheck(BaseModel):
    passed: bool
    notes: list[str]


class TraceItem(BaseModel):
    step: str
    summary: str
    details: TraceDetails


class AnswerResponse(BaseModel):
    answer: str
    message: ChatMessage
    confidence: Confidence
    self_check: SelfCheck
    assumptions: list[str]
    findings: list[str]
    warnings: list[str]
    trace: list[TraceItem]
