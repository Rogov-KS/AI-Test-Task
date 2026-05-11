from typing import Literal

from pydantic import BaseModel, Field

MessageRole = Literal["user", "assistant"]


class ChatMessage(BaseModel):
    role: MessageRole
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=50)
