from __future__ import annotations

import json
from typing import Any, Protocol

from openai import APIError, APITimeoutError, AsyncOpenAI

from backend.core import get_logger, settings

logger = get_logger(__name__)


class LLMGenerationError(RuntimeError):
    pass


class LLMProvider(Protocol):
    async def generate_json(
        self,
        *,
        messages: list[dict[str, str]],
        request_id: str | None = None,
        node: str | None = None,
        stage: str | None = None,
    ) -> dict[str, Any]: ...


def _extract_json_object(content: str) -> dict[str, Any]:
    stripped = content.strip()
    candidates = [stripped]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(stripped[start : end + 1])

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload

    raise LLMGenerationError("llm did not return a valid JSON object")


def _extract_text_from_message(message: Any) -> str | None:
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str) and item.strip():
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or item.get("input_text")
                if isinstance(text, str) and text.strip():
                    parts.append(text)
        if parts:
            return "\n".join(parts)

    function_call = getattr(message, "function_call", None)
    if function_call is not None:
        args = getattr(function_call, "arguments", None)
        if isinstance(args, str) and args.strip():
            return args

    tool_calls = getattr(message, "tool_calls", None)
    if isinstance(tool_calls, list):
        for call in tool_calls:
            function = getattr(call, "function", None)
            args = getattr(function, "arguments", None) if function is not None else None
            if isinstance(args, str) and args.strip():
                return args
    return None


def _summarize_obj(value: Any, max_items: int = 6, max_str: int = 300) -> Any:
    if isinstance(value, str):
        return _truncate_text(value, limit=max_str)
    if isinstance(value, list):
        items = [_summarize_obj(item, max_items=max_items, max_str=max_str) for item in value[:max_items]]
        if len(value) > max_items:
            items.append(f"...<truncated {len(value) - max_items} items>")
        return items
    if isinstance(value, dict):
        return {k: _summarize_obj(v, max_items=max_items, max_str=max_str) for k, v in value.items()}
    return value


def _truncate_text(value: str, limit: int = 2000) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]}...<truncated {len(value) - limit} chars>"


def _serialize_messages_for_log(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    serialized: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role", "unknown")
        content = message.get("content", "")
        serialized.append(
            {
                "role": role,
                "content": _truncate_text(content, limit=2000),
            }
        )
    return serialized


class OpenAICompatibleLLMProvider:
    def __init__(
        self,
        *,
        client: AsyncOpenAI | None = None,
        base_url: str = settings.llm_base_url,
        api_key: str = settings.llm_api_key,
        project: str = settings.llm_project,
        model: str = settings.llm_model,
        timeout_s: float = settings.llm_timeout_s,
        temperature: float = settings.llm_temperature,
    ) -> None:
        self._client = client or AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            project=project or None,
            timeout=timeout_s,
        )
        self._project = project
        self._model = self._resolve_model(model)
        self._temperature = temperature
        self._max_tokens = settings.llm_max_tokens

    def _resolve_model(self, model: str) -> str:
        if (
            self._project
            and "cloud.yandex.net" in self._client.base_url.host
            and not model.startswith("gpt://")
        ):
            return f"gpt://{self._project}/{model}"
        return model

    async def generate_json(
        self,
        *,
        messages: list[dict[str, str]],
        request_id: str | None = None,
        node: str | None = None,
        stage: str | None = None,
    ) -> dict[str, Any]:
        request_payload = {
            "model": self._model,
            "temperature": self._temperature,
            "response_format": {"type": "json_object"},
            "messages": messages,
            "max_tokens": self._max_tokens,
        }

        logger.info(
            "llm request started",
            extra={
                "request_id": request_id or "unknown",
                "node": node,
                "stage": stage,
                "llm_model": self._model,
                "llm_base_url": str(self._client.base_url),
                "llm_max_tokens": self._max_tokens,
                "llm_request_payload": {
                    "model": self._model,
                    "temperature": self._temperature,
                    "response_format": {"type": "json_object"},
                    "max_tokens": self._max_tokens,
                    "reasoning_disabled_requested": settings.llm_disable_reasoning,
                    "messages": _serialize_messages_for_log(messages),
                },
            },
        )
        try:
            response = await self._client.chat.completions.create(**request_payload)
        except (APIError, APITimeoutError, ValueError, TypeError) as exc:
            logger.exception(
                "llm request failed",
                extra={
                    "request_id": request_id or "unknown",
                    "node": node,
                    "stage": stage,
                    "llm_max_tokens": self._max_tokens,
                    "llm_request_payload": {
                        "model": self._model,
                        "temperature": self._temperature,
                        "response_format": {"type": "json_object"},
                        "max_tokens": self._max_tokens,
                        "reasoning_disabled_requested": settings.llm_disable_reasoning,
                        "messages": _serialize_messages_for_log(messages),
                    },
                },
            )
            raise LLMGenerationError("cloud llm request failed") from exc

        first_choice = response.choices[0]
        content = _extract_text_from_message(first_choice.message)
        if not isinstance(content, str) or not content.strip():
            raw_message = first_choice.message.model_dump() if hasattr(first_choice.message, "model_dump") else {}
            logger.warning(
                "llm returned empty content",
                extra={
                    "request_id": request_id or "unknown",
                    "node": node,
                    "stage": stage,
                    "finish_reason": getattr(first_choice, "finish_reason", None),
                    "raw_message": _summarize_obj(raw_message),
                },
            )
            raise LLMGenerationError("cloud llm returned empty content")
        logger.info(
            "llm request completed",
            extra={
                "request_id": request_id or "unknown",
                "node": node,
                "stage": stage,
                "content_length": len(content),
                "finish_reason": getattr(first_choice, "finish_reason", None),
            },
        )
        return _extract_json_object(content)
