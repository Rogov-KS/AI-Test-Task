from __future__ import annotations

import os

from pydantic import BaseModel, ConfigDict


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    app_name: str = "bridge-time-agent"
    log_level: str = "INFO"
    llm_base_url: str = "https://ai.api.cloud.yandex.net/v1"
    llm_api_key: str = os.environ.get("LLM_API_KEY", "")
    llm_project: str = "b1gi140rbiv38d5i42vu"
    llm_model: str = "qwen3-235b-a22b-fp8/latest"
    llm_timeout_s: float = 500.0
    llm_temperature: float = 0.2
    llm_max_tokens: int = 5000
    llm_disable_reasoning: bool = True
    agent_dialog_fallback_enabled: bool = False
    max_research_iterations: int = 3
    wikipedia_api_url: str = "https://en.wikipedia.org/w/api.php"
    wikipedia_timeout_s: float = 10.0
    wikipedia_user_agent: str = "bridge-time-agent/0.1 (+https://example.com)"


settings = Settings()
