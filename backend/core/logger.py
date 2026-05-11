from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

_LOG_RECORD_DEFAULT_FIELDS = set(logging.makeLogRecord({}).__dict__.keys())


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _LOG_RECORD_DEFAULT_FIELDS and not key.startswith("_")
        }
        payload.update(extras)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())

    if root.handlers:
        for handler in root.handlers:
            handler.setFormatter(JsonFormatter())
        return

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
