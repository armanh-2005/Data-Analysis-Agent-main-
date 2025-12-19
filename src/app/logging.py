from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict, Optional


# Context variable to attach run_id to every log record.
_RUN_ID: ContextVar[Optional[str]] = ContextVar("run_id", default=None)


def set_run_id(run_id: str) -> None:
    # Set run_id in context for current flow.
    _RUN_ID.set(run_id)


def clear_run_id() -> None:
    _RUN_ID.set(None)


class RunIdFilter(logging.Filter):
    # Adds run_id to log records.
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = _RUN_ID.get()
        return True


class JsonFormatter(logging.Formatter):
    # Structured JSON formatter for logs.
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "run_id": getattr(record, "run_id", None),
        }
        # Attach exception info if present.
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        # Optionally include extra fields passed through logger adapter or extra={}
        for key, value in record.__dict__.items():
            if key in {"msg", "args", "levelname", "levelno", "name", "pathname", "filename", "module",
                       "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created",
                       "msecs", "relativeCreated", "thread", "threadName", "processName", "process",
                       "run_id"}:
                continue
            # Keep extras JSON-serializable when possible.
            try:
                json.dumps(value, ensure_ascii=False)
                payload[key] = value
            except TypeError:
                payload[key] = str(value)

        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "INFO", json_logs: bool = True) -> None:
    # Configure root logging once.
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level.upper())

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RunIdFilter())

    if json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s run_id=%(run_id)s %(message)s"
        ))

    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
