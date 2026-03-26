"""
app/utils/logging.py — Structured JSON logging for production.
Uses Python's standard logging with a JSON formatter so logs
can be ingested by CloudWatch, Datadog, Loki, etc.
"""

import json
import logging
import os
import sys
import time
import uuid
from typing import Any


class JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    RESERVED = {"msg", "args", "exc_info", "exc_text", "stack_info", "levelno", "lineno"}

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # Merge any extra fields attached with logger.info("…", extra={…})
        for key, val in record.__dict__.items():
            if key not in logging.LogRecord.__dict__ and key not in self.RESERVED:
                payload[key] = val
        return json.dumps(payload, default=str)


def setup_logging(level: str | None = None) -> None:
    """Configure root logger with JSON output."""
    log_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())

    root = logging.getLogger()
    root.setLevel(log_level)
    # Remove any default handlers (e.g. from uvicorn)
    root.handlers = [handler]

    # Quieten noisy libs
    for lib in ("httpx", "httpcore", "faiss", "sentence_transformers",
                 "botocore", "urllib3", "multipart"):
        logging.getLogger(lib).setLevel(logging.WARNING)


def get_request_id() -> str:
    """Generate a short alphanumeric request ID."""
    return uuid.uuid4().hex[:12]
