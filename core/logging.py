"""Logging configuration for the toolkit.

Call :func:`configure_logging` once at process entry (e.g. the FastAPI
lifespan). Set ``LOG_FORMAT=json`` for structured one-line-per-record output
suitable for log aggregation; otherwise a human-readable formatter is used.
"""
from __future__ import annotations

import json
import logging
import os
import sys

# Standard ``LogRecord`` attributes, used to detect caller-supplied extras.
_RESERVED = set(
    logging.makeLogRecord({}).__dict__.keys()
) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        """Render a record as a JSON string, including any ``extra`` fields.

        Args:
            record: The log record to format.

        Returns:
            str: A JSON-encoded log line.
        """
        payload: dict[str, object] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED:
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str | None = None) -> None:
    """Configure the root logger once for the process.

    Args:
        level: Log level name; defaults to the ``LOG_LEVEL`` env var or ``INFO``.
    """
    log_level = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    handler = logging.StreamHandler(sys.stdout)

    if os.getenv("LOG_FORMAT", "").lower() == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)
