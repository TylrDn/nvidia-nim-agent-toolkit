"""Unit tests for logging configuration."""
from __future__ import annotations

import json
import logging

import pytest

from core.logging import JsonFormatter, configure_logging


def test_configure_logging_sets_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    configure_logging("DEBUG")
    assert logging.getLogger().level == logging.DEBUG


def test_configure_logging_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_FORMAT", "json")
    configure_logging("INFO")
    handler = logging.getLogger().handlers[0]
    assert isinstance(handler.formatter, JsonFormatter)


def test_json_formatter_includes_extras() -> None:
    record = logging.LogRecord("svc", logging.INFO, "path", 10, "hello", None, None)
    record.request_id = "abc123"
    payload = json.loads(JsonFormatter().format(record))
    assert payload["message"] == "hello"
    assert payload["level"] == "INFO"
    assert payload["request_id"] == "abc123"
