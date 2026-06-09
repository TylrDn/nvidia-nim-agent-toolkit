"""Shared pytest fixtures.

All fixtures here keep unit tests offline: NIM and Langfuse are mocked so no
network calls or credentials are required.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage


@pytest.fixture(autouse=True)
def _test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide dummy credentials so client construction never reads real env."""
    monkeypatch.setenv("NIM_API_KEY", "test-key")
    monkeypatch.setenv("NIM_BASE_URL", "https://nim.test/v1")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)


@pytest.fixture(autouse=True)
def mock_langfuse(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable Langfuse so callbacks resolve to an empty list during tests."""
    monkeypatch.setattr("nim.client.get_langfuse_handler", lambda: None)


def make_ai_message(content: str = "", tool_calls: list | None = None) -> AIMessage:
    """Build an ``AIMessage`` with optional tool calls for mocking LLM output.

    Args:
        content: Message text content.
        tool_calls: LangChain tool-call dicts, or ``None`` for a final answer.

    Returns:
        AIMessage: A message suitable for use as a mocked ``ainvoke`` return.
    """
    return AIMessage(content=content, tool_calls=tool_calls or [])


@pytest.fixture
def mock_llm() -> MagicMock:
    """Return a mock LangChain LLM whose ``ainvoke`` is an ``AsyncMock``."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=make_ai_message(content="mocked answer"))
    llm.bind_tools = MagicMock(return_value=llm)
    return llm
