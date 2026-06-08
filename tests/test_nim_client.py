"""Unit tests for NIM client — mocks the NIM endpoint."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from nim.client import NIMClient, NIMConfig


def test_get_llm_returns_chat_openai() -> None:
    from langchain_openai import ChatOpenAI
    client = NIMClient(NIMConfig(base_url="http://localhost:8000/v1",
                                 api_key="test", model="test-model"))
    llm = client.get_llm()
    assert isinstance(llm, ChatOpenAI)


@patch("httpx.get")
def test_health_check_ok(mock_get: MagicMock) -> None:
    mock_get.return_value = MagicMock(status_code=200)
    client = NIMClient(NIMConfig(base_url="http://localhost:8000/v1",
                                 api_key="test", model="test-model"))
    result = client.health_check()
    assert result["status"] == "ok"


@patch("httpx.get", side_effect=Exception("conn refused"))
def test_health_check_unreachable(mock_get: MagicMock) -> None:
    import httpx
    mock_get.side_effect = httpx.ConnectError("refused")
    client = NIMClient(NIMConfig(base_url="http://localhost:8000/v1",
                                 api_key="test", model="test-model"))
    result = client.health_check()
    assert result["status"] == "unreachable"
