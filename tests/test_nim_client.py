"""Unit tests for NIMClient."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from nim.client import NIMClient


def test_nim_client_init_from_env(monkeypatch):
    monkeypatch.setenv("NIM_API_KEY", "test-key")
    monkeypatch.setenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    monkeypatch.setenv("NIM_MODEL", "meta/llama-3.1-70b-instruct")
    client = NIMClient()
    assert client.api_key == "test-key"
    assert client.model == "meta/llama-3.1-70b-instruct"


def test_nim_client_override_params():
    client = NIMClient(
        model="mistralai/mistral-7b-instruct-v0.3",
        base_url="https://custom.nim.endpoint/v1",
        api_key="custom-key",
    )
    assert client.model == "mistralai/mistral-7b-instruct-v0.3"
    assert client.base_url == "https://custom.nim.endpoint/v1"


@patch("nim.client.ChatOpenAI")
def test_get_llm_returns_chat_openai(mock_chat):
    mock_chat.return_value = MagicMock()
    client = NIMClient(
        model="meta/llama-3.1-8b-instruct",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="test",
    )
    llm = client.get_llm()
    mock_chat.assert_called_once()
    assert llm is not None
