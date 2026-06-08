"""Unit tests for NIM client (mocked — no real API calls)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import os

import pytest


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("NIM_BASE_URL", "https://fake-nim.example.com/v1")
    monkeypatch.setenv("NIM_API_KEY", "test-key")
    monkeypatch.setenv("NIM_DEFAULT_MODEL", "meta/llama3-70b-instruct")


def test_nim_client_instantiation():
    from nim.client import NIMClient
    client = NIMClient()
    assert client.model == "meta/llama3-70b-instruct"
    assert "fake-nim" in client.base_url


def test_get_llm_returns_chat_openai():
    from nim.client import NIMClient
    from langchain_openai import ChatOpenAI
    client = NIMClient()
    llm = client.get_llm()
    assert isinstance(llm, ChatOpenAI)


@patch("httpx.get")
def test_list_models(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = {"data": [{"id": "meta/llama3-70b-instruct"}]}
    mock_get.return_value = mock_response

    from nim.client import NIMClient
    client = NIMClient()
    models = client.list_models()
    assert len(models) == 1
    assert models[0]["id"] == "meta/llama3-70b-instruct"
