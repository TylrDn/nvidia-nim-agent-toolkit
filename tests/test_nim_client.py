"""Unit tests for NIM client layer."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from nim.client import NIMClient, get_client


def test_nim_client_init():
    client = NIMClient(model="meta/llama-3.1-8b-instruct")
    assert client.model == "meta/llama-3.1-8b-instruct"
    assert client.temperature == 0.0


def test_get_client_singleton():
    c1 = get_client()
    c2 = get_client()
    assert c1 is c2


@patch("nim.client.OpenAI")
def test_chat_called(mock_openai):
    mock_instance = MagicMock()
    mock_openai.return_value = mock_instance
    mock_instance.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Hello"))]
    )
    client = NIMClient()
    client.chat([{"role": "user", "content": "hi"}])
    mock_instance.chat.completions.create.assert_called_once()
