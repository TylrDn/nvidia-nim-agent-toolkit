"""Unit tests for NIM client."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from nim.client import NIMClient


@pytest.fixture
def client() -> NIMClient:
    return NIMClient(model="meta/llama-3.1-70b-instruct")


def test_client_init(client: NIMClient) -> None:
    assert client.model == "meta/llama-3.1-70b-instruct"
    assert client.temperature == 0.0
    assert client.max_tokens == 2048


def test_from_config() -> None:
    config = {
        "model": "mistralai/mistral-7b-instruct-v0.3",
        "temperature": 0.1,
        "max_tokens": 512,
    }
    c = NIMClient.from_config(config)
    assert c.model == "mistralai/mistral-7b-instruct-v0.3"
    assert c.temperature == 0.1
    assert c.max_tokens == 512


@patch("nim.client.httpx.Client")
def test_health_check_ok(mock_httpx_cls: MagicMock, client: NIMClient) -> None:
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"data": [{"id": "meta/llama-3.1-70b-instruct"}]}
    mock_resp.raise_for_status.return_value = None
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=MagicMock(get=MagicMock(return_value=mock_resp)))
    mock_ctx.__exit__ = MagicMock(return_value=False)
    mock_httpx_cls.return_value = mock_ctx

    result = client.health_check()
    assert result["status"] == "ok"
    assert "meta/llama-3.1-70b-instruct" in result["models"]


@patch("nim.client.httpx.Client")
def test_health_check_error(mock_httpx_cls: MagicMock, client: NIMClient) -> None:
    mock_httpx_cls.side_effect = Exception("Connection refused")
    result = client.health_check()
    assert result["status"] == "error"
    assert "Connection refused" in result["detail"]
