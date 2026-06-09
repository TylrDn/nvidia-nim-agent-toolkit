"""Unit tests for the NIM client layer."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nim.client import DEFAULT_MODEL, NIMClient, get_callbacks, get_client, get_langfuse_handler


def test_nim_client_init_defaults() -> None:
    client = NIMClient()
    assert client.model == DEFAULT_MODEL
    assert client.temperature == 0.0
    assert client.api_key == "test-key"


def test_get_client_singleton() -> None:
    assert get_client() is get_client()


def test_api_key_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NIM_API_KEY", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "legacy-key")
    assert NIMClient().api_key == "legacy-key"


def test_langfuse_handler_disabled_without_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    # Undo the autouse patch to exercise the real factory with no keys set.
    monkeypatch.setattr("nim.client.get_langfuse_handler", get_langfuse_handler)
    assert get_langfuse_handler() is None
    assert get_callbacks() == []


@pytest.mark.asyncio
async def test_list_models_returns_sorted_ids() -> None:
    client = NIMClient()
    mock_async = MagicMock()
    mock_async.models.list = AsyncMock(
        return_value=SimpleNamespace(data=[SimpleNamespace(id="b"), SimpleNamespace(id="a")])
    )
    client._async_client = mock_async
    assert await client.list_models() == ["a", "b"]


@pytest.mark.asyncio
async def test_chat_completion_returns_content() -> None:
    client = NIMClient()
    mock_async = MagicMock()
    mock_async.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))]
        )
    )
    client._async_client = mock_async
    assert await client.chat_completion([{"role": "user", "content": "yo"}]) == "hi"


@pytest.mark.asyncio
async def test_batch_chat_completion() -> None:
    client = NIMClient()
    mock_async = MagicMock()
    mock_async.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="x"))]
        )
    )
    client._async_client = mock_async
    batch = [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]]
    assert await client.batch_chat_completion(batch) == ["x", "x"]


@pytest.mark.asyncio
async def test_stream_chat_yields_chunks() -> None:
    async def fake_stream():
        for piece in ("a", "b"):
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=piece))])

    client = NIMClient()
    mock_async = MagicMock()
    mock_async.chat.completions.create = AsyncMock(return_value=fake_stream())
    client._async_client = mock_async
    chunks = [chunk async for chunk in client.stream_chat([{"role": "user", "content": "x"}])]
    assert chunks == ["a", "b"]


def test_as_langchain_llm_attaches_no_callbacks_without_keys() -> None:
    llm = NIMClient(model="m").as_langchain_llm()
    assert llm.model_name == "m"
