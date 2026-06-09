"""Unit tests for the NIM readiness probes."""
from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

import nim.health_check as hc


def test_is_nim_ready_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hc.httpx, "get", lambda *a, **k: SimpleNamespace(status_code=200))
    assert hc.is_nim_ready() is True


def test_is_nim_ready_false_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: object, **k: object) -> None:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(hc.httpx, "get", boom)
    assert hc.is_nim_ready() is False


class _FakeAsyncClient:
    def __init__(self, status: int = 200, raise_error: bool = False) -> None:
        self._status = status
        self._raise = raise_error

    def __call__(self, *args: object, **kwargs: object) -> "_FakeAsyncClient":
        return self

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *args: object) -> bool:
        return False

    async def get(self, url: str) -> SimpleNamespace:
        if self._raise:
            raise httpx.ConnectError("refused")
        return SimpleNamespace(status_code=self._status)


@pytest.mark.asyncio
async def test_check_nim_health_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hc.httpx, "AsyncClient", _FakeAsyncClient(status=200))
    assert await hc.check_nim_health() is True


@pytest.mark.asyncio
async def test_check_nim_health_false_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hc.httpx, "AsyncClient", _FakeAsyncClient(raise_error=True))
    assert await hc.check_nim_health() is False
