"""Tests for the FastAPI server endpoints."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import api.server as server


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(server, "check_nim_health", AsyncMock(return_value=True))
    monkeypatch.setattr(server, "run_agent_async", AsyncMock(return_value="the answer"))
    with TestClient(server.app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "nim_reachable": True}
    assert "X-Request-ID" in response.headers


def test_query(client: TestClient) -> None:
    response = client.post("/query", json={"query": "hello"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "the answer"
    assert body["thread_id"]


def test_models(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = AsyncMock()
    fake_client.list_models = AsyncMock(return_value=["m1", "m2"])
    monkeypatch.setattr(server, "get_client", lambda: fake_client)
    response = client.get("/models")
    assert response.status_code == 200
    assert response.json() == {"models": ["m1", "m2"]}
