"""Unit tests for the Planner node."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.nodes.planner import planner_node
from orchestrator.state import AgentState
from tests.conftest import make_ai_message


def _state(query: str) -> AgentState:
    return {
        "messages": [],
        "user_query": query,
        "task_list": [],
        "current_task_index": 0,
        "task_results": [],
        "final_answer": "",
        "reviewer_score": 0.0,
        "retry_count": 0,
        "routing_key": "",
        "metadata": {},
    }


def _patch_llm(monkeypatch: pytest.MonkeyPatch, content: str) -> None:
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=make_ai_message(content=content))
    nim_client = MagicMock()
    nim_client.as_langchain_llm.return_value = llm
    monkeypatch.setattr(
        "orchestrator.nodes.planner.NIMClient", MagicMock(return_value=nim_client)
    )


@pytest.mark.asyncio
async def test_planner_returns_task_list(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_llm(
        monkeypatch,
        '[{"id": 1, "description": "fetch data", "agent": "api_agent", "depends_on": []}]',
    )
    result = await planner_node(_state("fetch stock data"))
    assert len(result["task_list"]) == 1
    assert result["task_list"][0]["agent"] == "api_agent"
    assert result["current_task_index"] == 0


@pytest.mark.asyncio
async def test_planner_falls_back_on_bad_json(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_llm(monkeypatch, "not json at all")
    result = await planner_node(_state("do something"))
    assert len(result["task_list"]) == 1
    assert result["task_list"][0]["agent"] == "doc_agent"
