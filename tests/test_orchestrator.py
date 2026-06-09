"""Unit tests for the reviewer node and post-reviewer routing."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langgraph.graph import END

from orchestrator.graph import _route_after_reviewer
from orchestrator.nodes.reviewer import reviewer_node
from orchestrator.state import AgentState
from tests.conftest import make_ai_message


def _state(**overrides) -> AgentState:
    base: AgentState = {
        "messages": [],
        "user_query": "q",
        "task_list": [{"id": 1, "description": "d", "agent": "doc_agent"}],
        "current_task_index": 0,
        "task_results": [{"task_id": 1, "agent": "doc_agent", "result": "r"}],
        "final_answer": "",
        "reviewer_score": 0.0,
        "retry_count": 0,
        "routing_key": "",
        "metadata": {},
    }
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


def _patch_reviewer_llm(monkeypatch: pytest.MonkeyPatch, content: str) -> None:
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=make_ai_message(content=content))
    nim_client = MagicMock()
    nim_client.as_langchain_llm.return_value = llm
    monkeypatch.setattr(
        "orchestrator.nodes.reviewer.NIMClient", MagicMock(return_value=nim_client)
    )


def test_route_after_reviewer() -> None:
    assert _route_after_reviewer({"routing_key": "__end__"}) == END
    assert _route_after_reviewer({"routing_key": "next_task"}) == "executor"
    assert _route_after_reviewer({"routing_key": "retry"}) == "executor"


@pytest.mark.asyncio
async def test_reviewer_completes_single_task(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_reviewer_llm(
        monkeypatch,
        '{"score": 0.9, "verdict": "complete", "final_answer": "done", "feedback": "ok"}',
    )
    result = await reviewer_node(_state())
    assert result["final_answer"] == "done"
    assert result["routing_key"] == "__end__"
    assert result["reviewer_score"] == 0.9


@pytest.mark.asyncio
async def test_reviewer_advances_when_more_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_reviewer_llm(monkeypatch, '{"score": 0.8, "verdict": "accept", "feedback": "ok"}')
    state = _state(
        task_list=[
            {"id": 1, "description": "a", "agent": "doc_agent"},
            {"id": 2, "description": "b", "agent": "api_agent"},
        ]
    )
    result = await reviewer_node(state)
    assert result["routing_key"] == "next_task"
    assert result["current_task_index"] == 1
