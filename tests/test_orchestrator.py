"""Unit tests for the orchestrator graph."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from orchestrator.state import AgentState
from orchestrator.nodes.planner import planner_node
from orchestrator.nodes.reviewer import should_continue


def _make_state(**overrides) -> AgentState:
    base: AgentState = {
        "messages": [],
        "user_intent": "test intent",
        "task_list": [],
        "current_task_index": 0,
        "task_results": [],
        "reviewer_score": 0.0,
        "retry_count": 0,
        "final_answer": "",
        "metadata": {},
    }
    base.update(overrides)
    return base


@patch("orchestrator.nodes.planner.NIMClient")
def test_planner_node_valid_json(mock_client_cls: MagicMock) -> None:
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='[{"id": 1, "description": "Fetch data", "tool": "api", "depends_on": []}]'
    )
    mock_client = MagicMock()
    mock_client.llm = mock_llm
    mock_client_cls.return_value = mock_client

    state = _make_state(user_intent="Fetch weather data")
    result = planner_node(state, client=mock_client)

    assert len(result["task_list"]) == 1
    assert result["task_list"][0]["tool"] == "api"
    assert result["current_task_index"] == 0


@patch("orchestrator.nodes.planner.NIMClient")
def test_planner_node_invalid_json_fallback(mock_client_cls: MagicMock) -> None:
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Not valid JSON at all")
    mock_client = MagicMock()
    mock_client.llm = mock_llm

    state = _make_state(user_intent="Do something")
    result = planner_node(state, client=mock_client)

    assert len(result["task_list"]) == 1
    assert result["task_list"][0]["tool"] == "none"


def test_should_continue_done() -> None:
    state = _make_state(current_task_index=3, task_list=[{}, {}, {}])
    assert should_continue(state) == "done"


def test_should_continue_retry() -> None:
    state = _make_state(
        current_task_index=0,
        task_list=[{}],
        reviewer_score=0.3,
        retry_count=1,
    )
    assert should_continue(state) == "retry"


def test_should_continue_continue() -> None:
    state = _make_state(
        current_task_index=0,
        task_list=[{}, {}],
        reviewer_score=0.9,
        retry_count=0,
    )
    assert should_continue(state) == "continue"
