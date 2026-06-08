"""Unit tests for orchestrator graph and state."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from orchestrator.state import AgentState


def test_agent_state_keys():
    state: AgentState = {
        "messages": [],
        "user_query": "test",
        "task_list": [],
        "current_task_index": 0,
        "task_results": [],
        "final_answer": "",
        "reviewer_score": 0.0,
        "retry_count": 0,
        "routing_key": "",
        "metadata": {},
    }
    assert state["user_query"] == "test"
    assert isinstance(state["task_list"], list)


@patch("orchestrator.nodes.planner.get_client")
def test_planner_returns_task_list(mock_get_client):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='[{"id": 1, "description": "fetch data", "agent": "api_agent", "depends_on": []}]'
    )
    mock_get_client.return_value.as_langchain_llm.return_value = mock_llm

    from orchestrator.nodes.planner import planner_node
    state: AgentState = {
        "messages": [], "user_query": "fetch stock data",
        "task_list": [], "current_task_index": 0,
        "task_results": [], "final_answer": "",
        "reviewer_score": 0.0, "retry_count": 0,
        "routing_key": "", "metadata": {},
    }
    result = planner_node(state)
    assert len(result["task_list"]) == 1
    assert result["task_list"][0]["agent"] == "api_agent"
