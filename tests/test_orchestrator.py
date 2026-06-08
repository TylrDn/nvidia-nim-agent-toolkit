"""Integration-style tests for the LangGraph orchestrator (mocked LLM)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage


def _make_mock_llm(content: str) -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content=content)
    llm.bind_tools.return_value = llm
    return llm


@patch("nim.client.NIMClient.get_llm")
def test_planner_node_parses_tasks(mock_get_llm: MagicMock) -> None:
    tasks = [{"id": 0, "description": "do something", "tool": "none"}]
    mock_get_llm.return_value = _make_mock_llm(json.dumps(tasks))

    from orchestrator.nodes.planner import planner_node
    from orchestrator.state import AgentState

    state: AgentState = {
        "messages": [],
        "user_request": "do something",
        "tasks": [],
        "current_task_idx": 0,
        "results": [],
        "reviewer_score": 0.0,
        "retry_count": 0,
        "final_answer": "",
    }
    result = planner_node(state)
    assert result["tasks"] == tasks
    assert result["current_task_idx"] == 0
