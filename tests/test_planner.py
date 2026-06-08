"""Unit tests for the Planner node."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from orchestrator.nodes.planner import planner_node
from orchestrator.state import AgentState


@patch("orchestrator.nodes.planner.get_default_llm")
def test_planner_returns_task_list(mock_llm_fn):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content="1. Fetch the data\n2. Analyse results\n3. Generate report"
    )
    mock_llm_fn.return_value = mock_llm

    state: AgentState = {
        "user_input": "Analyse our Q2 sales data",
        "plan": [],
        "current_task": 0,
        "tool_results": [],
        "messages": [],
        "reviewer_score": 0.0,
        "retry_count": 0,
        "final_output": None,
        "error": None,
    }

    result = planner_node(state)
    assert "plan" in result
    assert len(result["plan"]) == 3
    assert result["current_task"] == 0
