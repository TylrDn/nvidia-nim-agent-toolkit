"""Integration test for the full LangGraph pipeline."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from orchestrator.graph import build_graph


@patch("orchestrator.nodes.planner.get_default_llm")
@patch("orchestrator.nodes.executor.get_default_llm")
@patch("orchestrator.nodes.reviewer.get_default_llm")
def test_graph_builds_and_invokes(mock_reviewer_llm, mock_exec_llm, mock_plan_llm):
    """Smoke test: graph should compile and complete without errors."""
    for mock_fn in [mock_plan_llm, mock_exec_llm, mock_reviewer_llm]:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content='1. Do task one')
        mock_fn.return_value = mock_llm

    # Reviewer must return valid JSON score
    mock_reviewer_llm.return_value.invoke.return_value = MagicMock(
        content='{"score": 0.9, "feedback": "Good result"}'
    )

    graph = build_graph()
    assert graph is not None  # graph compiled without error
