"""Unit tests for the AgentState schema."""
from __future__ import annotations

from orchestrator.state import AgentState


def test_agent_state_structure() -> None:
    state: AgentState = {
        "messages": [],
        "user_query": "What is the weather in Seattle?",
        "task_list": [],
        "current_task_index": 0,
        "task_results": [],
        "final_answer": "",
        "reviewer_score": 0.0,
        "retry_count": 0,
        "routing_key": "",
        "metadata": {},
    }
    assert state["user_query"] == "What is the weather in Seattle?"
    assert isinstance(state["task_list"], list)
    assert state["current_task_index"] == 0
