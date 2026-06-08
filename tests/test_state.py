"""Unit tests for AgentState schema."""
from __future__ import annotations

from orchestrator.state import AgentState, SubTask


def test_subtask_creation():
    task = SubTask(
        task_id="t1",
        description="Fetch weather data",
        agent="api",
        status="pending",
        result=None,
    )
    assert task["agent"] == "api"
    assert task["status"] == "pending"


def test_agent_state_structure():
    state: AgentState = {
        "user_query": "What is the weather in Seattle?",
        "session_id": "abc123",
        "plan": [],
        "current_task_index": 0,
        "task_results": [],
        "reviewer_score": 0.0,
        "reviewer_feedback": "",
        "loop_count": 0,
        "max_loops": 3,
        "final_answer": None,
        "error": None,
    }
    assert state["user_query"] == "What is the weather in Seattle?"
    assert state["max_loops"] == 3
