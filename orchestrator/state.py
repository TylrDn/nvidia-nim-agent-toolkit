"""LangGraph TypedDict state schema for the multi-agent coordinator."""
from __future__ import annotations

from typing import Annotated, Any
from typing_extensions import TypedDict
import operator


class SubTask(TypedDict):
    task_id: str
    description: str
    agent: str  # "api" | "sql" | "doc"
    status: str  # "pending" | "running" | "done" | "failed"
    result: str | None


class AgentState(TypedDict):
    """Shared state threaded through every node in the LangGraph."""

    # User input
    user_query: str
    session_id: str

    # Planner output
    plan: list[SubTask]
    current_task_index: int

    # Executor output — accumulated across tasks
    task_results: Annotated[list[dict[str, Any]], operator.add]

    # Reviewer output
    reviewer_score: float  # 0.0 – 1.0
    reviewer_feedback: str
    loop_count: int
    max_loops: int

    # Final answer
    final_answer: str | None
    error: str | None
