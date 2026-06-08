"""TypedDict agent state schema shared across all LangGraph nodes."""
from __future__ import annotations

from typing import Annotated, Any

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Shared mutable state passed through every graph node."""

    # Accumulated message history (human + AI turns)
    messages: Annotated[list[Any], add_messages]

    # Original user request (immutable reference)
    user_request: str

    # Planner output — ordered list of subtask dicts
    tasks: list[dict[str, Any]]

    # Index of the task currently being executed
    current_task_idx: int

    # Collected results from the executor
    results: list[dict[str, Any]]

    # Reviewer score for the latest executor output (0.0–1.0)
    reviewer_score: float

    # Number of retry attempts on the current task
    retry_count: int

    # Terminal output delivered to the user
    final_answer: str
