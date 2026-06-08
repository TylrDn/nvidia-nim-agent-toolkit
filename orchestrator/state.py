"""LangGraph agent state schema.

All nodes in the multi-agent graph read from and write to this shared
TypedDict state. Keeping the schema in one place prevents drift.
"""
from __future__ import annotations

from typing import Annotated, Any
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Top-level state object shared across all graph nodes."""

    # Conversation history — add_messages merges incoming messages.
    messages: Annotated[list[BaseMessage], add_messages]

    # Original user intent, preserved across the full planning loop.
    user_intent: str

    # Ordered task list produced by the Planner node.
    task_list: list[dict[str, Any]]

    # Index of the task currently being executed.
    current_task_index: int

    # Accumulated results from all completed tasks.
    task_results: list[dict[str, Any]]

    # Reviewer score for the most recent Executor output (0.0 – 1.0).
    reviewer_score: float

    # Number of retry attempts for the current task.
    retry_count: int

    # Final synthesised answer, populated on terminal node.
    final_answer: str

    # Arbitrary metadata (trace IDs, timing, model used, etc.)
    metadata: dict[str, Any]
