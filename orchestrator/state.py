"""TypedDict agent state schema for the NIM multi-agent LangGraph."""
from __future__ import annotations

from typing import Annotated, Any
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Shared state passed between all nodes in the LangGraph StateGraph."""

    # Conversation message history — appended, never replaced
    messages: Annotated[list[BaseMessage], add_messages]

    # Original user query (immutable)
    user_query: str

    # Planner output — ordered list of subtasks
    task_list: list[dict[str, Any]]

    # Index of the currently executing task
    current_task_index: int

    # Accumulated results from completed tasks
    task_results: list[dict[str, Any]]

    # Final synthesized answer (set by reviewer on completion)
    final_answer: str

    # Reviewer score for the last executor output (0.0 – 1.0)
    reviewer_score: float

    # Number of retry attempts for the current task
    retry_count: int

    # Which agent/node should handle execution (set by planner)
    routing_key: str

    # Arbitrary metadata passthrough
    metadata: dict[str, Any]
