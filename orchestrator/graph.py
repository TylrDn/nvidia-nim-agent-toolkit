"""LangGraph StateGraph — multi-agent coordinator.

Wires together: Planner → Executor → Reviewer with conditional retry loop.

Usage::

    from orchestrator.graph import build_graph
    graph = build_graph()
    result = graph.invoke({"user_input": "Summarise our Q2 sales data"})
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END
from orchestrator.state import AgentState
from orchestrator.nodes.planner import planner_node
from orchestrator.nodes.executor import executor_node
from orchestrator.nodes.reviewer import reviewer_node

MAX_RETRIES = 3
REVIEWER_PASS_THRESHOLD = 0.7


def _should_retry(state: AgentState) -> str:
    """Conditional edge: retry if reviewer score low and retries remain."""
    if state["reviewer_score"] >= REVIEWER_PASS_THRESHOLD:
        return "done"
    if state["retry_count"] >= MAX_RETRIES:
        return "done"  # fail-open after max retries
    return "retry"


def build_graph() -> StateGraph:
    """Assemble and compile the multi-agent LangGraph.

    Returns:
        Compiled LangGraph ready for .invoke() or .stream().
    """
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("reviewer", reviewer_node)

    # Edges
    builder.set_entry_point("planner")
    builder.add_edge("planner", "executor")
    builder.add_edge("executor", "reviewer")
    builder.add_conditional_edges(
        "reviewer",
        _should_retry,
        {"retry": "executor", "done": END},
    )

    return builder.compile()
