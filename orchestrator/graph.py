"""LangGraph StateGraph — multi-agent coordinator (Planner → Executor → Reviewer)."""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from orchestrator.state import AgentState
from orchestrator.nodes.planner import planner_node
from orchestrator.nodes.executor import executor_node
from orchestrator.nodes.reviewer import reviewer_node


def _should_continue(state: AgentState) -> str:
    """Routing function: loop back to executor or end."""
    if state.get("error"):
        return END
    if state["reviewer_score"] >= 0.8:
        return END
    if state["loop_count"] >= state["max_loops"]:
        return END
    return "executor"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("reviewer", reviewer_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "reviewer")
    graph.add_conditional_edges(
        "reviewer",
        _should_continue,
        {"executor": "executor", END: END},
    )

    return graph.compile()


# Compiled graph singleton — import and invoke directly
app = build_graph()
