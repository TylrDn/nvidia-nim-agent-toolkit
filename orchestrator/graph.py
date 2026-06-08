"""LangGraph StateGraph — multi-agent coordinator (Planner → Executor → Reviewer)."""
from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph

from orchestrator.nodes.executor import executor_node
from orchestrator.nodes.planner import planner_node
from orchestrator.nodes.reviewer import reviewer_node, should_retry
from orchestrator.state import AgentState


def _advance_task(state: AgentState) -> AgentState:
    """Move to the next task; reset retry counter."""
    return {**state, "current_task_idx": state["current_task_idx"] + 1,
            "retry_count": 0}


def _increment_retry(state: AgentState) -> AgentState:
    """Increment retry counter without advancing task."""
    return {**state, "retry_count": state.get("retry_count", 0) + 1}


def _finalize(state: AgentState) -> AgentState:
    """Compose the final answer from all collected results."""
    summary = "\n".join(
        f"Task {r['task_id']}: {r['result'].get('output', str(r['result']))}"
        for r in state["results"]
    )
    final = AIMessage(content=summary)
    return {**state, "final_answer": summary,
            "messages": state["messages"] + [final]}


def build_graph() -> StateGraph:
    """Assemble and compile the multi-agent StateGraph."""
    g = StateGraph(AgentState)

    g.add_node("planner", planner_node)
    g.add_node("executor", executor_node)
    g.add_node("reviewer", reviewer_node)
    g.add_node("advance", _advance_task)
    g.add_node("retry", _increment_retry)
    g.add_node("finalize", _finalize)

    g.set_entry_point("planner")
    g.add_edge("planner", "executor")
    g.add_edge("executor", "reviewer")
    g.add_conditional_edges(
        "reviewer",
        should_retry,
        {"retry": "retry", "advance": "advance", "done": "finalize"},
    )
    g.add_edge("retry", "executor")
    g.add_edge("advance", "executor")
    g.add_edge("finalize", END)

    return g.compile()


# Singleton compiled graph — import and call `.invoke()` directly
graph = build_graph()
