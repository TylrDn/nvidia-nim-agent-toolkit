"""LangGraph StateGraph — Planner → Executor → Reviewer multi-agent loop."""
from __future__ import annotations

from typing import Any

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from orchestrator.state import AgentState
from orchestrator.nodes.planner import planner_node
from orchestrator.nodes.executor import executor_node
from orchestrator.nodes.reviewer import reviewer_node


def _route_after_reviewer(state: AgentState) -> str:
    """Conditional edge: decide what comes after the reviewer."""
    key = state.get("routing_key", "next_task")
    if key == "__end__":
        return END
    elif key == "retry":
        return "executor"
    else:  # next_task
        return "executor"


def build_graph(checkpointer: MemorySaver | None = None) -> Any:
    """Assemble and compile the multi-agent StateGraph."""
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("reviewer", reviewer_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "reviewer")
    graph.add_conditional_edges("reviewer", _route_after_reviewer)

    memory = checkpointer or MemorySaver()
    return graph.compile(checkpointer=memory)


def run_agent(query: str, thread_id: str = "default") -> str:
    """Run the multi-agent pipeline for a single user query."""
    app = build_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: AgentState = {
        "messages": [],
        "user_query": query,
        "task_list": [],
        "current_task_index": 0,
        "task_results": [],
        "final_answer": "",
        "reviewer_score": 0.0,
        "retry_count": 0,
        "routing_key": "",
        "metadata": {},
    }

    final = app.invoke(initial_state, config=config)
    return final.get("final_answer", "")
