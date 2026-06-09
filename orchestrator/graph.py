"""LangGraph StateGraph — Planner → Executor → Reviewer multi-agent loop."""
from __future__ import annotations

import asyncio
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from orchestrator.nodes.executor import executor_node
from orchestrator.nodes.planner import planner_node
from orchestrator.nodes.reviewer import reviewer_node
from orchestrator.state import AgentState


def _route_after_reviewer(state: AgentState) -> str:
    """Conditional edge: decide what comes after the reviewer.

    Args:
        state: Current graph state.

    Returns:
        str: ``END`` when complete, otherwise the next node name.
    """
    if state.get("routing_key") == "__end__":
        return END
    return "executor"


def build_graph(checkpointer: MemorySaver | None = None) -> Any:
    """Assemble and compile the multi-agent StateGraph.

    Args:
        checkpointer: Optional checkpointer; an in-memory saver is used by default.

    Returns:
        Any: The compiled LangGraph application.
    """
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("reviewer", reviewer_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "reviewer")
    graph.add_conditional_edges("reviewer", _route_after_reviewer, {"executor": "executor", END: END})

    memory = checkpointer or MemorySaver()
    return graph.compile(checkpointer=memory)


def _initial_state(query: str) -> AgentState:
    """Build the initial graph state for a query."""
    return {
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


async def run_agent_async(query: str, thread_id: str = "default") -> str:
    """Run the multi-agent pipeline asynchronously for a single query.

    Args:
        query: The user query.
        thread_id: Checkpointer thread identifier.

    Returns:
        str: The final synthesized answer.
    """
    app = build_graph()
    config = {"configurable": {"thread_id": thread_id}}
    final = await app.ainvoke(_initial_state(query), config=config)
    return final.get("final_answer", "")


def run_agent(query: str, thread_id: str = "default") -> str:
    """Synchronous wrapper around :func:`run_agent_async` for CLI/eval use.

    Args:
        query: The user query.
        thread_id: Checkpointer thread identifier.

    Returns:
        str: The final synthesized answer.
    """
    return asyncio.run(run_agent_async(query, thread_id=thread_id))
