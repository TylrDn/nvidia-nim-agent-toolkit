"""Executor node — dispatches tool calls to the appropriate sub-agent."""
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from orchestrator.state import AgentState


def executor_node(state: AgentState) -> dict[str, Any]:
    """Run the current task by routing to the correct agent."""
    idx = state["current_task_index"]
    task = state["task_list"][idx]
    agent_key = task.get("agent", "doc_agent")

    # Lazy import to avoid circular deps
    from agents.api_agent import run as api_run
    from agents.sql_agent import run as sql_run
    from agents.doc_agent import run as doc_run

    dispatch = {
        "api_agent": api_run,
        "sql_agent": sql_run,
        "doc_agent": doc_run,
    }

    runner = dispatch.get(agent_key, doc_run)
    result = runner(task["description"], state)

    task_results = list(state.get("task_results", []))
    task_results.append({"task_id": task["id"], "agent": agent_key, "result": result})

    msg = AIMessage(content=f"[{agent_key}] {result}")

    return {
        "task_results": task_results,
        "routing_key": agent_key,
        "messages": [msg],
    }
