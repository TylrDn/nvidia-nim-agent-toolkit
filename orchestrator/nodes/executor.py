"""Executor node — dispatches the current task to the appropriate agent."""
from __future__ import annotations

from orchestrator.state import AgentState


def executor_node(state: AgentState) -> AgentState:
    """Run the current subtask via the correct tool agent."""
    from agents.api_agent import run as run_api
    from agents.doc_agent import run as run_doc
    from agents.sql_agent import run as run_sql

    tasks = state["tasks"]
    idx = state["current_task_idx"]

    if idx >= len(tasks):
        return state

    task = tasks[idx]
    tool = task.get("tool", "none")

    dispatch = {"api": run_api, "sql": run_sql, "doc": run_doc}

    if tool in dispatch:
        result = dispatch[tool](task["description"])
    else:
        result = {"output": task["description"], "tool": "none"}

    return {
        **state,
        "results": state["results"] + [{"task_id": task["id"], "result": result}],
    }
