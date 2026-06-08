"""Executor node — dispatches the current subtask to the appropriate agent."""
from __future__ import annotations

from orchestrator.state import AgentState
from agents.api_agent import run_api_agent
from agents.sql_agent import run_sql_agent
from agents.doc_agent import run_doc_agent

_AGENT_DISPATCH = {
    "api": run_api_agent,
    "sql": run_sql_agent,
    "doc": run_doc_agent,
}


def executor_node(state: AgentState) -> dict:
    plan = state["plan"]
    idx = state["current_task_index"]

    # Find the next pending task
    pending = [i for i, t in enumerate(plan) if t["status"] == "pending"]
    if not pending:
        return {"task_results": []}

    task = plan[pending[0]]
    agent_fn = _AGENT_DISPATCH.get(task["agent"], run_doc_agent)

    try:
        result = agent_fn(task["description"])
        task["status"] = "done"
        task["result"] = result
    except Exception as exc:  # noqa: BLE001
        task["status"] = "failed"
        task["result"] = f"Error: {exc}"

    return {
        "plan": plan,
        "current_task_index": pending[0] + 1,
        "task_results": [{"task_id": task["task_id"], "result": task["result"]}],
    }
