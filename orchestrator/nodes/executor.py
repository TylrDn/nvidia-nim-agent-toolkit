"""Executor node — dispatches the current task to the appropriate sub-agent."""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage

from configs.loader import get_routing_config
from orchestrator.state import AgentState

logger = logging.getLogger(__name__)


async def executor_node(state: AgentState) -> dict[str, Any]:
    """Run the current task by routing to the correct sub-agent.

    Args:
        state: Current graph state.

    Returns:
        dict[str, Any]: Partial state update appending the task result.
    """
    idx = state["current_task_index"]
    task = state["task_list"][idx]
    fallback = get_routing_config().fallback_agent
    agent_key = task.get("agent", fallback)

    # Lazy import to avoid circular dependencies at module load.
    from agents.api_agent import run as api_run
    from agents.doc_agent import run as doc_run
    from agents.sql_agent import run as sql_run

    dispatch = {
        "api_agent": api_run,
        "sql_agent": sql_run,
        "doc_agent": doc_run,
    }
    runner = dispatch.get(agent_key, doc_run)
    logger.info("Executor dispatching task %s to %s", task.get("id"), agent_key)

    result = await runner(task["description"], state)

    task_results = list(state.get("task_results", []))
    task_results.append({"task_id": task["id"], "agent": agent_key, "result": result})

    return {
        "task_results": task_results,
        "routing_key": agent_key,
        "messages": [AIMessage(content=f"[{agent_key}] {result}")],
    }
