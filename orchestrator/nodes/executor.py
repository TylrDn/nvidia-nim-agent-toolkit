"""Executor node — dispatches the current task to the appropriate tool.

Routes based on the task's `tool` field:
  api  → api_agent
  sql  → sql_agent
  doc  → doc_agent
  none → direct LLM answer
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from nim.client import NIMClient
from orchestrator.state import AgentState

logger = logging.getLogger(__name__)

EXECUTOR_SYSTEM_PROMPT = """\
You are a focused task executor. Complete the given task using the information
and tool results provided. Be concise and factual. Do not invent data.
"""

MAX_RETRIES = 3


def executor_node(state: AgentState, client: NIMClient | None = None) -> dict[str, Any]:
    """LangGraph node: execute the current task."""
    from agents.api_agent import ApiAgent
    from agents.sql_agent import SqlAgent
    from agents.doc_agent import DocAgent

    llm_client = client or NIMClient()
    task_list = state["task_list"]
    idx = state["current_task_index"]

    if idx >= len(task_list):
        return {}

    task = task_list[idx]
    tool_type = task.get("tool", "none")
    description = task["description"]

    logger.info("Executor: task %d/%d tool=%s", idx + 1, len(task_list), tool_type)

    try:
        if tool_type == "api":
            agent = ApiAgent(llm_client)
            result_text = agent.run(description)
        elif tool_type == "sql":
            agent = SqlAgent(llm_client)
            result_text = agent.run(description)
        elif tool_type == "doc":
            agent = DocAgent(llm_client)
            result_text = agent.run(description)
        else:
            messages = [
                SystemMessage(content=EXECUTOR_SYSTEM_PROMPT),
                HumanMessage(content=description),
            ]
            result_text = llm_client.invoke(messages).content
    except Exception as exc:  # noqa: BLE001
        logger.error("Executor error on task %d: %s", idx + 1, exc)
        result_text = f"Error executing task: {exc}"

    task_result = {
        "task_id": task["id"],
        "description": description,
        "tool": tool_type,
        "output": result_text,
    }

    updated_results = state.get("task_results", []) + [task_result]
    return {
        "task_results": updated_results,
        "messages": [AIMessage(content=result_text)],
    }
