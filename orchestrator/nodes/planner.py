"""Planner node — decomposes user intent into an ordered task list."""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from nim.client import NIMClient
from orchestrator.state import AgentState

_SYSTEM_PROMPT = """\
You are a task-planning agent. Given a user request, decompose it into a
ordered JSON array of atomic subtasks.

Return ONLY valid JSON — an array of objects, each with:
  {"id": <int>, "description": <str>, "tool": "api" | "sql" | "doc" | "none"}

Do not include any markdown fences or explanatory text outside the JSON.
"""

_llm = NIMClient().get_llm()


def planner_node(state: AgentState) -> AgentState:
    """Break the user request into subtasks; write to state['tasks']."""
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=state["user_request"]),
    ]
    response = _llm.invoke(messages)
    try:
        tasks = json.loads(response.content)
    except json.JSONDecodeError:
        tasks = [{"id": 0, "description": state["user_request"], "tool": "none"}]

    return {
        **state,
        "tasks": tasks,
        "current_task_idx": 0,
        "results": [],
        "retry_count": 0,
        "messages": state["messages"] + [response],
    }
