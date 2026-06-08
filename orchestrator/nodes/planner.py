"""Planner node — decomposes user intent into an ordered task list."""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from nim.client import get_client
from orchestrator.state import AgentState


SYSTEM_PROMPT = """\
You are a task planning agent. Given a user query, decompose it into an ordered
list of discrete subtasks. Each subtask must specify:
- "id": sequential integer
- "description": what needs to be done
- "agent": one of ["api_agent", "sql_agent", "doc_agent"]
- "depends_on": list of task ids this task depends on (empty list if none)

Return ONLY a JSON array of task objects. No prose, no markdown fences.
"""


def planner_node(state: AgentState) -> dict[str, Any]:
    """Decompose user_query into task_list."""
    llm = get_client().as_langchain_llm()

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=state["user_query"]),
    ])

    try:
        task_list = json.loads(response.content)
    except json.JSONDecodeError:
        # Fallback: single pass-through task
        task_list = [{
            "id": 1,
            "description": state["user_query"],
            "agent": "doc_agent",
            "depends_on": [],
        }]

    return {
        "task_list": task_list,
        "current_task_index": 0,
        "task_results": [],
        "retry_count": 0,
        "messages": [response],
    }
