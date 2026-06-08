"""Planner node — decomposes user intent into an ordered list of subtasks."""
from __future__ import annotations

import json
import uuid

from langchain_core.messages import HumanMessage, SystemMessage

from nim.client import NIMClient
from orchestrator.state import AgentState, SubTask

SYSTEM_PROMPT = """\
You are a planning agent. Given a user query, decompose it into a list of
atomic subtasks. Each subtask must specify which specialist agent should
handle it:
  - "api"  — fetch data from a REST API
  - "sql"  — query a relational database
  - "doc"  — retrieve information from a document store

Respond ONLY with a JSON array of objects with keys:
  task_id, description, agent

Example:
[
  {"task_id": "t1", "description": "Get current weather for Seattle", "agent": "api"},
  {"task_id": "t2", "description": "Find Q1 sales data for Seattle region", "agent": "sql"}
]
"""

_client = NIMClient()
_llm = _client.get_llm()


def planner_node(state: AgentState) -> dict:
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=state["user_query"]),
    ]
    response = _llm.invoke(messages)
    try:
        raw_tasks = json.loads(response.content)
    except json.JSONDecodeError:
        # Fallback: single doc task
        raw_tasks = [{"task_id": "t1", "description": state["user_query"], "agent": "doc"}]

    plan: list[SubTask] = [
        SubTask(
            task_id=t.get("task_id", str(uuid.uuid4())[:8]),
            description=t["description"],
            agent=t.get("agent", "doc"),
            status="pending",
            result=None,
        )
        for t in raw_tasks
    ]
    return {
        "plan": plan,
        "current_task_index": 0,
        "loop_count": 0,
        "max_loops": state.get("max_loops", 3),
    }
