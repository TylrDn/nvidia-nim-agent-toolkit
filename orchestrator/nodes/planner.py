"""Planner node — decomposes user intent into an ordered task list.

Produces a JSON array of tasks, each with:
  - id: sequential task identifier
  - description: what to do
  - tool: which tool category to use (api | sql | doc | none)
  - depends_on: list of task ids that must complete first
"""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from nim.client import NIMClient
from orchestrator.state import AgentState

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """\
You are a precise task planner for an enterprise AI assistant.

Given a user intent, decompose it into an ordered list of atomic tasks.
Respond ONLY with a valid JSON array. Each element must have:
  - "id": integer starting at 1
  - "description": clear action description
  - "tool": one of ["api", "sql", "doc", "none"]
  - "depends_on": list of task ids (empty list if no dependency)

Do NOT include any text outside the JSON array.
"""


def planner_node(state: AgentState, client: NIMClient | None = None) -> dict[str, Any]:
    """LangGraph node: plan tasks from the user intent."""
    llm = (client or NIMClient()).llm
    intent = state["user_intent"]

    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=f"User intent: {intent}"),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    try:
        task_list = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Planner returned non-JSON; falling back to single task.")
        task_list = [
            {
                "id": 1,
                "description": intent,
                "tool": "none",
                "depends_on": [],
            }
        ]

    logger.info("Planner produced %d tasks for intent: %s", len(task_list), intent)
    return {
        "task_list": task_list,
        "current_task_index": 0,
        "task_results": [],
        "retry_count": 0,
    }
