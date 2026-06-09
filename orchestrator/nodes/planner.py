"""Planner node — decomposes user intent into an ordered task list."""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from configs.loader import get_agent_config
from nim.client import NIMClient, get_callbacks
from orchestrator.state import AgentState

logger = logging.getLogger(__name__)

AGENT_NAME = "planner"


async def planner_node(state: AgentState) -> dict[str, Any]:
    """Decompose ``user_query`` into an ordered ``task_list``.

    Args:
        state: Current graph state.

    Returns:
        dict[str, Any]: Partial state update with the task list and counters reset.
    """
    config = get_agent_config(AGENT_NAME)
    llm = NIMClient(model=config.model, temperature=config.temperature).as_langchain_llm()

    response = await llm.ainvoke(
        [
            SystemMessage(content=config.system_prompt),
            HumanMessage(content=state["user_query"]),
        ],
        config={"callbacks": get_callbacks()},
    )

    content = response.content if isinstance(response.content, str) else ""
    try:
        task_list = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Planner returned non-JSON; falling back to single task")
        task_list = [
            {
                "id": 1,
                "description": state["user_query"],
                "agent": "doc_agent",
                "depends_on": [],
            }
        ]

    logger.info("Planner produced %d task(s)", len(task_list))
    return {
        "task_list": task_list,
        "current_task_index": 0,
        "task_results": [],
        "retry_count": 0,
        "messages": [response],
    }
