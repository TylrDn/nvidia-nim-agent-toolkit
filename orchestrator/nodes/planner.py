"""Planner node — decomposes user intent into ordered subtasks.

Receives the user_input from AgentState, calls NIM to produce
a numbered task list, and writes it back to state.plan.
"""
from __future__ import annotations

import logging
from orchestrator.state import AgentState
from nim.client import get_default_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a task planning agent. Given a user request, decompose it
into a concise ordered list of actionable subtasks. Return ONLY a numbered list,
one task per line. Be specific and tool-oriented."""


def planner_node(state: AgentState) -> dict:
    """LangGraph node: generate a task plan from user_input.

    Args:
        state: Current AgentState with user_input populated.

    Returns:
        Partial state update with ``plan`` and reset ``current_task``.
    """
    logger.info("Planner: decomposing task for input: %s", state["user_input"][:80])
    llm = get_default_llm()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": state["user_input"]},
    ]

    response = llm.invoke(messages)
    raw_plan = response.content

    # Parse numbered list into clean task strings
    tasks = [
        line.split(".", 1)[-1].strip()
        for line in raw_plan.strip().splitlines()
        if line.strip() and line[0].isdigit()
    ]

    logger.info("Planner: generated %d tasks", len(tasks))
    return {"plan": tasks, "current_task": 0, "retry_count": 0}
