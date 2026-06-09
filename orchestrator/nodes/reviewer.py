"""Reviewer node — scores executor output and decides the next action."""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from configs.loader import get_agent_config, get_routing_config
from nim.client import NIMClient, get_callbacks
from orchestrator.state import AgentState

logger = logging.getLogger(__name__)

AGENT_NAME = "reviewer"


async def reviewer_node(state: AgentState) -> dict[str, Any]:
    """Score the latest result and route: retry, advance, or complete.

    Args:
        state: Current graph state.

    Returns:
        dict[str, Any]: Partial state update with score and routing decision.
    """
    config = get_agent_config(AGENT_NAME)
    routing = get_routing_config()
    llm = NIMClient(model=config.model, temperature=config.temperature).as_langchain_llm()

    task_results = state.get("task_results", [])
    latest_result = task_results[-1]["result"] if task_results else ""
    idx = state["current_task_index"]
    task_list = state["task_list"]

    prompt = f"User query: {state['user_query']}\n\nLatest result:\n{latest_result}"
    response = await llm.ainvoke(
        [
            SystemMessage(content=config.system_prompt),
            HumanMessage(content=prompt),
        ],
        config={"callbacks": get_callbacks()},
    )

    content = response.content if isinstance(response.content, str) else ""
    try:
        review = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Reviewer returned non-JSON; defaulting to accept")
        review = {"score": 0.8, "verdict": "accept", "final_answer": "", "feedback": "parse error"}

    score = float(review.get("score", 0.8))
    verdict = review.get("verdict", "accept")
    retry_count = state.get("retry_count", 0)

    updates: dict[str, Any] = {"reviewer_score": score, "messages": [response]}

    if verdict == "complete" or idx >= len(task_list) - 1:
        final = review.get("final_answer") or _synthesize(state, latest_result)
        updates["final_answer"] = final
        updates["routing_key"] = "__end__"
        logger.info("Reviewer marked pipeline complete (score=%.2f)", score)
    elif verdict == "retry" and retry_count < routing.max_retries:
        updates["retry_count"] = retry_count + 1
        updates["routing_key"] = "retry"
        logger.info("Reviewer requested retry %d/%d", retry_count + 1, routing.max_retries)
    else:
        updates["current_task_index"] = idx + 1
        updates["retry_count"] = 0
        updates["routing_key"] = "next_task"
        logger.info("Reviewer advancing to task index %d", idx + 1)

    return updates


def _synthesize(state: AgentState, latest: str) -> str:
    """Concatenate all task results into a coherent fallback answer.

    Args:
        state: Current graph state.
        latest: The most recent result, used if no results are accumulated.

    Returns:
        str: The synthesized answer.
    """
    parts = [r["result"] for r in state.get("task_results", [])]
    return "\n\n".join(parts) or latest
