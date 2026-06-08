"""Reviewer node — scores executor output and decides next action."""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from nim.client import get_client
from orchestrator.state import AgentState


SYSTEM_PROMPT = """\
You are a quality reviewer. Given the user's original query and the latest
agent result, evaluate the result and return a JSON object with:
- "score": float 0.0-1.0 (1.0 = fully answers the query)
- "verdict": "accept" | "retry" | "complete"
  - "accept": result is good enough, move to next task
  - "retry": result is insufficient, retry this task (max 2 retries)
  - "complete": all tasks done, synthesize final answer
- "final_answer": string (only populated when verdict is "complete")
- "feedback": brief reason for the score

Return ONLY the JSON object.
"""

MAX_RETRIES = 2


def reviewer_node(state: AgentState) -> dict[str, Any]:
    """Score latest result and route: retry, advance, or complete."""
    llm = get_client().as_langchain_llm()

    task_results = state.get("task_results", [])
    latest_result = task_results[-1]["result"] if task_results else ""
    idx = state["current_task_index"]
    task_list = state["task_list"]

    prompt = f"User query: {state['user_query']}\n\nLatest result:\n{latest_result}"
    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    try:
        review = json.loads(response.content)
    except json.JSONDecodeError:
        review = {"score": 0.8, "verdict": "accept", "final_answer": "", "feedback": "parse error"}

    score = float(review.get("score", 0.8))
    verdict = review.get("verdict", "accept")
    retry_count = state.get("retry_count", 0)

    updates: dict[str, Any] = {"reviewer_score": score, "messages": [response]}

    if verdict == "complete" or idx >= len(task_list) - 1:
        final = review.get("final_answer") or _synthesize(state, latest_result)
        updates["final_answer"] = final
        updates["routing_key"] = "__end__"
    elif verdict == "retry" and retry_count < MAX_RETRIES:
        updates["retry_count"] = retry_count + 1
        updates["routing_key"] = "retry"
    else:
        updates["current_task_index"] = idx + 1
        updates["retry_count"] = 0
        updates["routing_key"] = "next_task"

    return updates


def _synthesize(state: AgentState, latest: str) -> str:
    """Fallback: concatenate all task results into a coherent answer."""
    parts = [r["result"] for r in state.get("task_results", [])]
    return "\n\n".join(parts) or latest
