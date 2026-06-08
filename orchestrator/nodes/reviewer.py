"""Reviewer node — validates executor output and decides next action.

Scores the most recent task result (0.0 – 1.0) and routes to:
  - next task (score >= threshold)
  - retry (score < threshold and retries remaining)
  - synthesise final answer (all tasks done)
"""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from nim.client import NIMClient
from orchestrator.state import AgentState

logger = logging.getLogger(__name__)

SCORE_THRESHOLD = 0.65
MAX_RETRIES = 3

REVIEWER_SYSTEM_PROMPT = """\
You are a rigorous quality reviewer for an AI task pipeline.

Given a task description and its output, score the output quality from 0.0 to 1.0.
Respond ONLY with a JSON object: {"score": <float>, "reason": "<one sentence>"}

Scoring guide:
  1.0 — fully correct, complete, and relevant
  0.7 — mostly correct with minor gaps
  0.5 — partially correct but missing key information
  0.3 — attempted but largely incorrect
  0.0 — empty, errored, or completely off-topic
"""

SYNTHESIS_SYSTEM_PROMPT = """\
You are a synthesis assistant. Given an original user intent and a list of completed
task results, compose a clear, concise, and accurate final answer.
Do not include internal task IDs or tool names in your response.
"""


def reviewer_node(state: AgentState, client: NIMClient | None = None) -> dict[str, Any]:
    """LangGraph node: review latest result and update routing state."""
    llm_client = client or NIMClient()
    llm = llm_client.llm

    task_results = state.get("task_results", [])
    task_list = state.get("task_list", [])
    idx = state.get("current_task_index", 0)
    retry_count = state.get("retry_count", 0)

    if not task_results:
        return {"reviewer_score": 0.0}

    latest = task_results[-1]
    task_desc = latest["description"]
    task_output = latest["output"]

    review_messages = [
        SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Task: {task_desc}\n\nOutput:\n{task_output}"
        ),
    ]

    raw = llm.invoke(review_messages).content.strip()
    try:
        review = json.loads(raw)
        score = float(review.get("score", 0.5))
        reason = review.get("reason", "")
    except (json.JSONDecodeError, ValueError):
        score = 0.5
        reason = "Could not parse reviewer response."

    logger.info(
        "Reviewer: task %d score=%.2f reason=%s",
        idx + 1,
        score,
        reason,
    )

    updates: dict[str, Any] = {"reviewer_score": score}

    all_tasks_done = (idx + 1) >= len(task_list)

    if score >= SCORE_THRESHOLD or all_tasks_done:
        # Advance to next task or synthesise if all done.
        next_idx = idx + 1
        updates["current_task_index"] = next_idx
        updates["retry_count"] = 0

        if next_idx >= len(task_list):
            # All tasks complete — synthesise final answer.
            intent = state.get("user_intent", "")
            results_summary = "\n".join(
                f"- {r['description']}: {r['output']}" for r in task_results
            )
            synthesis_messages = [
                SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"User intent: {intent}\n\nTask results:\n{results_summary}"
                ),
            ]
            final = llm.invoke(synthesis_messages).content
            updates["final_answer"] = final
    else:
        # Score below threshold — retry if budget allows.
        new_retry = retry_count + 1
        updates["retry_count"] = new_retry
        if new_retry >= MAX_RETRIES:
            logger.warning("Max retries reached for task %d — advancing anyway.", idx + 1)
            updates["current_task_index"] = idx + 1
            updates["retry_count"] = 0

    return updates


def should_continue(state: AgentState) -> str:
    """LangGraph conditional edge: decide which node to go to next."""
    idx = state.get("current_task_index", 0)
    task_list = state.get("task_list", [])

    if idx >= len(task_list):
        return "done"

    score = state.get("reviewer_score", 1.0)
    retry_count = state.get("retry_count", 0)

    if score < SCORE_THRESHOLD and retry_count < MAX_RETRIES:
        return "retry"

    return "continue"
