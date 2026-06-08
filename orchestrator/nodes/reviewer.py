"""Reviewer node — validates executor output and routes retry or done.

Scores the last tool result on a 0.0–1.0 scale. If below threshold,
the graph loops back to the executor. After MAX_RETRIES, exits regardless.
"""
from __future__ import annotations

import logging
import json
from orchestrator.state import AgentState
from nim.client import get_default_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a quality reviewer for AI agent outputs.
Given a task and its result, score the result from 0.0 to 1.0 where:
- 1.0 = complete, accurate, actionable
- 0.5 = partially complete or needs clarification
- 0.0 = wrong, empty, or errored

Return ONLY a JSON object: {"score": <float>, "feedback": "<one sentence>"}"""


def reviewer_node(state: AgentState) -> dict:
    """LangGraph node: score the latest tool result.

    Args:
        state: Current AgentState with tool_results populated.

    Returns:
        Partial state update with reviewer_score and incremented retry_count if needed.
    """
    results = state.get("tool_results", [])
    if not results:
        return {"reviewer_score": 0.0, "retry_count": state.get("retry_count", 0) + 1}

    last = results[-1]
    llm = get_default_llm()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Task: {last['task']}\nResult: {last['output']}"},
    ]

    try:
        response = llm.invoke(messages)
        data = json.loads(response.content)
        score = float(data.get("score", 0.5))
        logger.info("Reviewer score: %.2f — %s", score, data.get("feedback", ""))
    except Exception as exc:
        logger.warning("Reviewer parse error: %s — defaulting score to 0.5", exc)
        score = 0.5

    return {
        "reviewer_score": score,
        "retry_count": state.get("retry_count", 0) + (1 if score < 0.7 else 0),
    }
