"""Reviewer node — scores executor output and decides to retry or advance."""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from nim.client import NIMClient
from orchestrator.state import AgentState

_SYSTEM_PROMPT = """\
You are a quality-review agent. Given a task description and the result
produced by the executor, score the result from 0.0 to 1.0.

Return ONLY a JSON object: {"score": <float>, "reasoning": <str>}
"""

_RETRY_THRESHOLD = 0.6
_MAX_RETRIES = 2

_llm = NIMClient().get_llm()


def reviewer_node(state: AgentState) -> AgentState:
    """Score latest result and update reviewer_score in state."""
    import json

    tasks = state["tasks"]
    idx = state["current_task_idx"]
    results = state["results"]

    if not results or idx >= len(tasks):
        return {**state, "reviewer_score": 1.0}

    task = tasks[idx]
    latest_result = results[-1]["result"]

    prompt = f"Task: {task['description']}\n\nResult: {latest_result}"
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]
    response = _llm.invoke(messages)
    try:
        parsed = json.loads(response.content)
        score = float(parsed.get("score", 0.5))
    except (json.JSONDecodeError, ValueError):
        score = 0.5

    return {
        **state,
        "reviewer_score": score,
        "messages": state["messages"] + [response],
    }


def should_retry(state: AgentState) -> str:
    """Conditional edge: 'retry', 'advance', or 'done'."""
    score = state.get("reviewer_score", 1.0)
    retry_count = state.get("retry_count", 0)
    idx = state["current_task_idx"]
    total = len(state["tasks"])

    if score < _RETRY_THRESHOLD and retry_count < _MAX_RETRIES:
        return "retry"
    if idx + 1 < total:
        return "advance"
    return "done"
