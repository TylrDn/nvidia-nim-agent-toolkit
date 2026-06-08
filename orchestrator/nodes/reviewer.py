"""Reviewer node — scores task results and decides to loop or terminate."""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from nim.client import NIMClient
from orchestrator.state import AgentState

SYSTEM_PROMPT = """\
You are a quality reviewer agent. Given the original user query and a list
of task results, evaluate whether the results collectively answer the query.

Respond ONLY with a JSON object:
{
  "score": <float 0.0–1.0>,
  "feedback": "<one sentence explaining the score>",
  "final_answer": "<synthesized answer if score >= 0.8, else empty string>"
}
"""

_client = NIMClient()
_llm = _client.get_llm()


def reviewer_node(state: AgentState) -> dict:
    results_text = json.dumps(state["task_results"], indent=2)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=f"User query: {state['user_query']}\n\nTask results:\n{results_text}"
        ),
    ]
    response = _llm.invoke(messages)
    try:
        parsed = json.loads(response.content)
    except json.JSONDecodeError:
        parsed = {"score": 0.5, "feedback": "Could not parse reviewer response.", "final_answer": ""}

    return {
        "reviewer_score": float(parsed.get("score", 0.5)),
        "reviewer_feedback": parsed.get("feedback", ""),
        "final_answer": parsed.get("final_answer") or None,
        "loop_count": state.get("loop_count", 0) + 1,
    }
