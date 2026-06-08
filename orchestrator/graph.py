"""LangGraph StateGraph — multi-agent coordinator.

Wires the Planner → Executor → Reviewer loop with conditional edges
based on the reviewer's quality score.

Graph topology:
  START → planner → executor → reviewer
                               ├─ continue → executor   (next task)
                               ├─ retry    → executor   (same task, retry)
                               └─ done     → END
"""
from __future__ import annotations

import logging
from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph
from langchain_core.messages import HumanMessage

from nim.client import NIMClient
from orchestrator.state import AgentState
from orchestrator.nodes.planner import planner_node
from orchestrator.nodes.executor import executor_node
from orchestrator.nodes.reviewer import reviewer_node, should_continue

logger = logging.getLogger(__name__)


def build_graph(client: NIMClient | None = None) -> Any:
    """Construct and compile the multi-agent StateGraph."""
    nim = client or NIMClient()

    graph = StateGraph(AgentState)

    graph.add_node("planner", partial(planner_node, client=nim))
    graph.add_node("executor", partial(executor_node, client=nim))
    graph.add_node("reviewer", partial(reviewer_node, client=nim))

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "reviewer")

    graph.add_conditional_edges(
        "reviewer",
        should_continue,
        {
            "continue": "executor",
            "retry": "executor",
            "done": END,
        },
    )

    return graph.compile()


def run(intent: str, client: NIMClient | None = None) -> dict[str, Any]:
    """Run the full agent pipeline for a given user intent."""
    app = build_graph(client)
    initial_state: AgentState = {
        "messages": [HumanMessage(content=intent)],
        "user_intent": intent,
        "task_list": [],
        "current_task_index": 0,
        "task_results": [],
        "reviewer_score": 0.0,
        "retry_count": 0,
        "final_answer": "",
        "metadata": {},
    }
    logger.info("Starting agent run: %s", intent)
    result = app.invoke(initial_state)
    logger.info("Agent run complete. Final answer length: %d chars", len(result.get("final_answer", "")))
    return result
