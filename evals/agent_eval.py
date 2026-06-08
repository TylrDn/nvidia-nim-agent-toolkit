"""LangSmith tracing + correctness evaluation for the multi-agent graph."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from langsmith import Client
from langsmith.evaluation import evaluate

from orchestrator.graph import graph

_LS_CLIENT = Client()
_DATASET_NAME = os.getenv("LANGSMITH_DATASET", "nim-agent-eval-v1")
_EXPERIMENT_PREFIX = os.getenv("LANGSMITH_EXPERIMENT_PREFIX", "nim-agent")


# ---------------------------------------------------------------------------
# Ground-truth dataset (seed inline; in production load from JSON file)
# ---------------------------------------------------------------------------
_SEED_EXAMPLES = [
    {
        "inputs": {"user_request": "What is 2 + 2?"},
        "outputs": {"expected_keywords": ["4", "four"]},
    },
    {
        "inputs": {"user_request": "Fetch the top post from https://jsonplaceholder.typicode.com/posts/1"},
        "outputs": {"expected_keywords": ["userId", "title"]},
    },
]


def _ensure_dataset() -> str:
    """Create dataset if it doesn't exist; return its name."""
    datasets = list(_LS_CLIENT.list_datasets(dataset_name=_DATASET_NAME))
    if not datasets:
        ds = _LS_CLIENT.create_dataset(_DATASET_NAME,
                                       description="NIM agent eval ground truth")
        _LS_CLIENT.create_examples(
            inputs=[e["inputs"] for e in _SEED_EXAMPLES],
            outputs=[e["outputs"] for e in _SEED_EXAMPLES],
            dataset_id=ds.id,
        )
    return _DATASET_NAME


# ---------------------------------------------------------------------------
# Target function — wraps the graph invoke call
# ---------------------------------------------------------------------------
def _run_agent(inputs: dict[str, Any]) -> dict[str, Any]:
    state = graph.invoke({
        "messages": [],
        "user_request": inputs["user_request"],
        "tasks": [],
        "current_task_idx": 0,
        "results": [],
        "reviewer_score": 0.0,
        "retry_count": 0,
        "final_answer": "",
    })
    return {"final_answer": state["final_answer"]}


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------
def _keyword_evaluator(run: Any, example: Any) -> dict[str, Any]:
    """Pass if any expected keyword appears in the final answer."""
    answer = (run.outputs or {}).get("final_answer", "").lower()
    keywords = (example.outputs or {}).get("expected_keywords", [])
    passed = any(kw.lower() in answer for kw in keywords)
    return {"key": "keyword_match", "score": int(passed)}


# ---------------------------------------------------------------------------
# Main eval runner
# ---------------------------------------------------------------------------
def run_eval() -> None:
    dataset_name = _ensure_dataset()
    results = evaluate(
        _run_agent,
        data=dataset_name,
        evaluators=[_keyword_evaluator],
        experiment_prefix=_EXPERIMENT_PREFIX,
        max_concurrency=2,
    )
    print(json.dumps({"summary": str(results)}, indent=2))


if __name__ == "__main__":
    run_eval()
