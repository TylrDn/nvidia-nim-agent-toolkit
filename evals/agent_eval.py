"""LangSmith-backed evaluation harness for the NIM multi-agent toolkit."""
from __future__ import annotations

import os
from typing import Any

from langsmith import Client
from langsmith.evaluation import evaluate, LangChainStringEvaluator
from langsmith.schemas import Run, Example

from orchestrator.graph import run_agent


LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
DATASET_NAME = os.getenv("EVAL_DATASET_NAME", "nim-agent-toolkit-eval")


# ---------------------------------------------------------------------------
# Dataset bootstrap (run once to seed LangSmith)
# ---------------------------------------------------------------------------

DEMO_EXAMPLES = [
    {
        "input": {"query": "What is the current Bitcoin price?"},
        "output": {"answer": "The current Bitcoin price is"},  # prefix match
    },
    {
        "input": {"query": "List all tables in the database"},
        "output": {"answer": "Tables:"},
    },
    {
        "input": {"query": "Summarize the most recent documents about NVIDIA NIM"},
        "output": {"answer": "NVIDIA NIM"},
    },
]


def seed_dataset(client: Client) -> str:
    """Create the LangSmith eval dataset if it doesn't exist."""
    datasets = {d.name: d for d in client.list_datasets()}
    if DATASET_NAME in datasets:
        return datasets[DATASET_NAME].id

    dataset = client.create_dataset(DATASET_NAME, description="NIM agent toolkit eval set")
    for ex in DEMO_EXAMPLES:
        client.create_example(
            inputs=ex["input"],
            outputs=ex["output"],
            dataset_id=dataset.id,
        )
    return dataset.id


# ---------------------------------------------------------------------------
# Target function
# ---------------------------------------------------------------------------

def agent_target(inputs: dict[str, Any]) -> dict[str, Any]:
    """Run the agent and return its final answer."""
    answer = run_agent(inputs["query"])
    return {"answer": answer}


# ---------------------------------------------------------------------------
# Custom evaluators
# ---------------------------------------------------------------------------

def correctness_evaluator(run: Run, example: Example) -> dict:
    """Simple prefix-match correctness check."""
    predicted = (run.outputs or {}).get("answer", "")
    expected = (example.outputs or {}).get("answer", "")
    score = 1 if predicted.startswith(expected) else 0
    return {"key": "correctness", "score": score}


def non_empty_evaluator(run: Run, example: Example) -> dict:
    """Penalise empty or error responses."""
    answer = (run.outputs or {}).get("answer", "")
    score = 0 if not answer or answer.lower().startswith("error") else 1
    return {"key": "non_empty", "score": score}


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------

def run_eval() -> None:
    client = Client(api_key=LANGSMITH_API_KEY)
    seed_dataset(client)

    results = evaluate(
        agent_target,
        data=DATASET_NAME,
        evaluators=[correctness_evaluator, non_empty_evaluator],
        experiment_prefix="nim-agent-toolkit",
        metadata={"model": os.getenv("NIM_MODEL", "meta/llama-3.1-70b-instruct")},
    )

    print(f"\n=== Eval Results ===")
    for r in results:
        print(r)


if __name__ == "__main__":
    run_eval()
