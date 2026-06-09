"""LangSmith-backed evaluation harness for the NIM multi-agent toolkit.

Offline only — never imported by the main pipeline path. Run as a CLI:

    python -m evals.agent_eval --output-file results/eval.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.schemas import Example, Run

from orchestrator.graph import run_agent

load_dotenv()
logger = logging.getLogger(__name__)

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
DATASET_NAME = os.getenv("EVAL_DATASET_NAME", "nim-agent-toolkit-eval")

DEMO_EXAMPLES = [
    {
        "input": {"query": "What is the current Bitcoin price?"},
        "output": {"answer": "The current Bitcoin price is"},
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
    """Create the LangSmith eval dataset if it does not already exist.

    Args:
        client: An authenticated LangSmith client.

    Returns:
        str: The dataset id.
    """
    datasets = {d.name: d for d in client.list_datasets()}
    if DATASET_NAME in datasets:
        return datasets[DATASET_NAME].id

    dataset = client.create_dataset(DATASET_NAME, description="NIM agent toolkit eval set")
    for example in DEMO_EXAMPLES:
        client.create_example(
            inputs=example["input"],
            outputs=example["output"],
            dataset_id=dataset.id,
        )
    return dataset.id


def agent_target(inputs: dict[str, Any]) -> dict[str, Any]:
    """Run the agent and return its final answer.

    Args:
        inputs: A mapping containing the ``query`` key.

    Returns:
        dict[str, Any]: ``{"answer": <final answer>}``.
    """
    return {"answer": run_agent(inputs["query"])}


def correctness_evaluator(run: Run, example: Example) -> dict[str, Any]:
    """Prefix-match correctness check.

    Args:
        run: The evaluation run with the model output.
        example: The dataset example with the expected output.

    Returns:
        dict[str, Any]: A LangSmith score dict.
    """
    predicted = (run.outputs or {}).get("answer", "")
    expected = (example.outputs or {}).get("answer", "")
    return {"key": "correctness", "score": int(predicted.startswith(expected))}


def non_empty_evaluator(run: Run, example: Example) -> dict[str, Any]:
    """Penalize empty or error responses.

    Args:
        run: The evaluation run with the model output.
        example: The dataset example (unused).

    Returns:
        dict[str, Any]: A LangSmith score dict.
    """
    answer = (run.outputs or {}).get("answer", "")
    score = 0 if not answer or answer.lower().startswith("error") else 1
    return {"key": "non_empty", "score": score}


def run_eval(output_file: str) -> None:
    """Run the evaluation and write a JSON summary to ``output_file``.

    Args:
        output_file: Path to write the JSON results to.
    """
    client = Client(api_key=LANGSMITH_API_KEY)
    seed_dataset(client)

    results = evaluate(
        agent_target,
        data=DATASET_NAME,
        evaluators=[correctness_evaluator, non_empty_evaluator],
        experiment_prefix="nim-agent-toolkit",
        metadata={"model": os.getenv("NIM_DEFAULT_MODEL", "meta/llama-3.1-70b-instruct")},
    )

    summary = [{"example": str(r)} for r in results]
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump({"dataset": DATASET_NAME, "results": summary}, handle, indent=2)
    logger.info("Wrote %d eval result(s) to %s", len(summary), output_file)


def main() -> None:
    """CLI entry point for the evaluation harness."""
    parser = argparse.ArgumentParser(description="Run the NIM agent LangSmith eval.")
    parser.add_argument(
        "--output-file",
        default="results/agent_eval.json",
        help="Path to write the JSON eval summary.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_eval(args.output_file)


if __name__ == "__main__":
    main()
