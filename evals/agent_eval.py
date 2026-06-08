"""Agent evaluation harness with LangSmith tracing and correctness checks."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()

# ---------------------------------------------------------------------------
# Eval dataset — extend with domain-specific QA pairs
# ---------------------------------------------------------------------------

EVAL_DATASET = [
    {
        "query": "What is the capital of France?",
        "expected_keywords": ["Paris"],
    },
    {
        "query": "Summarize the concept of retrieval-augmented generation.",
        "expected_keywords": ["retrieval", "generation", "context"],
    },
    {
        "query": "List the top 3 NVIDIA GPU architectures for AI inference.",
        "expected_keywords": ["Hopper", "Ada", "Ampere"],
    },
]


@dataclass
class EvalResult:
    query: str
    answer: str
    passed: bool
    missing_keywords: list[str] = field(default_factory=list)
    score: float = 0.0


def run_eval(num_cases: int = len(EVAL_DATASET)) -> list[EvalResult]:
    """Run evaluation cases through the agent graph and score results."""
    from orchestrator.graph import app

    langfuse = _get_langfuse()
    results: list[EvalResult] = []

    for case in EVAL_DATASET[:num_cases]:
        state = {
            "user_query": case["query"],
            "session_id": "eval",
            "task_results": [],
            "loop_count": 0,
            "max_loops": 2,
        }
        output = app.invoke(state)
        answer = output.get("final_answer") or ""
        missing = [
            kw for kw in case["expected_keywords"]
            if kw.lower() not in answer.lower()
        ]
        passed = len(missing) == 0
        score = 1.0 - (len(missing) / max(len(case["expected_keywords"]), 1))

        result = EvalResult(
            query=case["query"],
            answer=answer,
            passed=passed,
            missing_keywords=missing,
            score=score,
        )
        results.append(result)

        if langfuse:
            langfuse.score(
                name="keyword_coverage",
                value=score,
                comment=f"Missing: {missing}",
            )

    _print_report(results)
    return results


def _get_langfuse() -> Langfuse | None:
    if os.environ.get("LANGFUSE_PUBLIC_KEY"):
        return Langfuse()
    return None


def _print_report(results: list[EvalResult]) -> None:
    passed = sum(1 for r in results if r.passed)
    avg_score = sum(r.score for r in results) / max(len(results), 1)
    print(f"\nEval Results: {passed}/{len(results)} passed | Avg score: {avg_score:.2f}")
    for r in results:
        status = "✅" if r.passed else "❌"
        print(f"  {status} {r.query[:60]}")
        if r.missing_keywords:
            print(f"     Missing: {r.missing_keywords}")


if __name__ == "__main__":
    run_eval()
