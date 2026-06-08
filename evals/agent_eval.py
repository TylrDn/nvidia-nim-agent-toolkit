"""Agent evaluation harness with Langfuse tracing.

Runs a suite of test cases through the full multi-agent pipeline
and scores each result for correctness, relevancy, and groundedness.
Outputs a structured evaluation report to evals/reports/.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

REPORTS_DIR = Path("evals/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

LANGFUSE_ENABLED = (
    os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
)


@dataclass
class EvalCase:
    id: str
    intent: str
    expected_keywords: list[str]
    expected_tool: str  # api | sql | doc | none | any
    description: str = ""


@dataclass
class EvalResult:
    case_id: str
    intent: str
    final_answer: str
    task_count: int
    reviewer_score: float
    keyword_hit_rate: float
    latency_sec: float
    passed: bool
    error: str = ""
    trace_url: str = ""


DEFAULT_EVAL_SUITE: list[EvalCase] = [
    EvalCase(
        id="eval_001",
        intent="What is the current weather in San Francisco?",
        expected_keywords=["weather", "temperature", "San Francisco"],
        expected_tool="api",
        description="Basic API lookup",
    ),
    EvalCase(
        id="eval_002",
        intent="List the top 5 customers by total order value from the database.",
        expected_keywords=["customer", "order", "value"],
        expected_tool="sql",
        description="SQL aggregation query",
    ),
    EvalCase(
        id="eval_003",
        intent="What are the key principles of retrieval-augmented generation?",
        expected_keywords=["retrieval", "augmented", "generation", "context"],
        expected_tool="doc",
        description="Document retrieval Q&A",
    ),
    EvalCase(
        id="eval_004",
        intent="Summarise the quarterly sales report and identify the top-performing region.",
        expected_keywords=["sales", "region", "quarterly"],
        expected_tool="any",
        description="Multi-step compound task",
    ),
    EvalCase(
        id="eval_005",
        intent="What is 2 + 2?",
        expected_keywords=["4", "four"],
        expected_tool="none",
        description="Direct LLM answer (no tool)",
    ),
]


def _score_keywords(answer: str, keywords: list[str]) -> float:
    """Return fraction of keywords present in the answer (case-insensitive)."""
    if not keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return hits / len(keywords)


def run_eval(
    cases: list[EvalCase] | None = None,
    client: Any | None = None,
) -> list[EvalResult]:
    """Run evaluation suite and return results."""
    from orchestrator.graph import run as run_pipeline
    from nim.client import NIMClient

    nim = client or NIMClient()
    suite = cases or DEFAULT_EVAL_SUITE
    results: list[EvalResult] = []

    if LANGFUSE_ENABLED:
        try:
            from langfuse import Langfuse

            lf = Langfuse(
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            )
            logger.info("Langfuse tracing enabled.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Langfuse init failed — running without tracing: %s", exc)
            lf = None
    else:
        lf = None

    for case in suite:
        logger.info("Running eval case %s: %s", case.id, case.intent)
        trace_url = ""
        error = ""
        start = time.perf_counter()

        try:
            if lf:
                trace = lf.trace(name=f"eval_{case.id}", input={"intent": case.intent})
                trace_url = trace.get_trace_url() if hasattr(trace, "get_trace_url") else ""

            state = run_pipeline(intent=case.intent, client=nim)
            final_answer = state.get("final_answer", "")
            task_count = len(state.get("task_list", []))
            reviewer_score = state.get("reviewer_score", 0.0)

            if lf:
                trace.update(output={"final_answer": final_answer, "reviewer_score": reviewer_score})

        except Exception as exc:  # noqa: BLE001
            logger.error("Eval %s failed: %s", case.id, exc)
            final_answer = ""
            task_count = 0
            reviewer_score = 0.0
            error = str(exc)

        latency = time.perf_counter() - start
        kw_rate = _score_keywords(final_answer, case.expected_keywords)
        passed = kw_rate >= 0.5 and reviewer_score >= 0.5 and not error

        result = EvalResult(
            case_id=case.id,
            intent=case.intent,
            final_answer=final_answer,
            task_count=task_count,
            reviewer_score=reviewer_score,
            keyword_hit_rate=kw_rate,
            latency_sec=round(latency, 3),
            passed=passed,
            error=error,
            trace_url=trace_url,
        )
        results.append(result)
        status = "PASS" if passed else "FAIL"
        logger.info(
            "[%s] %s | score=%.2f kw=%.2f latency=%.2fs",
            status, case.id, reviewer_score, kw_rate, latency,
        )

    _write_report(results)
    return results


def _write_report(results: list[EvalResult]) -> None:
    """Write JSON evaluation report to evals/reports/."""
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    report_path = REPORTS_DIR / f"eval_report_{ts}.json"
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    report = {
        "timestamp": ts,
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 3) if total else 0.0,
        "avg_latency_sec": round(sum(r.latency_sec for r in results) / total, 3) if total else 0.0,
        "results": [asdict(r) for r in results],
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Eval report written to %s — %d/%d passed", report_path, passed, total)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_eval()
