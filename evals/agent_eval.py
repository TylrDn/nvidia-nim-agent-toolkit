"""LangSmith tracing + correctness checks for the multi-agent system.

Runs end-to-end evaluation scenarios and uploads results to LangSmith
for review. Tracks accuracy, tool call rate, and latency per scenario.
"""
from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from orchestrator.graph import build_graph

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class EvalScenario:
    name: str
    user_input: str
    expected_keywords: list[str] = field(default_factory=list)


# Default eval scenarios — extend with domain-specific cases
DEFAULT_SCENARIOS: list[EvalScenario] = [
    EvalScenario(
        name="api_fetch",
        user_input="Fetch the current weather in San Francisco using the Open-Meteo API.",
        expected_keywords=["temperature", "weather"],
    ),
    EvalScenario(
        name="sql_query",
        user_input="How many users signed up last month?",
        expected_keywords=["users", "count"],
    ),
    EvalScenario(
        name="doc_search",
        user_input="What is our company refund policy?",
        expected_keywords=["refund", "policy"],
    ),
]


def run_eval(
    scenarios: list[EvalScenario] | None = None,
) -> list[dict[str, Any]]:
    """Run eval scenarios and return results.

    Args:
        scenarios: List of scenarios to run. Defaults to DEFAULT_SCENARIOS.

    Returns:
        List of result dicts with name, passed, latency_ms, output.
    """
    graph = build_graph()
    results = []

    for scenario in (scenarios or DEFAULT_SCENARIOS):
        logger.info("Running eval: %s", scenario.name)
        start = time.time()

        try:
            state = graph.invoke({"user_input": scenario.user_input})
            output = state.get("final_output", "")
            passed = all(kw.lower() in output.lower() for kw in scenario.expected_keywords)
        except Exception as exc:
            output = str(exc)
            passed = False

        latency_ms = int((time.time() - start) * 1000)
        results.append({
            "name": scenario.name,
            "passed": passed,
            "latency_ms": latency_ms,
            "output": output[:500],
        })
        logger.info("%s: %s (%dms)", scenario.name, "PASS" if passed else "FAIL", latency_ms)

    return results


if __name__ == "__main__":
    results = run_eval()
    passed = sum(1 for r in results if r["passed"])
    print(f"\nResults: {passed}/{len(results)} passed")
    for r in results:
        status = "✅" if r["passed"] else "❌"
        print(f"  {status} {r['name']} ({r['latency_ms']}ms)")
