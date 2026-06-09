"""Tests for the LangGraph pipeline assembly and async execution."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.graph import build_graph, run_agent_async
from tests.conftest import make_ai_message


def test_graph_compiles() -> None:
    assert build_graph() is not None


def _patch_node_llm(monkeypatch: pytest.MonkeyPatch, module: str, content: str) -> None:
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=make_ai_message(content=content))
    nim_client = MagicMock()
    nim_client.as_langchain_llm.return_value = llm
    monkeypatch.setattr(f"{module}.NIMClient", MagicMock(return_value=nim_client))


@pytest.mark.asyncio
async def test_pipeline_runs_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_node_llm(
        monkeypatch,
        "orchestrator.nodes.planner",
        '[{"id": 1, "description": "look it up", "agent": "doc_agent", "depends_on": []}]',
    )
    _patch_node_llm(
        monkeypatch,
        "orchestrator.nodes.reviewer",
        '{"score": 0.95, "verdict": "complete", "final_answer": "42", "feedback": "ok"}',
    )
    # Sub-agent run is mocked so the executor returns without touching tools/network.
    monkeypatch.setattr(
        "agents.doc_agent.run", AsyncMock(return_value="doc result"), raising=True
    )

    answer = await run_agent_async("what is the answer?", thread_id="t1")
    assert answer == "42"
