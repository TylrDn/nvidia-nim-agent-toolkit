"""Unit tests for the shared tool-calling agent loop."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.tools import StructuredTool

from agents.base import run_tool_agent
from tests.conftest import make_ai_message


def _echo_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=lambda text: f"echo:{text}",
        name="echo",
        description="Echo the input text.",
    )


def _patch_base_llm(monkeypatch: pytest.MonkeyPatch, responses: list) -> MagicMock:
    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=responses)
    llm.bind_tools.return_value = llm
    nim_client = MagicMock()
    nim_client.as_langchain_llm.return_value = llm
    monkeypatch.setattr("agents.base.NIMClient", MagicMock(return_value=nim_client))
    return llm


@pytest.mark.asyncio
async def test_agent_returns_direct_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_base_llm(monkeypatch, [make_ai_message(content="final answer")])
    result = await run_tool_agent(
        agent_name="api_agent",
        query="hi",
        tools=[_echo_tool()],
        system_prompt="be helpful",
        model="m",
        max_iterations=5,
    )
    assert result == "final answer"


@pytest.mark.asyncio
async def test_agent_executes_tool_then_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    tool_call = {"name": "echo", "args": {"text": "hello"}, "id": "call_1", "type": "tool_call"}
    llm = _patch_base_llm(
        monkeypatch,
        [
            make_ai_message(content="", tool_calls=[tool_call]),
            make_ai_message(content="done after tool"),
        ],
    )
    result = await run_tool_agent(
        agent_name="api_agent",
        query="echo hello",
        tools=[_echo_tool()],
        system_prompt="be helpful",
        model="m",
        max_iterations=5,
    )
    assert result == "done after tool"
    assert llm.ainvoke.await_count == 2


@pytest.mark.parametrize(
    "module, expected_name",
    [
        ("agents.api_agent", "api_agent"),
        ("agents.sql_agent", "sql_agent"),
        ("agents.doc_agent", "doc_agent"),
    ],
)
@pytest.mark.asyncio
async def test_agent_wrappers_pass_config(
    monkeypatch: pytest.MonkeyPatch, module: str, expected_name: str
) -> None:
    captured: dict = {}

    async def fake_run_tool_agent(**kwargs) -> str:
        captured.update(kwargs)
        return "wrapped"

    monkeypatch.setattr(f"{module}.run_tool_agent", fake_run_tool_agent)
    run = __import__(module, fromlist=["run"]).run

    result = await run("a task")
    assert result == "wrapped"
    assert captured["agent_name"] == expected_name
    assert captured["model"]  # loaded from agents.yaml
    assert captured["system_prompt"]


@pytest.mark.asyncio
async def test_unknown_tool_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    bad_call = {"name": "missing", "args": {}, "id": "c1", "type": "tool_call"}
    _patch_base_llm(
        monkeypatch,
        [
            make_ai_message(content="", tool_calls=[bad_call]),
            make_ai_message(content="recovered"),
        ],
    )
    result = await run_tool_agent(
        agent_name="api_agent",
        query="x",
        tools=[_echo_tool()],
        system_prompt="p",
        model="m",
        max_iterations=5,
    )
    assert result == "recovered"
