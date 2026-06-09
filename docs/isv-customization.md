# ISV Customization Guide

This toolkit is designed so that partners can add new capabilities with small,
well-contained changes. This guide walks through adding a new specialist agent
end to end: a tool, the YAML config, the executor wiring, and a test.

## The customization surface

| Layer | File(s) | What you change |
|---|---|---|
| Tools | `tools/*.py` | The Python functions an agent can call |
| Agent persona | `configs/agents.yaml` | Model, prompt, tool list, iteration cap |
| Agent entry point | `agents/*.py` | A thin async `run()` wrapper |
| Dispatch | `orchestrator/nodes/executor.py` | Route a task's `agent` key to the runner |
| Tests | `tests/*.py` | Verify behavior offline (mock NIM) |

Models and prompts are never hardcoded in Python — they are read from
`configs/agents.yaml` and validated at startup by `configs/loader.py`.

## 1. Create a StructuredTool

Add your tool function and expose it through a `get_*_tools()` helper. Use a
Pydantic input model and `langchain_core.tools.StructuredTool` (never the
deprecated `langchain.tools` import).

```python
# tools/weather_tools.py
from __future__ import annotations

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class WeatherInput(BaseModel):
    city: str = Field(description="City name to look up")


def _get_weather(city: str) -> str:
    try:
        response = httpx.get(f"https://wttr.in/{city}?format=3", timeout=15)
        response.raise_for_status()
        return response.text
    except httpx.HTTPError as exc:
        return f"Weather lookup error: {exc}"


def get_weather_tools() -> list[StructuredTool]:
    """Return the weather StructuredTools."""
    return [
        StructuredTool.from_function(
            func=_get_weather,
            name="get_weather",
            description="Look up the current weather for a city.",
            args_schema=WeatherInput,
        ),
    ]
```

## 2. Register the agent persona in `configs/agents.yaml`

Add an entry under `agents:`. Required fields are `model` and `system_prompt`;
`tools`, `max_iterations`, `temperature`, and `max_tokens` are optional.

```yaml
agents:
  weather_agent:
    model: meta/llama-3.1-8b-instruct
    temperature: 0.0
    max_iterations: 4
    tools: [get_weather]
    description: Current-conditions weather specialist
    system_prompt: |
      You are a weather specialist. Use the available tools to look up current
      conditions and answer concisely.
```

## 3. Add the agent entry point

Create a thin async wrapper that loads its config and calls the shared
tool-calling loop in `agents/base.py`.

```python
# agents/weather_agent.py
from __future__ import annotations

from typing import Any

from agents.base import run_tool_agent
from configs.loader import get_agent_config
from tools.weather_tools import get_weather_tools

AGENT_NAME = "weather_agent"


async def run(query: str, state: Any = None) -> str:
    config = get_agent_config(AGENT_NAME)
    return await run_tool_agent(
        agent_name=AGENT_NAME,
        query=query,
        tools=get_weather_tools(),
        system_prompt=config.system_prompt,
        model=config.model,
        max_iterations=config.max_iterations,
        temperature=config.temperature,
    )
```

## 4. Wire it into the executor

Add your runner to the dispatch map in
`orchestrator/nodes/executor.py` so the planner can route tasks to it by the
`agent` key it emits:

```python
from agents.weather_agent import run as weather_run

dispatch = {
    "api_agent": api_run,
    "sql_agent": sql_run,
    "doc_agent": doc_run,
    "weather_agent": weather_run,
}
```

If you want the planner to choose this agent, mention `weather_agent` in the
planner `system_prompt`'s list of available agents in `configs/agents.yaml`.

## 5. Write a test

Keep tests offline by mocking the tool-calling loop or the LLM. See
`tests/test_agents.py` for the pattern.

```python
import pytest


@pytest.mark.asyncio
async def test_weather_agent_passes_config(monkeypatch):
    captured = {}

    async def fake_run_tool_agent(**kwargs):
        captured.update(kwargs)
        return "sunny"

    monkeypatch.setattr("agents.weather_agent.run_tool_agent", fake_run_tool_agent)
    from agents.weather_agent import run

    assert await run("weather in Paris") == "sunny"
    assert captured["agent_name"] == "weather_agent"
```

## Verify

Run the same gate CI uses before opening a PR:

```bash
ruff check .
mypy nim/ orchestrator/ agents/ tools/ api/ configs/ core/ --ignore-missing-imports
pytest tests/ -m "not integration" --cov --cov-fail-under=80
```
