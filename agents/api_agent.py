"""REST API tool-calling agent backed by NIM.

Configuration (model, prompt, tools, iteration cap) is loaded from
``configs/agents.yaml`` — no values are hardcoded here.
"""
from __future__ import annotations

from typing import Any

from agents.base import run_tool_agent
from configs.loader import get_agent_config
from tools.api_tools import get_api_tools

AGENT_NAME = "api_agent"


async def run(query: str, state: Any = None) -> str:
    """Run the API agent for a single task description.

    Args:
        query: The task description to handle.
        state: Unused; present for a uniform sub-agent signature.

    Returns:
        str: The agent's final answer.
    """
    config = get_agent_config(AGENT_NAME)
    return await run_tool_agent(
        agent_name=AGENT_NAME,
        query=query,
        tools=get_api_tools(),
        system_prompt=config.system_prompt,
        model=config.model,
        max_iterations=config.max_iterations,
        temperature=config.temperature,
    )
