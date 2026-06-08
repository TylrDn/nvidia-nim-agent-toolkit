"""REST API tool-calling agent.

A LangChain tool-calling agent that queries external REST APIs on behalf
of the orchestrator. Wraps raw HTTP calls as structured LangChain tools.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from nim.client import get_default_llm
from tools.api_tools import api_tool

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an API agent. Use the api_request tool to fetch
data from REST APIs. Always validate the response before returning."""


def run_api_agent(task: str, context: dict[str, Any] | None = None) -> str:
    """Run the API agent for a given task.

    Args:
        task: Natural language description of the API call to make.
        context: Optional dict of additional context (base_url, headers, etc.).

    Returns:
        String result from the API call.
    """
    # TODO: inject context into tool config dynamically
    llm = get_default_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    agent = create_tool_calling_agent(llm, [api_tool], prompt)
    executor = AgentExecutor(agent=agent, tools=[api_tool], verbose=True)
    result = executor.invoke({"input": task})
    return result["output"]
