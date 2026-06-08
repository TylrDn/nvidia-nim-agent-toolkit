"""REST API tool-calling agent."""
from __future__ import annotations

import httpx
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

from nim.client import NIMClient
from tools.api_tools import get_api_tools

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an API agent. Use the provided tools to fetch data from REST APIs. "
               "Return a concise, factual answer. {agent_scratchpad}"),
    ("human", "{input}"),
])


def run_api_agent(task_description: str) -> str:
    """Run the API agent on a task and return the result string."""
    client = NIMClient()
    llm = client.get_llm()
    tools = get_api_tools()

    agent = create_tool_calling_agent(llm, tools, PROMPT)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)
    result = executor.invoke({"input": task_description})
    return str(result.get("output", ""))
