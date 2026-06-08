"""REST API tool-calling agent.

Accepts a natural-language task description and executes it by
making structured HTTP calls via the api_tools StructuredTool wrappers.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from nim.client import NIMClient
from tools.api_tools import get_api_tools

logger = logging.getLogger(__name__)

API_AGENT_SYSTEM = """\
You are a REST API specialist. Use the provided tools to fulfil the task.
Always prefer structured tool calls over guessing. Be concise.
"""


class ApiAgent:
    """LangChain tool-calling agent backed by REST API tools."""

    def __init__(self, client: NIMClient | None = None) -> None:
        self.client = client or NIMClient()
        self._executor: AgentExecutor | None = None

    @property
    def executor(self) -> AgentExecutor:
        if self._executor is None:
            tools = get_api_tools()
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", API_AGENT_SYSTEM),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            )
            agent = create_tool_calling_agent(self.client.llm, tools, prompt)
            self._executor = AgentExecutor(
                agent=agent,
                tools=tools,
                max_iterations=5,
                verbose=False,
            )
        return self._executor

    def run(self, task: str) -> str:
        """Execute a natural-language API task and return the result."""
        logger.info("ApiAgent executing: %s", task)
        try:
            result = self.executor.invoke({"input": task})
            return str(result.get("output", ""))
        except Exception as exc:  # noqa: BLE001
            logger.error("ApiAgent error: %s", exc)
            return f"API agent error: {exc}"
