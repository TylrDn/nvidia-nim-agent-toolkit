"""Text-to-SQL agent with SQLAlchemy backend.

Translates natural-language questions into SQL, executes them against
a configured database, and returns formatted results.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from nim.client import NIMClient
from tools.sql_tools import get_sql_tools

logger = logging.getLogger(__name__)

SQL_AGENT_SYSTEM = """\
You are a SQL expert. Use the provided tools to answer questions about data.
1. First use the schema tool to understand available tables.
2. Then construct and execute a precise SQL query.
3. Return results in a clear, readable format.
Never modify data — only SELECT queries are permitted.
"""


class SqlAgent:
    """LangChain tool-calling agent backed by SQL query tools."""

    def __init__(self, client: NIMClient | None = None) -> None:
        self.client = client or NIMClient()
        self._executor: AgentExecutor | None = None

    @property
    def executor(self) -> AgentExecutor:
        if self._executor is None:
            tools = get_sql_tools()
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", SQL_AGENT_SYSTEM),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            )
            agent = create_tool_calling_agent(self.client.llm, tools, prompt)
            self._executor = AgentExecutor(
                agent=agent,
                tools=tools,
                max_iterations=6,
                verbose=False,
            )
        return self._executor

    def run(self, task: str) -> str:
        """Execute a natural-language SQL task and return the result."""
        logger.info("SqlAgent executing: %s", task)
        try:
            result = self.executor.invoke({"input": task})
            return str(result.get("output", ""))
        except Exception as exc:  # noqa: BLE001
            logger.error("SqlAgent error: %s", exc)
            return f"SQL agent error: {exc}"
