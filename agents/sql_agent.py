"""Text-to-SQL agent backed by NIM + SQLAlchemy."""
from __future__ import annotations

from typing import Any

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from nim.client import get_client
from tools.sql_tools import get_sql_tools


SYSTEM = """\
You are a SQL specialist agent. Translate the user's natural language request
into SQL queries, execute them using the available tools, and return the results
as structured data or a clear summary.
Never guess schema — always introspect the database first.
"""


class SQLAgent:
    """LangChain OpenAI-tools agent for text-to-SQL via NIM."""

    def __init__(self) -> None:
        self.llm = get_client().as_langchain_llm()
        self.tools = get_sql_tools()
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])
        agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        self.executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True, max_iterations=5)

    def run(self, query: str, **kwargs: Any) -> str:
        result = self.executor.invoke({"input": query})
        return result.get("output", "")


_instance: SQLAgent | None = None


def run(query: str, state: Any = None) -> str:
    global _instance
    if _instance is None:
        _instance = SQLAgent()
    return _instance.run(query)
