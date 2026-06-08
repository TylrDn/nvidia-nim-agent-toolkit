"""REST API tool-calling agent backed by NIM."""
from __future__ import annotations

from typing import Any

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage

from nim.client import get_client
from tools.api_tools import get_api_tools


SYSTEM = """\
You are an API specialist agent. Use the available tools to make REST API calls
and retrieve data needed to answer the user's request.
Always prefer structured tool calls over speculation.
Return a concise, factual answer based only on tool outputs.
"""


class APIAgent:
    """LangChain OpenAI-tools agent wired to NIM for REST API interactions."""

    def __init__(self) -> None:
        self.llm = get_client().as_langchain_llm()
        self.tools = get_api_tools()
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


_instance: APIAgent | None = None


def run(query: str, state: Any = None) -> str:
    """Module-level entry point used by the executor node."""
    global _instance
    if _instance is None:
        _instance = APIAgent()
    return _instance.run(query)
