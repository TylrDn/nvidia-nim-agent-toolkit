"""Document store retrieval agent backed by NIM."""
from __future__ import annotations

from typing import Any

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from nim.client import get_client
from tools.doc_tools import get_doc_tools


SYSTEM = """\
You are a document retrieval agent. Search the document store to find relevant
information and synthesize a concise, grounded answer.
Only answer from retrieved documents. Say "not found" if no relevant docs exist.
"""


class DocAgent:
    """LangChain OpenAI-tools agent for document retrieval via NIM."""

    def __init__(self) -> None:
        self.llm = get_client().as_langchain_llm()
        self.tools = get_doc_tools()
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


_instance: DocAgent | None = None


def run(query: str, state: Any = None) -> str:
    global _instance
    if _instance is None:
        _instance = DocAgent()
    return _instance.run(query)
