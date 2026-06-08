"""Document store retrieval agent.

Answers questions by retrieving relevant passages from a vector store
and grounding the LLM response in those passages.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from nim.client import NIMClient
from tools.doc_tools import get_doc_tools

logger = logging.getLogger(__name__)

DOC_AGENT_SYSTEM = """\
You are a document retrieval specialist. Use the retrieval tools to find
relevant information before answering. Always ground your answer in the
retrieved passages. If no relevant passage is found, say so explicitly.
"""


class DocAgent:
    """LangChain tool-calling agent backed by document retrieval tools."""

    def __init__(self, client: NIMClient | None = None) -> None:
        self.client = client or NIMClient()
        self._executor: AgentExecutor | None = None

    @property
    def executor(self) -> AgentExecutor:
        if self._executor is None:
            tools = get_doc_tools()
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", DOC_AGENT_SYSTEM),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            )
            agent = create_tool_calling_agent(self.client.llm, tools, prompt)
            self._executor = AgentExecutor(
                agent=agent,
                tools=tools,
                max_iterations=4,
                verbose=False,
            )
        return self._executor

    def run(self, task: str) -> str:
        """Execute a natural-language document retrieval task."""
        logger.info("DocAgent executing: %s", task)
        try:
            result = self.executor.invoke({"input": task})
            return str(result.get("output", ""))
        except Exception as exc:  # noqa: BLE001
            logger.error("DocAgent error: %s", exc)
            return f"Doc agent error: {exc}"
