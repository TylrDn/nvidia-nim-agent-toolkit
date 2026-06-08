"""Document store retrieval agent.

Performs semantic search over an in-memory or persisted document store
and synthesises an answer from retrieved chunks.
"""
from __future__ import annotations

import logging
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from nim.client import get_default_llm
from tools.doc_tools import doc_tool

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a document retrieval agent. Use the doc_search tool
to find relevant information and synthesise a precise answer.
Cite the source chunk when possible."""


def run_doc_agent(query: str) -> str:
    """Search documents and synthesise an answer.

    Args:
        query: Natural language question to answer from documents.

    Returns:
        Synthesised answer with source references.
    """
    # TODO: wire to persistent vector store (pgvector or Milvus)
    llm = get_default_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    agent = create_tool_calling_agent(llm, [doc_tool], prompt)
    executor = AgentExecutor(agent=agent, tools=[doc_tool], verbose=True)
    result = executor.invoke({"input": query})
    return result["output"]
