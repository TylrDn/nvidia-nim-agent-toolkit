"""Text-to-SQL agent (SQLAlchemy backend).

Converts natural language queries to SQL, executes them against
a configured database, and returns structured results.
"""
from __future__ import annotations

import logging
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from nim.client import get_default_llm
from tools.sql_tools import sql_tool

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a SQL agent. Convert the user's question into a
SQL query, execute it, and return the results in a clear format.
Never execute DROP, DELETE, or TRUNCATE statements."""


def run_sql_agent(question: str) -> str:
    """Convert a natural language question to SQL and execute it.

    Args:
        question: Natural language question about the database.

    Returns:
        Query results as a formatted string.
    """
    # TODO: add schema injection so the LLM knows table structures
    llm = get_default_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    agent = create_tool_calling_agent(llm, [sql_tool], prompt)
    executor = AgentExecutor(agent=agent, tools=[sql_tool], verbose=True)
    result = executor.invoke({"input": question})
    return result["output"]
