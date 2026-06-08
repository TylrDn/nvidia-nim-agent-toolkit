"""Text-to-SQL agent backed by SQLAlchemy."""
from __future__ import annotations

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

from nim.client import NIMClient
from tools.sql_tools import get_sql_tools

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a SQL agent. Use the provided tools to query the database. "
               "Always explain your query before running it. {agent_scratchpad}"),
    ("human", "{input}"),
])


def run_sql_agent(task_description: str) -> str:
    """Run the SQL agent on a task and return the result string."""
    client = NIMClient()
    llm = client.get_llm()
    tools = get_sql_tools()

    agent = create_tool_calling_agent(llm, tools, PROMPT)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)
    result = executor.invoke({"input": task_description})
    return str(result.get("output", ""))
