"""Document store retrieval agent backed by ChromaDB."""
from __future__ import annotations

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

from nim.client import NIMClient
from tools.doc_tools import get_doc_tools

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a document retrieval agent. Use the provided tools to search "
               "the document store and synthesize an answer. {agent_scratchpad}"),
    ("human", "{input}"),
])


def run_doc_agent(task_description: str) -> str:
    """Run the document agent on a task and return the result string."""
    client = NIMClient()
    llm = client.get_llm()
    tools = get_doc_tools()

    agent = create_tool_calling_agent(llm, tools, PROMPT)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)
    result = executor.invoke({"input": task_description})
    return str(result.get("output", ""))
