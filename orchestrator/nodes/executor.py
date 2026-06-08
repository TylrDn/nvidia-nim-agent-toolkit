"""Executor node — dispatches tool calls for the current task.

Reads the current task from state.plan[state.current_task],
selects the appropriate agent/tool, runs it, and appends the
result to state.tool_results.
"""
from __future__ import annotations

import logging
from orchestrator.state import AgentState
from nim.client import get_default_llm
from tools.api_tools import api_tool
from tools.sql_tools import sql_tool
from tools.doc_tools import doc_tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

AVAILABLE_TOOLS = [api_tool, sql_tool, doc_tool]

SYSTEM_PROMPT = """You are an execution agent. Complete the assigned task using
the available tools. Be precise and return structured results."""


def executor_node(state: AgentState) -> dict:
    """LangGraph node: execute the current task using tool-calling agent.

    Args:
        state: Current AgentState with plan and current_task populated.

    Returns:
        Partial state update with appended tool_results and incremented current_task.
    """
    plan = state["plan"]
    idx = state["current_task"]

    if idx >= len(plan):
        logger.info("Executor: all tasks complete")
        return {"final_output": _assemble_output(state["tool_results"])}

    task = plan[idx]
    logger.info("Executor: running task %d/%d: %s", idx + 1, len(plan), task[:80])

    llm = get_default_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, AVAILABLE_TOOLS, prompt)
    executor = AgentExecutor(agent=agent, tools=AVAILABLE_TOOLS, verbose=True)

    try:
        result = executor.invoke({"input": task})
        tool_result = {"task": task, "output": result["output"], "status": "success"}
    except Exception as exc:
        logger.error("Executor error on task %d: %s", idx, exc)
        tool_result = {"task": task, "output": str(exc), "status": "error"}

    return {
        "tool_results": [tool_result],
        "current_task": idx + 1,
        "retry_count": state.get("retry_count", 0),
    }


def _assemble_output(results: list[dict]) -> str:
    """Combine tool results into a final readable response."""
    # TODO: improve assembly with LLM synthesis step
    return "\n\n".join(
        f"Task: {r['task']}\nResult: {r['output']}" for r in results
    )
