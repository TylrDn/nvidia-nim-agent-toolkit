"""Shared async tool-calling loop for NIM sub-agents.

Replaces the deprecated ``langchain.agents.AgentExecutor`` with an explicit,
traceable loop: bind tools to a ``ChatOpenAI`` instance, invoke, dispatch any
tool calls, and repeat until the model returns a final answer or the iteration
cap is reached. Langfuse callbacks are attached to every invocation.
"""
from __future__ import annotations

import logging
from typing import Any, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from nim.client import NIMClient, get_callbacks

logger = logging.getLogger(__name__)


async def run_tool_agent(
    *,
    agent_name: str,
    query: str,
    tools: list[StructuredTool],
    system_prompt: str,
    model: str,
    max_iterations: int,
    temperature: float = 0.0,
) -> str:
    """Run a tool-calling agent loop until completion or the iteration cap.

    Args:
        agent_name: Name of the agent, used for logging.
        query: The user request to handle.
        tools: StructuredTools the agent may call.
        system_prompt: Persona and instructions.
        model: NIM model name.
        max_iterations: Maximum tool-calling rounds before forcing an answer.
        temperature: Sampling temperature.

    Returns:
        str: The agent's final natural-language answer.
    """
    llm = NIMClient(model=model, temperature=temperature).as_langchain_llm()
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {tool.name: tool for tool in tools}
    callbacks = get_callbacks()

    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query),
    ]

    for iteration in range(1, max_iterations + 1):
        response = cast(
            AIMessage, await llm_with_tools.ainvoke(messages, config={"callbacks": callbacks})
        )
        messages.append(response)

        if not response.tool_calls:
            logger.info("[%s] completed in %d iteration(s)", agent_name, iteration)
            return _as_text(response.content)

        logger.info(
            "[%s] iteration %d: %d tool call(s)",
            agent_name,
            iteration,
            len(response.tool_calls),
        )
        for tool_call in response.tool_calls:
            output = await _invoke_tool(tool_map, cast("dict[str, Any]", tool_call))
            messages.append(
                ToolMessage(content=str(output), tool_call_id=tool_call["id"])
            )

    logger.warning("[%s] hit max_iterations=%d; forcing final answer", agent_name, max_iterations)
    final = cast(AIMessage, await llm_with_tools.ainvoke(messages, config={"callbacks": callbacks}))
    return _as_text(final.content)


async def _invoke_tool(tool_map: dict[str, StructuredTool], tool_call: dict[str, Any]) -> str:
    """Dispatch a single tool call, returning its output as text.

    Args:
        tool_map: Mapping of tool name to StructuredTool.
        tool_call: A LangChain tool-call dict with ``name``, ``args``, ``id``.

    Returns:
        str: The tool output, or an error string if the tool is unknown/raises.
    """
    name = tool_call["name"]
    tool = tool_map.get(name)
    if tool is None:
        logger.warning("Unknown tool requested: %s", name)
        return f"Error: unknown tool '{name}'"
    try:
        return str(await tool.ainvoke(tool_call["args"]))
    except Exception as exc:  # noqa: BLE001 - surface tool errors back to the model
        logger.exception("Tool '%s' raised", name)
        return f"Error executing tool '{name}': {exc}"


def _as_text(content: Any) -> str:
    """Coerce LangChain message content (str or content blocks) to plain text.

    Args:
        content: Message content, which may be a string or a list of blocks.

    Returns:
        str: A plain-text representation.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        ]
        return "".join(parts)
    return str(content)
