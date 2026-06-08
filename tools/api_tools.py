"""REST API StructuredTool wrapper.

Exposes HTTP GET/POST as a LangChain StructuredTool so the executor
agent can call external APIs via tool-calling.
"""
from __future__ import annotations

import httpx
from langchain_core.tools import tool


@tool
def api_request(url: str, method: str = "GET", payload: dict | None = None) -> str:
    """Make an HTTP request to a REST API endpoint.

    Args:
        url: Full URL to request.
        method: HTTP method — GET or POST.
        payload: Optional JSON body for POST requests.

    Returns:
        Response body as a string (JSON or text).
    """
    # TODO: add auth header injection from config
    try:
        if method.upper() == "POST":
            resp = httpx.post(url, json=payload, timeout=30)
        else:
            resp = httpx.get(url, timeout=30)
        resp.raise_for_status()
        return resp.text
    except httpx.HTTPError as exc:
        return f"HTTP error: {exc}"


api_tool = api_request
