"""Text-to-SQL agent backed by SQLAlchemy."""
from __future__ import annotations

import os
from typing import Any

from langchain.tools import StructuredTool
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

from nim.client import NIMClient

_DB_URL = os.getenv("DATABASE_URL", "sqlite:///./demo.db")
_engine = create_engine(_DB_URL)


class SQLQueryInput(BaseModel):
    query: str = Field(..., description="A read-only SQL SELECT statement")


def _run_sql(query: str) -> dict[str, Any]:
    if not query.strip().upper().startswith("SELECT"):
        return {"error": "Only SELECT queries are permitted."}
    try:
        with _engine.connect() as conn:
            rows = conn.execute(text(query)).fetchall()
            return {"rows": [dict(r._mapping) for r in rows]}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


_sql_tool = StructuredTool.from_function(
    func=_run_sql,
    name="sql_query",
    description="Execute a read-only SQL SELECT query and return the results.",
    args_schema=SQLQueryInput,
)

_llm = NIMClient().get_llm().bind_tools([_sql_tool])

_SYSTEM = (
    "You are a SQL agent. Translate the user's natural-language request into a "
    "valid SQL SELECT query, execute it with the sql_query tool, and return the "
    "results as structured JSON."
)


def run(task_description: str) -> dict[str, Any]:
    """Execute the SQL agent for a single task."""
    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=task_description),
    ]
    response = _llm.invoke(messages)
    return {"output": response.content, "tool": "sql"}
