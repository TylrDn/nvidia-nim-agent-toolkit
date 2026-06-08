"""Standalone SQL tools importable outside the agent context."""
from __future__ import annotations

import os
from typing import Any

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

_engine = create_engine(os.getenv("DATABASE_URL", "sqlite:///./demo.db"))


class SQLInput(BaseModel):
    query: str = Field(..., description="Read-only SELECT statement")


def sql_query(query: str) -> dict[str, Any]:
    if not query.strip().upper().startswith("SELECT"):
        return {"error": "Only SELECT queries allowed."}
    try:
        with _engine.connect() as conn:
            rows = conn.execute(text(query)).fetchall()
            return {"rows": [dict(r._mapping) for r in rows]}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


SQL_QUERY_TOOL = StructuredTool.from_function(
    func=sql_query,
    name="sql_query",
    description="Execute a read-only SQL SELECT query.",
    args_schema=SQLInput,
)
