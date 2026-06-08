"""SQL StructuredTool wrappers for the SQL agent."""
from __future__ import annotations

import os
from typing import Optional

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, inspect, text

DB_URL = os.getenv("DATABASE_URL", "sqlite:///./demo.db")
_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(DB_URL)
    return _engine


class QueryInput(BaseModel):
    sql: str = Field(description="SQL SELECT statement to execute")


class TableInput(BaseModel):
    table_name: Optional[str] = Field(default=None, description="Table name to describe; omit to list all tables")


def _run_query(sql: str) -> str:
    try:
        engine = _get_engine()
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            rows = result.fetchmany(100)
            cols = list(result.keys())
            lines = ["\t".join(cols)]
            lines += ["\t".join(str(v) for v in row) for row in rows]
            return "\n".join(lines)
    except Exception as e:
        return f"SQL error: {e}"


def _describe_table(table_name: str | None = None) -> str:
    try:
        engine = _get_engine()
        insp = inspect(engine)
        if table_name:
            cols = insp.get_columns(table_name)
            return "\n".join(f"{c['name']} {c['type']}" for c in cols)
        else:
            return "\n".join(insp.get_table_names())
    except Exception as e:
        return f"Schema error: {e}"


def get_sql_tools() -> list:
    return [
        StructuredTool.from_function(
            func=_run_query,
            name="sql_query",
            description="Execute a SQL SELECT query. Returns up to 100 rows as tab-separated text.",
            args_schema=QueryInput,
        ),
        StructuredTool.from_function(
            func=_describe_table,
            name="sql_describe",
            description="List all tables or describe columns of a specific table.",
            args_schema=TableInput,
        ),
    ]
