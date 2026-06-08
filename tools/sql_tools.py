"""StructuredTool wrappers for SQL database queries via SQLAlchemy."""
from __future__ import annotations

import os

from langchain.tools import StructuredTool
from pydantic import BaseModel
from sqlalchemy import create_engine, text

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        db_url = os.environ.get("DB_URL", "sqlite:///./agent.db")
        _engine = create_engine(db_url)
    return _engine


class RunQueryInput(BaseModel):
    query: str


def _run_query(query: str) -> str:
    """Execute a read-only SQL query and return results as a string."""
    engine = _get_engine()
    with engine.connect() as conn:
        result = conn.execute(text(query))
        rows = result.fetchmany(50)  # limit rows
        cols = list(result.keys())
        return str({"columns": cols, "rows": [dict(zip(cols, r)) for r in rows]})


class ListTablesInput(BaseModel):
    pass


def _list_tables() -> str:
    """Return a list of available tables in the database."""
    engine = _get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
        return str([row[0] for row in result])


def get_sql_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            func=_run_query,
            name="run_sql_query",
            description="Execute a SQL SELECT query against the database. Return results as JSON.",
            args_schema=RunQueryInput,
        ),
        StructuredTool.from_function(
            func=_list_tables,
            name="list_tables",
            description="List all tables available in the database.",
            args_schema=ListTablesInput,
        ),
    ]
