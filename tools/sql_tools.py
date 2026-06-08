"""SQL tool wrappers as LangChain StructuredTools.

Provides schema inspection and SELECT-only query execution via
SQLAlchemy. Configure the target database via DATABASE_URL env var.
"""
from __future__ import annotations

import os
import logging
from typing import Optional

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./demo.db")


class SchemaInput(BaseModel):
    table_name: Optional[str] = Field(
        default=None,
        description="Table name to inspect. Leave blank to list all tables.",
    )


class QueryInput(BaseModel):
    query: str = Field(description="Valid SELECT SQL query to execute")
    max_rows: int = Field(default=50, description="Maximum rows to return")


def get_schema(table_name: Optional[str] = None) -> str:
    """Return schema info for a table or list all tables."""
    try:
        from sqlalchemy import create_engine, inspect, text

        engine = create_engine(DATABASE_URL)
        inspector = inspect(engine)

        if table_name:
            columns = inspector.get_columns(table_name)
            pk = inspector.get_pk_constraint(table_name)
            fks = inspector.get_foreign_keys(table_name)
            lines = [f"Table: {table_name}"]
            lines += [f"  {c['name']} {c['type']} {'(PK)' if c['name'] in pk.get('constrained_columns', []) else ''}" for c in columns]
            if fks:
                lines += [f"  FK: {fk['constrained_columns']} -> {fk['referred_table']}.{fk['referred_columns']}" for fk in fks]
            return "\n".join(lines)
        else:
            tables = inspector.get_table_names()
            return "Available tables: " + ", ".join(tables) if tables else "No tables found."
    except Exception as exc:  # noqa: BLE001
        return f"Schema error: {exc}"


def execute_query(query: str, max_rows: int = 50) -> str:
    """Execute a SELECT query and return formatted results."""
    q = query.strip().upper()
    if not q.startswith("SELECT"):
        return "Error: only SELECT queries are permitted."
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            result = conn.execute(text(query))
            rows = result.fetchmany(max_rows)
            if not rows:
                return "Query returned no results."
            cols = list(result.keys())
            header = " | ".join(cols)
            sep = "-" * len(header)
            body = "\n".join(" | ".join(str(v) for v in row) for row in rows)
            return f"{header}\n{sep}\n{body}"
    except Exception as exc:  # noqa: BLE001
        return f"Query error: {exc}"


def get_sql_tools() -> list[StructuredTool]:
    """Return the list of SQL StructuredTools for agent use."""
    return [
        StructuredTool.from_function(
            func=get_schema,
            name="get_db_schema",
            description="Inspect database schema. Pass a table name or leave blank to list all tables.",
            args_schema=SchemaInput,
        ),
        StructuredTool.from_function(
            func=execute_query,
            name="execute_sql",
            description="Execute a SELECT SQL query and return tabular results. No data modification allowed.",
            args_schema=QueryInput,
        ),
    ]
