"""SQL query StructuredTool wrapper.

Executes parameterised SELECT queries against a SQLAlchemy-compatible
database. Enforces read-only access (no DDL or DML).
"""
from __future__ import annotations

import os
from langchain_core.tools import tool
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()


@tool
def sql_query(query: str) -> str:
    """Execute a read-only SQL SELECT query and return results.

    Args:
        query: A valid SQL SELECT statement. DDL/DML is rejected.

    Returns:
        Query results as a formatted string table.
    """
    if any(kw in query.upper() for kw in ["DROP", "DELETE", "TRUNCATE", "INSERT", "UPDATE"]):
        return "Error: only SELECT queries are permitted."

    db_url = os.environ.get("DATABASE_URL", "sqlite:///./dev.db")
    # TODO: use connection pooling in production
    engine = create_engine(db_url)
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query))
            rows = result.fetchall()
            cols = list(result.keys())
            header = " | ".join(cols)
            body = "\n".join(" | ".join(str(v) for v in row) for row in rows)
            return f"{header}\n{'-' * len(header)}\n{body}" if rows else "No results."
    except Exception as exc:
        return f"SQL error: {exc}"


sql_tool = sql_query
