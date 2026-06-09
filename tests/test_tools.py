"""Unit tests for the StructuredTool wrappers (offline)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from sqlalchemy import create_engine, text

import tools.api_tools as api_tools
import tools.doc_tools as doc_tools
import tools.sql_tools as sql_tools


# --- SQL tools -------------------------------------------------------------

def test_sql_query_and_describe(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE users (id INTEGER, name TEXT)"))
        conn.execute(text("INSERT INTO users VALUES (1, 'alice')"))
    monkeypatch.setattr(sql_tools, "_engine", engine)

    assert "alice" in sql_tools._run_query("SELECT * FROM users")
    assert "name" in sql_tools._describe_table("users")
    assert "users" in sql_tools._describe_table()
    assert "SQL error" in sql_tools._run_query("SELECT * FROM missing")
    assert len(sql_tools.get_sql_tools()) == 2


# --- API tools -------------------------------------------------------------

def test_http_get_success(monkeypatch: pytest.MonkeyPatch) -> None:
    response = MagicMock()
    response.json.return_value = {"ok": True}
    monkeypatch.setattr(api_tools.httpx, "get", lambda *a, **k: response)
    assert "ok" in api_tools._http_get("http://example.test")


def test_http_get_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: object, **k: object) -> None:
        raise RuntimeError("down")

    monkeypatch.setattr(api_tools.httpx, "get", boom)
    assert "HTTP GET error" in api_tools._http_get("http://example.test")


def test_http_post_success(monkeypatch: pytest.MonkeyPatch) -> None:
    response = MagicMock()
    response.json.return_value = {"created": 1}
    monkeypatch.setattr(api_tools.httpx, "post", lambda *a, **k: response)
    assert "created" in api_tools._http_post("http://example.test", {"a": 1})
    assert len(api_tools.get_api_tools()) == 2


# --- Doc tools -------------------------------------------------------------

def test_search_docs_returns_passages(monkeypatch: pytest.MonkeyPatch) -> None:
    store = MagicMock()
    store.similarity_search.return_value = [
        Document(page_content="hello world", metadata={"source": "doc1"})
    ]
    monkeypatch.setattr(doc_tools, "_get_vectorstore", lambda: store)
    out = doc_tools._search_docs("greeting")
    assert "hello world" in out
    assert "doc1" in out


def test_search_docs_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    store = MagicMock()
    store.similarity_search.return_value = []
    monkeypatch.setattr(doc_tools, "_get_vectorstore", lambda: store)
    assert "No relevant documents" in doc_tools._search_docs("q")
    assert len(doc_tools.get_doc_tools()) == 1
