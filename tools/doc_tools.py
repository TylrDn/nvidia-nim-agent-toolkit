"""StructuredTool wrappers for document store retrieval via ChromaDB."""
from __future__ import annotations

import os

import chromadb
from langchain.tools import StructuredTool
from pydantic import BaseModel

_chroma_client = None


def _get_chroma():
    global _chroma_client
    if _chroma_client is None:
        host = os.environ.get("CHROMA_HOST", "localhost")
        port = int(os.environ.get("CHROMA_PORT", "8000"))
        _chroma_client = chromadb.HttpClient(host=host, port=port)
    return _chroma_client


class SearchDocsInput(BaseModel):
    query: str
    collection: str = "default"
    n_results: int = 5


def _search_docs(query: str, collection: str = "default", n_results: int = 5) -> str:
    """Semantic search over a ChromaDB collection. Returns top-k document chunks."""
    client = _get_chroma()
    col = client.get_or_create_collection(collection)
    results = col.query(query_texts=[query], n_results=n_results)
    docs = results.get("documents", [[]])[0]
    return "\n\n---\n\n".join(docs)


class ListCollectionsInput(BaseModel):
    pass


def _list_collections() -> str:
    """List available ChromaDB collections."""
    client = _get_chroma()
    return str([c.name for c in client.list_collections()])


def get_doc_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            func=_search_docs,
            name="search_documents",
            description="Semantic search over the document store. Returns relevant text chunks.",
            args_schema=SearchDocsInput,
        ),
        StructuredTool.from_function(
            func=_list_collections,
            name="list_collections",
            description="List all document collections available in the vector store.",
            args_schema=ListCollectionsInput,
        ),
    ]
