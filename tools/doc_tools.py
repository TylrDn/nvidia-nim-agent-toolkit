"""Document store retrieval StructuredTool.

Performs keyword or semantic search over an in-memory FAISS store.
In production, swap the store for pgvector or Milvus.
"""
from __future__ import annotations

from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from nim.client import NIMClient

# Module-level store — populated via load_documents() before agent runs
_store: FAISS | None = None


def load_documents(texts: list[str]) -> None:
    """Load raw text chunks into the in-memory FAISS store.

    Args:
        texts: List of plain-text document chunks.
    """
    global _store
    # TODO: replace with NIM embedding endpoint
    embeddings = OpenAIEmbeddings(
        base_url=NIMClient().base_url,
        api_key=NIMClient().api_key,  # type: ignore[arg-type]
        model="nv-embedqa-e5-v5",
    )
    _store = FAISS.from_texts(texts, embeddings)


@tool
def doc_search(query: str, k: int = 4) -> str:
    """Search documents for the most relevant chunks.

    Args:
        query: Natural language search query.
        k: Number of top chunks to return.

    Returns:
        Top-k document chunks concatenated as a string.
    """
    if _store is None:
        return "Document store not initialised. Call load_documents() first."
    docs = _store.similarity_search(query, k=k)
    return "\n---\n".join(d.page_content for d in docs)


doc_tool = doc_search
