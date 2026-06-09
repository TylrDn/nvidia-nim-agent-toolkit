"""Sub-agents: async tool-calling entry points dispatched by the executor."""
from agents.api_agent import run as run_api_agent
from agents.doc_agent import run as run_doc_agent
from agents.sql_agent import run as run_sql_agent

__all__ = ["run_api_agent", "run_sql_agent", "run_doc_agent"]
