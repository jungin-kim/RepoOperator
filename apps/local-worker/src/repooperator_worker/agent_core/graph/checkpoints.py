"""Checkpoint wiring for the RepoOperator LangGraph runtime."""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver

from repooperator_worker.agent_core.graph_checkpoints import EventServiceLangGraphSaver


_DEFAULT_LANGGRAPH_CHECKPOINTER = EventServiceLangGraphSaver()


def get_default_langgraph_checkpointer() -> InMemorySaver:
    return _DEFAULT_LANGGRAPH_CHECKPOINTER


__all__ = ["EventServiceLangGraphSaver", "get_default_langgraph_checkpointer"]
