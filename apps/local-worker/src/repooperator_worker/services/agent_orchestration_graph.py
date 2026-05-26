"""Compatibility adapter for the named orchestration graph entry point."""

from __future__ import annotations

from typing import Any, Iterator

from repooperator_worker.agent_core.langgraph_runtime import (
    run_langgraph_controller,
    stream_langgraph_controller,
)
from repooperator_worker.agent_core.repository_review import (
    MAX_REPOSITORY_REVIEW_BYTES,
    REPOSITORY_REVIEW_BINARY_SUFFIXES,
    REPOSITORY_REVIEW_SUFFIXES,
)
from repooperator_worker.schemas import AgentRunRequest, AgentRunResponse


def run_agent_orchestration_graph(request: AgentRunRequest) -> AgentRunResponse:
    """Deprecated adapter for callers that still import the former graph name."""
    return run_langgraph_controller(request)


def stream_agent_orchestration_graph(request: AgentRunRequest, *, run_id: str | None = None) -> Iterator[dict[str, Any]]:
    """Deprecated adapter for callers that still import the former stream name."""
    yield from stream_langgraph_controller(request, run_id=run_id)
