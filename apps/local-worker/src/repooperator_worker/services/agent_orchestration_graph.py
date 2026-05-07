"""Compatibility adapter for the former orchestration graph entry point.

Active execution is handled by agent_core.controller_graph. This module keeps
the old run/stream import path only.
"""

from __future__ import annotations

from typing import Any, Iterator

from repooperator_worker.agent_core.controller_graph import (
    run_controller_graph,
    stream_controller_graph,
)
from repooperator_worker.agent_core.repository_review import (
    MAX_REPOSITORY_REVIEW_BYTES,
    REPOSITORY_REVIEW_BINARY_SUFFIXES,
    REPOSITORY_REVIEW_SUFFIXES,
)
from repooperator_worker.schemas import AgentRunRequest, AgentRunResponse


def run_agent_orchestration_graph(request: AgentRunRequest) -> AgentRunResponse:
    """Deprecated adapter for callers that still import the former graph name."""
    return run_controller_graph(request)


def stream_agent_orchestration_graph(request: AgentRunRequest, *, run_id: str | None = None) -> Iterator[dict[str, Any]]:
    """Deprecated adapter for callers that still import the former stream name."""
    yield from stream_controller_graph(request, run_id=run_id)
