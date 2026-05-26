"""Cancellation checkpoints for LangGraph agent runs."""

from __future__ import annotations

from repooperator_worker.agent_core.events import append_activity_event
from repooperator_worker.agent_core.state import AgentCoreState
from repooperator_worker.schemas import AgentRunRequest
from repooperator_worker.services.event_service import get_run


def check_cancel(state: AgentCoreState, request: AgentRunRequest) -> None:
    try:
        run = get_run(state.run_id) or {}
    except OSError:
        run = {}
    if run.get("status") not in {"cancelled", "cancelling"}:
        return
    state.cancellation_requested = True
    state.stop_reason = "cancelled"
    append_activity_event(
        run_id=state.run_id,
        request=request,
        activity_id="langgraph-cancelled",
        event_type="activity_completed",
        phase="Finished",
        label="Run cancelled",
        status="cancelled",
        observation="Cancellation was requested. RepoOperator stopped at the next safe checkpoint.",
    )
