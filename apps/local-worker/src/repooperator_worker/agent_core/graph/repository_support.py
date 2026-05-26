"""Repository guards for LangGraph agent runs."""

from __future__ import annotations

from repooperator_worker.schemas import AgentRunRequest
from repooperator_worker.services.active_repository import get_active_repository


def validate_active_repository(request: AgentRunRequest) -> None:
    try:
        active = get_active_repository()
    except Exception:
        active = None
    if active is None:
        return
    requested = str(request.project_path)
    active_path = str(active.project_path)
    if requested != active_path:
        raise ValueError(
            "The active repository changed before this run started. "
            "Open the repository again or start a new thread for the stale request."
        )
    if request.branch and active.branch and request.branch != active.branch:
        raise ValueError("The active branch changed before this run started.")
