"""Authoritative agent entry point for /agent/run."""

from __future__ import annotations

import logging
from pathlib import Path

from repooperator_worker.config import get_settings
from repooperator_worker.schemas import AgentRunRequest, AgentRunResponse
from repooperator_worker.services.model_client import OpenAICompatibleModelClient

logger = logging.getLogger(__name__)


def run_agent_task(request: AgentRunRequest) -> AgentRunResponse:
    """Run the authoritative LangGraph agent path.

    User-facing validation errors are allowed to propagate. Any unexpected
    runtime failure becomes a structured ``agent_error`` so the caller can show
    a retryable error without falling back to another runtime.
    """
    try:
        from repooperator_worker.agent_core.langgraph_runtime import run_langgraph_controller

        return run_langgraph_controller(request)
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("LangGraph agent orchestration failed: %s", exc)
        return _agent_error_response(request, exc)


def _agent_error_response(request: AgentRunRequest, exc: Exception) -> AgentRunResponse:
    settings = get_settings()
    try:
        model_name = OpenAICompatibleModelClient().model_name
    except (ValueError, RuntimeError):
        model_name = settings.configured_model_name or "unknown"

    return AgentRunResponse(
        project_path=request.project_path,
        git_provider=request.git_provider,
        active_repository_source=request.git_provider,
        active_repository_path=request.project_path,
        active_branch=request.branch,
        task=request.task,
        model=model_name,
        branch=request.branch,
        repo_root_name=Path(request.project_path).name or request.project_path,
        context_summary="",
        top_level_entries=[],
        readme_included=False,
        diff_included=False,
        is_git_repository=True,
        files_read=[],
        response=(
            "RepoOperator hit an agent routing error before it could complete this request. "
            "Retry once, and if it keeps happening check the Debug Events / Runs panel."
        ),
        response_type="agent_error",
        intent_classification="agent_error",
        graph_path="langgraph:error",
        agent_flow="langgraph",
        proposal_error_details=str(exc),
        validation_status="agent_error",
    )
