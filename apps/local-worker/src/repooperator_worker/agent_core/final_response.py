from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from repooperator_worker.config import get_settings
from repooperator_worker.schemas import AgentRunRequest, AgentRunResponse
from repooperator_worker.services.common import resolve_project_path
from repooperator_worker.services.json_safe import json_safe
from repooperator_worker.services.model_client import OpenAICompatibleModelClient


def build_agent_response(
    request: AgentRunRequest,
    *,
    response: str,
    response_type: str = "assistant_answer",
    model: str | None = None,
    files_read: list[str] | None = None,
    graph_path: str = "agent_core",
    intent_classification: str | None = None,
    activity_events: list[dict[str, Any]] | None = None,
    run_id: str | None = None,
    stop_reason: str = "completed",
    loop_iteration: int = 1,
    **updates: Any,
) -> AgentRunResponse:
    try:
        repo_path = resolve_project_path(request.project_path)
    except ValueError:
        repo_path = Path(request.project_path)
    top_level_entries: list[str] = []
    if repo_path.exists():
        try:
            top_level_entries = sorted(path.name for path in repo_path.iterdir())[:80]
        except OSError:
            top_level_entries = []
    is_git = (repo_path / ".git").exists()
    branch = request.branch or _git_branch(repo_path)
    model_name = model or _model_name()
    payload: dict[str, Any] = {
        "project_path": request.project_path,
        "git_provider": request.git_provider,
        "active_repository_source": request.git_provider,
        "active_repository_path": request.project_path,
        "active_branch": branch,
        "task": request.task,
        "model": model_name,
        "branch": branch,
        "repo_root_name": repo_path.name or request.project_path,
        "context_summary": "",
        "top_level_entries": top_level_entries,
        "readme_included": "README.md" in top_level_entries or "readme.md" in {item.lower() for item in top_level_entries},
        "diff_included": False,
        "is_git_repository": is_git,
        "files_read": files_read or [],
        "response": response,
        "response_type": response_type,
        "intent_classification": intent_classification,
        "graph_path": graph_path,
        "agent_flow": "langgraph",
        "activity_events": activity_events or [],
        "run_id": run_id,
        "stop_reason": stop_reason,
        "loop_iteration": loop_iteration,
    }
    payload.update(json_safe(updates))
    return AgentRunResponse(**json_safe(payload))


def _model_name() -> str:
    try:
        return OpenAICompatibleModelClient().model_name
    except Exception:
        settings = get_settings()
        return settings.configured_model_name or "unknown"


def _git_branch(repo_path: Path) -> str | None:
    if not (repo_path / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=repo_path,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    branch = result.stdout.strip()
    return branch or None
