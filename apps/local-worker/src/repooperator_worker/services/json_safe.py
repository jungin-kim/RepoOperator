from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any


def json_safe(value: Any) -> Any:
    """Return a JSON-serializable copy of common app payload objects."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, BaseException):
        return f"{value.__class__.__name__}: {value}"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return json_safe(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value))
    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            return json_safe(value.model_dump(mode="json"))
        except Exception:
            try:
                return json_safe(value.model_dump())
            except Exception:
                return safe_repr(value)
    if isinstance(value, dict):
        return {str(json_safe(key)): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(item) for item in value]
    return safe_repr(value)


def safe_agent_response_payload(response: Any) -> dict[str, Any]:
    """Return a JSON-valid AgentRunResponse-like payload without losing readable evidence."""
    try:
        payload = response.model_dump(mode="json")
        json.dumps(payload, ensure_ascii=False)
        return payload
    except Exception:
        payload = json_safe(response)
        if not isinstance(payload, dict):
            payload = {}
    preserved = {
        "project_path": getattr(response, "project_path", payload.get("project_path", "")),
        "git_provider": getattr(response, "git_provider", payload.get("git_provider", None)),
        "active_repository_source": getattr(response, "active_repository_source", payload.get("active_repository_source", None)),
        "active_repository_path": getattr(response, "active_repository_path", payload.get("active_repository_path", None)),
        "active_branch": getattr(response, "active_branch", payload.get("active_branch", None)),
        "task": getattr(response, "task", payload.get("task", "")),
        "model": getattr(response, "model", payload.get("model", "unknown")),
        "branch": getattr(response, "branch", payload.get("branch", None)),
        "repo_root_name": getattr(response, "repo_root_name", payload.get("repo_root_name", "")),
        "context_summary": getattr(response, "context_summary", payload.get("context_summary", "")),
        "top_level_entries": getattr(response, "top_level_entries", payload.get("top_level_entries", [])),
        "readme_included": getattr(response, "readme_included", payload.get("readme_included", False)),
        "diff_included": getattr(response, "diff_included", payload.get("diff_included", False)),
        "is_git_repository": getattr(response, "is_git_repository", payload.get("is_git_repository", False)),
        "files_read": getattr(response, "files_read", payload.get("files_read", [])),
        "response": getattr(response, "response", payload.get("response", "")),
        "response_type": getattr(response, "response_type", payload.get("response_type", "assistant_answer")),
        "activity_events": getattr(response, "activity_events", payload.get("activity_events", [])),
        "stop_reason": getattr(response, "stop_reason", payload.get("stop_reason", None)),
        "loop_iteration": getattr(response, "loop_iteration", payload.get("loop_iteration", 0)),
        "intent_classification": getattr(response, "intent_classification", payload.get("intent_classification", None)),
        "skills_used": getattr(response, "skills_used", payload.get("skills_used", [])),
        "graph_path": getattr(response, "graph_path", payload.get("graph_path", None)),
        "run_id": getattr(response, "run_id", payload.get("run_id", None)),
        "agent_flow": getattr(response, "agent_flow", payload.get("agent_flow", "langgraph")),
    }
    payload.update(json_safe(preserved))
    payload["metadata_serialization_error"] = True
    safe_payload = json_safe(payload)
    json.dumps(safe_payload, ensure_ascii=False)
    return safe_payload


def safe_repr(value: Any, *, limit: int = 500) -> str:
    try:
        text = repr(value)
    except Exception:
        text = f"<unrepresentable {value.__class__.__name__}>"
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "..."
