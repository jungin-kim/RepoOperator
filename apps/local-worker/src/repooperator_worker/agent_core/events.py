from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any

from repooperator_worker.schemas import AgentRunRequest
from repooperator_worker.services.event_service import append_run_event


TERMINAL_ACTIVITY_STATUSES = {"completed", "failed", "cancelled", "timed_out", "waiting"}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def activity_event(
    *,
    run_id: str,
    request: AgentRunRequest,
    activity_id: str,
    event_type: str,
    phase: str,
    label: str,
    status: str = "running",
    visibility: str | None = None,
    display: str | None = None,
    current_action: str | None = None,
    observation: str | None = None,
    next_action: str | None = None,
    detail: str = "",
    detail_delta: str | None = None,
    observation_delta: str | None = None,
    next_action_delta: str | None = None,
    safe_reasoning_summary: str | None = None,
    safe_reasoning_summary_delta: str | None = None,
    evidence_needed: list[str] | None = None,
    uncertainty: list[str] | None = None,
    safety_note: str | None = None,
    operation: str | None = None,
    action_type: str | None = None,
    tool_name: str | None = None,
    started_at: str | None = None,
    ended_at: str | None = None,
    duration_ms: int | None = None,
    related_files: list[str] | None = None,
    related_search_query: str | None = None,
    related_command: list[str] | str | None = None,
    command: list[str] | str | None = None,
    proposal_id: str | None = None,
    aggregate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = utc_now()
    resolved_visibility = visibility or ("user" if event_type == "work_trace" else "debug")
    resolved_display = display or ("primary" if event_type == "work_trace" else "secondary")
    event = {
        "id": f"{run_id}-event-{uuid.uuid4().hex[:10]}",
        "type": "progress_delta",
        "event_type": event_type,
        "activity_id": activity_id,
        "run_id": run_id,
        "thread_id": request.thread_id,
        "repo": request.project_path,
        "branch": request.branch,
        "phase": phase,
        "label": label,
        "visibility": resolved_visibility,
        "display": resolved_display,
        "status": status,
        "current_action": current_action,
        "observation": observation,
        "next_action": next_action,
        "detail": detail,
        "detail_delta": detail_delta,
        "observation_delta": observation_delta,
        "next_action_delta": next_action_delta,
        "safe_reasoning_summary": safe_reasoning_summary,
        "safe_reasoning_summary_delta": safe_reasoning_summary_delta,
        "summary_delta": safe_reasoning_summary_delta,
        "evidence_needed": evidence_needed or [],
        "uncertainty": uncertainty or [],
        "safety_note": safety_note,
        "operation": operation,
        "action_type": action_type,
        "tool_name": tool_name,
        "started_at": started_at or now,
        "updated_at": now,
        "ended_at": ended_at or (now if status in TERMINAL_ACTIVITY_STATUSES else None),
        "duration_ms": duration_ms,
        "related_search_query": related_search_query,
        "related_files": related_files or [],
        "files": related_files or [],
        "related_command": related_command,
        "command": command if command is not None else related_command,
        "proposal_id": proposal_id,
        "aggregate": aggregate,
    }
    return {key: value for key, value in event.items() if value is not None}


def append_activity_event(**kwargs: Any) -> dict[str, Any]:
    event = activity_event(**kwargs)
    try:
        return append_run_event(str(kwargs["run_id"]), event)
    except OSError:
        return event
    except PermissionError:
        return event


def append_work_trace(
    *,
    run_id: str,
    request: AgentRunRequest,
    activity_id: str,
    phase: str,
    label: str,
    status: str = "running",
    safe_reasoning_summary: str | None = None,
    current_action: str | None = None,
    observation: str | None = None,
    next_action: str | None = None,
    evidence_needed: list[str] | None = None,
    uncertainty: list[str] | None = None,
    safety_note: str | None = None,
    operation: str | None = None,
    action_type: str | None = None,
    tool_name: str | None = None,
    related_files: list[str] | None = None,
    related_search_query: str | None = None,
    command: list[str] | str | None = None,
    proposal_id: str | None = None,
    started_at: str | None = None,
    ended_at: str | None = None,
    duration_ms: int | None = None,
    aggregate: dict[str, Any] | None = None,
    visibility: str = "user",
    display: str = "primary",
) -> dict[str, Any]:
    """Persist a concise user-visible work trace event.

    Work traces are progress_delta-compatible so existing clients can merge and
    rehydrate them by activity_id. They must contain only safe summaries of
    what was checked, observed, or chosen.
    """
    return append_activity_event(
        run_id=run_id,
        request=request,
        activity_id=activity_id,
        event_type="work_trace",
        phase=phase,
        label=label,
        status=status,
        visibility=visibility,
        display=display,
        safe_reasoning_summary=_truncate_text(safe_reasoning_summary, 360),
        current_action=_truncate_text(current_action, 260),
        observation=_truncate_text(observation, 360),
        next_action=_truncate_text(next_action, 260),
        evidence_needed=_truncate_list(evidence_needed, item_limit=120, max_items=6),
        uncertainty=_truncate_list(uncertainty, item_limit=140, max_items=6),
        safety_note=_truncate_text(safety_note, 260),
        operation=_truncate_text(operation, 80),
        action_type=_truncate_text(action_type, 80),
        tool_name=_truncate_text(tool_name, 80),
        related_files=_truncate_list(related_files, item_limit=200, max_items=12),
        related_search_query=_truncate_text(related_search_query, 300),
        command=command,
        related_command=command,
        proposal_id=_truncate_text(proposal_id, 200),
        started_at=started_at,
        ended_at=ended_at,
        duration_ms=duration_ms,
        aggregate=aggregate,
    )


def merge_activity_states(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for event in events:
        activity_id = str(event.get("activity_id") or event.get("id") or "")
        if not activity_id:
            continue
        if activity_id not in states:
            states[activity_id] = dict(event)
            order.append(activity_id)
        else:
            merged = states[activity_id]
            for key, value in event.items():
                if value not in (None, "", [], {}):
                    if key.endswith("_delta") and merged.get(key):
                        merged[key] = str(merged[key]) + str(value)
                    else:
                        merged[key] = value
            states[activity_id] = merged
    return [states[item] for item in order]


def _truncate_text(value: str | None, limit: int) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    if not text:
        return None
    if _contains_nonpublic_reasoning_marker(text):
        return "A safe work summary was unavailable."
    return text if len(text) <= limit else text[: max(0, limit - 3)].rstrip() + "..."


def _truncate_list(values: list[str] | None, *, item_limit: int, max_items: int) -> list[str]:
    out: list[str] = []
    for value in values or []:
        text = _truncate_text(str(value), item_limit)
        if text:
            out.append(text)
        if len(out) >= max_items:
            break
    return out


def _contains_nonpublic_reasoning_marker(text: str) -> bool:
    lowered = text.lower()
    markers = (
        "<think>",
        "chain-" + "of-thought",
        "chain " + "of thought",
        "private " + "reasoning",
        "hidden " + "reasoning",
    )
    return any(marker in lowered for marker in markers)


def duration_ms(started_at: str | None, ended_at: str | None) -> int | None:
    if not started_at or not ended_at:
        return None
    try:
        start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        end = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return max(0, int((end - start).total_seconds() * 1000))
