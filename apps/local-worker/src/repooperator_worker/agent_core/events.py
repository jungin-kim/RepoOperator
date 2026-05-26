from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any

from repooperator_worker.schemas import AgentRunRequest
from repooperator_worker.services.event_service import append_run_event


TERMINAL_ACTIVITY_STATUSES = {"completed", "failed", "cancelled", "timed_out", "waiting"}

RUN_STATUSES = {"pending", "running", "waiting_approval", "cancelling", "completed", "failed", "cancelled", "timed_out"}
ACTION_RESULT_STATUSES = {"success", "failed", "skipped", "blocked", "waiting_approval", "cancelled", "timed_out"}
PROGRESS_STATUSES = {"queued", "running", "completed", "failed", "waiting", "waiting_approval", "cancelled"}
PROPOSAL_STATUSES = {"draft", "valid", "invalid", "repairable", "awaiting_approval", "approved", "applied", "rejected", "failed"}
VALIDATION_KINDS = {"change_set", "post_apply", "command", "git"}
VALIDATION_STATUSES = {"passed", "failed", "skipped", "blocked", "warning"}

EVENT_KIND_GRAPH_TRANSITION = "graph_transition"
EVENT_KIND_TOOL_ACTION = "tool_action"
EVENT_KIND_ACTION_RESULT = "action_result"
EVENT_KIND_VALIDATION = "validation"
EVENT_KIND_PROPOSAL = "proposal"
EVENT_KIND_APPROVAL = "approval"
EVENT_KIND_GIT = "git"
EVENT_KIND_WEB = "web"
EVENT_KIND_FINAL_ANSWER = "final_answer"
EVENT_KIND_DEBUG_RATIONALE = "debug_rationale"

EVENT_AUDIENCE_PRIMARY = "primary"
EVENT_AUDIENCE_SECONDARY = "secondary"
EVENT_AUDIENCE_DEBUG = "debug"
EVENT_AUDIENCE_INTERNAL = "internal"

EVENT_KINDS = {
    EVENT_KIND_GRAPH_TRANSITION,
    EVENT_KIND_TOOL_ACTION,
    EVENT_KIND_ACTION_RESULT,
    EVENT_KIND_VALIDATION,
    EVENT_KIND_PROPOSAL,
    EVENT_KIND_APPROVAL,
    EVENT_KIND_GIT,
    EVENT_KIND_WEB,
    EVENT_KIND_FINAL_ANSWER,
    EVENT_KIND_DEBUG_RATIONALE,
}
EVENT_AUDIENCES = {EVENT_AUDIENCE_PRIMARY, EVENT_AUDIENCE_SECONDARY, EVENT_AUDIENCE_DEBUG, EVENT_AUDIENCE_INTERNAL}


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
    kind: str | None = None,
    audience: str | None = None,
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
    resolved_kind = _resolve_event_kind(
        explicit_kind=kind,
        event_type=event_type,
        operation=operation,
        action_type=action_type,
        tool_name=tool_name,
        related_files=related_files,
        command=command if command is not None else related_command,
        proposal_id=proposal_id,
        aggregate=aggregate,
        safe_reasoning_summary=safe_reasoning_summary or safe_reasoning_summary_delta,
    )
    resolved_audience = _resolve_event_audience(
        explicit_audience=audience,
        visibility=visibility,
        display=display,
        event_type=event_type,
        kind=resolved_kind,
    )
    resolved_visibility = visibility or _visibility_for_audience(resolved_audience)
    resolved_display = display or _display_for_audience(resolved_audience)
    event = {
        "id": f"{run_id}-event-{uuid.uuid4().hex[:10]}",
        "type": "progress_delta",
        "event_type": event_type,
        "kind": resolved_kind,
        "audience": resolved_audience,
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
    kind: str | None = None,
    audience: str | None = None,
    visibility: str | None = None,
    display: str | None = None,
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
        kind=kind,
        audience=audience,
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


def normalize_validation_kind(value: Any) -> str | None:
    text = str(value or "").strip().lower().replace("-", "_")
    if text == "post_apply_validation":
        text = "post_apply"
    if text in VALIDATION_KINDS:
        return text
    return None


def normalize_validation_status(value: Any) -> str | None:
    text = str(value or "").strip().lower().replace("-", "_")
    if text in {"passed", "pass", "success", "succeeded", "valid"}:
        return "passed"
    if text in {"failed", "fail", "failure", "error", "invalid"}:
        return "failed"
    if text in {
        "skipped",
        "skip",
        "not_run",
        "notrun",
        "none",
        "selected",
        "skipped_no_validation_command",
        "skipped_no_safe_command_selected",
    }:
        return "skipped"
    if text in {"blocked", "waiting_approval", "approval_denied", "denied", "cancelled", "timed_out"}:
        return "blocked"
    if text in {"warning", "warn", "repairable"}:
        return "warning"
    if text in VALIDATION_STATUSES:
        return text
    return None


def normalize_validation_result(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None
    kind = normalize_validation_kind(result.get("kind") or result.get("source"))
    if not kind:
        return None
    status = normalize_validation_status(result.get("status"))
    if not status:
        return None
    normalized = dict(result)
    normalized["kind"] = kind
    normalized.setdefault("source", kind)
    if normalized.get("status") != status:
        normalized.setdefault("raw_status", normalized.get("status"))
        normalized["status"] = status
    return normalized


def _resolve_event_kind(
    *,
    explicit_kind: str | None,
    event_type: str,
    operation: str | None,
    action_type: str | None,
    tool_name: str | None,
    related_files: list[str] | None,
    command: list[str] | str | None,
    proposal_id: str | None,
    aggregate: dict[str, Any] | None,
    safe_reasoning_summary: str | None,
) -> str:
    if explicit_kind in EVENT_KINDS:
        return str(explicit_kind)
    aggregate = aggregate if isinstance(aggregate, dict) else {}
    aggregate_validation = aggregate.get("validation_result") if isinstance(aggregate.get("validation_result"), dict) else None
    if normalize_validation_result(aggregate_validation):
        return EVENT_KIND_VALIDATION
    aggregate_action = str(aggregate.get("action_type") or aggregate.get("tool") or aggregate.get("operation") or "")
    action = str(action_type or tool_name or operation or aggregate_action or "")
    if event_type == "graph_transition":
        if _is_git_action(action):
            return EVENT_KIND_GIT
        if _is_web_action(action):
            return EVENT_KIND_WEB
        if _is_proposal_action(action) or proposal_id:
            return EVENT_KIND_PROPOSAL
        if "approval" in action:
            return EVENT_KIND_APPROVAL
        return EVENT_KIND_GRAPH_TRANSITION
    if event_type == "action_result":
        return EVENT_KIND_ACTION_RESULT
    if event_type == "work_trace":
        if _is_git_action(action):
            return EVENT_KIND_GIT
        if _is_web_action(action):
            return EVENT_KIND_WEB
        if _is_proposal_action(action) or proposal_id:
            return EVENT_KIND_PROPOSAL
        if _has_concrete_work_signal(action, related_files, command, proposal_id, aggregate):
            return EVENT_KIND_TOOL_ACTION
        if safe_reasoning_summary:
            return EVENT_KIND_DEBUG_RATIONALE
        return EVENT_KIND_TOOL_ACTION
    return EVENT_KIND_GRAPH_TRANSITION


def _resolve_event_audience(
    *,
    explicit_audience: str | None,
    visibility: str | None,
    display: str | None,
    event_type: str,
    kind: str,
) -> str:
    if explicit_audience in EVENT_AUDIENCES:
        return str(explicit_audience)
    if visibility == "internal" or display == "hidden":
        return EVENT_AUDIENCE_INTERNAL
    if visibility == "debug" or kind == EVENT_KIND_DEBUG_RATIONALE:
        return EVENT_AUDIENCE_DEBUG
    if display == "secondary":
        return EVENT_AUDIENCE_SECONDARY
    if display == "primary" or visibility == "user":
        return EVENT_AUDIENCE_PRIMARY
    if kind == EVENT_KIND_VALIDATION:
        return EVENT_AUDIENCE_PRIMARY
    if event_type == "work_trace" and kind != EVENT_KIND_DEBUG_RATIONALE:
        return EVENT_AUDIENCE_PRIMARY
    return EVENT_AUDIENCE_DEBUG


def _visibility_for_audience(audience: str) -> str:
    if audience == EVENT_AUDIENCE_INTERNAL:
        return "internal"
    if audience == EVENT_AUDIENCE_DEBUG:
        return "debug"
    return "user"


def _display_for_audience(audience: str) -> str:
    if audience == EVENT_AUDIENCE_INTERNAL:
        return "hidden"
    if audience in {EVENT_AUDIENCE_DEBUG, EVENT_AUDIENCE_SECONDARY}:
        return "secondary"
    return "primary"


def _has_concrete_work_signal(
    action: str,
    related_files: list[str] | None,
    command: list[str] | str | None,
    proposal_id: str | None,
    aggregate: dict[str, Any],
) -> bool:
    if action or proposal_id or command or related_files:
        return True
    concrete_keys = {
        "action_type",
        "tool",
        "operation",
        "query",
        "queries",
        "text_queries",
        "file_path",
        "path",
        "directory",
        "entries_count",
        "display_command",
        "command",
        "exit_code",
        "returncode",
        "edit_archive",
        "source_count",
        "sources",
    }
    return any(key in aggregate and aggregate.get(key) not in (None, "", [], {}) for key in concrete_keys)


def _is_web_action(action: str) -> bool:
    return action in {"search_web", "fetch_url", "summarize_web_evidence", "web_search", "web_fetch"}


def _is_git_action(action: str) -> bool:
    return action.startswith("git_") or action in {"github_create_pr", "gitlab_create_mr", "inspect_git_state"}


def _is_proposal_action(action: str) -> bool:
    return action in {
        "generate_edit",
        "generate_change_set",
        "validate_change_set",
        "plan_change_set",
        "repair_change_set",
        "apply_change_set",
        "change_set_apply",
    }


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
