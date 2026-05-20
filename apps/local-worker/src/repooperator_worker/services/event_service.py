from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from repooperator_worker.schemas import AgentRunRequest, AgentRunResponse
from repooperator_worker.services.common import get_repooperator_home_dir
from repooperator_worker.services.json_safe import json_safe


def _runs_dir() -> Path:
    path = get_repooperator_home_dir() / "runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _runs_file() -> Path:
    return _runs_dir() / "runs.jsonl"


_RUN_LOCK = RLock()


def _run_dir(run_id: str) -> Path:
    safe = "".join(ch for ch in run_id if ch.isalnum() or ch in {"_", "-"})
    path = _runs_dir() / safe
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_meta_file(run_id: str) -> Path:
    return _run_dir(run_id) / "meta.json"


def _run_events_file(run_id: str) -> Path:
    return _run_dir(run_id) / "events.jsonl"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def new_run_id() -> str:
    return f"run_{uuid.uuid4().hex[:12]}"


def start_active_run(
    *,
    run_id: str,
    request: AgentRunRequest,
    thread_id: str | None = None,
) -> dict[str, Any]:
    effective_thread_id = thread_id or request.thread_id
    record = {
        "id": run_id,
        "thread_id": effective_thread_id,
        "repo": request.project_path,
        "branch": request.branch,
        "task_summary": summarize_user_message(request.task),
        "request_snapshot": json_safe(request.model_dump(mode="json")),
        "status": "running",
        "started_at": _now_iso(),
        "completed_at": None,
        "final_result": None,
        "error": None,
    }
    with _RUN_LOCK:
        try:
            _run_meta_file(run_id).write_text(json.dumps(json_safe(record), ensure_ascii=False, sort_keys=True), encoding="utf-8")
        except OSError:
            pass
    return record


def append_run_event(run_id: str, event: dict[str, Any]) -> dict[str, Any]:
    with _RUN_LOCK:
        events = list_run_events(run_id)
        sequence = int(event.get("sequence") or len(events) + 1)
        meta = get_run(run_id) or {}
        record = json_safe({
            **event,
            "run_id": run_id,
            "thread_id": event.get("thread_id") or meta.get("thread_id"),
            "repo": event.get("repo") or meta.get("repo"),
            "branch": event.get("branch") or meta.get("branch"),
            "sequence": sequence,
            "timestamp": event.get("timestamp") or _now_iso(),
            "persisted": True,
        })
        if not record.get("id"):
            record["id"] = f"{run_id}-event-{sequence}"
        try:
            with _run_events_file(run_id).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        except OSError:
            pass
    return record


def complete_active_run(
    *,
    run_id: str,
    status: str,
    final_result: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    with _RUN_LOCK:
        meta = get_run(run_id) or {"id": run_id}
        if meta.get("status") in {"cancelled", "cancelling"} and status == "completed":
            status = "cancelled"
        completed_at = _now_iso()
        _finalize_running_events(run_id, status=status, ended_at=completed_at)
        meta.update(
            {
                "status": status,
                "completed_at": completed_at,
                "final_result": json_safe(final_result),
                "error": error,
            }
        )
        try:
            _run_meta_file(run_id).write_text(json.dumps(json_safe(meta), ensure_ascii=False, sort_keys=True), encoding="utf-8")
        except OSError:
            pass
    return meta


def request_run_cancellation(run_id: str) -> dict[str, Any]:
    meta = get_run(run_id) or {}
    if meta.get("status") in {"completed", "failed", "cancelled"}:
        return meta
    meta["status"] = "cancelling"
    meta["error"] = meta.get("error")
    try:
        _run_meta_file(run_id).write_text(json.dumps(json_safe(meta), ensure_ascii=False, sort_keys=True), encoding="utf-8")
    except OSError:
        pass
    append_run_event(
        run_id,
        {
            "type": "progress_delta",
            "event_type": "cancellation_requested",
            "thread_id": meta.get("thread_id"),
            "repo": meta.get("repo"),
            "branch": meta.get("branch"),
            "phase": "Finished",
            "label": "Cancellation requested",
            "detail": "RepoOperator will stop this run at the next safe checkpoint.",
            "status": "waiting",
        },
    )
    return meta


def record_run_steering(run_id: str, content: str) -> dict[str, Any]:
    event = append_run_event(
        run_id,
        {
            "type": "progress_delta",
            "event_type": "steering_received",
            "phase": "Planning",
            "label": "Received steering instruction",
            "detail": summarize_user_message(content, max_len=220),
            "status": "completed",
        },
    )
    meta = get_run(run_id) or {"id": run_id}
    steering = list(meta.get("steering_instructions") or [])
    steering.append(
        {
            "content": summarize_user_message(content, max_len=500),
            "created_at": _now_iso(),
            "status": "recorded",
            "accepted": None,
            "parse_status": "pending",
            "reason": "Recorded for parsing at the next safe checkpoint.",
        }
    )
    meta["steering_instructions"] = steering
    try:
        _run_meta_file(run_id).write_text(json.dumps(json_safe(meta), ensure_ascii=False, sort_keys=True), encoding="utf-8")
    except OSError:
        pass
    return event


def get_run(run_id: str) -> dict[str, Any] | None:
    path = _run_meta_file(run_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def list_run_events(run_id: str, *, after_sequence: int = 0) -> list[dict[str, Any]]:
    path = _run_events_file(run_id)
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if int(event.get("sequence") or 0) > after_sequence:
            events.append(event)
    return events


def list_activity_states(run_id: str) -> list[dict[str, Any]]:
    """Return UI-facing merged activity cards for a run.

    Raw events remain append-only in ``list_run_events``. This view merges by
    stable ``activity_id`` so Reading -> Reviewing -> Completed transitions
    update one card instead of creating duplicate cards.
    """
    states: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for event in list_run_events(run_id):
        activity_id = event.get("activity_id")
        if not activity_id:
            continue
        key = str(activity_id)
        if key not in states:
            states[key] = dict(event)
            order.append(key)
            continue
        merged = states[key]
        for field, value in event.items():
            if value in (None, "", [], {}):
                continue
            if field.endswith("_delta") and merged.get(field):
                merged[field] = str(merged[field]) + str(value)
            else:
                merged[field] = value
        states[key] = merged
    return [states[key] for key in order]


def get_active_runs(thread_id: str | None = None) -> list[dict[str, Any]]:
    active: list[dict[str, Any]] = []
    active_statuses = {"pending", "running", "waiting_approval", "cancelling"}
    for meta_path in _runs_dir().glob("*/meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            continue
        if meta.get("status") not in active_statuses:
            continue
        if thread_id and meta.get("thread_id") != thread_id:
            continue
        active.append(meta)
    return sorted(active, key=lambda item: str(item.get("started_at") or ""), reverse=True)


def summarize_user_message(message: str, *, max_len: int = 180) -> str:
    cleaned = " ".join(message.split())
    if len(cleaned) > max_len:
        return cleaned[: max_len - 1].rstrip() + "..."
    return cleaned


def record_agent_run(
    *,
    run_id: str,
    request: AgentRunRequest,
    response: AgentRunResponse | None,
    status: str,
    latency_ms: int,
    error: str | None = None,
) -> dict[str, Any]:
    record = {
        "id": run_id,
        "timestamp": _now_iso(),
        "repo": request.project_path,
        "branch": request.branch,
        "user_message_summary": summarize_user_message(request.task),
        "intent": response.intent_classification if response else None,
        "graph_path": response.graph_path if response else None,
        "agent_flow": response.agent_flow if response else "langgraph",
        "model": response.model if response else None,
        "status": status,
        "latency_ms": latency_ms,
        "files_read": response.files_read if response else [],
        "thread_context_files": response.thread_context_files if response else [],
        "thread_context_symbols": response.thread_context_symbols if response else [],
        "proposal_id": response.proposal_relative_path if response else None,
        "error": error,
    }
    try:
        with _runs_file().open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(json_safe(record), ensure_ascii=False, sort_keys=True) + "\n")
    except OSError:
        pass
    return record


def record_event(
    *,
    event_type: str,
    repo: str | None = None,
    branch: str | None = None,
    status: str = "ok",
    summary: str = "",
    files: list[str] | None = None,
    tool: str | None = None,
    command: list[str] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    record = {
        "id": new_run_id(),
        "timestamp": _now_iso(),
        "type": event_type,
        "repo": repo,
        "branch": branch,
        "status": status,
        "summary": summarize_user_message(summary),
        "files_read": files or [],
        "tool": tool,
        "command": command,
        "error": error,
    }
    try:
        with _runs_file().open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(json_safe(record), ensure_ascii=False, sort_keys=True) + "\n")
    except OSError:
        pass
    return record


def list_recent_runs(limit: int = 50) -> list[dict[str, Any]]:
    path = _runs_file()
    if not path.exists():
        return []
    runs: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            runs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return list(reversed(runs[-limit:]))


def _finalize_running_events(run_id: str, *, status: str, ended_at: str) -> None:
    path = _run_events_file(run_id)
    if not path.exists():
        return
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return
    finalized: list[str] = []
    final_status = "completed"
    if status == "cancelled":
        final_status = "cancelled"
    elif status == "failed":
        final_status = "failed"
    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            finalized.append(line)
            continue
        if event.get("type") == "progress_delta" and event.get("status") == "running":
            event["status"] = final_status
            event["ended_at"] = event.get("ended_at") or ended_at
            if event.get("duration_ms") is None:
                event["duration_ms"] = _duration_ms(event.get("started_at"), event["ended_at"])
        finalized.append(json.dumps(json_safe(event), ensure_ascii=False, sort_keys=True))
    try:
        path.write_text("\n".join(finalized) + ("\n" if finalized else ""), encoding="utf-8")
    except OSError:
        pass


def _duration_ms(started_at: str | None, ended_at: str | None) -> int | None:
    if not started_at or not ended_at:
        return None
    try:
        start = _parse_iso(started_at)
        end = _parse_iso(ended_at)
    except ValueError:
        return None
    return max(0, int((end - start).total_seconds() * 1000))


def _parse_iso(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed
