from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from threading import RLock, Thread
from typing import Any, Iterator

from repooperator_worker.schemas import AgentRunRequest, AgentRunResponse
from repooperator_worker.services.common import get_repooperator_home_dir
from repooperator_worker.services.event_service import (
    append_run_event,
    complete_active_run,
    get_active_runs,
    get_run,
    list_run_events,
    new_run_id,
    request_run_cancellation,
    record_agent_run,
    record_event,
    start_active_run,
    summarize_user_message,
)
from repooperator_worker.agent_core.actions import ActionResult
from repooperator_worker.services.json_safe import json_safe, safe_agent_response_payload, safe_repr
from repooperator_worker.services.memory_service import maybe_record_from_agent_run
from repooperator_worker.services.thread_context_service import update_thread_context


_COORDINATOR_LOCK = RLock()


def start_run(request: AgentRunRequest, *, stream: bool = False) -> AgentRunResponse:
    """Run the authoritative sync agent lifecycle and persist run state/events."""
    run_id = new_run_id()
    start_active_run(run_id=run_id, request=request, thread_id=request.thread_id)
    append_activity(
        run_id,
        request=request,
        phase="Thinking",
        label="Started agent run",
        status="running",
        event_type="run_started",
    )
    start = time.perf_counter()
    response: AgentRunResponse | None = None
    try:
        from repooperator_worker.agent_core.langgraph_runtime import run_langgraph_controller

        response = run_langgraph_controller(request, run_id=run_id).model_copy(update={"run_id": run_id})
        final_payload = _safe_final_result(response, run_id=run_id)
        _record_response_events(run_id, request, response)
        if _is_waiting_for_approval(response):
            wait_for_approval(run_id, _approval_resume_payload(response, request))
            record_agent_run(
                run_id=run_id,
                request=request,
                response=response,
                status="waiting_approval",
                latency_ms=int((time.perf_counter() - start) * 1000),
            )
            return response
        maybe_record_from_agent_run(request, response)
        update_thread_context(request, response)
        terminal_status = "cancelled" if response.stop_reason == "cancelled" else "completed"
        complete_active_run(
            run_id=run_id,
            status=terminal_status,
            final_result=final_payload,
        )
        if terminal_status == "completed":
            _drain_queue_after_run(request)
        record_agent_run(
            run_id=run_id,
            request=request,
            response=response,
            status="ok",
            latency_ms=int((time.perf_counter() - start) * 1000),
        )
        return response
    except Exception as exc:
        complete_active_run(run_id=run_id, status="failed", error=str(exc))
        record_agent_run(
            run_id=run_id,
            request=request,
            response=response,
            status="error",
            latency_ms=int((time.perf_counter() - start) * 1000),
            error=str(exc),
        )
        raise


def stream_run(request: AgentRunRequest) -> tuple[str, Iterator[str]]:
    """Start a background streamed run and return an SSE iterator."""
    run_id = new_run_id()
    start_active_run(run_id=run_id, request=request, thread_id=request.thread_id)
    deferred_finalization: dict[str, Any] = {}
    deferred_finalization_lock = RLock()

    def defer_stream_finalization(
        *,
        status: str,
        final_result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        with deferred_finalization_lock:
            deferred_finalization.clear()
            deferred_finalization.update(
                {
                    "status": status,
                    "final_result": json_safe(final_result),
                    "error": error,
                }
            )

    def take_deferred_stream_finalization() -> dict[str, Any] | None:
        with deferred_finalization_lock:
            if not deferred_finalization:
                return None
            payload = dict(deferred_finalization)
            deferred_finalization.clear()
            return payload

    def worker() -> None:
        final_result: dict | None = None
        started = time.perf_counter()
        try:
            from repooperator_worker.agent_core.langgraph_runtime import stream_langgraph_controller

            for event in stream_langgraph_controller(request, run_id=run_id):
                if isinstance(event, str):
                    try:
                        event = json.loads(event)
                    except json.JSONDecodeError:
                        continue
                if not isinstance(event, dict):
                    continue
                if should_cancel(run_id) and event.get("type") != "final_message":
                    append_activity(
                        run_id,
                        request=request,
                        phase="Finished",
                        label="Run cancelled",
                        status="failed",
                        event_type="run_cancelled",
                    )
                    defer_stream_finalization(status="cancelled", error="Cancelled by user.")
                    return
                if event.get("type") == "final_message":
                    final_result = json_safe(event.get("result"))
                if not event.get("persisted"):
                    append_run_event(run_id, event)
            if isinstance(final_result, dict):
                try:
                    response = AgentRunResponse.model_validate(final_result)
                    if _is_waiting_for_approval(response):
                        wait_for_approval(run_id, _approval_resume_payload(response, request))
                        record_agent_run(
                            run_id=run_id,
                            request=request,
                            response=response,
                            status="waiting_approval",
                            latency_ms=int((time.perf_counter() - started) * 1000),
                        )
                        return
                    update_thread_context(request, response)
                    maybe_record_from_agent_run(request, response)
                    record_agent_run(
                        run_id=run_id,
                        request=request,
                        response=response,
                        status="ok",
                        latency_ms=int((time.perf_counter() - started) * 1000),
                    )
                except Exception:
                    pass
            run_meta = get_run(run_id) or {}
            terminal_status = "cancelled" if (
                isinstance(final_result, dict) and final_result.get("stop_reason") == "cancelled"
            ) or run_meta.get("status") == "cancelling" else "completed"
            if terminal_status == "cancelled":
                defer_stream_finalization(status="cancelled", final_result=final_result)
            else:
                complete_active_run(run_id=run_id, status=terminal_status, final_result=final_result)
                if terminal_status == "completed":
                    _drain_queue_after_run(request)
        except Exception as exc:  # noqa: BLE001
            append_run_event(run_id, {"type": "error", "message": str(exc), "status": "failed"})
            complete_active_run(run_id=run_id, status="failed", error=str(exc))
            record_agent_run(
                run_id=run_id,
                request=request,
                response=None,
                status="error",
                latency_ms=int((time.perf_counter() - started) * 1000),
                error=str(exc),
            )

    Thread(target=worker, daemon=True).start()

    def generate() -> Iterator[str]:
        last_sequence = 0
        while True:
            events = list_run_events(run_id, after_sequence=last_sequence)
            for event in events:
                last_sequence = int(event.get("sequence") or last_sequence)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if not events:
                deferred = take_deferred_stream_finalization()
                if deferred is not None:
                    complete_active_run(
                        run_id=run_id,
                        status=str(deferred.get("status") or "completed"),
                        final_result=deferred.get("final_result"),
                        error=deferred.get("error"),
                    )
                    break
            run = get_run(run_id)
            active_stream_statuses = {"pending", "running", "waiting_approval", "cancelling"}
            if run and run.get("status") not in active_stream_statuses and not events:
                break
            time.sleep(0.25)
        yield "data: [DONE]\n\n"

    return run_id, generate()


def get_active_run(thread_id: str | None = None) -> list[dict[str, Any]]:
    return get_active_runs(thread_id=thread_id)


def list_events(run_id: str, after_sequence: int = 0) -> list[dict[str, Any]]:
    return list_run_events(run_id, after_sequence=after_sequence)


def enqueue_message(thread_id: str | None, repo: str, branch: str | None, content: str) -> dict[str, Any]:
    if not thread_id:
        raise ValueError("A thread id is required to queue a message.")
    if not content.strip():
        raise ValueError("Queued message must not be empty.")
    item = {
        "id": f"queue_{uuid.uuid4().hex[:12]}",
        "thread_id": thread_id,
        "repo": repo,
        "branch": branch,
        "content": content.strip(),
        "status": "queued",
        "created_at": _now_iso(),
    }
    with _COORDINATOR_LOCK:
        queue = _read_queue()
        queue.append(item)
        _write_queue(queue)
    record_event(event_type="queued_message_created", repo=repo, branch=branch, summary=summarize_user_message(content))
    return item


def list_queue(thread_id: str | None = None, repo: str | None = None, branch: str | None = None) -> list[dict[str, Any]]:
    queue = _read_queue()
    result: list[dict[str, Any]] = []
    for item in queue:
        if item.get("status") != "queued":
            continue
        if thread_id and item.get("thread_id") != thread_id:
            continue
        if repo and item.get("repo") != repo:
            continue
        if branch and item.get("branch") not in {branch, None, ""}:
            continue
        result.append(item)
    return result


def cancel_queued_message(queue_id: str) -> dict[str, Any]:
    with _COORDINATOR_LOCK:
        queue = _read_queue()
        for item in queue:
            if item.get("id") == queue_id and item.get("status") == "queued":
                item["status"] = "cancelled"
                item["completed_at"] = _now_iso()
                _write_queue(queue)
                record_event(
                    event_type="queued_message_cancelled",
                    repo=item.get("repo"),
                    branch=item.get("branch"),
                    summary=summarize_user_message(str(item.get("content") or "")),
                )
                return item
    raise ValueError("Queued message not found.")


def steer_run(run_id: str, *, content: str | None = None, queued_message_id: str | None = None) -> dict[str, Any]:
    run = get_run(run_id)
    if run is None:
        raise ValueError("Run not found.")
    if run.get("status") not in {"running", "cancelling"}:
        raise ValueError("Only running agent runs can accept steering.")
    steering_content = (content or "").strip()
    queue_item: dict[str, Any] | None = None
    if queued_message_id:
        queue_item = _queued_item(queued_message_id)
        if queue_item is None:
            raise ValueError("Queued message not found.")
        steering_content = str(queue_item.get("content") or "").strip()
    if not steering_content:
        raise ValueError("Steering content must not be empty.")
    meta = _append_steering(run_id, steering_content)
    if queue_item is not None:
        _mark_queue_item(queued_message_id or "", "steered")
    append_activity(
        run_id,
        phase="Planning",
        label="Received steering instruction",
        detail=summarize_user_message(steering_content, max_len=220),
        status="completed",
        event_type="steering_recorded",
    )
    return {"status": "recorded", "run_id": run_id, "steering": meta}


def cancel_run(run_id: str) -> dict[str, Any]:
    run = get_run(run_id)
    if run is None:
        raise ValueError("Run not found.")
    if run.get("status") not in {"running", "cancelling", "waiting_approval"}:
        return run
    _set_control_flag(run_id, "cancellation_requested", True)
    return request_run_cancellation(run_id)


def should_cancel(run_id: str) -> bool:
    control = _read_control(run_id)
    run = get_run(run_id) or {}
    return bool(control.get("cancellation_requested")) or run.get("status") in {"cancelled", "cancelling"}


def consume_steering(run_id: str) -> list[dict[str, Any]]:
    control = _read_control(run_id)
    steering = list(control.get("steering_instructions") or [])
    if steering:
        control["steering_instructions"] = []
        _write_control(run_id, control)
    return steering


def append_activity(
    run_id: str,
    *,
    request: AgentRunRequest | None = None,
    phase: str,
    label: str,
    status: str,
    event_type: str = "activity",
    detail: str = "",
    related_files: list[str] | None = None,
    related_command: list[str] | None = None,
    aggregate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stable_activity_id = f"{event_type}:{label.lower().replace(' ', '-')}"
    return append_run_event(
        run_id,
        {
            "id": f"{run_id}-{event_type}-{uuid.uuid4().hex[:8]}",
            "type": "progress_delta",
            "event_type": event_type,
            "activity_id": stable_activity_id,
            "phase": phase,
            "label": label,
            "detail": detail,
            "status": status,
            "thread_id": request.thread_id if request else None,
            "repo": request.project_path if request else None,
            "branch": request.branch if request else None,
            "related_files": related_files or [],
            "files": related_files or [],
            "related_command": related_command,
            "aggregate": aggregate,
            "started_at": _now_iso(),
            "ended_at": _now_iso() if status in {"completed", "failed"} else None,
        },
    )


def append_activity_started(run_id: str, **kwargs: Any) -> dict[str, Any]:
    return append_activity(run_id, event_type="activity_started", status="running", **kwargs)


def append_activity_update(run_id: str, **kwargs: Any) -> dict[str, Any]:
    return append_activity(run_id, event_type="activity_updated", **kwargs)


def append_activity_delta(run_id: str, **kwargs: Any) -> dict[str, Any]:
    return append_activity(run_id, event_type="activity_delta", **kwargs)


def append_activity_completed(run_id: str, **kwargs: Any) -> dict[str, Any]:
    return append_activity(run_id, event_type="activity_completed", status="completed", **kwargs)


def check_cancel(run_id: str) -> bool:
    return should_cancel(run_id)


def wait_for_approval(run_id: str, approval: dict[str, Any]) -> dict[str, Any]:
    meta = get_run(run_id) or {"id": run_id}
    meta["status"] = "waiting_approval"
    meta["pending_approval"] = approval
    _run_meta = get_repooperator_home_dir() / "runs" / run_id / "meta.json"
    try:
        _run_meta.write_text(json.dumps(meta, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    except OSError:
        pass
    append_activity(
        run_id,
        phase="Editing" if approval.get("kind") == "change_set_apply" else "Commands",
        label="Waiting for change-set approval" if approval.get("kind") == "change_set_apply" else "Waiting for command approval",
        status="waiting",
        event_type="approval_waiting",
        detail=str(approval.get("reason") or ("Change set requires approval." if approval.get("kind") == "change_set_apply" else "Command requires approval.")),
    )
    return meta


def resume_approval(run_id: str, decision: dict[str, Any]) -> dict[str, Any]:
    run = get_run(run_id)
    if run is None:
        raise ValueError("Run not found.")
    if run.get("status") != "waiting_approval":
        raise ValueError("Run is not waiting for approval.")
    pending = dict(run.get("pending_approval") or {})
    if pending.get("runtime") != "langgraph":
        raise ValueError("Run approval is not owned by the LangGraph runtime.")
    request_snapshot = run.get("request_snapshot") or pending.get("request_snapshot")
    if not isinstance(request_snapshot, dict):
        raise ValueError("Waiting approval run is missing its request snapshot.")
    request = AgentRunRequest(**request_snapshot)
    normalized = _normalize_resume_decision(decision)
    started = time.perf_counter()
    from repooperator_worker.agent_core.langgraph_runtime import resume_langgraph_controller

    response = resume_langgraph_controller(request, run_id=run_id, approval_decision=normalized).model_copy(update={"run_id": run_id})
    final_payload = _safe_final_result(response, run_id=run_id)
    _record_response_events(run_id, request, response)
    if _is_waiting_for_approval(response):
        wait_for_approval(run_id, _approval_resume_payload(response, request))
        terminal_status = "waiting_approval"
    else:
        terminal_status = "cancelled" if response.stop_reason == "cancelled" else "completed"
        complete_active_run(run_id=run_id, status=terminal_status, final_result=final_payload)
        if terminal_status == "completed":
            maybe_record_from_agent_run(request, response)
            update_thread_context(request, response)
            _drain_queue_after_run(request)
    record_agent_run(
        run_id=run_id,
        request=request,
        response=response,
        status=terminal_status,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
    return final_payload


def record_action_result(run_id: str, result: ActionResult | dict[str, Any]) -> dict[str, Any]:
    payload = json_safe(result.model_dump() if hasattr(result, "model_dump") else dict(result))
    return append_run_event(run_id, {"type": "action_result", "event_type": "action_result", "status": payload.get("status"), "result": payload})


def complete_run(run_id: str, *, status: str, final_result: dict[str, Any] | None = None, error: str | None = None) -> dict[str, Any]:
    return complete_active_run(run_id=run_id, status=status, final_result=json_safe(final_result), error=error)


def _is_waiting_for_approval(response: AgentRunResponse) -> bool:
    return response.response_type == "command_approval" or response.stop_reason == "waiting_approval"


def _approval_resume_payload(response: AgentRunResponse, request: AgentRunRequest) -> dict[str, Any]:
    payload = dict(response.command_approval or {})
    if response.change_set_proposal and response.stop_reason == "waiting_approval":
        payload.update(
            {
                "kind": "change_set_apply",
                "proposal_id": response.change_set_proposal.get("proposal_id"),
                "change_set_proposal": response.change_set_proposal,
                "reason": "Applying this validated change set will modify files and requires approval.",
            }
        )
    payload.update(
        {
            "runtime": "langgraph",
            "run_id": response.run_id,
            "thread_id": request.thread_id,
            "repo": request.project_path,
            "branch": request.branch,
            "request_snapshot": json_safe(request.model_dump(mode="json")),
        }
    )
    return json_safe(payload)


def _normalize_resume_decision(decision: dict[str, Any]) -> dict[str, Any]:
    raw = str(decision.get("decision") or decision.get("approval") or "").strip().lower()
    if raw in {"yes", "yes_session", "allow", "approved", "approve"}:
        normalized = "allow"
    else:
        normalized = "deny"
    return {**json_safe(decision), "decision": normalized}


def _safe_final_result(response: AgentRunResponse, *, run_id: str) -> dict[str, Any]:
    try:
        payload = safe_agent_response_payload(response)
        json.dumps(payload, ensure_ascii=False)
        return payload
    except Exception as exc:  # noqa: BLE001
        payload = json_safe(response)
        payload["response"] = (
            "The review completed, but RepoOperator hit an internal metadata serialization error. "
            "The readable summary is below...\n\n"
            + str(payload.get("response") or response.response)
        )
        append_run_event(
            run_id,
            {
                "type": "error",
                "event_type": "metadata_serialization_error",
                "status": "failed",
                "message": safe_repr(exc, limit=220),
            },
        )
        return json_safe(payload)


def _record_response_events(run_id: str, request: AgentRunRequest, response: AgentRunResponse) -> None:
    if _is_waiting_for_approval(response):
        phase = "Editing" if response.change_set_proposal else "Commands"
        label = "Waiting for change-set approval" if response.change_set_proposal else "Waiting for command approval"
        append_activity(
            run_id,
            request=request,
            phase=phase,
            label=label,
            detail=response.stop_reason or response.response_type,
            status="waiting",
            event_type="approval_waiting",
            related_command=(response.command_approval or {}).get("command") if response.command_approval else None,
        )
        return
    append_activity(
        run_id,
        request=request,
        phase="Finished",
        label="Completed task",
        detail=response.stop_reason or response.response_type,
        status="completed" if response.stop_reason != "cancelled" else "failed",
        event_type="run_completed" if response.stop_reason != "cancelled" else "run_cancelled",
    )
    for file_path in response.files_read:
        append_activity(
            run_id,
            request=request,
            phase="Reading",
            label=f"Read {file_path}",
            status="completed",
            event_type="file_read",
            related_files=[file_path],
        )
    for record in response.edit_archive:
        file_path = str(record.get("file_path") or "")
        append_activity(
            run_id,
            request=request,
            phase="Editing",
            label=f"{'Applied' if record.get('status') == 'applied' else 'Proposed'} {Path(file_path).name or file_path}",
            detail=f"+{record.get('additions', 0)} -{record.get('deletions', 0)}",
            status="completed",
            event_type="file_edit",
            related_files=[file_path] if file_path else [],
            aggregate={
                "action_type": "generate_edit",
                "edit_archive": [record],
                "additions": record.get("additions"),
                "deletions": record.get("deletions"),
                "diff_available": bool(record.get("diff")),
                "status": record.get("status"),
                "proposal_id": record.get("proposal_id"),
                "edit_summary": record.get("summary"),
            },
        )
    append_activity(
        run_id,
        request=request,
        phase="Finished",
        label="Completed task",
        detail=response.stop_reason or response.response_type,
        status="completed",
        event_type="run_completed",
    )


def _drain_queue_after_run(request: AgentRunRequest) -> None:
    next_item = None
    with _COORDINATOR_LOCK:
        queue = _read_queue()
        for item in queue:
            if (
                item.get("status") == "queued"
                and item.get("thread_id") == request.thread_id
                and item.get("repo") == request.project_path
                and item.get("branch") in {request.branch, None, ""}
            ):
                item["status"] = "running"
                item["started_at"] = _now_iso()
                next_item = dict(item)
                break
        _write_queue(queue)
    if next_item is None:
        return
    record_event(
        event_type="queued_message_started",
        repo=request.project_path,
        branch=request.branch,
        summary=summarize_user_message(str(next_item.get("content") or "")),
    )
    queued_request = request.model_copy(update={"task": str(next_item.get("content") or "")})
    try:
        start_run(queued_request, stream=False)
        _mark_queue_item(str(next_item.get("id")), "completed")
    except Exception as exc:  # noqa: BLE001
        _mark_queue_item(str(next_item.get("id")), "failed", error=str(exc))


def _queue_path() -> Path:
    path = get_repooperator_home_dir() / "threads"
    path.mkdir(parents=True, exist_ok=True)
    return path / "queue.json"


def _read_queue() -> list[dict[str, Any]]:
    path = _queue_path()
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return []
    return payload if isinstance(payload, list) else []


def _write_queue(queue: list[dict[str, Any]]) -> None:
    _queue_path().write_text(json.dumps(queue, ensure_ascii=False, indent=2), encoding="utf-8")


def _queued_item(queue_id: str) -> dict[str, Any] | None:
    for item in _read_queue():
        if item.get("id") == queue_id and item.get("status") == "queued":
            return item
    return None


def _mark_queue_item(queue_id: str, status: str, *, error: str | None = None) -> None:
    with _COORDINATOR_LOCK:
        queue = _read_queue()
        for item in queue:
            if item.get("id") == queue_id:
                item["status"] = status
                item["completed_at"] = _now_iso()
                if error:
                    item["error"] = error
        _write_queue(queue)


def _control_path(run_id: str) -> Path:
    safe = "".join(ch for ch in run_id if ch.isalnum() or ch in {"_", "-"})
    path = get_repooperator_home_dir() / "runs" / safe
    path.mkdir(parents=True, exist_ok=True)
    return path / "control.json"


def _read_control(run_id: str) -> dict[str, Any]:
    path = _control_path(run_id)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_control(run_id: str, control: dict[str, Any]) -> None:
    _control_path(run_id).write_text(json.dumps(control, ensure_ascii=False, indent=2), encoding="utf-8")


def _set_control_flag(run_id: str, key: str, value: Any) -> None:
    control = _read_control(run_id)
    control[key] = value
    _write_control(run_id, control)


def _append_steering(run_id: str, content: str) -> dict[str, Any]:
    control = _read_control(run_id)
    steering = list(control.get("steering_instructions") or [])
    item = {
        "id": f"steer_{uuid.uuid4().hex[:12]}",
        "content": summarize_user_message(content, max_len=500),
        "created_at": _now_iso(),
        "status": "recorded",
        "accepted": None,
        "parse_status": "pending",
        "reason": "Recorded for parsing at the next safe checkpoint.",
    }
    steering.append(item)
    control["steering_instructions"] = steering
    _write_control(run_id, control)
    return item


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
