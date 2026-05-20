from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from repooperator_worker.agent_core.actions import AgentAction, ActionResult, new_action_id
from repooperator_worker.agent_core.planner import TaskFrame
from repooperator_worker.agent_core.request_understanding import RequestUnderstanding
from repooperator_worker.agent_core.state import AgentSubtask, ClassifierResult
from repooperator_worker.schemas import AgentRunRequest, AgentRunResponse
from repooperator_worker.services.json_safe import json_safe


def request_to_snapshot(request: AgentRunRequest | dict[str, Any]) -> dict[str, Any]:
    if isinstance(request, AgentRunRequest):
        return json_safe(request.model_dump(mode="json"))
    return json_safe(dict(request or {}))


def request_from_snapshot(snapshot: dict[str, Any] | AgentRunRequest) -> AgentRunRequest:
    if isinstance(snapshot, AgentRunRequest):
        return snapshot
    return AgentRunRequest(**dict(snapshot or {}))


def action_to_snapshot(action: AgentAction | dict[str, Any] | None) -> dict[str, Any] | None:
    if action is None:
        return None
    if isinstance(action, AgentAction):
        return json_safe(action.model_dump())
    return json_safe(dict(action or {}))


def action_from_snapshot(snapshot: dict[str, Any] | AgentAction | None) -> AgentAction | None:
    if snapshot is None:
        return None
    if isinstance(snapshot, AgentAction):
        return snapshot
    payload = dict(snapshot or {})
    return AgentAction(
        type=payload.get("type") or "final_answer",
        reason_summary=str(payload.get("reason_summary") or "Continue RepoOperator graph action."),
        action_id=str(payload.get("action_id") or new_action_id()),
        target_files=[str(item) for item in payload.get("target_files") or []],
        target_symbols=[str(item) for item in payload.get("target_symbols") or []],
        command=[str(item) for item in payload.get("command") or []] if payload.get("command") is not None else None,
        expected_output=payload.get("expected_output"),
        safety_requirements=[str(item) for item in payload.get("safety_requirements") or []],
        requires_approval=bool(payload.get("requires_approval") or False),
        created_at=str(payload.get("created_at") or ""),
        payload=json_safe(payload.get("payload") or {}),
    )


def result_to_snapshot(result: ActionResult | dict[str, Any] | None) -> dict[str, Any] | None:
    if result is None:
        return None
    if isinstance(result, ActionResult):
        return json_safe(result.model_dump())
    return json_safe(dict(result or {}))


def result_from_snapshot(snapshot: dict[str, Any] | ActionResult | None) -> ActionResult | None:
    if snapshot is None:
        return None
    if isinstance(snapshot, ActionResult):
        return snapshot
    payload = dict(snapshot or {})
    return ActionResult(
        action_id=str(payload.get("action_id") or ""),
        status=payload.get("status") or "skipped",
        observation=str(payload.get("observation") or ""),
        files_read=[str(item) for item in payload.get("files_read") or []],
        files_changed=[str(item) for item in payload.get("files_changed") or []],
        command_result=json_safe(payload.get("command_result")) if payload.get("command_result") is not None else None,
        edit_records=[json_safe(item) for item in payload.get("edit_records") or []],
        errors=[str(item) for item in payload.get("errors") or []],
        next_recommended_action=payload.get("next_recommended_action"),
        duration_ms=int(payload.get("duration_ms") or 0),
        payload=json_safe(payload.get("payload") or {}),
    )


def response_to_snapshot(response: AgentRunResponse | dict[str, Any] | None) -> dict[str, Any] | None:
    if response is None:
        return None
    if isinstance(response, AgentRunResponse):
        return json_safe(response.model_dump(mode="json"))
    return json_safe(dict(response or {}))


def response_from_snapshot(snapshot: dict[str, Any] | AgentRunResponse | None) -> AgentRunResponse | None:
    if snapshot is None:
        return None
    if isinstance(snapshot, AgentRunResponse):
        return snapshot
    return AgentRunResponse(**dict(snapshot or {}))


def classifier_to_snapshot(classifier: ClassifierResult | dict[str, Any] | None) -> dict[str, Any]:
    if classifier is None:
        return json_safe(asdict(ClassifierResult()))
    if isinstance(classifier, ClassifierResult):
        return json_safe(asdict(classifier))
    return json_safe(dict(classifier or {}))


def classifier_from_snapshot(snapshot: dict[str, Any] | ClassifierResult | None) -> ClassifierResult:
    if isinstance(snapshot, ClassifierResult):
        return snapshot
    payload = dict(snapshot or {})
    return ClassifierResult(
        intent=str(payload.get("intent") or "ambiguous"),
        confidence=float(payload.get("confidence") or 0.0),
        target_files=[str(item) for item in payload.get("target_files") or []],
        target_symbols=[str(item) for item in payload.get("target_symbols") or []],
        requested_action=str(payload.get("requested_action") or ""),
        needs_tool=payload.get("needs_tool"),
        needs_clarification=bool(payload.get("needs_clarification") or False),
        clarification_question=payload.get("clarification_question"),
        raw=json_safe(payload.get("raw") or {}),
    )


def request_understanding_to_snapshot(value: RequestUnderstanding | dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, RequestUnderstanding):
        return json_safe(value.model_dump())
    return json_safe(dict(value or {}))


def request_understanding_from_snapshot(value: dict[str, Any] | RequestUnderstanding | None) -> RequestUnderstanding | None:
    if value is None or isinstance(value, RequestUnderstanding):
        return value
    return RequestUnderstanding(**dict(value or {}))


def task_frame_to_snapshot(frame: TaskFrame | dict[str, Any] | None) -> dict[str, Any] | None:
    if frame is None:
        return None
    if isinstance(frame, TaskFrame):
        return json_safe(asdict(frame))
    return json_safe(dict(frame or {}))


def task_frame_from_snapshot(frame: dict[str, Any] | TaskFrame | None) -> TaskFrame | None:
    if frame is None or isinstance(frame, TaskFrame):
        return frame
    return TaskFrame(**dict(frame or {}))


def subtask_to_snapshot(subtask: AgentSubtask | dict[str, Any]) -> dict[str, Any]:
    if isinstance(subtask, AgentSubtask):
        return json_safe(asdict(subtask))
    return json_safe(dict(subtask or {}))


def subtask_from_snapshot(snapshot: dict[str, Any] | AgentSubtask) -> AgentSubtask:
    if isinstance(snapshot, AgentSubtask):
        return snapshot
    return AgentSubtask(**dict(snapshot or {}))


def graph_value_to_snapshot(value: Any) -> Any:
    if isinstance(value, AgentRunRequest):
        return request_to_snapshot(value)
    if isinstance(value, AgentRunResponse):
        return response_to_snapshot(value)
    if isinstance(value, AgentAction):
        return action_to_snapshot(value)
    if isinstance(value, ActionResult):
        return result_to_snapshot(value)
    if isinstance(value, ClassifierResult):
        return classifier_to_snapshot(value)
    if isinstance(value, RequestUnderstanding):
        return request_understanding_to_snapshot(value)
    if isinstance(value, TaskFrame):
        return task_frame_to_snapshot(value)
    if isinstance(value, AgentSubtask):
        return subtask_to_snapshot(value)
    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value))
    return json_safe(value)


def ensure_json_safe_graph_state(state: dict[str, Any]) -> dict[str, Any]:
    return {str(key): graph_value_to_snapshot(value) for key, value in dict(state or {}).items()}
