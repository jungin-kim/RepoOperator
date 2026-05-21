"""Adapters between LangGraph state and RepoOperator core services."""

from __future__ import annotations

import copy
from typing import Any

from repooperator_worker.agent_core.actions import AgentAction, ActionResult
from repooperator_worker.agent_core.graph_state import (
    action_from_snapshot,
    action_to_snapshot,
    classifier_from_snapshot,
    classifier_to_snapshot,
    request_from_snapshot,
    request_to_snapshot,
    request_understanding_from_snapshot,
    request_understanding_to_snapshot,
    result_from_snapshot,
    result_to_snapshot,
    subtask_from_snapshot,
    subtask_to_snapshot,
    task_frame_from_snapshot,
)
from repooperator_worker.agent_core.hooks import HookManager
from repooperator_worker.agent_core.state import AgentCoreState
from repooperator_worker.agent_core.tool_orchestrator import ToolOrchestrator
from repooperator_worker.agent_core.tools.registry import get_default_tool_registry
from repooperator_worker.agent_core.graph.state import APPEND_REDUCER_FIELDS, UNIQUE_APPEND_REDUCER_FIELDS, RepoOperatorGraphState
from repooperator_worker.agent_core.understanding_context import (
    append_visible_rationale,
    evidence_basis_update,
    rationale_basis_refs_for_action,
    rationale_safety_note_for_action,
    rationale_summary_for_action,
    rationale_uncertainty_for_action,
)
from repooperator_worker.schemas import AgentRunRequest
from repooperator_worker.services.event_service import append_run_event
from repooperator_worker.services.json_safe import json_safe

def _execute_if_action_type(state: RepoOperatorGraphState, action_types: set[str], subgraph: str, node_name: str) -> dict[str, Any]:
    action = _pending_action(state)
    if action and action.type in action_types:
        return _execute_pending_action(state, subgraph=subgraph, node_name=node_name)
    return {"events_to_emit": [_graph_transition_event(state, node_name, subgraph=subgraph, operation="skip")]}

def _execute_pending_action(state: RepoOperatorGraphState, *, subgraph: str | None, node_name: str | None = None) -> dict[str, Any]:
    action = _pending_action(state)
    if not action:
        return {
            "routing_stage": "after_tool_result",
            "events_to_emit": [_graph_transition_event(state, node_name or "execute_tool", subgraph=subgraph, operation="skip")],
        }
    request = _request(state)
    core = _core_state_from_graph(state)
    rationale_update: dict[str, Any] = {}
    if action.type != "final_answer":
        rationale_update = append_visible_rationale(
            state,
            node=node_name or "execute_tool",
            action=action,
            summary=rationale_summary_for_action(action, fallback="I am running the next visible action through the safe tool boundary."),
            basis_refs=rationale_basis_refs_for_action(action),
            safety_note=rationale_safety_note_for_action(action),
            uncertainty=rationale_uncertainty_for_action(action),
        )
    orchestrator = ToolOrchestrator(
        run_id=str(state.get("run_id") or "run_controller"),
        request=request,
        registry=get_default_tool_registry(),
        hook_manager=HookManager(),
    )
    result = orchestrator.execute_action(action)
    _append_action_event(str(state.get("run_id") or "run_controller"), action, result)
    core.actions_taken.append(action)
    core.action_results.append(result)
    _controller().observe_result(core, action, result, request)
    _controller().update_plan(core, action, result, request)
    _controller().check_cancel(core, request)
    update = _updates_from_core_after_action(state, core, action, result)
    operation = _action_operation(action.type)
    update["events_to_emit"] = [
        _graph_transition_event(
            state,
            node_name or "execute_tool",
            subgraph=subgraph,
            operation=operation,
            action_type=action.type,
            activity_id=f"action:{action.action_id}",
            status=result.status,
            files=list(result.files_read or result.files_changed or action.target_files),
            command=action.command,
            validation_result={"status": result.status, "errors": result.errors} if result.errors else None,
        )
    ]
    update["pending_action"] = None
    combined = _merge_updates(rationale_update, update)
    next_state = _merge_updates(dict(state), combined)
    return _merge_updates(combined, evidence_basis_update(next_state, trigger_node=node_name or "execute_tool"))

def _execute_ad_hoc_action(state: RepoOperatorGraphState, action: AgentAction, *, subgraph: str | None, node_name: str) -> dict[str, Any]:
    working = {**dict(state), "pending_action": action_to_snapshot(action)}
    return _execute_pending_action(working, subgraph=subgraph, node_name=node_name)

def _merge_updates(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if key in APPEND_REDUCER_FIELDS or key in UNIQUE_APPEND_REDUCER_FIELDS:
            existing = list(merged.get(key) or [])
            incoming = value if isinstance(value, list) else [value]
            merged[key] = [*existing, *incoming]
        elif key == "evidence_store" and isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged

def _invoke_subgraph_delta(builder: Any, state: RepoOperatorGraphState) -> dict[str, Any]:
    before = copy.copy(dict(state))
    after = builder().compile().invoke(state)
    return _delta_state(before, after)

def _delta_state(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    update: dict[str, Any] = {}
    append_fields = APPEND_REDUCER_FIELDS | UNIQUE_APPEND_REDUCER_FIELDS
    for key, value in after.items():
        if key in append_fields:
            old = before.get(key) or []
            new = value or []
            if len(new) > len(old):
                update[key] = list(new[len(old):])
            continue
        if before.get(key) != value:
            update[key] = value
    return update

def _updates_from_core(before: RepoOperatorGraphState, core: AgentCoreState) -> dict[str, Any]:
    update: dict[str, Any] = {
        "context_packet": core.context_packet,
        "request_understanding_snapshot": request_understanding_to_snapshot(core.request_understanding),
        "classifier_snapshot": classifier_to_snapshot(core.classifier_result),
        "plan": list(core.plan),
        "current_subtask_id": core.current_subtask_id,
        "pending_approval": core.pending_approval,
        "cancellation_requested": core.cancellation_requested,
        "skills_used": list(core.skills_used),
        "memories_used": list(core.memories_used),
        "recommendation_context": core.recommendation_context,
        "stop_reason": core.stop_reason,
        "final_response": core.final_response,
        "loop_iteration": core.loop_iteration,
        "max_loop_iterations": core.max_loop_iterations,
        "max_file_reads": core.max_file_reads,
        "max_commands": core.max_commands,
        "max_edits": core.max_edits,
        "subtasks": [subtask_to_snapshot(subtask) for subtask in core.subtasks],
        "current_step": core.current_step,
        "zero_result_queries": list(core.zero_result_queries),
        "failed_action_signatures": list(core.failed_action_signatures),
        "strategy_shifts": list(core.strategy_shifts),
    }
    for field_name in APPEND_REDUCER_FIELDS | UNIQUE_APPEND_REDUCER_FIELDS:
        if not hasattr(core, field_name):
            continue
        old = before.get(field_name) or []
        new = getattr(core, field_name) or []
        if len(new) > len(old):
            if field_name == "actions_taken":
                update[field_name] = [action_to_snapshot(item) for item in new[len(old):]]
            elif field_name == "action_results":
                update[field_name] = [result_to_snapshot(item) for item in new[len(old):]]
            else:
                update[field_name] = list(new[len(old):])
    return {key: value for key, value in update.items() if value is not None or key in {"pending_approval", "stop_reason"}}

def _updates_from_core_after_action(
    before: RepoOperatorGraphState,
    core: AgentCoreState,
    action: AgentAction,
    result: ActionResult,
) -> dict[str, Any]:
    update = _updates_from_core(before, core)
    update["actions_taken"] = [action_to_snapshot(action)]
    update["action_results"] = [result_to_snapshot(result)]
    if result.files_changed:
        update["files_changed"] = list(result.files_changed)
    if result.files_read:
        update["files_read"] = list(result.files_read)
    if result.status == "waiting_approval":
        decision = result.payload.get("permission_decision") if isinstance(result.payload, dict) else None
        metadata = decision.get("metadata") if isinstance(decision, dict) and isinstance(decision.get("metadata"), dict) else {}
        update["pending_approval"] = result.command_result or {
            "kind": action.type,
            "reason": result.observation,
            "approval_payload": metadata.get("approval_payload") or action.payload,
            "tool_name": action.type,
        }
    if result.status in {"cancelled", "timed_out"}:
        update["stop_reason"] = result.status
    return update

def _core_state_from_graph(state: RepoOperatorGraphState) -> AgentCoreState:
    request = _request(state)
    core = AgentCoreState(
        run_id=str(state.get("run_id") or "run_controller"),
        thread_id=state.get("thread_id"),
        repo=str(state.get("repo") or request.project_path),
        branch=state.get("branch"),
        user_task=request.task,
    )
    core.classifier_result = classifier_from_snapshot(state.get("classifier_snapshot") or state.get("classifier_result"))  # type: ignore[typeddict-item]
    core.request_understanding = request_understanding_from_snapshot(
        state.get("request_understanding_snapshot") or state.get("request_understanding")  # type: ignore[typeddict-item]
    )
    core.plan = list(state.get("plan") or [])
    core.current_step = state.get("current_step")
    core.observations = list(state.get("observations") or [])
    core.actions_taken = [action for action in (action_from_snapshot(item) for item in state.get("actions_taken") or []) if action]
    core.action_results = [result for result in (result_from_snapshot(item) for item in state.get("action_results") or []) if result]
    core.files_read = list(state.get("files_read") or [])
    core.files_changed = list(state.get("files_changed") or [])
    core.commands_run = list(state.get("commands_run") or [])
    core.pending_approval = state.get("pending_approval")
    core.cancellation_requested = bool(state.get("cancellation_requested") or False)
    core.skills_used = list(state.get("skills_used") or [])
    core.memories_used = list(state.get("memories_used") or [])
    core.recommendation_context = state.get("recommendation_context")
    core.context_packet = state.get("context_packet")
    core.stop_reason = state.get("stop_reason")
    core.final_response = str(state.get("final_response") or "")
    core.loop_iteration = int(state.get("loop_iteration") or 0)
    core.max_loop_iterations = int(state.get("max_loop_iterations") or (state.get("budgets") or {}).get("max_loop_iterations") or 8)
    core.max_file_reads = int(state.get("max_file_reads") or (state.get("budgets") or {}).get("max_file_reads") or 40)
    core.max_commands = int(state.get("max_commands") or (state.get("budgets") or {}).get("max_commands") or 8)
    core.max_edits = int(state.get("max_edits") or (state.get("budgets") or {}).get("max_edits") or 6)
    core.subtasks = [subtask_from_snapshot(item) for item in state.get("subtasks") or []]
    core.current_subtask_id = state.get("current_subtask_id")
    core.zero_result_queries = list(state.get("zero_result_queries") or [])
    core.failed_action_signatures = list(state.get("failed_action_signatures") or [])
    core.strategy_shifts = list(state.get("strategy_shifts") or [])
    return core

def _request(state: RepoOperatorGraphState) -> AgentRunRequest:
    snapshot = state.get("request_snapshot")
    if isinstance(snapshot, dict):
        return request_from_snapshot(snapshot)
    request = state.get("request")  # type: ignore[typeddict-item]
    if isinstance(request, AgentRunRequest):
        return request
    raise ValueError("RepoOperatorGraphState requires request_snapshot.")

def _pending_action(state: RepoOperatorGraphState) -> AgentAction | None:
    return action_from_snapshot(state.get("pending_action"))

def _task_frame(state: RepoOperatorGraphState) -> Any | None:
    return task_frame_from_snapshot(state.get("task_frame_snapshot") or state.get("task_frame"))  # type: ignore[typeddict-item]

def _latest_result(state: RepoOperatorGraphState) -> ActionResult | None:
    results = state.get("action_results") or []
    return result_from_snapshot(results[-1]) if results else None

def _graph_transition_event(
    state: RepoOperatorGraphState,
    node: str,
    *,
    subgraph: str | None = None,
    operation: str,
    action_type: str | None = None,
    activity_id: str | None = None,
    status: str = "completed",
    files: list[str] | None = None,
    command: list[str] | None = None,
    validation_result: dict[str, Any] | None = None,
    next_node: str | None = None,
    aggregate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    graph_metadata = {
        "graph_node": node,
        "subgraph": subgraph,
        "subtask_id": state.get("current_subtask_id"),
        "validation_result": validation_result,
        "change_set_summary": aggregate,
        "next_node": next_node,
    }
    from repooperator_worker.agent_core.events import activity_event

    event = activity_event(
        run_id=str(state.get("run_id") or "run_controller"),
        request=_request(state),
        activity_id=activity_id or f"graph:{node}",
        event_type="graph_transition",
        phase="Thinking",
        label=_graph_event_label(node, operation),
        status=status,
        visibility="debug",
        display="secondary",
        operation=operation,
        action_type=action_type,
        related_files=files or [],
        command=command,
        aggregate={key: value for key, value in graph_metadata.items() if value not in (None, [], {})},
    )
    event.update({key: value for key, value in graph_metadata.items() if value not in (None, [], {})})
    event = json_safe(event)
    _append_graph_event_safe(str(state.get("run_id") or "run_controller"), event)
    return event

def _graph_event_label(node: str, operation: str) -> str:
    labels = {
        "load_context": "Loaded runtime context",
        "capability_discovery": "Discovered capabilities",
        "context_pack": "Packed model context",
        "understand_request": "Framed request",
        "build_task_plan": "Built task plan",
        "route_next": "Selected next step",
        "supervisor": "Delegated bounded work",
        "gather_evidence": "Gathered evidence",
        "analysis_graph": "Analyzed repository evidence",
        "execute_tool": "Ran safe tool boundary",
        "validate_result": "Validated result",
        "plan_change_set": "Planned proposal",
        "generate_change_set": "Generated proposal",
        "validate_change_set": "Validated proposal",
        "repair_change_set": "Repaired proposal",
        "ask_clarification": "Prepared clarification",
        "await_approval": "Waiting for approval",
        "await_change_approval": "Waiting for approval",
        "apply_change_set": "Applied change set",
        "post_apply_validation": "Checked applied changes",
        "select_validation_commands": "Selected validation commands",
        "preview_command": "Previewed command",
        "await_validation_approval": "Waiting for validation approval",
        "run_validation_command": "Ran validation command",
        "parse_validation_result": "Parsed validation result",
        "web_research_graph": "Researched web evidence",
        "git_workflow_graph": "Prepared git workflow",
        "route_git_workflow": "Routed git workflow",
        "git_status": "Read git status",
        "git_diff": "Read git diff",
        "propose_commit_summary": "Prepared commit summary",
        "await_commit_approval": "Waiting for commit approval",
        "await_push_approval": "Waiting for push approval",
        "await_pr_approval": "Waiting for PR/MR approval",
        "routine_enqueue_node": "Checked routine enqueue",
        "decompose_task": "Decomposed task",
        "dispatch_work_units": "Dispatched work units",
        "reduce_work_reports": "Reduced work reports",
        "final_synthesis": "Built final response",
    }
    return labels.get(node, operation.replace("_", " ").title())

def _append_graph_event_safe(run_id: str, event: dict[str, Any]) -> None:
    try:
        append_run_event(run_id, event)
    except OSError:
        return

def _append_action_event(run_id: str, action: AgentAction, result: ActionResult) -> None:
    try:
        append_run_event(
            run_id,
            {
                "type": "action_result",
                "event_type": "action_result",
                "status": result.status,
                "action": action.model_dump(),
                "result": result.model_dump(),
            },
        )
    except OSError:
        return

def _action_operation(action_type: str) -> str:
    try:
        from repooperator_worker.agent_core.task_policy import action_operation

        return action_operation(action_type)
    except Exception:
        return action_type

def _with_checkpoint_bump(update: dict[str, Any]) -> dict[str, Any]:
    update["checkpoint_sequence"] = int(update.get("checkpoint_sequence") or 0) + 1
    return update

def _is_langgraph_checkpointer(value: Any) -> bool:
    return value is not None and hasattr(value, "put") and hasattr(value, "get_tuple")

def _controller() -> Any:
    from repooperator_worker.agent_core import controller_graph

    return controller_graph
