"""Validation nodes for RepoOperator LangGraph."""

from __future__ import annotations

import shlex
from typing import Any

from repooperator_worker.agent_core.actions import AgentAction
from repooperator_worker.agent_core.graph.adapters import _execute_if_action_type, _execute_pending_action, _graph_transition_event, _latest_result, _merge_updates, _request, _with_checkpoint_bump
from repooperator_worker.agent_core.graph.state import RepoOperatorGraphState
from repooperator_worker.agent_core.graph_state import action_to_snapshot, result_from_snapshot
from repooperator_worker.agent_core.graph.nodes.apply import await_approval_node
from repooperator_worker.agent_core.understanding_context import append_visible_rationale, evidence_basis_update
from repooperator_worker.agent_core.validation_selector import ValidationCommandSelector
from repooperator_worker.services.permissions_service import permission_profile

def execute_tool_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    update = _execute_pending_action(state, subgraph=None)
    update["routing_stage"] = "after_tool_result"
    return _with_checkpoint_bump(update)

def validate_result_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    latest = _latest_result(state)
    validation: dict[str, Any] = {
        "status": latest.status if latest else "skipped",
        "action_id": latest.action_id if latest else None,
        "errors": list(latest.errors if latest else []),
    }
    stop_reason = state.get("stop_reason")
    if latest and latest.status == "waiting_approval":
        stop_reason = "waiting_approval"
    elif latest and latest.status in {"cancelled", "timed_out"}:
        stop_reason = latest.status
    update = {
            "validation_results": [validation],
            "stop_reason": stop_reason,
            "routing_stage": "after_validation",
            "events_to_emit": [
                _graph_transition_event(
                    state,
                    "validate_result",
                    subgraph="validation_graph",
                    operation="validate_result",
                    status=validation["status"],
                    validation_result=validation,
                )
            ],
        }
    next_state = _merge_updates(dict(state), update)
    update = _merge_updates(update, evidence_basis_update(next_state, trigger_node="validate_result"))
    update = _merge_updates(
        update,
        append_visible_rationale(
            next_state,
            node="validate_result",
            action=None,
            summary="I recorded the tool result status before deciding whether to continue, ask approval, or synthesize the answer.",
            basis_refs=[{"kind": "validation", "id": "validation:latest_result"}],
            safety_note=None,
            uncertainty=validation["errors"],
        ),
    )
    return _with_checkpoint_bump(update)

def validation_choose_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    return {"events_to_emit": [_graph_transition_event(state, "choose_validation", subgraph="validation_graph", operation="choose_validation")]}

def validation_preview_command_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    return _execute_if_action_type(state, {"preview_command", "inspect_git_state"}, "validation_graph", "preview_command")

def validation_approval_interrupt_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    if state.get("pending_approval"):
        return await_approval_node(state)
    return {
        "events_to_emit": [
            _graph_transition_event(state, "approval_interrupt_if_needed", subgraph="validation_graph", operation="approval_not_needed")
        ]
    }

def validation_run_safe_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    return _execute_if_action_type(state, {"run_approved_command"}, "validation_graph", "run_safe_validation")

def validation_parse_errors_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    latest = _latest_result(state)
    errors = list(latest.errors if latest else [])
    update = {
        "validation_results": [{"kind": "command", "status": latest.status if latest else "skipped", "errors": errors}],
        "events_to_emit": [_graph_transition_event(state, "parse_errors", subgraph="validation_graph", operation="parse_errors")],
    }
    next_state = _merge_updates(dict(state), update)
    return _merge_updates(update, evidence_basis_update(next_state, trigger_node="parse_errors"))

def validation_update_result_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    return {
        "validation_done": True,
        "events_to_emit": [_graph_transition_event(state, "update_validation_result", subgraph="validation_graph", operation="update_validation_result")]
    }

def validation_route_next_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    return {
        "events_to_emit": [_graph_transition_event(state, "route_validation_next", subgraph="validation_graph", operation="route_validation_next")]
    }

def select_validation_commands_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    if state.get("apply_status") != "applied":
        return _post_apply_validation_status_update(state, "not_run", "No applied change set was available for post-apply validation.")

    try:
        profile = permission_profile()
        permission_mode = str(profile.get("mode") or "default")
    except Exception:
        permission_mode = "default"
    selection = ValidationCommandSelector().select(
        project_path=str(state.get("repo") or _request(state).project_path),
        changed_files=[str(path) for path in state.get("files_changed") or []],
        user_request=_request(state).task,
        permission_mode=permission_mode,
    )
    payload = selection.model_dump()
    selected = payload.get("selected") if isinstance(payload.get("selected"), dict) else None
    status = "selected" if selected else "skipped_no_validation_command"
    update: dict[str, Any] = {
        "validation_command_selection": payload,
        "post_apply_validation_status": status,
        "routing_stage": "after_apply",
        "events_to_emit": [
            _graph_transition_event(
                state,
                "select_validation_commands",
                operation="select_validation_commands",
                status="completed",
                files=list(state.get("files_changed") or []),
                validation_result={"status": status, "candidate_count": len(payload.get("candidates") or []), "selected": selected},
            )
        ],
    }
    if selected:
        command = [str(part) for part in selected.get("command") or []]
        update["pending_action"] = action_to_snapshot(
            AgentAction(
                type="preview_command",
                reason_summary=str(selected.get("reason") or "Preview selected validation command before execution."),
                command=command,
                expected_output="Validation command safety classification.",
                payload={"validation_candidate": selected, "reason_summary": str(selected.get("reason") or "")},
            )
        )
    else:
        update["validation_results"] = [{"kind": "post_apply_validation", "status": status, "errors": [], "selection": payload}]
    next_state = _merge_updates(dict(state), update)
    update = _merge_updates(update, evidence_basis_update(next_state, trigger_node="select_validation_commands"))
    update = _merge_updates(
        update,
        append_visible_rationale(
            next_state,
            node="select_validation_commands",
            action=None,
            summary=selection.reason,
            basis_refs=[{"kind": "file", "path": path} for path in selection.changed_files],
            safety_note="Validation commands are selected before execution and still pass through command policy preview.",
            uncertainty=[] if selected else [selection.reason],
        ),
    )
    return _with_checkpoint_bump(update)

def preview_selected_validation_command_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    update = _execute_if_action_type(state, {"preview_command"}, None, "preview_command")
    latest = result_from_snapshot((update.get("action_results") or [None])[-1])
    selection = dict(state.get("validation_command_selection") or {})
    selected = dict(selection.get("selected") or {})
    if not latest:
        return update
    command_result = latest.command_result or {}
    selected.update({"preview_result": command_result, "preview_status": latest.status})
    selection["selected"] = selected
    if latest.status == "waiting_approval":
        pending = dict(update.get("pending_approval") or command_result)
        pending.update(
            {
                "kind": "validation_command_approval",
                "tool_name": "run_validation_command",
                "validation_candidate": selected,
                "reason": command_result.get("reason") or latest.observation or "Validation command requires approval.",
            }
        )
        update.update(
            {
                "pending_approval": pending,
                "validation_command_selection": selection,
                "post_apply_validation_status": "waiting_approval",
                "stop_reason": "waiting_approval",
            }
        )
        return update
    if latest.status == "success" and command_result.get("command"):
        command = [str(part) for part in command_result.get("command") or []]
        update.update(
            {
                "validation_command_selection": selection,
                "pending_action": action_to_snapshot(
                    AgentAction(
                        type="run_validation_command",
                        reason_summary=f"Run selected validation command `{shlex.join(command)}`.",
                        command=command,
                        expected_output="Post-apply validation output.",
                        payload={
                            "approval_id": command_result.get("approval_id"),
                            "validation_candidate": selected,
                            "reason_summary": selected.get("reason") or command_result.get("reason") or "",
                        },
                    )
                ),
            }
        )
        return update
    update.update(
        {
            "validation_command_selection": selection,
            "post_apply_validation_status": "failed",
            "stop_reason": "failed",
        }
    )
    return update

def await_validation_approval_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    if state.get("pending_approval"):
        return await_approval_node(state)
    return {
        "events_to_emit": [
            _graph_transition_event(state, "await_validation_approval", operation="approval_not_needed")
        ]
    }

def run_selected_validation_command_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    if state.get("stop_reason") == "approval_denied":
        return _post_apply_validation_status_update(state, "approval_denied", "Validation command approval was denied.")
    return _execute_if_action_type(state, {"run_validation_command"}, None, "run_validation_command")

def parse_validation_result_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    latest = _latest_result(state)
    selection = dict(state.get("validation_command_selection") or {})
    selected = dict(selection.get("selected") or {})
    command_source = selected.get("command")
    if not command_source and latest and latest.command_result:
        command_source = latest.command_result.get("command")
    command = [str(part) for part in command_source or []]
    errors: list[str] = []
    output = ""
    if latest and latest.command_result:
        output = str(latest.command_result.get("stderr") or latest.command_result.get("stdout") or latest.observation or "")
    elif latest:
        output = latest.observation
    if latest and latest.status == "failed":
        errors.append(output or "Validation command failed.")
    status = _validation_status_from_latest(latest, state, bool(selected))
    result = {
        "kind": "post_apply_validation",
        "status": status,
        "command": command,
        "display_command": shlex.join(command) if command else selected.get("display_command"),
        "candidate": selected or None,
        "candidate_commands": selection.get("candidates") or [],
        "errors": errors,
        "output": output[-4000:] if output else "",
    }
    proposal = dict(state.get("change_set_proposal") or {})
    if proposal:
        proposal["post_apply_validation_status"] = status
    final_response = _final_response_with_validation_status(str(state.get("final_response") or ""), result)
    update = {
        "validation_results": [result],
        "validation_command_selection": selection,
        "post_apply_validation_status": status,
        "change_set_proposal": proposal or state.get("change_set_proposal"),
        "validation_done": True,
        "routing_stage": "after_validation",
        "final_response": final_response,
        "stop_reason": "failed" if status == "failed" else (None if state.get("stop_reason") != "approval_denied" else "approval_denied"),
        "events_to_emit": [
            _graph_transition_event(
                state,
                "parse_validation_result",
                operation="parse_validation_result",
                status="completed" if status != "failed" else "failed",
                command=command or None,
                validation_result=result,
            )
        ],
    }
    next_state = _merge_updates(dict(state), update)
    update = _merge_updates(update, evidence_basis_update(next_state, trigger_node="parse_validation_result"))
    return _with_checkpoint_bump(update)

def _post_apply_validation_status_update(state: RepoOperatorGraphState, status: str, reason: str) -> dict[str, Any]:
    proposal = dict(state.get("change_set_proposal") or {})
    if proposal:
        proposal["post_apply_validation_status"] = status
    result = {"kind": "post_apply_validation", "status": status, "errors": [], "reason": reason}
    return _with_checkpoint_bump(
        {
            "post_apply_validation_status": status,
            "change_set_proposal": proposal or state.get("change_set_proposal"),
            "validation_results": [result],
            "validation_done": True,
            "routing_stage": "after_validation",
            "events_to_emit": [
                _graph_transition_event(
                    state,
                    "select_validation_commands",
                    operation="select_validation_commands",
                    validation_result=result,
                )
            ],
        }
    )

def _validation_status_from_latest(latest: Any, state: RepoOperatorGraphState, had_selected_command: bool) -> str:
    if state.get("stop_reason") == "approval_denied":
        return "approval_denied"
    if not had_selected_command:
        return "skipped_no_validation_command"
    if latest is None:
        return "not_run"
    if latest.status == "success":
        return "passed"
    if latest.status == "failed":
        return "failed"
    if latest.status == "waiting_approval":
        return "waiting_approval"
    return str(latest.status or "not_run")

def _final_response_with_validation_status(existing: str, result: dict[str, Any]) -> str:
    command = str(result.get("display_command") or "").strip()
    status = str(result.get("status") or "not_run")
    if status == "passed":
        line = f"Post-apply validation passed with `{command}`." if command else "Post-apply validation passed."
    elif status == "failed":
        detail = "; ".join(str(item) for item in result.get("errors") or []) or "validation command failed"
        line = f"Post-apply validation failed with `{command}`: {detail}" if command else f"Post-apply validation failed: {detail}"
    elif status == "approval_denied":
        line = "Post-apply validation was not run because command approval was denied."
    elif status == "skipped_no_validation_command":
        line = "Post-apply validation was skipped because no candidate command was selected."
    else:
        line = f"Post-apply validation status: {status}."
    if line in existing:
        return existing
    return (existing.rstrip() + "\n" + line).strip() if existing.strip() else line
