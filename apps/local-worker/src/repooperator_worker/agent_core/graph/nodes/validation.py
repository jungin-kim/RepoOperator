"""Validation nodes for RepoOperator LangGraph."""

from __future__ import annotations

from typing import Any

from repooperator_worker.agent_core.graph.adapters import _execute_if_action_type, _execute_pending_action, _graph_transition_event, _latest_result, _merge_updates, _with_checkpoint_bump
from repooperator_worker.agent_core.graph.state import RepoOperatorGraphState
from repooperator_worker.agent_core.graph.nodes.apply import await_approval_node
from repooperator_worker.agent_core.understanding_context import append_visible_rationale, evidence_basis_update

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
