"""Change-set planning and edit subgraph nodes for RepoOperator LangGraph."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END

from repooperator_worker.agent_core.actions import ActionResult
from repooperator_worker.agent_core.change_set import (
    change_set_from_payload,
    plan_change_set,
    proposal_from_edit_result,
    validate_change_set as validate_change_set_model,
)
from repooperator_worker.agent_core.graph.adapters import (
    _execute_if_action_type,
    _graph_transition_event,
    _invoke_subgraph_delta,
    _latest_result,
    _pending_action,
    _request,
    _with_checkpoint_bump,
)
from repooperator_worker.agent_core.graph.state import RepoOperatorGraphState
from repooperator_worker.services.json_safe import json_safe

def plan_change_set_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    action = _pending_action(state)
    proposal = plan_change_set(
        list(action.target_files if action else []),
        action.reason_summary if action else "Plan proposal-only change set.",
    ).model_dump()
    proposal["action_id"] = action.action_id if action else None
    return _with_checkpoint_bump(
        {
            "change_set_proposal": proposal,
            "proposal_id": proposal.get("proposal_id"),
            "proposal_status": proposal.get("status"),
            "apply_status": "not_applied",
            "proposed_changes": [proposal],
            "routing_stage": "after_change_plan",
            "events_to_emit": [
                _graph_transition_event(
                    state,
                    "plan_change_set",
                    subgraph="edit_graph",
                    operation="plan_change_set",
                    files=list((proposal.get("plan") or {}).get("target_files") or []),
                )
            ],
        }
    )

def generate_change_set_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    from repooperator_worker.agent_core.graph.builder import build_edit_graph

    update = _invoke_subgraph_delta(build_edit_graph, state)
    update["routing_stage"] = "after_change_plan"
    update.setdefault("events_to_emit", []).append(
        _graph_transition_event(state, "generate_change_set", subgraph="edit_graph", operation="generate_change_set")
    )
    return _with_checkpoint_bump(update)

def validate_change_set_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    latest = _latest_result(state)
    proposal = _change_set_from_latest_result(state, latest) or state.get("change_set_proposal") or {}
    if isinstance(proposal, dict) and proposal.get("changes"):
        typed = change_set_from_payload(proposal)
        validation_model = validate_change_set_model(typed, repo=str(state.get("repo") or _request(state).project_path))
        typed.validation = validation_model
        typed.status = validation_model.status
        typed.validation_status = validation_model.status
        typed.proposal_error = "; ".join(validation_model.errors) if validation_model.errors else None
        proposal = typed.model_dump()
    validation = {
        "kind": "change_set",
        "status": (proposal.get("status") if proposal else None) or (latest.status if latest else "skipped"),
        "action_id": latest.action_id if latest else None,
        "proposal_files": [str(item.get("path")) for item in proposal.get("changes") or [] if isinstance(item, dict)],
        "errors": list((proposal.get("validation") or {}).get("errors") or []),
    }
    pending_approval = None
    stop_reason = state.get("stop_reason")
    final_response = state.get("final_response") or ""
    if validation["status"] == "valid" and proposal.get("changes") and not proposal.get("applied"):
        proposal_id = str(proposal.get("proposal_id") or "")
        pending_approval = {
            "kind": "change_set_apply",
            "proposal_id": proposal_id,
            "change_set_proposal": json_safe(proposal),
            "reason": "Applying this validated change set will modify files and requires approval.",
        }
        stop_reason = "waiting_approval"
        final_response = _final_text_for_change_set(state, proposal)
    return _with_checkpoint_bump(
        {
            "change_set_proposal": proposal,
            "pending_approval": pending_approval if pending_approval is not None else state.get("pending_approval"),
            "proposal_id": proposal.get("proposal_id") if isinstance(proposal, dict) else None,
            "proposal_status": validation["status"],
            "apply_status": "pending" if pending_approval else state.get("apply_status"),
            "stop_reason": stop_reason,
            "final_response": final_response,
            "validation_results": [validation],
            "proposal_errors": validation["errors"],
            "routing_stage": "after_change_plan",
            "events_to_emit": [
                _graph_transition_event(
                    state,
                    "validate_change_set",
                    subgraph="edit_graph",
                    operation="validate_change_set",
                    status=validation["status"],
                    files=validation["proposal_files"],
                    validation_result=validation,
                )
            ],
        }
    )

def repair_change_set_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    attempts = int(state.get("repair_attempts") or 0) + 1
    return _with_checkpoint_bump(
        {
            "repair_attempts": attempts,
            "risk_notes": ["Change-set repair requested after validation failed."],
            "routing_stage": "after_change_plan",
            "events_to_emit": [
                _graph_transition_event(state, "repair_change_set", subgraph="edit_graph", operation="repair_change_set", status="completed")
            ],
        }
    )

def edit_locate_targets_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    action = _pending_action(state)
    return {
        "events_to_emit": [
            _graph_transition_event(
                state,
                "locate_targets",
                subgraph="edit_graph",
                operation="locate_targets",
                files=list(action.target_files if action else []),
            )
        ]
    }

def route_edit_next_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    action = _pending_action(state)
    if not action:
        return {
            "edit_done": True,
            "events_to_emit": [_graph_transition_event(state, "route_edit_next", subgraph="edit_graph", operation="edit_complete")],
        }
    return {
        "events_to_emit": [
            _graph_transition_event(
                state,
                "route_edit_next",
                subgraph="edit_graph",
                operation="route_edit_next",
                action_type=action.type,
            )
        ]
    }

def route_edit_next(state: RepoOperatorGraphState) -> str:
    action = _pending_action(state)
    if not action:
        return END
    if action.type in {"generate_change_set", "generate_edit"}:
        if not state.get("change_set_proposal"):
            return "locate_targets"
        return "generate_change_set"
    return END

def edit_plan_change_set_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    action = _pending_action(state)
    proposal = plan_change_set(
        list(action.target_files if action else []),
        action.reason_summary if action else "Plan proposal-only change set.",
    ).model_dump()
    return {
        "change_set_proposal": proposal,
        "proposed_changes": [proposal],
        "events_to_emit": [_graph_transition_event(state, "plan_change_set", subgraph="edit_graph", operation="plan_change_set")],
    }

def edit_generate_change_set_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    return _execute_if_action_type(state, {"generate_change_set", "generate_edit"}, "edit_graph", "generate_change_set")

def edit_validate_change_set_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    latest = _latest_result(state)
    proposal = _change_set_from_latest_result(state, latest)
    validation = {
        "kind": "change_set",
        "status": (proposal.get("status") if proposal else None) or (latest.status if latest else "skipped"),
        "action_id": latest.action_id if latest else None,
        "errors": list((proposal.get("validation") or {}).get("errors") or []),
    }
    return {
        "change_set_proposal": proposal or state.get("change_set_proposal"),
        "validation_results": [validation],
        "proposal_errors": validation["errors"],
        "events_to_emit": [
            _graph_transition_event(state, "validate_change_set", subgraph="edit_graph", operation="validate_change_set", validation_result=validation)
        ],
    }

def edit_repair_change_set_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    attempts = int(state.get("repair_attempts") or 0) + 1
    return {
        "repair_attempts": attempts,
        "attempts": [{"kind": "repair", "attempt": attempts, "status": "blocked" if attempts > 1 else "queued"}],
        "events_to_emit": [_graph_transition_event(state, "repair_change_set", subgraph="edit_graph", operation="repair_change_set")]
    }

def route_edit_after_validation(state: RepoOperatorGraphState) -> str:
    proposal = state.get("change_set_proposal") or {}
    errors = list((proposal.get("validation") or {}).get("errors") or state.get("proposal_errors") or [])
    if errors and int(state.get("repair_attempts") or 0) < 1:
        return "repair_change_set"
    return END

def _change_set_from_latest_result(state: RepoOperatorGraphState, result: ActionResult | None) -> dict[str, Any] | None:
    if not result:
        return None
    if isinstance(result.payload.get("change_set_proposal"), dict):
        return json_safe(result.payload.get("change_set_proposal"))
    edit_proposals = result.payload.get("edit_proposals") or []
    if edit_proposals:
        plan_summary = str(((state.get("change_set_proposal") or {}).get("plan") or {}).get("summary") or "Prepare proposal-only edits.")
        return proposal_from_edit_result(edit_proposals, repo=str(state.get("repo") or _request(state).project_path), plan_summary=plan_summary).model_dump()
    if result.payload.get("proposal_error"):
        proposal = state.get("change_set_proposal") or plan_change_set([], "Prepare proposal-only edits.").model_dump()
        error = str(result.payload.get("proposal_error") or "")
        proposal.update({"status": "invalid", "proposal_error": error, "validation": {"status": "invalid", "errors": [error], "warnings": []}})
        return proposal
    proposal = state.get("change_set_proposal")
    if isinstance(proposal, dict) and proposal.get("changes"):
        typed = change_set_from_payload(proposal)
        validation = validate_change_set_model(typed, repo=str(state.get("repo") or _request(state).project_path))
        typed.validation = validation
        typed.status = validation.status
        typed.validation_status = validation.status
        proposal = typed.model_dump()
        return proposal
    return None

def _final_text_for_change_set(state: RepoOperatorGraphState, proposal: dict[str, Any]) -> str:
    del state
    changes = [item for item in proposal.get("changes") or [] if isinstance(item, dict)]
    files = [f"- {str(item.get('operation') or 'modify')}: `{str(item.get('path') or '')}`" for item in changes]
    validation = proposal.get("validation") if isinstance(proposal.get("validation"), dict) else {}
    validation_status = str(validation.get("status") or proposal.get("status") or "pending")
    return "\n".join(
        [
            "I prepared a ChangeSetProposal. No files were modified.",
            "",
            "Proposed files:",
            *(files or ["- No files"]),
            "",
            f"Validation result: {validation_status}.",
            "Review the diff and approve Apply changes to write it to disk.",
        ]
    )
