"""Approval and change-set apply nodes for RepoOperator LangGraph."""

from __future__ import annotations

import shlex
from typing import Any

from langgraph.types import interrupt

from repooperator_worker.agent_core.actions import AgentAction
from repooperator_worker.agent_core.graph.adapters import (
    _execute_pending_action,
    _graph_transition_event,
    _request,
    _with_checkpoint_bump,
)
from repooperator_worker.agent_core.graph.state import RepoOperatorGraphState
from repooperator_worker.agent_core.graph_state import action_to_snapshot, result_from_snapshot
from repooperator_worker.services.json_safe import json_safe

def await_approval_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    payload = _approval_interrupt_payload(state)
    decision = interrupt(payload)
    normalized = _normalize_approval_decision(decision)
    pending = state.get("pending_approval") or {}
    if pending.get("kind") == "change_set_apply" or payload.get("kind") == "change_set_apply":
        proposal_id = str(pending.get("proposal_id") or payload.get("proposal_id") or "")
        if normalized.get("decision") == "allow":
            return _with_checkpoint_bump(
                {
                    "pending_action": action_to_snapshot(
                        AgentAction(
                            type="apply_change_set",
                            reason_summary="Apply approved ChangeSetProposal.",
                            expected_output="Files written through approved change-set apply path.",
                            payload={
                                "proposal_id": proposal_id,
                                "approval_decision": normalized,
                                "change_set_snapshot": state.get("change_set_proposal"),
                            },
                        )
                    ),
                    "pending_approval": None,
                    "stop_reason": None,
                    "routing_stage": "after_interrupt_resume",
                    "edit_mode": "apply_approved",
                    "apply_status": "pending",
                    "approval_decision": normalized,
                    "events_to_emit": [
                        _graph_transition_event(
                            state,
                            "await_change_approval",
                            operation="approval_resume",
                            status="completed",
                            files=[str(item.get("path")) for item in (state.get("change_set_proposal") or {}).get("changes") or [] if isinstance(item, dict)],
                            aggregate={"proposal_id": proposal_id, "kind": "change_set_apply"},
                        )
                    ],
                }
            )
        proposal = dict(state.get("change_set_proposal") or {})
        if proposal:
            proposal.update({"status": "rejected", "apply_status": "rejected"})
        return _with_checkpoint_bump(
            {
                "stop_reason": "approval_denied",
                "final_response": "The change-set proposal was not applied. No files were modified.",
                "pending_approval": None,
                "change_set_proposal": proposal or state.get("change_set_proposal"),
                "proposal_status": "rejected",
                "apply_status": "rejected",
                "routing_stage": "after_approval",
                "approval_decision": normalized,
                "events_to_emit": [
                    _graph_transition_event(
                        state,
                        "await_change_approval",
                        operation="approval_gate",
                        status="completed",
                        aggregate={"proposal_id": proposal_id, "decision": "deny"},
                    )
                ],
            }
        )
    if pending.get("kind") in {"git_commit", "git_push", "github_create_pr", "gitlab_create_mr"} or payload.get("kind") in {"git_commit", "git_push", "github_create_pr", "gitlab_create_mr"}:
        kind = str(pending.get("kind") or payload.get("kind") or "")
        if normalized.get("decision") == "allow":
            action_type = kind
            action_payload = {**json_safe(pending.get("approval_payload") or {}), "approval_decision": normalized}
            if kind == "git_push":
                action_payload.update({"remote": pending.get("remote") or "origin", "branch": pending.get("branch") or state.get("branch") or "HEAD"})
            return _with_checkpoint_bump(
                {
                    "pending_action": action_to_snapshot(
                        AgentAction(
                            type=action_type,
                            reason_summary=f"Run {kind} after explicit approval.",
                            expected_output=f"{kind} result.",
                            payload=action_payload,
                        )
                    ),
                    "pending_approval": None,
                    "stop_reason": None,
                    "routing_stage": "after_interrupt_resume",
                    "approval_decision": normalized,
                    "events_to_emit": [
                        _graph_transition_event(
                            state,
                            "await_approval",
                            operation="approval_resume",
                            status="completed",
                            aggregate={"kind": kind, "decision": "allow"},
                        )
                    ],
                }
            )
        return _with_checkpoint_bump(
            {
                "stop_reason": "approval_denied",
                "final_response": f"I did not run {kind} because approval was denied. No git write was performed.",
                "pending_approval": None,
                "routing_stage": "after_approval",
                "approval_decision": normalized,
                "events_to_emit": [
                    _graph_transition_event(
                        state,
                        "await_approval",
                        operation="approval_gate",
                        status="completed",
                        aggregate={"kind": kind, "decision": "deny"},
                    )
                ],
            }
        )
    if pending.get("kind") in {"search_web", "fetch_url"} or payload.get("kind") in {"search_web", "fetch_url"}:
        kind = str(pending.get("kind") or payload.get("kind") or "")
        if normalized.get("decision") == "allow":
            return _with_checkpoint_bump(
                {
                    "pending_action": action_to_snapshot(
                        AgentAction(
                            type=kind,
                            reason_summary=f"Run {kind} after explicit network approval.",
                            expected_output="Untrusted web evidence with source metadata.",
                            payload={**json_safe(pending.get("approval_payload") or {}), "approval_decision": normalized},
                        )
                    ),
                    "pending_approval": None,
                    "stop_reason": None,
                    "routing_stage": "after_interrupt_resume",
                    "approval_decision": normalized,
                    "events_to_emit": [_graph_transition_event(state, "await_approval", operation="approval_resume", status="completed", aggregate={"kind": kind, "decision": "allow"})],
                }
            )
        return _with_checkpoint_bump(
            {
                "stop_reason": "approval_denied",
                "final_response": f"I did not run {kind} because network approval was denied.",
                "pending_approval": None,
                "routing_stage": "after_approval",
                "approval_decision": normalized,
                "events_to_emit": [_graph_transition_event(state, "await_approval", operation="approval_gate", status="completed", aggregate={"kind": kind, "decision": "deny"})],
            }
        )
    if normalized.get("decision") == "allow":
        command = list((state.get("pending_approval") or {}).get("command") or payload.get("command") or [])
        approval_id = str((state.get("pending_approval") or {}).get("approval_id") or payload.get("approval_id") or "")
        return _with_checkpoint_bump(
            {
                "pending_action": action_to_snapshot(
                    AgentAction(
                        type="run_approved_command",
                        reason_summary="Run command after user approval.",
                        command=command,
                        expected_output="Command output after approval.",
                        payload={"approval_id": approval_id, "approval_decision": normalized},
                    )
                ),
                "pending_approval": None,
                "stop_reason": None,
                "routing_stage": "after_interrupt_resume",
                "approval_decision": normalized,
                "events_to_emit": [
                    _graph_transition_event(
                        state,
                        "await_approval",
                        operation="approval_resume",
                        status="completed",
                        command=command,
                    )
                ],
            }
        )
    final_response = "I did not run the command because approval was denied. No command was executed."
    return _with_checkpoint_bump(
        {
            "stop_reason": "approval_denied",
            "final_response": final_response,
            "pending_approval": None,
            "routing_stage": "after_approval",
            "approval_decision": normalized,
            "events_to_emit": [
                _graph_transition_event(
                    state,
                    "await_approval",
                    operation="approval_gate",
                    status="completed",
                    command=payload.get("command"),
                )
            ],
        }
    )

def apply_change_set_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    update = _execute_pending_action(state, subgraph=None, node_name="apply_change_set")
    result = result_from_snapshot((update.get("action_results") or [None])[-1])
    payload = result.payload if result else {}
    proposal = payload.get("change_set_proposal") if isinstance(payload.get("change_set_proposal"), dict) else state.get("change_set_proposal")
    files_changed = list(payload.get("files_modified") or []) + list(payload.get("files_created") or []) + list(payload.get("files_deleted") or [])
    for item in payload.get("files_renamed") or []:
        if isinstance(item, dict) and item.get("to"):
            files_changed.append(str(item.get("to")))
    applied = bool(payload.get("applied"))
    update.update(
        {
            "change_set_proposal": proposal,
            "files_changed": files_changed,
            "edit_mode": "applied" if applied else "blocked",
            "apply_status": "applied" if applied else "failed",
            "proposal_status": "applied" if applied else "valid",
            "applied_change_set_id": payload.get("applied_change_set_id"),
            "stop_reason": None if applied else "failed",
            "routing_stage": "after_tool_result",
            "final_response": _final_text_for_applied_change_set(state, payload) if applied else _final_text_for_failed_apply(payload),
        }
    )
    return _with_checkpoint_bump(update)

def post_apply_validation_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    status = "not_run"
    if state.get("apply_status") == "applied":
        status = "skipped_no_safe_command_selected"
    proposal = dict(state.get("change_set_proposal") or {})
    if proposal:
        proposal["post_apply_validation_status"] = status
    return _with_checkpoint_bump(
        {
            "post_apply_validation_status": status,
            "change_set_proposal": proposal or state.get("change_set_proposal"),
            "validation_results": [{"kind": "post_apply", "status": status, "errors": []}],
            "routing_stage": "after_apply",
            "events_to_emit": [
                _graph_transition_event(
                    state,
                    "post_apply_validation",
                    operation="post_apply_validation",
                    status="completed",
                    validation_result={"status": status, "errors": []},
                )
            ],
        }
    )

def _approval_interrupt_payload(state: RepoOperatorGraphState) -> dict[str, Any]:
    approval = state.get("pending_approval") or {}
    if approval.get("kind") == "change_set_apply":
        proposal = state.get("change_set_proposal") if isinstance(state.get("change_set_proposal"), dict) else approval.get("change_set_proposal")
        files = [str(item.get("path")) for item in (proposal or {}).get("changes") or [] if isinstance(item, dict)]
        proposal_id = str(approval.get("proposal_id") or (proposal or {}).get("proposal_id") or "")
        return json_safe(
            {
                "kind": "change_set_apply",
                "run_id": state.get("run_id"),
                "thread_id": state.get("thread_id"),
                "proposal_id": proposal_id,
                "change_set_proposal": proposal,
                "files": files,
                "risk": approval.get("reason") or "Applying this proposal modifies files and requires approval.",
                "resume_token": f"{state.get('run_id')}:change_set_apply:{proposal_id}",
            }
        )
    if approval.get("kind") in {"git_commit", "git_push", "github_create_pr", "gitlab_create_mr"}:
        kind = str(approval.get("kind"))
        return json_safe(
            {
                "kind": kind,
                "run_id": state.get("run_id"),
                "thread_id": state.get("thread_id"),
                "files": approval.get("files") or [],
                "message": approval.get("message"),
                "remote": approval.get("remote"),
                "branch": approval.get("branch") or state.get("branch"),
                "target_branch": approval.get("target_branch"),
                "risk": approval.get("reason") or f"{kind} requires explicit approval.",
                "resume_token": f"{state.get('run_id')}:{kind}:{approval.get('branch') or approval.get('message') or ''}",
            }
        )
    if approval.get("kind") in {"search_web", "fetch_url"}:
        kind = str(approval.get("kind"))
        return json_safe(
            {
                "kind": kind,
                "run_id": state.get("run_id"),
                "thread_id": state.get("thread_id"),
                "files": [],
                "risk": approval.get("reason") or "Network access requires approval.",
                "approval_payload": approval.get("approval_payload") or {},
                "resume_token": f"{state.get('run_id')}:{kind}",
            }
        )
    command = list(approval.get("command") or [])
    return json_safe(
        {
            "kind": "command_approval",
            "run_id": state.get("run_id"),
            "thread_id": state.get("thread_id"),
            "command": command,
            "approval_id": approval.get("approval_id"),
            "files": [],
            "risk": approval.get("reason") or approval.get("risk") or "Command requires approval before execution.",
            "resume_token": f"{state.get('run_id')}:command:{approval.get('approval_id') or shlex.join(command)}",
        }
    )

def _normalize_approval_decision(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        decision = str(value.get("decision") or value.get("approval") or value.get("action") or "").strip().lower()
        if decision in {"allow", "approved", "approve", "yes"}:
            return {**json_safe(value), "decision": "allow"}
        return {**json_safe(value), "decision": "deny"}
    if str(value).strip().lower() in {"allow", "approved", "approve", "yes", "true"}:
        return {"decision": "allow"}
    return {"decision": "deny"}

def _final_text_for_applied_change_set(state: RepoOperatorGraphState, payload: dict[str, Any]) -> str:
    del state
    modified = [str(item) for item in payload.get("files_modified") or []]
    created = [str(item) for item in payload.get("files_created") or []]
    deleted = [str(item) for item in payload.get("files_deleted") or []]
    renamed = [f"{item.get('from')} -> {item.get('to')}" for item in payload.get("files_renamed") or [] if isinstance(item, dict)]
    lines = ["Applied the approved ChangeSetProposal. Files were modified."]
    if modified:
        lines.append("Modified: " + ", ".join(f"`{path}`" for path in modified))
    if created:
        lines.append("Created: " + ", ".join(f"`{path}`" for path in created))
    if deleted:
        lines.append("Deleted: " + ", ".join(f"`{path}`" for path in deleted))
    if renamed:
        lines.append("Renamed: " + ", ".join(f"`{path}`" for path in renamed))
    validation = payload.get("validation_result") if isinstance(payload.get("validation_result"), dict) else {}
    lines.append(f"Validation result: {validation.get('status') or 'valid'} before apply.")
    return "\n".join(lines)

def _final_text_for_failed_apply(payload: dict[str, Any]) -> str:
    errors = "; ".join(str(item) for item in payload.get("errors") or []) or "unknown apply error"
    return f"The approved ChangeSetProposal could not be applied. No success was recorded. Error: {errors}"
