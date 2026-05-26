"""Routing helpers for the RepoOperator LangGraph topology."""

from __future__ import annotations

import time
from typing import Any

from repooperator_worker.agent_core.graph_routes import choose_graph_next_action
from repooperator_worker.agent_core.graph_state import action_to_snapshot, task_frame_to_snapshot
from repooperator_worker.agent_core.graph.adapters import (
    _core_state_from_graph,
    _graph_transition_event,
    _latest_result,
    _pending_action,
    _request,
    _updates_from_core,
    _with_checkpoint_bump,
)
from repooperator_worker.agent_core.graph.support import check_cancel, emit_action_decision, should_continue
from repooperator_worker.agent_core.graph.nodes.git import _git_workflow_requested
from repooperator_worker.agent_core.graph.nodes.supervisor import _should_use_supervisor
from repooperator_worker.agent_core.graph.nodes.web import _has_web_evidence, _web_research_available, _web_research_needed
from repooperator_worker.agent_core.graph.state import RepoOperatorGraphState
from repooperator_worker.agent_core.planner import build_task_frame
from repooperator_worker.agent_core.understanding_context import (
    append_visible_rationale,
    rationale_basis_refs_for_action,
    rationale_safety_note_for_action,
    rationale_summary_for_action,
    rationale_uncertainty_for_action,
)

def route_next_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    request = _request(state)
    core = _core_state_from_graph(state)
    existing_action = _pending_action(state)
    if existing_action:
        route = route_by_stage(state)
        update = {
            "next_node": route,
            "events_to_emit": [
                _graph_transition_event(
                    state,
                    "route_next",
                    operation="route_existing_action",
                    action_type=existing_action.type,
                    next_node=route,
                )
            ],
        }
        update.update(
            append_visible_rationale(
                state,
                node="route_next",
                action=existing_action,
                summary=rationale_summary_for_action(existing_action, fallback=f"I am continuing the pending {existing_action.type} action via {route}."),
                basis_refs=rationale_basis_refs_for_action(existing_action),
                safety_note=rationale_safety_note_for_action(existing_action),
                uncertainty=rationale_uncertainty_for_action(existing_action),
            )
        )
        return _with_checkpoint_bump(update)
    if state.get("stop_reason") == "waiting_approval":
        return {"next_node": "await_approval", "events_to_emit": [_graph_transition_event(state, "route_next", operation="approval_gate")]}
    if state.get("stop_reason") in {"needs_clarification"}:
        return {"next_node": "ask_clarification", "events_to_emit": [_graph_transition_event(state, "route_next", operation="clarification")]}
    if state.get("stop_reason") in {"approval_denied"}:
        return {"next_node": "final_synthesis", "events_to_emit": [_graph_transition_event(state, "route_next", operation="approval_denied")]}

    can_continue = should_continue(
        core,
        request=request,
        started=float(state.get("graph_started_at") or time.perf_counter()),
        max_wall_clock_seconds=300,
    )
    update = _updates_from_core(state, core)
    if not can_continue:
        update.update({"next_node": "final_synthesis", "pending_action": None})
        update["events_to_emit"] = [_graph_transition_event(state, "route_next", operation="stop_budget")]
        update.update(
            append_visible_rationale(
                state,
                node="route_next",
                action=None,
                summary="I reached a run boundary, so I am stopping tool selection and moving to final synthesis.",
                basis_refs=[],
                safety_note="Stopping does not bypass approval or tool safety.",
                uncertainty=[],
            )
        )
        return _with_checkpoint_bump(update)

    check_cancel(core, request)
    if core.cancellation_requested:
        update = _updates_from_core(state, core)
        update.update({"next_node": "final_synthesis", "pending_action": None})
        update["events_to_emit"] = [_graph_transition_event(state, "route_next", operation="cancelled")]
        return _with_checkpoint_bump(update)

    if _git_workflow_requested(state) and not state.get("git_workflow") and not state.get("pending_action"):
        update.update({"next_node": "git_workflow_graph", "pending_action": None})
        update["events_to_emit"] = [_graph_transition_event(state, "route_next", operation="route_git_workflow", next_node="git_workflow_graph")]
        update.update(
            append_visible_rationale(
                state,
                node="route_next",
                action=None,
                summary="The user explicitly asked for a git workflow, so I am starting with read-only status and diff before any approval-gated write.",
                basis_refs=[],
                safety_note="Git writes remain behind explicit approval.",
                uncertainty=[],
            )
        )
        return _with_checkpoint_bump(update)

    from repooperator_worker.agent_core.steering import consume_steering_for_state

    consume_steering_for_state(core, request)
    action = choose_graph_next_action(core, request)
    core.current_step = action.reason_summary
    if action.type != "final_answer":
        emit_action_decision(core, request, action)
    update = _updates_from_core(state, core)
    action_snapshot = action_to_snapshot(action)
    route = route_by_stage({**dict(state), **update, "pending_action": action_snapshot})
    update.update(
        {
            "pending_action": action_snapshot,
            "next_node": route,
            "current_step": action.reason_summary,
            "task_frame_snapshot": task_frame_to_snapshot(build_task_frame(request, core)),
            "events_to_emit": [
                _graph_transition_event(
                    state,
                    "route_next",
                    operation="route",
                    action_type=action.type,
                    status="completed",
                    next_node=route,
                )
            ],
        }
    )
    update.update(
        append_visible_rationale(
            {**dict(state), **update},
            node="route_next",
            action=action,
            summary=rationale_summary_for_action(action, fallback=f"I selected {action.type} as the next visible action because it matches the current evidence need."),
            basis_refs=rationale_basis_refs_for_action(action),
            safety_note=rationale_safety_note_for_action(action),
            uncertainty=rationale_uncertainty_for_action(action),
        )
    )
    return _with_checkpoint_bump(update)

def route_to_next_node(state: RepoOperatorGraphState) -> str:
    return str(state.get("next_node") or "final_synthesis")

def route_by_stage(state: RepoOperatorGraphState) -> str:
    stage = state.get("routing_stage") or "after_understanding"
    if stage == "after_interrupt_resume":
        return route_after_interrupt_resume(state)
    if stage == "after_evidence":
        return route_after_evidence(state)
    if stage == "after_tool_result":
        return route_after_tool_result(state)
    if stage == "after_validation":
        return route_after_validation(state)
    if stage == "after_change_plan":
        return route_after_change_plan(state)
    if stage == "after_apply":
        return route_after_apply(state)
    if stage == "after_approval":
        return route_after_approval(state)
    return route_after_understanding(state)

def route_after_understanding(state: RepoOperatorGraphState) -> str:
    if _should_use_supervisor(state):
        return "decompose_task"
    if _git_workflow_requested(state) and not state.get("pending_action"):
        return "git_workflow_graph"
    return _route_to_final_or_action(state)

def route_after_evidence(state: RepoOperatorGraphState) -> str:
    if _web_research_needed(state) and _web_research_available(state) and not _has_web_evidence(state):
        return "web_research_graph"
    return route_to_final_or_continue(state)

def route_after_tool_result(state: RepoOperatorGraphState) -> str:
    latest = _latest_result(state)
    if latest and latest.status == "waiting_approval":
        return "await_approval"
    if latest and latest.status in {"cancelled", "timed_out"}:
        return "final_synthesis"
    return route_to_final_or_continue(state)

def route_after_validation(state: RepoOperatorGraphState) -> str:
    latest = _latest_result(state)
    if latest and latest.status == "waiting_approval":
        return "await_approval"
    if state.get("stop_reason") in {"cancelled", "timed_out", "failed"}:
        return "final_synthesis"
    if _should_start_post_apply_git_workflow(state):
        return "git_workflow_graph"
    return route_to_final_or_continue(state)

def route_after_change_plan(state: RepoOperatorGraphState) -> str:
    latest = _latest_result(state)
    if latest and latest.status == "waiting_approval":
        return "await_approval"
    proposal = state.get("change_set_proposal") or {}
    errors = list((proposal.get("validation") or {}).get("errors") or state.get("proposal_errors") or [])
    if (errors or (latest and latest.status == "failed")) and int(state.get("repair_attempts") or 0) < 1:
        return "repair_change_set"
    if isinstance(proposal, dict) and proposal.get("changes") and str(proposal.get("status")) == "valid" and not proposal.get("applied"):
        return "await_change_approval"
    return route_to_final_or_continue(state)

def route_after_approval(state: RepoOperatorGraphState) -> str:
    if state.get("pending_approval"):
        return "final_synthesis"
    return route_to_final_or_continue(state)

def route_after_apply(state: RepoOperatorGraphState) -> str:
    return route_to_final_or_continue(state)

def route_after_interrupt_resume(state: RepoOperatorGraphState) -> str:
    if _pending_action(state):
        if _pending_action(state).type == "apply_change_set":
            return "apply_change_set"
        if _pending_action(state).type in {"git_commit", "git_push", "github_create_pr", "gitlab_create_mr"}:
            return "git_workflow_graph"
        if _pending_action(state).type == "run_validation_command":
            return "run_validation_command"
        return "execute_tool"
    if state.get("stop_reason") == "approval_denied":
        return "final_synthesis"
    return route_to_final_or_continue(state)

def route_to_final_or_continue(state: RepoOperatorGraphState) -> str:
    if state.get("stop_reason") in {"cancelled", "timed_out", "max_loop_iterations", "max_file_reads", "max_commands", "waiting_approval", "approval_denied"}:
        return "final_synthesis"
    return _route_to_final_or_action(state)

def _route_to_final_or_action(state: RepoOperatorGraphState) -> str:
    action = _pending_action(state)
    if not action:
        return "final_synthesis"
    if action.type == "final_answer":
        return "final_synthesis"
    if action.type == "ask_clarification":
        return "ask_clarification"
    if action.type in {"inspect_repo_tree", "search_files", "search_text", "read_file", "inspect_symbol", "analyze_file"}:
        return "gather_evidence"
    if action.type in {"search_web", "fetch_url", "summarize_web_evidence"}:
        return "web_research_graph"
    if action.type == "analyze_repository":
        return "analysis_graph"
    if action.type in {"generate_change_set", "generate_edit", "validate_change_set", "validate_edit"}:
        return "plan_change_set"
    if action.type == "apply_change_set":
        return "apply_change_set"
    if action.type in {"git_status", "git_diff", "git_log", "git_branch_create", "git_commit", "git_push", "github_create_pr", "gitlab_create_mr"}:
        return "execute_tool"
    if action.type in {"preview_command", "inspect_git_state", "run_approved_command", "run_validation_command", "request_command_approval"}:
        return "execute_tool"
    return "execute_tool"

def _should_start_post_apply_git_workflow(state: RepoOperatorGraphState) -> bool:
    if state.get("apply_status") != "applied":
        return False
    if state.get("post_apply_validation_status") == "failed":
        return False
    workflow = state.get("git_workflow") if isinstance(state.get("git_workflow"), dict) else {}
    return not bool(workflow.get("commit_proposed") or workflow.get("blocked"))
