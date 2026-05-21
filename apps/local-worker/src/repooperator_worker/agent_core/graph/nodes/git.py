"""Git workflow nodes for RepoOperator LangGraph."""

from __future__ import annotations

from typing import Any

from repooperator_worker.agent_core.graph.adapters import _execute_if_action_type, _graph_transition_event, _invoke_subgraph_delta, _merge_updates, _request, _with_checkpoint_bump
from repooperator_worker.agent_core.graph.state import RepoOperatorGraphState
from repooperator_worker.agent_core.graph.nodes.apply import await_approval_node
from repooperator_worker.agent_core.understanding_context import append_visible_rationale, evidence_basis_update

def git_workflow_graph_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    from repooperator_worker.agent_core.graph.builder import build_git_workflow_graph

    update = _invoke_subgraph_delta(build_git_workflow_graph, state)
    update["routing_stage"] = "after_approval" if update.get("pending_approval") else "after_tool_result"
    next_state = _merge_updates(dict(state), update)
    update = _merge_updates(update, evidence_basis_update(next_state, trigger_node="git_workflow_graph"))
    update.setdefault("events_to_emit", []).append(
        _graph_transition_event(state, "git_workflow_graph", subgraph="git_workflow_graph", operation="git_workflow")
    )
    return _with_checkpoint_bump(update)

def git_propose_commit_summary_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    workflow = dict(state.get("git_workflow") or {})
    files = list(state.get("files_changed") or [])
    message = workflow.get("commit_message") or _generated_commit_message(state)
    workflow.update({"commit_message": message, "files": files, "commit_proposed": True})
    pending = {
        "kind": "git_commit",
        "message": message,
        "files": files,
        "reason": "Creating a local commit requires explicit approval.",
        "approval_payload": {"message": message, "files": files},
    }
    update = {
        "git_workflow": workflow,
        "pending_approval": pending,
        "stop_reason": "waiting_approval",
        "final_response": f"Validation passed. Proposed commit message before approval:\n\n{message}",
        "events_to_emit": [_graph_transition_event(state, "propose_commit_summary", subgraph="git_workflow_graph", operation="propose_commit_summary", files=files, aggregate={"message": message, "files": files})],
    }
    update = _merge_updates(
        update,
        append_visible_rationale(
            {**dict(state), **update},
            node="propose_commit_summary",
            action=None,
            summary="A git commit would write repository history, so I prepared the message and stopped for approval.",
            basis_refs=[{"kind": "file", "path": path} for path in files],
            safety_note="Git commit requires explicit approval.",
            uncertainty=[],
        ),
    )
    return update

def git_await_commit_approval_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    if state.get("pending_approval"):
        return await_approval_node(state)
    return {"events_to_emit": [_graph_transition_event(state, "await_commit_approval", subgraph="git_workflow_graph", operation="approval_not_needed")]}

def git_commit_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    return _execute_if_action_type(state, {"git_commit"}, "git_workflow_graph", "git_commit")

def git_await_push_approval_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    if not _git_push_requested(state):
        return {"events_to_emit": [_graph_transition_event(state, "await_push_approval", subgraph="git_workflow_graph", operation="push_not_requested")]}
    branch = state.get("branch") or _current_branch_hint(state) or "HEAD"
    pending = {
        "kind": "git_push",
        "remote": "origin",
        "branch": branch,
        "reason": f"Pushing branch {branch} to origin requires explicit approval.",
    }
    update = {"pending_approval": pending, "stop_reason": "waiting_approval", "events_to_emit": [_graph_transition_event(state, "await_push_approval", subgraph="git_workflow_graph", operation="await_push_approval", aggregate=pending)]}
    update = _merge_updates(
        update,
        append_visible_rationale(
            {**dict(state), **update},
            node="await_push_approval",
            action=None,
            summary=f"Pushing branch {branch} would write to a remote, so I am asking for approval first.",
            basis_refs=[],
            safety_note="Remote git writes require explicit approval.",
            uncertainty=[],
        ),
    )
    return update

def git_push_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    return _execute_if_action_type(state, {"git_push"}, "git_workflow_graph", "git_push")

def git_await_pr_approval_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    return {"events_to_emit": [_graph_transition_event(state, "await_pr_approval", subgraph="git_workflow_graph", operation="pr_not_requested")]}

def git_create_review_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    return _execute_if_action_type(state, {"github_create_pr", "gitlab_create_mr"}, "git_workflow_graph", "create_pr_or_mr")

def _git_workflow_requested(state: RepoOperatorGraphState) -> bool:
    text = _request(state).task.lower()
    return any(term in text for term in ("commit", "push", "pull request", "merge request", "pr", "mr"))

def _git_push_requested(state: RepoOperatorGraphState) -> bool:
    text = _request(state).task.lower()
    return any(term in text for term in ("push", "pull request", "merge request", "pr", "mr"))

def _generated_commit_message(state: RepoOperatorGraphState) -> str:
    files = [str(item) for item in state.get("files_changed") or [] if str(item)]
    proposal = state.get("change_set_proposal") if isinstance(state.get("change_set_proposal"), dict) else {}
    summary = ((proposal.get("plan") or {}).get("summary") if isinstance(proposal.get("plan"), dict) else None) or "Apply RepoOperator change set"
    suffix = f" ({len(files)} file{'s' if len(files) != 1 else ''})" if files else ""
    return f"{str(summary).strip()[:64]}{suffix}"

def _current_branch_hint(state: RepoOperatorGraphState) -> str | None:
    for result in reversed(state.get("action_results") or []):
        payload = result.get("payload") if isinstance(result, dict) else {}
        if isinstance(payload, dict) and payload.get("branch"):
            return str(payload.get("branch"))
    return None
