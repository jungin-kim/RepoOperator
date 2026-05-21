"""Git workflow nodes for RepoOperator LangGraph."""

from __future__ import annotations

import re
from typing import Any

from langgraph.graph import END

from repooperator_worker.agent_core.actions import AgentAction
from repooperator_worker.agent_core.graph.adapters import _execute_ad_hoc_action, _execute_if_action_type, _graph_transition_event, _invoke_subgraph_delta, _merge_updates, _pending_action, _request, _with_checkpoint_bump
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

def git_route_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    return {"events_to_emit": [_graph_transition_event(state, "route_git_workflow", subgraph="git_workflow_graph", operation="route_git_workflow")]}

def route_git_workflow_next(state: RepoOperatorGraphState) -> str:
    action = _pending_action(state)
    if action and action.type == "git_commit":
        return "git_commit"
    if action and action.type == "git_push":
        return "git_push"
    if action and action.type in {"github_create_pr", "gitlab_create_mr"}:
        return "create_pr_or_mr"

    workflow = dict(state.get("git_workflow") or {})
    if not workflow.get("status_checked"):
        return "git_status"
    if not workflow.get("diff_checked"):
        return "git_diff"
    if not workflow.get("commit_proposed") and not workflow.get("blocked"):
        return "propose_commit_summary"
    return END

def git_status_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    update = _execute_ad_hoc_action(
        state,
        AgentAction(type="git_status", reason_summary="Read git status after applying changes.", expected_output="Working tree status."),
        subgraph="git_workflow_graph",
        node_name="git_status",
    )
    result = (update.get("action_results") or [{}])[-1]
    payload = result.get("payload") if isinstance(result, dict) else {}
    workflow = dict(state.get("git_workflow") or {})
    workflow.update({"status_checked": True, "status_result": payload, "status_text": (result or {}).get("observation") if isinstance(result, dict) else ""})
    update["git_workflow"] = workflow
    return update

def git_diff_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    files = [str(path) for path in state.get("files_changed") or []]
    update = _execute_ad_hoc_action(
        state,
        AgentAction(
            type="git_diff",
            reason_summary="Read git diff for applied change-set files.",
            expected_output="Local diff for commit summary.",
            payload={"relative_paths": files},
            target_files=files,
        ),
        subgraph="git_workflow_graph",
        node_name="git_diff",
    )
    result = (update.get("action_results") or [{}])[-1]
    payload = result.get("payload") if isinstance(result, dict) else {}
    workflow = dict(state.get("git_workflow") or {})
    workflow.update({"diff_checked": True, "diff_result": payload, "diff_excerpt": str((payload or {}).get("diff") or "")[:8000]})
    update["git_workflow"] = workflow
    return update

def git_propose_commit_summary_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    if not _git_write_context_allowed(state):
        workflow = dict(state.get("git_workflow") or {})
        workflow.update({"blocked": True, "block_reason": "Git writes require an applied change set unless the user explicitly asked for a git-only workflow."})
        return {
            "git_workflow": workflow,
            "final_response": "I did not prepare a git write. Git commit, push, and PR/MR actions require an applied change set or an explicit git-only request.",
            "events_to_emit": [_graph_transition_event(state, "propose_commit_summary", subgraph="git_workflow_graph", operation="git_write_blocked", aggregate=workflow)],
        }
    workflow = dict(state.get("git_workflow") or {})
    files = list(state.get("files_changed") or [])
    message = workflow.get("commit_message") or _generated_commit_message(state)
    validation_status = str(state.get("post_apply_validation_status") or "unknown")
    commit_summary = {
        "message": message,
        "files": files,
        "validation_status": validation_status,
        "status_text": workflow.get("status_text") or "",
        "diff_excerpt": workflow.get("diff_excerpt") or "",
    }
    workflow.update({"commit_message": message, "files": files, "commit_proposed": True, "commit_summary": commit_summary})
    pending = {
        "kind": "git_commit",
        "message": message,
        "files": files,
        "reason": "Creating a local commit requires explicit approval.",
        "approval_payload": {"message": message, "files": files, "validation_status": validation_status},
        "commit_summary": commit_summary,
    }
    update = {
        "git_workflow": workflow,
        "pending_approval": pending,
        "stop_reason": "waiting_approval",
        "final_response": _format_commit_summary_preview(commit_summary),
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
    update = _execute_if_action_type(state, {"git_commit"}, "git_workflow_graph", "git_commit")
    result = (update.get("action_results") or [{}])[-1]
    payload = result.get("payload") if isinstance(result, dict) else {}
    workflow = dict(state.get("git_workflow") or {})
    workflow.update({"commit_completed": (result or {}).get("status") == "success" if isinstance(result, dict) else False, "commit_result": payload, "commit_sha": (payload or {}).get("commit_sha")})
    update["git_workflow"] = workflow
    return update

def git_await_push_approval_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    if not _git_push_requested(state):
        return {"events_to_emit": [_graph_transition_event(state, "await_push_approval", subgraph="git_workflow_graph", operation="push_not_requested")]}
    branch = state.get("branch") or _current_branch_hint(state) or "HEAD"
    pending = {
        "kind": "git_push",
        "remote": "origin",
        "branch": branch,
        "reason": f"Pushing branch {branch} to origin requires explicit approval.",
        "files": list(state.get("files_changed") or []),
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
    update = _execute_if_action_type(state, {"git_push"}, "git_workflow_graph", "git_push")
    result = (update.get("action_results") or [{}])[-1]
    payload = result.get("payload") if isinstance(result, dict) else {}
    workflow = dict(state.get("git_workflow") or {})
    workflow.update({"push_completed": (result or {}).get("status") == "success" if isinstance(result, dict) else False, "push_result": payload})
    update["git_workflow"] = workflow
    return update

def git_await_pr_approval_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    if not _review_requested(state):
        return {"events_to_emit": [_graph_transition_event(state, "await_pr_approval", subgraph="git_workflow_graph", operation="pr_not_requested")]}
    branch = state.get("branch") or _current_branch_hint(state) or "HEAD"
    target_branch = _target_branch_hint(state)
    workflow = dict(state.get("git_workflow") or {})
    summary = workflow.get("commit_summary") if isinstance(workflow.get("commit_summary"), dict) else {}
    provider_kind = "gitlab_create_mr" if _merge_request_requested(state) else "github_create_pr"
    title = str(summary.get("message") or workflow.get("commit_message") or "RepoOperator change set")
    body = _review_body(summary)
    pending = {
        "kind": provider_kind,
        "source_branch": branch,
        "target_branch": target_branch,
        "title": title,
        "body": body,
        "description": body,
        "files": list(state.get("files_changed") or []),
        "reason": f"Creating a {'merge request' if provider_kind == 'gitlab_create_mr' else 'pull request'} requires explicit approval.",
        "approval_payload": {"source_branch": branch, "target_branch": target_branch, "title": title, "body": body, "description": body},
    }
    update = {"pending_approval": pending, "stop_reason": "waiting_approval", "events_to_emit": [_graph_transition_event(state, "await_pr_approval", subgraph="git_workflow_graph", operation="await_pr_approval", aggregate=pending)]}
    update = _merge_updates(
        update,
        append_visible_rationale(
            {**dict(state), **update},
            node="await_pr_approval",
            action=None,
            summary="Creating a PR/MR would write remote review state, so I am asking for approval first.",
            basis_refs=[{"kind": "file", "path": path} for path in state.get("files_changed") or []],
            safety_note="Remote review writes require explicit approval.",
            uncertainty=[],
        ),
    )
    return update

def git_create_review_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    return _execute_if_action_type(state, {"github_create_pr", "gitlab_create_mr"}, "git_workflow_graph", "create_pr_or_mr")

def _git_workflow_requested(state: RepoOperatorGraphState) -> bool:
    text = _request(state).task.lower()
    return bool(re.search(r"\b(commit|push)\b|pull request|merge request|\bpr\b|\bmr\b", text))

def _git_push_requested(state: RepoOperatorGraphState) -> bool:
    text = _request(state).task.lower()
    return bool(re.search(r"\bpush\b|pull request|merge request|\bpr\b|\bmr\b", text))

def _review_requested(state: RepoOperatorGraphState) -> bool:
    text = _request(state).task.lower()
    return bool(re.search(r"pull request|merge request|\bpr\b|\bmr\b", text))

def _merge_request_requested(state: RepoOperatorGraphState) -> bool:
    text = _request(state).task.lower()
    return bool(re.search(r"merge request|\bmr\b", text))

def _git_write_context_allowed(state: RepoOperatorGraphState) -> bool:
    return bool(state.get("apply_status") == "applied" or state.get("applied_change_set_id") or _git_workflow_requested(state))

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

def _target_branch_hint(state: RepoOperatorGraphState) -> str:
    text = _request(state).task
    match = re.search(r"\b(?:into|to|target(?: branch)?)\s+([A-Za-z0-9._/-]+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip(".,")
    return "main"

def _format_commit_summary_preview(summary: dict[str, Any]) -> str:
    files = [str(path) for path in summary.get("files") or []]
    lines = [
        "Commit approval required. RepoOperator has not committed anything.",
        "",
        f"Proposed message: {summary.get('message') or 'RepoOperator change set'}",
        f"Validation status: {summary.get('validation_status') or 'unknown'}",
    ]
    if files:
        lines.append("Changed files: " + ", ".join(f"`{path}`" for path in files))
    return "\n".join(lines)

def _review_body(summary: dict[str, Any]) -> str:
    files = [str(path) for path in summary.get("files") or []]
    lines = [
        "Summary:",
        f"- {summary.get('message') or 'RepoOperator change set'}",
        "",
        "Validation:",
        f"- Post-apply validation status: {summary.get('validation_status') or 'unknown'}",
    ]
    if files:
        lines.extend(["", "Changed files:", *[f"- {path}" for path in files]])
    return "\n".join(lines)
