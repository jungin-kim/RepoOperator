from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from repooperator_worker.agent_core.actions import AgentAction, ActionResult
from repooperator_worker.agent_core.planner import (
    _has_action,
    _has_command_preview,
    _has_command_run,
    _has_search_for,
    _latest_command_preview,
    _latest_unrun_read_only_preview,
    _preview_read_only,
    build_task_frame,
    candidate_files_from_results,
    command_needed_for_task,
    current_edit_target_files,
    current_search_candidate_files,
    edit_requested,
    emit_target_resolution,
    known_context_files,
    pending_commit_context,
    propose_next_action_with_model,
    project_summary_files,
    resolve_target_files,
)
from repooperator_worker.agent_core.graph import support as graph_support
from repooperator_worker.agent_core.state import AgentCoreState
from repooperator_worker.agent_core.task_policy import (
    block_current_subtask,
    ensure_subtasks,
    minimum_evidence_missing_for_task,
    next_evidence_gathering_action,
    next_recovery_action,
    should_ask_clarification_now,
)
from repooperator_worker.schemas import AgentRunRequest
from repooperator_worker.services.json_safe import json_safe


def choose_graph_next_action(state: AgentCoreState, request: AgentRunRequest) -> AgentAction:
    frame = build_task_frame(request, state)
    ensure_subtasks(state, request, frame)
    state.recommendation_context = json_safe({"task_frame": frame, "context_packet": state.context_packet})

    if state.actions_taken and state.action_results:
        previous_action = state.actions_taken[-1]
        previous_result = state.action_results[-1]
        if _is_ineffective_graph_result(previous_action, previous_result):
            recovery = next_recovery_action(state, request, frame, previous_action, previous_result)
            if recovery and not _repeats_graph_action(state, recovery):
                return recovery

    for chooser in (
        _next_explicit_target_action,
        _next_symbol_action,
        _next_policy_evidence_action,
        _next_model_planner_action,
        _next_command_action,
        _next_search_candidate_action,
        _next_edit_action,
        _next_project_summary_action,
    ):
        action = chooser(state, request, frame)
        if action and not _repeats_graph_action(state, action):
            return action

    return AgentAction(type="final_answer", reason_summary="Enough evidence is available for a grounded answer.")


def _next_explicit_target_action(state: AgentCoreState, request: AgentRunRequest, frame: Any) -> AgentAction | None:
    resolved = resolve_target_files(request, frame.mentioned_files, preferred=known_context_files(request, state))
    unread = [path for path in resolved if path not in state.files_read]
    if unread:
        emit_target_resolution(state, request, frame.mentioned_files, resolved)
        return AgentAction(
            type="read_file",
            reason_summary="Read resolved target files before answering.",
            target_files=unread,
            expected_output="File contents for grounded answer.",
        )

    unresolved = [
        item
        for item in frame.mentioned_files
        if item and not any(Path(path).name.lower() == Path(item).name.lower() or path.lower() == item.lower() for path in resolved)
    ]
    if unresolved and not _has_search_for(state, unresolved):
        return AgentAction(
            type="search_files",
            reason_summary="Resolve mentioned files before asking for clarification.",
            expected_output="Repo-relative candidate paths.",
            payload={"queries": unresolved},
        )

    explicit_candidates = current_search_candidate_files(state, min_score=35.0)
    explicit_candidate_unread = [
        path
        for path in explicit_candidates
        if path not in state.files_read and any(Path(path).name.lower() == Path(item).name.lower() for item in unresolved)
    ]
    if explicit_candidate_unread:
        return AgentAction(
            type="read_file",
            reason_summary="Read the resolved high-confidence target file.",
            target_files=explicit_candidate_unread[:1],
            expected_output="File contents for grounded answer.",
        )

    if frame.mentioned_files and resolved and all(path in state.files_read for path in resolved) and not edit_requested(frame):
        return AgentAction(type="final_answer", reason_summary="Answer from the explicitly requested file evidence.")

    if unresolved and _has_search_for(state, unresolved):
        return _clarification_action(state, request, frame, missing_files=unresolved)
    return None


def _next_symbol_action(state: AgentCoreState, request: AgentRunRequest, frame: Any) -> AgentAction | None:
    del request
    if frame.mentioned_symbols and not state.files_read and not _has_search_for(state, frame.mentioned_symbols):
        return AgentAction(
            type="search_files",
            reason_summary="Resolve mentioned symbols before answering.",
            target_symbols=frame.mentioned_symbols,
            expected_output="Repo-relative candidate paths.",
            payload={"queries": frame.mentioned_symbols},
        )
    return None


def _next_policy_evidence_action(state: AgentCoreState, request: AgentRunRequest, frame: Any) -> AgentAction | None:
    evidence_action = next_evidence_gathering_action(state, request, frame)
    if evidence_action:
        return evidence_action
    if should_ask_clarification_now(state, request, frame):
        return _clarification_action(state, request, frame)
    return None


def _next_model_planner_action(state: AgentCoreState, request: AgentRunRequest, frame: Any) -> AgentAction | None:
    return propose_next_action_with_model(
        request,
        state,
        frame,
        model_client_factory=graph_support.OpenAICompatibleModelClient,
    )


def _next_command_action(state: AgentCoreState, request: AgentRunRequest, frame: Any) -> AgentAction | None:
    del request
    unrun_preview = _latest_unrun_read_only_preview(state)
    if unrun_preview:
        return AgentAction(
            type="run_approved_command",
            reason_summary="Run read-only command after policy preview.",
            command=list(unrun_preview.command_result.get("command") or []),
            expected_output="Command output for the user request.",
        )
    command = command_needed_for_task(frame, state)
    if not command:
        return None
    if not _has_command_preview(state, command):
        return AgentAction(
            type="inspect_git_state" if command[:1] == ["git"] else "preview_command",
            reason_summary="Preview the safe command needed for missing evidence.",
            command=command,
            expected_output="Command safety classification.",
        )
    preview = _latest_command_preview(state, command)
    if preview and preview.status == "success" and _preview_read_only(preview.command_result) and not _has_command_run(state, command):
        return AgentAction(
            type="run_approved_command",
            reason_summary="Run read-only command after policy preview.",
            command=command,
            expected_output="Command output for the user request.",
        )
    if pending_commit_context(frame) and _has_command_run(state, ["git", "log", "--oneline", "-n", "5"]) and not _has_command_preview(state, ["git", "status", "--short"]):
        return AgentAction(
            type="inspect_git_state",
            reason_summary="Inspect git status before discussing a possible commit.",
            command=["git", "status", "--short"],
            expected_output="Working tree status.",
        )
    return AgentAction(type="final_answer", reason_summary="Answer from command evidence.")


def _next_search_candidate_action(state: AgentCoreState, request: AgentRunRequest, frame: Any) -> AgentAction | None:
    del request
    searched_candidates = candidate_files_from_results(state, edit_related=edit_requested(frame))
    candidate_unread = [path for path in searched_candidates if path not in state.files_read]
    if candidate_unread:
        read_limit = 1 if edit_requested(frame) else 4
        return AgentAction(
            type="read_file",
            reason_summary="Read best candidate files found by repository search.",
            target_files=candidate_unread[:read_limit],
            expected_output="Candidate file contents.",
        )
    return None


def _next_edit_action(state: AgentCoreState, request: AgentRunRequest, frame: Any) -> AgentAction | None:
    if not edit_requested(frame):
        return None
    edit_targets = current_edit_target_files(state, frame, request)
    if edit_targets:
        if not (_has_action(state, "generate_change_set") or _has_action(state, "generate_edit")):
            return AgentAction(
                type="generate_edit",
                reason_summary="Prepare a ChangeSetProposal for validated current edit targets.",
                target_files=edit_targets,
                expected_output="Validated ChangeSetProposal with before/after diff summary.",
                payload={"task_frame": json_safe(frame), "current_edit_targets": edit_targets},
            )
        return AgentAction(type="final_answer", reason_summary="Report the proposed edit without claiming it was applied.")
    return _clarification_action(state, request, frame)


def _next_project_summary_action(state: AgentCoreState, request: AgentRunRequest, frame: Any) -> AgentAction | None:
    del frame
    if not state.files_read and not _has_action(state, "inspect_repo_tree"):
        return AgentAction(type="inspect_repo_tree", reason_summary="Inspect repository inventory before answering.")
    project_files = project_summary_files(request)
    unread_project_files = [path for path in project_files if path not in state.files_read]
    if unread_project_files and len(state.files_read) < 4:
        return AgentAction(
            type="read_file",
            reason_summary="Read high-signal project files for a project-level answer.",
            target_files=unread_project_files[:4],
            expected_output="Project purpose and technology evidence.",
        )
    return None


def _clarification_action(state: AgentCoreState, request: AgentRunRequest, frame: Any, *, missing_files: list[str] | None = None) -> AgentAction:
    missing = missing_files or minimum_evidence_missing_for_task(state, request, frame)
    checked = _checked_evidence_summary(state)
    question = _clarification_question(missing, checked, frame)
    block_current_subtask(state, question)
    payload: dict[str, Any] = {"question": question, "missing_evidence": missing, "checked_evidence": checked}
    if state.edit_target_candidates:
        payload["candidate_files_considered"] = [
            {
                "path": item.get("path"),
                "score": item.get("score"),
                "role": item.get("role"),
                "already_read": item.get("already_read"),
                "blocked_reason": item.get("blocked_reason"),
            }
            for item in state.edit_target_candidates[:8]
            if isinstance(item, dict)
        ]
    if missing_files:
        payload["missing_files"] = missing_files
    return AgentAction(
        type="ask_clarification",
        reason_summary="Ask a precise clarification after safe evidence gathering did not resolve the target.",
        payload=payload,
    )


def _repeats_graph_action(state: AgentCoreState, action: AgentAction) -> bool:
    signature = _graph_action_signature(action)
    if not signature:
        return False
    ineffective = 0
    for previous, result in zip(state.actions_taken, state.action_results):
        if _graph_action_signature(previous) != signature:
            continue
        if _is_ineffective_graph_result(previous, result):
            ineffective += 1
    return ineffective >= 1


def _graph_action_signature(action: AgentAction) -> tuple[str, str] | None:
    if action.type == "search_text":
        return (action.type, _normalize_search_query(action.payload.get("query") or ""))
    if action.type == "search_files":
        queries = [_normalize_search_query(item) for item in action.payload.get("queries") or [] if _normalize_search_query(item)]
        text_queries = [_normalize_search_query(item) for item in action.payload.get("text_queries") or [] if _normalize_search_query(item)]
        return (action.type, "|".join([*queries, *text_queries]))
    if action.type == "read_file":
        return (action.type, "|".join(sorted(action.target_files)))
    if action.command:
        return (action.type, shlex.join(action.command))
    return None


def _is_ineffective_graph_result(action: AgentAction, result: ActionResult) -> bool:
    if result.status in {"failed", "skipped", "timed_out"}:
        return True
    if action.type == "search_files":
        return not bool(result.payload.get("candidates"))
    if action.type == "search_text":
        return not bool(result.payload.get("matches"))
    return False


def _checked_evidence_summary(state: AgentCoreState) -> list[str]:
    checked: list[str] = []
    if any(action.type == "inspect_repo_tree" for action in state.actions_taken):
        checked.append("repository structure")
    searched: list[str] = []
    for action in state.actions_taken:
        if action.type == "search_files":
            searched.extend(str(item) for item in [*(action.payload.get("queries") or []), *(action.payload.get("text_queries") or [])] if str(item))
        elif action.type == "search_text":
            query = str(action.payload.get("query") or "")
            if query:
                searched.append(query)
    if searched:
        checked.append("searches: " + ", ".join(_dedupe_text(searched[:8])))
    if state.files_read:
        checked.append("files read: " + ", ".join(state.files_read[-8:]))
    return checked


def _clarification_question(missing: list[str], checked: list[str], frame: Any) -> str:
    missing_text = "; ".join(missing) if missing else "the remaining target or scope"
    checked_text = "; ".join(checked) if checked else "no repository evidence could be gathered"
    if edit_requested(frame):
        return (
            "I could not identify a safe implementation target from the repository evidence. "
            f"Checked: {checked_text}. Missing: {missing_text}. "
            "Please name the file, module, route, handler, or component that should own the change."
        )
    return (
        "I need one more detail before I can answer accurately. "
        f"Checked: {checked_text}. Missing: {missing_text}. "
        "Please narrow the file, module, or workflow to inspect next."
    )


def _normalize_search_query(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _dedupe_text(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out
