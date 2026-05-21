from __future__ import annotations

import json
import os
import re
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from repooperator_worker.agent_core.agent_loop import AgentLoop, AgentLoopDeps
from repooperator_worker.agent_core.actions import AgentAction, ActionResult
from repooperator_worker.agent_core.context_service import get_default_context_service
from repooperator_worker.agent_core.events import append_activity_event, append_work_trace
from repooperator_worker.agent_core.final_synthesis import (
    _answer_with_model,
    validate_or_repair_final_answer,
)
from repooperator_worker.agent_core.final_response import build_agent_response
from repooperator_worker.agent_core.hooks import HookManager
from repooperator_worker.agent_core.planner import (
    TaskFrame,
    _format_command_result,
    _format_edit_proposal,
    _has_action,
    _has_command_preview,
    _has_command_run,
    _has_search_for,
    _latest_command_preview,
    _latest_command_result,
    _latest_edit_proposal,
    _latest_unrun_read_only_preview,
    _preview_read_only,
    _repository_review_response,
    build_task_frame,
    candidate_files_from_results,
    command_needed_for_task,
    current_edit_target_files,
    current_search_candidate_files,
    edit_requested,
    emit_target_resolution,
    known_context_files,
    likely_feature_context_files,
    likely_edit_file_queries,
    normalize_search_query,
    pending_commit_context,
    project_summary_files,
    propose_next_action_with_model as planner_propose_next_action_with_model,
    resolve_target_files,
)
from repooperator_worker.agent_core.state import AgentCoreState
from repooperator_worker.agent_core.steering import consume_steering_for_state
from repooperator_worker.agent_core.task_policy import (
    action_operation,
    block_current_subtask,
    ensure_subtasks,
    first_batch_files,
    group_inventory,
    minimum_evidence_missing_for_task,
    next_evidence_gathering_action,
    next_recovery_action,
    record_ineffective_action,
    repository_file_inventory,
    should_ask_clarification_now,
    update_subtasks_after_action,
)
from repooperator_worker.agent_core.tool_orchestrator import ToolOrchestrator
from repooperator_worker.agent_core.tools.registry import get_default_tool_registry
from repooperator_worker.schemas import AgentRunRequest, AgentRunResponse
from repooperator_worker.services.model_client import OpenAICompatibleModelClient
from repooperator_worker.services.event_service import append_run_event, get_run, list_run_events
from repooperator_worker.services.json_safe import json_safe, safe_agent_response_payload, safe_repr
from repooperator_worker.services.skills_service import enabled_skill_context
from repooperator_worker.services.active_repository import get_active_repository


@dataclass(frozen=True)
class LoopBudget:
    max_loop_iterations: int
    max_file_reads: int
    max_commands: int
    max_edits: int = 6
    reason: str = "default"


HARD_MAX_LOOP_ITERATIONS = 18

def run_controller_graph(
    request: AgentRunRequest,
    *,
    run_id: str | None = None,
    stream_final_answer: bool = False,
) -> AgentRunResponse:
    if _use_langgraph_runtime():
        from repooperator_worker.agent_core.langgraph_runtime import run_langgraph_controller

        return run_langgraph_controller(request, run_id=run_id, stream_final_answer=stream_final_answer)
    run_id = run_id or "run_controller"
    _validate_active_repository(request)
    state = _initial_state(request, run_id)
    skills_context, skills_used = enabled_skill_context()
    state.skills_used = skills_used
    registry = get_default_tool_registry()
    hook_manager = HookManager()
    context_service = get_default_context_service()
    orchestrator = ToolOrchestrator(run_id=run_id, request=request, registry=registry, hook_manager=hook_manager)

    def append_action_event(action: AgentAction, result: ActionResult) -> None:
        _append_run_event_safe(
            run_id,
            {
                "type": "action_result",
                "event_type": "action_result",
                "status": result.status,
                "action": action.model_dump(),
                "result": result.model_dump(),
            },
        )

    def synthesize(state_for_answer: AgentCoreState, request_for_answer: AgentRunRequest) -> str:
        on_delta = _stream_final_delta(run_id) if stream_final_answer else None
        packet_context = ""
        if isinstance(state_for_answer.context_packet, dict):
            packet_context = str(state_for_answer.context_packet.get("skills_context") or "")
        return build_final_answer_text(
            state_for_answer,
            request_for_answer,
            skills_context=packet_context or skills_context,
            on_delta=on_delta,
        )

    loop = AgentLoop(
        AgentLoopDeps(
            context_service=context_service,
            tool_registry=registry,
            tool_orchestrator=orchestrator,
            hook_manager=hook_manager,
            load_context=load_context,
            classify=classify,
            create_initial_plan=create_initial_plan,
            emit_plan_update=emit_plan_update,
            should_continue=should_continue,
            check_cancel=check_cancel,
            consume_steering=consume_steering_for_state,
            choose_next_action=controller_choose_next_action,
            execute_action=orchestrator.execute_action,
            append_action_event=append_action_event,
            observe_result=observe_result,
            update_plan=update_plan,
            build_final_answer=synthesize,
            validate_final_answer=validate_or_repair_final_answer,
            build_final_response=build_final_response,
        )
    )
    return loop.run(state, request)


def load_context(state: AgentCoreState, request: AgentRunRequest) -> None:
    packet = get_default_context_service().collect(request)
    state.context_packet = packet.model_dump()
    high_signal = sorted(packet.high_signal_files)
    instructions = sorted(packet.project_instructions)
    state.observations.append("Loaded request context for the active repository.")
    append_activity_event(
        run_id=state.run_id,
        request=request,
        activity_id="controller-load-context",
        event_type="activity_completed",
        phase="Thinking",
        label="Loaded context",
        status="completed",
        observation="Loaded request, repository, branch, thread, and skill context.",
        aggregate={
            "repo_root_name": packet.repo_root_name,
            "branch": packet.branch,
            "high_signal_files": high_signal,
            "project_instruction_files": instructions,
            "prior_files_read": packet.prior_files_read,
            "prior_commands_run": packet.prior_commands_run,
            "git_status_available": bool(packet.git_status_summary),
            "recent_commits_available": bool(packet.recent_commits_summary),
        },
    )


def classify(state: AgentCoreState, request: AgentRunRequest) -> None:
    from repooperator_worker.agent_core.request_understanding import (
        understand_request,
        request_understanding_to_classifier_result,
    )
    ru = understand_request(request)
    state.request_understanding = ru
    state.classifier_result = request_understanding_to_classifier_result(ru, request)
    frame = build_task_frame(request, state)
    budget = determine_loop_budget(frame, request, state.context_packet)
    state.max_loop_iterations = budget.max_loop_iterations
    state.max_file_reads = budget.max_file_reads
    state.max_commands = budget.max_commands
    state.max_edits = budget.max_edits
    ensure_subtasks(state, request, frame)
    state.recommendation_context = json_safe({"task_frame": frame, "context_packet": state.context_packet})
    append_activity_event(
        run_id=state.run_id,
        request=request,
        activity_id="controller-frame-request",
        event_type="activity_completed",
        phase="Thinking",
        label="Framed request",
        status="completed",
        observation=f"Goal framed with {len(frame.mentioned_files)} mentioned file(s) and {len(frame.likely_capabilities)} likely capability hint(s).",
        aggregate={"task_frame": json_safe(frame), "loop_budget": json_safe(budget)},
    )


def determine_loop_budget(frame: TaskFrame, request: AgentRunRequest, context_packet: dict[str, Any] | None = None) -> LoopBudget:
    del request, context_packet
    explicit_file = bool(frame.mentioned_files)
    edit_like = edit_requested(frame)
    requested_outputs = {str(item).strip().lower() for item in frame.requested_outputs}
    tool_hints = {str(item).strip() for item in frame.likely_needed_tools}
    repo_wide = "analyze_repository" in tool_hints or "repository_review" in requested_outputs or ("code_review" in requested_outputs and not explicit_file)

    if repo_wide:
        return _bounded_loop_budget(16, 32, 6, reason="repo-wide review")
    if edit_like and explicit_file:
        return _bounded_loop_budget(8, 12, 4, reason="feature/edit with explicit file")
    if edit_like:
        return _bounded_loop_budget(12, 16, 4, reason="feature/edit discovery")
    if explicit_file:
        return _bounded_loop_budget(5, 8, 3, reason="explicit file question")
    return _bounded_loop_budget(6, 8, 3, reason="project summary")


def _bounded_loop_budget(max_loop_iterations: int, max_file_reads: int, max_commands: int, *, reason: str) -> LoopBudget:
    return LoopBudget(
        max_loop_iterations=min(max_loop_iterations, HARD_MAX_LOOP_ITERATIONS),
        max_file_reads=max_file_reads,
        max_commands=max_commands,
        reason=reason,
    )


def create_initial_plan(state: AgentCoreState) -> None:
    if state.subtasks:
        state.plan = [f"{item.title}: {item.status}" for item in state.subtasks]
        return
    state.plan = [
        "Frame the user's goal",
        "Resolve missing evidence",
        "Use safe primitive actions",
        "Answer only from gathered evidence or ask a precise clarification",
    ]


def should_continue(state: AgentCoreState, *, started: float, max_wall_clock_seconds: int, request: AgentRunRequest | None = None) -> bool:
    if state.stop_reason or state.cancellation_requested:
        return False
    if state.loop_iteration >= state.max_loop_iterations:
        if request and _should_extend_for_unread_feature_entrypoint(state, request):
            state.max_loop_iterations = min(HARD_MAX_LOOP_ITERATIONS, state.max_loop_iterations + 2)
            if state.loop_iteration < state.max_loop_iterations:
                state.loop_iteration += 1
                return True
        state.stop_reason = "max_loop_iterations"
        return False
    if len(state.files_read) >= state.max_file_reads:
        state.stop_reason = "max_file_reads"
        return False
    if len(state.commands_run) >= state.max_commands:
        state.stop_reason = "max_commands"
        return False
    if time.perf_counter() - started > max_wall_clock_seconds:
        state.stop_reason = "timed_out"
        return False
    state.loop_iteration += 1
    return True


def _should_extend_for_unread_feature_entrypoint(state: AgentCoreState, request: AgentRunRequest) -> bool:
    if state.max_loop_iterations >= HARD_MAX_LOOP_ITERATIONS:
        return False
    frame = build_task_frame(request, state)
    if not edit_requested(frame) or frame.mentioned_files:
        return False
    context_files = likely_feature_context_files(request)
    return "main.py" in context_files and "main.py" not in state.files_read


def check_cancel(state: AgentCoreState, request: AgentRunRequest) -> None:
    try:
        run = get_run(state.run_id) or {}
    except OSError:
        run = {}
    if run.get("status") not in {"cancelled", "cancelling"}:
        return
    state.cancellation_requested = True
    state.stop_reason = "cancelled"
    append_activity_event(
        run_id=state.run_id,
        request=request,
        activity_id="controller-cancelled",
        event_type="activity_completed",
        phase="Finished",
        label="Run cancelled",
        status="cancelled",
        observation="Cancellation was requested. RepoOperator stopped at the next safe checkpoint.",
    )


def controller_choose_next_action(state: AgentCoreState, request: AgentRunRequest) -> AgentAction:
    frame = build_task_frame(request, state)
    ensure_subtasks(state, request, frame)
    state.recommendation_context = json_safe({"task_frame": frame, "context_packet": state.context_packet})

    if state.actions_taken and state.action_results:
        previous_action = state.actions_taken[-1]
        previous_result = state.action_results[-1]
        if _is_ineffective_result(previous_action, previous_result):
            recovery = next_recovery_action(state, request, frame, previous_action, previous_result)
            if recovery and not _repeats_ineffective_action(state, recovery):
                return recovery

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

    unresolved = [item for item in frame.mentioned_files if item and not any(Path(path).name.lower() == Path(item).name.lower() or path.lower() == item.lower() for path in resolved)]
    if unresolved and not _has_search_for(state, unresolved):
        return AgentAction(
            type="search_files",
            reason_summary="Resolve mentioned files before asking for clarification.",
            expected_output="Repo-relative candidate paths.",
            payload={"queries": unresolved},
        )

    explicit_candidates = current_search_candidate_files(state, min_score=35.0)
    explicit_candidate_unread = [
        path for path in explicit_candidates
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
        return AgentAction(
            type="ask_clarification",
            reason_summary="Ask a precise file clarification after repository search did not find targets.",
            payload={"missing_files": unresolved},
        )

    if frame.mentioned_symbols and not state.files_read and not _has_search_for(state, frame.mentioned_symbols):
        return AgentAction(
            type="search_files",
            reason_summary="Resolve mentioned symbols before answering.",
            target_symbols=frame.mentioned_symbols,
            expected_output="Repo-relative candidate paths.",
            payload={"queries": frame.mentioned_symbols},
        )

    evidence_action = next_evidence_gathering_action(state, request, frame)
    if evidence_action and not _repeats_ineffective_action(state, evidence_action):
        return evidence_action

    if should_ask_clarification_now(state, request, frame):
        missing = minimum_evidence_missing_for_task(state, request, frame)
        checked = _checked_evidence_summary(state)
        question = _clarification_question(missing, checked, frame)
        block_current_subtask(state, question)
        return AgentAction(
            type="ask_clarification",
            reason_summary="Ask a precise clarification after safe evidence gathering did not resolve the target.",
            payload={"question": question, "missing_evidence": missing, "checked_evidence": checked},
        )

    planned = planner_propose_next_action_with_model(request, state, frame, model_client_factory=OpenAICompatibleModelClient)
    if planned and not _repeats_ineffective_action(state, planned):
        return planned

    unrun_preview = _latest_unrun_read_only_preview(state)
    if unrun_preview:
        return AgentAction(
            type="run_approved_command",
            reason_summary="Run read-only command after policy preview.",
            command=list(unrun_preview.command_result.get("command") or []),
            expected_output="Command output for the user request.",
        )

    command = command_needed_for_task(frame, state)
    if command:
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

    if edit_requested(frame):
        edit_targets = current_edit_target_files(state, frame, request)
        if edit_targets:
            if not _has_action(state, "generate_edit"):
                return AgentAction(
                    type="generate_edit",
                    reason_summary="Prepare a proposed patch for validated current edit targets.",
                    target_files=edit_targets,
                    expected_output="Proposed diff and before/after summary.",
                    payload={"task_frame": json_safe(frame), "current_edit_targets": edit_targets},
                )
            return AgentAction(type="final_answer", reason_summary="Report the proposed edit without claiming it was applied.")
        missing = minimum_evidence_missing_for_task(state, request, frame)
        checked = _checked_evidence_summary(state)
        question = _clarification_question(missing, checked, frame)
        block_current_subtask(state, question)
        return AgentAction(
            type="ask_clarification",
            reason_summary="Ask which implementation area to change after evidence gathering did not find a safe target.",
            payload={"question": question, "missing_evidence": missing, "checked_evidence": checked},
        )

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

    if not state.files_read and unresolved:
        return AgentAction(
            type="ask_clarification",
            reason_summary="Ask a precise file clarification after repository search did not find targets.",
            payload={"missing_files": unresolved},
        )
    return AgentAction(type="final_answer", reason_summary="Enough evidence is available for a grounded answer.")


def _repeats_ineffective_action(state: AgentCoreState, action: AgentAction) -> bool:
    signature = _action_signature(action)
    if not signature:
        return False
    ineffective = 0
    for previous, result in zip(state.actions_taken, state.action_results):
        if _action_signature(previous) != signature:
            continue
        if _is_ineffective_result(previous, result):
            ineffective += 1
    return ineffective >= 1


def _action_signature(action: AgentAction) -> tuple[str, str] | None:
    if action.type == "search_text":
        return (action.type, normalize_search_query(action.payload.get("query") or ""))
    if action.type == "search_files":
        queries = [normalize_search_query(item) for item in action.payload.get("queries") or [] if normalize_search_query(item)]
        text_queries = [normalize_search_query(item) for item in action.payload.get("text_queries") or [] if normalize_search_query(item)]
        return (action.type, "|".join([*queries, *text_queries]))
    if action.type == "read_file":
        return (action.type, "|".join(sorted(action.target_files)))
    if action.command:
        return (action.type, shlex.join(action.command))
    return None


def _is_ineffective_result(action: AgentAction, result: ActionResult) -> bool:
    if result.status in {"failed", "skipped", "timed_out"}:
        return True
    if action.type == "search_files":
        return not bool(result.payload.get("candidates"))
    if action.type == "search_text":
        return not bool(result.payload.get("matches"))
    return False


def _zero_result_search_count(state: AgentCoreState) -> int:
    count = 0
    for action, result in zip(state.actions_taken, state.action_results):
        if action.type not in {"search_files", "search_text"}:
            continue
        if _is_ineffective_result(action, result):
            count += 1
    return count


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


def _clarification_question(missing: list[str], checked: list[str], frame: TaskFrame) -> str:
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


def observe_result(state: AgentCoreState, action: AgentAction, result: ActionResult, request: AgentRunRequest) -> None:
    record_ineffective_action(state, action, result)
    if result.files_read:
        for path in result.files_read:
            if path not in state.files_read:
                state.files_read.append(path)
    if result.files_changed:
        for path in result.files_changed:
            if path not in state.files_changed:
                state.files_changed.append(path)
    if result.command_result and result.command_result.get("display_command"):
        command = str(result.command_result.get("display_command"))
        if result.status == "success" and result.command_result.get("exit_code") is not None:
            state.commands_run.append(command)
    if result.status == "waiting_approval":
        if result.command_result:
            state.pending_approval = result.command_result
        else:
            decision = result.payload.get("permission_decision") if isinstance(result.payload, dict) else {}
            metadata = decision.get("metadata") if isinstance(decision, dict) and isinstance(decision.get("metadata"), dict) else {}
            state.pending_approval = {
                "kind": action.type,
                "reason": result.observation,
                "approval_payload": decision.get("approval_payload") or metadata.get("approval_payload") or action.payload,
                "tool_name": action.type,
            }
    observation = _safe_observation(action, result)
    if observation:
        state.observations.append(observation)
        append_activity_event(
            run_id=state.run_id,
            request=request,
            activity_id=f"controller-observe:{action.action_id}",
            event_type="activity_completed",
            phase="Observing",
            label="Recorded observation",
            status="completed",
            observation=observation,
            related_files=result.files_read,
            related_command=action.command,
        )
    if action.type == "analyze_repository":
        response = result.payload.get("response")
        if isinstance(response, AgentRunResponse):
            state.final_response = response.response
        elif isinstance(response, dict):
            state.final_response = str(response.get("response") or "")
    update_subtasks_after_action(state, action, result, action_operation(action.type))


def update_plan(state: AgentCoreState, action: AgentAction, result: ActionResult, request: AgentRunRequest) -> None:
    if state.subtasks:
        state.plan = [f"{item.title}: {item.status}" for item in state.subtasks]
    if result.status == "waiting_approval":
        state.plan.append("Wait for user approval before running the command")
    elif result.next_recommended_action:
        state.plan.append(f"Consider next safe action: {result.next_recommended_action}")
    elif result.status == "success":
        state.plan.append(f"Completed: {action.type}")
    emit_plan_update(state, request, "Updated plan")


def emit_plan_update(state: AgentCoreState, request: AgentRunRequest, label: str) -> None:
    append_activity_event(
        run_id=state.run_id,
        request=request,
        activity_id="controller-plan",
        event_type="activity_updated",
        phase="Planning",
        label=label,
        status="running",
        observation="; ".join(state.plan[-4:]),
        aggregate={"plan_steps": list(state.plan), "loop_iteration": state.loop_iteration},
    )


def build_final_answer_text(
    state: AgentCoreState,
    request: AgentRunRequest,
    *,
    skills_context: str = "",
    on_delta: Any | None = None,
) -> str:
    if state.cancellation_requested or state.stop_reason == "cancelled":
        completed = "; ".join(state.observations[-4:]) or "No action completed before cancellation."
        return f"Run cancelled. Completed work before stopping: {completed}"
    if state.pending_approval:
        return _format_command_preview(list(state.pending_approval.get("command") or []), state.pending_approval)
    repository_review = _repository_review_response(state)
    if repository_review:
        return repository_review.response
    edit_proposal = _latest_edit_proposal(state)
    if edit_proposal:
        return _format_edit_proposal(edit_proposal)
    proposal_error = _latest_proposal_error(state)
    if proposal_error:
        return (
            "I could not prepare a validated proposal-only patch. "
            f"Reason: {proposal_error}. No files were modified."
        )
    command_result = _latest_command_result(state)
    if command_result:
        return _format_command_result(command_result, pending_commit=pending_commit_context(build_task_frame(request, state)))
    if _is_broad_analysis_request(build_task_frame(request, state)) and state.files_read:
        return _format_broad_analysis_answer(state, request)
    if state.stop_reason in {"failed", "timed_out", "max_loop_iterations", "max_file_reads", "max_commands"}:
        return _build_evidence_limited_answer(state, request)
    contents: dict[str, str] = {}
    for result in state.action_results:
        contents.update(result.payload.get("contents") or {})
    repo_observation = "\n".join(state.observations[-6:])
    append_work_trace(
        run_id=state.run_id,
        request=request,
        activity_id="final-synthesis-preparing",
        phase="Finished",
        label="Preparing final answer",
        status="running",
        safe_reasoning_summary="Preparing an evidence-based answer from the gathered files and observations.",
        current_action="Synthesize the final response from collected evidence.",
        related_files=list(contents.keys()),
    )
    answer = _answer_with_model(request, contents, state=state, repo_observation=repo_observation, skills_context=skills_context, on_delta=on_delta)
    append_activity_event(
        run_id=state.run_id,
        request=request,
        activity_id="final-synthesis-prepared",
        event_type="activity_completed",
        phase="Finished",
        label="Prepared evidence-based answer",
        status="completed",
        safe_reasoning_summary="Prepared an evidence-based answer from gathered files and observations.",
        observation="Prepared the final answer from gathered evidence.",
    )
    return answer


def _build_evidence_limited_answer(state: AgentCoreState, request: AgentRunRequest) -> str:
    frame = build_task_frame(request, state)
    checked: list[str] = []
    if state.files_read:
        checked.append("read " + ", ".join(f"`{path}`" for path in state.files_read[-6:]))
    if state.commands_run:
        checked.append("ran " + ", ".join(f"`{cmd}`" for cmd in state.commands_run[-3:]))
    recent_observations = [item for item in state.observations[-4:] if item]
    if recent_observations:
        checked.append("observed " + "; ".join(recent_observations))

    missing: list[str] = []
    if edit_requested(frame):
        context_files = likely_feature_context_files(request)
        if "main.py" in context_files and "main.py" not in state.files_read:
            missing.append("`main.py` should be read before choosing an implementation target")
        if not current_edit_target_files(state, frame, request):
            missing.append("the exact file or component that should own the requested feature is still not confirmed")
        if not _latest_edit_proposal(state):
            missing.append("no proposal-only edit was generated")
    elif not state.files_read and not state.commands_run:
        missing.append("no repository file or command evidence was gathered")

    lines = ["I do not have enough confirmed evidence to give a final implementation answer yet."]
    lines.append("Checked: " + (("; ".join(checked)) if checked else "no completed repository evidence."))
    if missing:
        lines.append("Missing evidence: " + "; ".join(_dedupe_text(missing)) + ".")
    if edit_requested(frame):
        lines.append("Next safe step: confirm the target file or let me continue by reading the likely entrypoint and then preparing a proposal-only patch.")
    else:
        lines.append("Next safe step: confirm the specific file or workflow to inspect next, or let me continue gathering repository evidence.")
    return "\n".join(lines)


def _latest_proposal_error(state: AgentCoreState) -> str | None:
    for result in reversed(state.action_results):
        error = result.payload.get("proposal_error")
        if error:
            return str(error)
    return None


def _is_broad_analysis_request(frame: TaskFrame) -> bool:
    text = " ".join([frame.user_goal, *frame.requested_outputs]).lower()
    return any(term in text for term in ("all", "every", "whole", "entire", "directory structure", "source tree", "codebase", "전체", "모든")) and any(
        term in text for term in ("source", "file", "module", "project", "repo", "directory", "folder", "파일", "모듈", "구조")
    )


def _format_broad_analysis_answer(state: AgentCoreState, request: AgentRunRequest) -> str:
    contents: dict[str, str] = {}
    for result in state.action_results:
        contents.update(result.payload.get("contents") or {})
    inventory = repository_file_inventory(request)
    groups = group_inventory(inventory)
    analyzed = list(contents.keys())
    remaining_groups: list[str] = []
    for group, files in groups.items():
        remaining = [path for path in files if path not in analyzed]
        if remaining:
            remaining_groups.append(f"- {group}: {len(remaining)} remaining ({', '.join(remaining[:6])})")
    role_lines = ["| File | Role |", "| --- | --- |"]
    for path, content in contents.items():
        role_lines.append(f"| `{path}` | {_file_role_summary(path, content)} |")
    analyzed_text = ", ".join(f"`{path}`" for path in analyzed) or "none"
    remaining_text = "\n".join(remaining_groups) if remaining_groups else "- No remaining readable groups found in the current inventory."
    return "\n".join(
        [
            "I analyzed the first bounded batch of repository files.",
            "",
            f"Analyzed batch: {analyzed_text}",
            "",
            "File role table:",
            *role_lines,
            "",
            "Remaining groups/files:",
            remaining_text,
            "",
            "Next safe continuation: read the next bounded batch from the remaining groups and extend the same table.",
        ]
    )


def _file_role_summary(path: str, content: str) -> str:
    name = Path(path).name.lower()
    suffix = Path(path).suffix.lower()
    if name.startswith("readme") or suffix in {".md", ".rst", ".txt"}:
        return "documentation or project overview"
    if suffix in {".json", ".toml", ".yaml", ".yml", ".gradle", ".xml"} or name in {"makefile", "dockerfile"}:
        return "configuration/build metadata"
    functions = re.findall(r"^\s*(?:def|function)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", content, flags=re.MULTILINE)
    classes = re.findall(r"^\s*(?:class|interface|struct)\s+([A-Za-z_][A-Za-z0-9_]*)\b", content, flags=re.MULTILINE)
    exported = re.findall(r"\bexport\s+(?:function|class|const)\s+([A-Za-z_][A-Za-z0-9_]*)", content)
    names = [*functions[:3], *classes[:3], *exported[:3]]
    if names:
        return "source module defining " + ", ".join(_dedupe_text(names[:5]))
    return "source or support file"


def _dedupe_text(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def build_final_response(state: AgentCoreState, request: AgentRunRequest) -> AgentRunResponse:
    review_response = _repository_review_response(state)
    if review_response and state.stop_reason not in {"cancelled", "waiting_approval"}:
        return _response_json_safe(review_response.model_copy(update={"loop_iteration": state.loop_iteration}), request)
    response_type = "command_approval" if state.pending_approval else "assistant_answer"
    graph_path = "agent_core:" + (
        "cancelled" if state.stop_reason == "cancelled"
        else "command_preview" if state.pending_approval
        else "read_file_answer" if state.files_read
        else "general_answer"
    )
    return _response_json_safe(build_agent_response(
        request,
        response=state.final_response,
        response_type=response_type,
        files_read=state.files_read,
        graph_path=graph_path,
        intent_classification=state.classifier_result.intent,
        run_id=state.run_id,
        skills_used=state.skills_used,
        stop_reason=state.stop_reason or "completed",
        loop_iteration=max(1, state.loop_iteration),
        command_approval=state.pending_approval,
        commands_planned=[shlex.join(list(state.pending_approval.get("command") or []))] if state.pending_approval else [],
        commands_run=state.commands_run,
        activity_events=[],
    ), request)


def _validate_active_repository(request: AgentRunRequest) -> None:
    try:
        active = get_active_repository()
    except Exception:
        active = None
    if active is None:
        return
    requested = str(request.project_path)
    active_path = str(active.project_path)
    if requested != active_path:
        raise ValueError(
            "The active repository changed before this run started. "
            "Open the repository again or start a new thread for the stale request."
        )
    if request.branch and active.branch and request.branch != active.branch:
        raise ValueError("The active branch changed before this run started.")


def stream_controller_graph(request: AgentRunRequest, *, run_id: str | None = None) -> Iterator[dict[str, Any]]:
    if _use_langgraph_runtime():
        from repooperator_worker.agent_core.langgraph_runtime import stream_langgraph_controller

        yield from stream_langgraph_controller(request, run_id=run_id)
        return
    before_sequence = _latest_sequence(run_id) if run_id else 0
    response = run_controller_graph(request, run_id=run_id, stream_final_answer=True)
    for event in list_run_events(run_id or response.run_id or "", after_sequence=before_sequence):
        if event.get("type") == "assistant_delta":
            before_sequence = int(event.get("sequence") or before_sequence)
            yield event
    if not _streamed_assistant_delta(run_id or ""):
        for chunk in _chunk_text(response.response):
            yield {"type": "assistant_delta", "delta": chunk, "streaming_mode": "post_hoc_chunking"}
    final = _response_json_safe(response.model_copy(update={"activity_events": []}), request)
    yield {"type": "final_message", "result": safe_agent_response_payload(final)}


def propose_next_action_with_model(request: AgentRunRequest, state: AgentCoreState, task_frame: TaskFrame) -> AgentAction | None:
    return planner_propose_next_action_with_model(request, state, task_frame, model_client_factory=OpenAICompatibleModelClient)


def _initial_state(request: AgentRunRequest, run_id: str) -> AgentCoreState:
    return AgentCoreState(
        run_id=run_id,
        thread_id=request.thread_id,
        repo=request.project_path,
        branch=request.branch,
        user_task=request.task,
    )


def _use_langgraph_runtime() -> bool:
    configured = os.getenv("REPOOPERATOR_AGENT_RUNTIME")
    default = os.getenv("REPOOPERATOR_AGENT_RUNTIME_DEFAULT", "legacy")
    return (configured if configured is not None else default).strip().lower() == "langgraph"


def _safe_observation(action: AgentAction, result: ActionResult) -> str:
    if result.status == "waiting_approval":
        return "Command preview requires approval before execution."
    if result.files_read:
        files = ", ".join(result.files_read)
        return f"Read {files}."
    if action.type == "generate_edit" and result.status == "success":
        proposals = result.payload.get("edit_proposals") or []
        files = ", ".join(str(item.get("file")) for item in proposals if isinstance(item, dict))
        return f"Prepared proposed edit for {files}. No files were written."
    if action.type == "search_files" and result.status == "success":
        candidates = result.payload.get("candidates") or []
        return f"Found candidate files: {', '.join(candidates[:8])}."
    if action.type == "analyze_repository" and result.status == "success":
        return "Completed repository-wide review and collected per-file evidence."
    if result.command_result and result.command_result.get("exit_code") is not None:
        return f"Ran `{result.command_result.get('display_command')}` with exit code {result.command_result.get('exit_code')}."
    if result.observation:
        return " ".join(str(result.observation).split())[:500]
    return ""


def _stream_final_delta(run_id: str):
    def emit(delta: str) -> None:
        _append_run_event_safe(run_id, {"type": "assistant_delta", "delta": delta, "streaming_mode": "model_stream"})

    return emit


def _streamed_assistant_delta(run_id: str) -> bool:
    if not run_id:
        return False
    return any(event.get("type") == "assistant_delta" for event in list_run_events(run_id))


def _latest_sequence(run_id: str | None) -> int:
    if not run_id:
        return 0
    events = list_run_events(run_id)
    return max((int(event.get("sequence") or 0) for event in events), default=0)


def _append_run_event_safe(run_id: str, event: dict[str, Any]) -> dict[str, Any]:
    try:
        return append_run_event(run_id, json_safe(event))
    except OSError:
        return json_safe(event)


def _response_json_safe(response: AgentRunResponse, request: AgentRunRequest) -> AgentRunResponse:
    try:
        payload = safe_agent_response_payload(response)
        json.dumps(payload, ensure_ascii=False)
        return response
    except Exception as exc:  # noqa: BLE001
        safe_payload = json_safe(response)
        safe_payload["response"] = (
            "The review completed, but RepoOperator hit an internal metadata serialization error. "
            "The readable summary is below...\n\n"
            + str(safe_payload.get("response") or response.response)
        )
        safe_payload["activity_events"] = json_safe(safe_payload.get("activity_events") or [])
        safe_payload["stop_reason"] = safe_payload.get("stop_reason") or "completed_with_metadata_error"
        _append_run_event_safe(
            response.run_id or "run_controller",
            {
                "type": "error",
                "event_type": "metadata_serialization_error",
                "status": "failed",
                "message": safe_repr(exc, limit=220),
            },
        )
        return build_agent_response(
            request,
            response=str(safe_payload.get("response") or ""),
            response_type=str(safe_payload.get("response_type") or "assistant_answer"),
            files_read=list(safe_payload.get("files_read") or []),
            graph_path=str(safe_payload.get("graph_path") or "agent_core:metadata_sanitized"),
            intent_classification=safe_payload.get("intent_classification"),
            run_id=safe_payload.get("run_id"),
            skills_used=list(safe_payload.get("skills_used") or []),
            stop_reason=str(safe_payload.get("stop_reason") or "completed_with_metadata_error"),
            loop_iteration=int(safe_payload.get("loop_iteration") or 1),
            activity_events=list(safe_payload.get("activity_events") or []),
        )


def _format_command_preview(command: list[str], preview: dict[str, Any]) -> str:
    text = " ".join(command)
    if preview.get("needs_approval"):
        return f"`{text}` requires approval before RepoOperator can run it. Reason: {preview.get('reason') or 'command policy'}"
    return f"`{text}` is allowed by command policy. I did not run a mutating command."


def _chunk_text(text: str, chunk_size: int = 96) -> Iterator[str]:
    for start in range(0, len(text or ""), chunk_size):
        chunk = text[start : start + chunk_size]
        if chunk:
            yield chunk
