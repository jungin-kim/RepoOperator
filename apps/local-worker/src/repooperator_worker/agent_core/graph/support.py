"""LangGraph-native support services for RepoOperator agent runs."""

from __future__ import annotations

import json
import re
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from repooperator_worker.agent_core.actions import AgentAction, ActionResult
from repooperator_worker.agent_core.context_service import get_default_context_service
from repooperator_worker.agent_core.context_packer import pack_context
from repooperator_worker.agent_core.events import append_activity_event, append_work_trace
from repooperator_worker.agent_core.final_response import build_agent_response
from repooperator_worker.agent_core.final_synthesis import (
    _answer_with_model,
    validate_or_repair_final_answer,
)
from repooperator_worker.agent_core.planner import (
    TaskFrame,
    _format_command_result,
    _format_edit_proposal,
    _latest_command_result,
    _latest_edit_proposal,
    _repository_review_response,
    build_task_frame,
    current_edit_target_files,
    edit_requested,
    likely_feature_context_files,
    pending_commit_context,
)
from repooperator_worker.agent_core.state import AgentCoreState
from repooperator_worker.agent_core.task_policy import (
    action_operation,
    ensure_subtasks,
    group_inventory,
    repository_file_inventory,
    update_subtasks_after_action,
)
from repooperator_worker.schemas import AgentRunRequest, AgentRunResponse
from repooperator_worker.services.active_repository import get_active_repository
from repooperator_worker.services.event_service import append_run_event, get_run
from repooperator_worker.services.json_safe import json_safe, safe_agent_response_payload, safe_repr
from repooperator_worker.services.model_client import OpenAICompatibleModelClient


@dataclass(frozen=True)
class LoopBudget:
    max_loop_iterations: int
    max_file_reads: int
    max_commands: int
    max_edits: int = 6
    reason: str = "default"


HARD_MAX_LOOP_ITERATIONS = 18


def validate_active_repository(request: AgentRunRequest) -> None:
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


def load_context(state: AgentCoreState, request: AgentRunRequest) -> None:
    packet = get_default_context_service().collect(request)
    state.context_packet = packet.model_dump()
    high_signal = sorted(packet.high_signal_files)
    instructions = sorted(packet.project_instructions)
    state.observations.append("Loaded request context for the active repository.")
    append_activity_event(
        run_id=state.run_id,
        request=request,
        activity_id="langgraph-load-context",
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
        request_understanding_to_classifier_result,
        understand_request,
    )

    refresh_context_pack_for_core(state, request, "summary", "understand_request")
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
        activity_id="langgraph-frame-request",
        event_type="activity_completed",
        phase="Thinking",
        label="Framed request",
        status="completed",
        observation=f"Goal framed with {len(frame.mentioned_files)} mentioned file(s) and {len(frame.likely_capabilities)} likely capability hint(s).",
        aggregate={"task_frame": json_safe(frame), "loop_budget": json_safe(budget)},
    )


def refresh_context_pack_for_core(state: AgentCoreState, request: AgentRunRequest, kind: str, trigger_node: str) -> dict[str, Any]:
    base_context = state.context_packet if isinstance(state.context_packet, dict) else {}
    packet = pack_context(kind, request, state=_core_context_pack_state(state), base_context=base_context)
    state.context_packet = json_safe({**dict(base_context or {}), **packet})
    report = packet.get("context_pack_report") if isinstance(packet.get("context_pack_report"), dict) else {}
    summary = {
        "kind": packet.get("kind") or kind,
        "trigger_node": trigger_node,
        "compression_ratio": report.get("compression_ratio"),
        "estimated_input_tokens": report.get("estimated_input_tokens"),
        "estimated_output_reserve": report.get("estimated_output_reserve"),
        "included_sections": report.get("included_sections") or [],
        "excluded_sections": report.get("excluded_sections") or [],
        "warnings": report.get("warnings") or [],
        "retained_files": report.get("retained_files") or [],
        "omitted_files": report.get("omitted_files") or [],
        "retained_web_sources": report.get("retained_web_sources") or [],
    }
    append_activity_event(
        run_id=state.run_id,
        request=request,
        activity_id=f"context-pack:{trigger_node}",
        event_type="graph_transition",
        phase="Thinking",
        label="Packed model context",
        status="completed",
        visibility="debug",
        display="secondary",
        operation="context_pack",
        aggregate=json_safe(summary),
    )
    return json_safe(packet)


def _core_context_pack_state(state: AgentCoreState) -> dict[str, Any]:
    action_results = [result.model_dump() for result in state.action_results]
    actions_taken = [action.model_dump() for action in state.actions_taken]
    proposal: dict[str, Any] | None = None
    for result in reversed(state.action_results):
        candidate = result.payload.get("change_set_proposal") if isinstance(result.payload, dict) else None
        if isinstance(candidate, dict):
            proposal = candidate
            break
    if proposal is None:
        for result in reversed(state.action_results):
            proposals = result.payload.get("edit_proposals") if isinstance(result.payload, dict) else None
            if isinstance(proposals, list) and proposals:
                proposal = {"proposal_id": result.action_id, "status": result.status, "changes": proposals}
                break
    return {
        "actions_taken": actions_taken,
        "action_results": action_results,
        "files_read": list(state.files_read),
        "files_changed": list(state.files_changed),
        "commands_run": list(state.commands_run),
        "pending_approval": state.pending_approval,
        "change_set_proposal": proposal,
        "observations": list(state.observations),
    }


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


def should_continue(
    state: AgentCoreState,
    *,
    started: float,
    max_wall_clock_seconds: int,
    request: AgentRunRequest | None = None,
) -> bool:
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
        activity_id="langgraph-cancelled",
        event_type="activity_completed",
        phase="Finished",
        label="Run cancelled",
        status="cancelled",
        observation="Cancellation was requested. RepoOperator stopped at the next safe checkpoint.",
    )


def observe_result(state: AgentCoreState, action: AgentAction, result: ActionResult, request: AgentRunRequest) -> None:
    from repooperator_worker.agent_core.task_policy import record_ineffective_action

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
            activity_id=f"langgraph-observe:{action.action_id}",
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
        activity_id="langgraph-plan",
        event_type="activity_updated",
        phase="Planning",
        label=label,
        status="running",
        observation="; ".join(state.plan[-4:]),
        aggregate={"plan_steps": list(state.plan), "loop_iteration": state.loop_iteration},
    )


def emit_action_decision(state: AgentCoreState, request: AgentRunRequest, action: AgentAction) -> None:
    note = action.payload.get("visible_work_note") if isinstance(action.payload, dict) else None
    note = note if isinstance(note, dict) else {}
    phase = "Safety" if action.type in {"preview_command", "inspect_git_state", "run_approved_command"} else "Decision"
    safety_note = str(note.get("safety_note") or "")
    if not safety_note and action.type in {"preview_command", "inspect_git_state", "run_approved_command"}:
        safety_note = "Commands are checked through policy before any execution."
    if not safety_note and action.type in {"generate_edit", "generate_change_set"}:
        safety_note = "Edit generation is proposal-only; no files are written by this action."
    append_work_trace(
        run_id=state.run_id,
        request=request,
        activity_id=f"action:{action.action_id}",
        phase=phase,
        label=_action_label(action),
        status="completed",
        safe_reasoning_summary=str(note.get("why_this_action") or action.reason_summary),
        current_action=_action_current_text(action),
        next_action=action.expected_output,
        evidence_needed=[str(item) for item in note.get("evidence_needed") or []],
        uncertainty=[str(item) for item in note.get("uncertainty") or []],
        safety_note=safety_note or None,
        related_files=list(action.target_files),
        command=action.command,
        aggregate={"action_type": action.type},
    )


def _action_current_text(action: AgentAction) -> str:
    if action.type == "read_file" and action.target_files:
        return "Read " + ", ".join(f"`{path}`" for path in action.target_files[:6]) + "."
    if action.type in {"search_files", "search_text"}:
        queries = []
        if isinstance(action.payload, dict):
            queries = [str(item) for item in action.payload.get("queries") or []]
            query = action.payload.get("query")
            if query:
                queries.append(str(query))
        return "Search repository evidence" + (f" for {', '.join(queries[:4])}." if queries else ".")
    if action.type in {"generate_edit", "generate_change_set"} and action.target_files:
        return "Prepare a proposal-only patch for " + ", ".join(f"`{path}`" for path in action.target_files[:4]) + "."
    if action.command:
        return "Preview or run command through policy: `" + " ".join(action.command) + "`."
    if action.type == "ask_clarification":
        return "Ask for the missing information needed to proceed safely."
    if action.type == "final_answer":
        return "Prepare the final answer from gathered evidence."
    return action.reason_summary


def _action_label(action: AgentAction) -> str:
    if action.type == "read_file" and action.target_files:
        first = action.target_files[0]
        return f"Reading {first}" if len(action.target_files) == 1 else "Reading repository files"
    if action.type in {"search_files", "search_text"}:
        return "Searching for repository evidence"
    if action.type in {"preview_command", "inspect_git_state", "run_approved_command"}:
        return "Checking command safety"
    if action.type in {"generate_edit", "generate_change_set"}:
        return "Preparing proposal-only edit"
    if action.type in {"inspect_repo_tree", "analyze_repository"}:
        return "Inspecting repository structure"
    if action.type == "ask_clarification":
        return "Preparing clarification"
    if action.type == "final_answer":
        return "Preparing final answer"
    return "Working on the next repository step"


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


def build_final_response(state: AgentCoreState, request: AgentRunRequest) -> AgentRunResponse:
    review_response = _repository_review_response(state)
    if review_response and state.stop_reason not in {"cancelled", "waiting_approval"}:
        return _response_json_safe(review_response.model_copy(update={"loop_iteration": state.loop_iteration, "agent_flow": "langgraph"}), request)
    response_type = "command_approval" if state.pending_approval else "assistant_answer"
    graph_path = "agent_core:" + (
        "cancelled" if state.stop_reason == "cancelled"
        else "command_preview" if state.pending_approval
        else "read_file_answer" if state.files_read
        else "general_answer"
    )
    return _response_json_safe(
        build_agent_response(
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
            agent_flow="langgraph",
        ),
        request,
    )


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
            agent_flow="langgraph",
        )


def _format_command_preview(command: list[str], preview: dict[str, Any]) -> str:
    text = " ".join(command)
    if preview.get("needs_approval"):
        return f"`{text}` requires approval before RepoOperator can run it. Reason: {preview.get('reason') or 'command policy'}"
    return f"`{text}` is allowed by command policy. I did not run a mutating command."


def _dedupe_text(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out
