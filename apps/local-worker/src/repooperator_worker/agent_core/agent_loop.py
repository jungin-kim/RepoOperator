from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from repooperator_worker.agent_core.actions import AgentAction, ActionResult
from repooperator_worker.agent_core.context_service import ContextService
from repooperator_worker.agent_core.events import append_work_trace
from repooperator_worker.agent_core.hooks import HookManager
from repooperator_worker.agent_core.state import AgentCoreState
from repooperator_worker.agent_core.tool_orchestrator import ToolOrchestrator
from repooperator_worker.agent_core.tools.registry import ToolRegistry
from repooperator_worker.schemas import AgentRunRequest, AgentRunResponse


@dataclass
class AgentLoopDeps:
    context_service: ContextService
    tool_registry: ToolRegistry
    tool_orchestrator: ToolOrchestrator
    hook_manager: HookManager
    load_context: Callable[[AgentCoreState, AgentRunRequest], None]
    classify: Callable[[AgentCoreState, AgentRunRequest], None]
    create_initial_plan: Callable[[AgentCoreState], None]
    emit_plan_update: Callable[[AgentCoreState, AgentRunRequest, str], None]
    should_continue: Callable[..., bool]
    check_cancel: Callable[[AgentCoreState, AgentRunRequest], None]
    consume_steering: Callable[[AgentCoreState, AgentRunRequest], None]
    choose_next_action: Callable[[AgentCoreState, AgentRunRequest], AgentAction]
    execute_action: Callable[[AgentAction], ActionResult]
    append_action_event: Callable[[AgentAction, ActionResult], None]
    observe_result: Callable[[AgentCoreState, AgentAction, ActionResult, AgentRunRequest], None]
    update_plan: Callable[[AgentCoreState, AgentAction, ActionResult, AgentRunRequest], None]
    build_final_answer: Callable[[AgentCoreState, AgentRunRequest], str]
    validate_final_answer: Callable[[str, AgentCoreState, AgentRunRequest], str]
    build_final_response: Callable[[AgentCoreState, AgentRunRequest], AgentRunResponse]


class AgentLoop:
    def __init__(self, deps: AgentLoopDeps, *, max_wall_clock_seconds: int = 300) -> None:
        self.deps = deps
        self.max_wall_clock_seconds = max_wall_clock_seconds

    def run(self, state: AgentCoreState, request: AgentRunRequest) -> AgentRunResponse:
        started = time.perf_counter()
        self.deps.load_context(state, request)
        self.deps.classify(state, request)
        self.deps.create_initial_plan(state)
        self.deps.emit_plan_update(state, request, "Created initial plan")

        while self.deps.should_continue(state, request=request, started=started, max_wall_clock_seconds=self.max_wall_clock_seconds):
            self.deps.check_cancel(state, request)
            if state.cancellation_requested:
                break
            self.deps.consume_steering(state, request)
            action = self.deps.choose_next_action(state, request)
            state.current_step = action.reason_summary
            if action.type != "final_answer":
                _emit_action_decision(state, request, action)
            if action.type == "final_answer":
                break
            if action.type == "ask_clarification":
                state.stop_reason = "needs_clarification"
                missing = ", ".join(action.payload.get("missing_files") or [])
                state.final_response = (
                    action.payload.get("question")
                    or state.classifier_result.clarification_question
                    or (f"I could not find {missing}. Please confirm the repo-relative path or choose one of the candidates I found." if missing else "Could you clarify which files or workflow you want me to inspect?")
                )
                break

            state.actions_taken.append(action)
            result = self.deps.execute_action(action)
            state.action_results.append(result)
            self.deps.append_action_event(action, result)
            self.deps.observe_result(state, action, result, request)
            self.deps.update_plan(state, action, result, request)
            self.deps.check_cancel(state, request)
            if state.cancellation_requested:
                break
            if result.status == "waiting_approval":
                state.stop_reason = "waiting_approval"
                break
            if result.status in {"failed", "cancelled", "timed_out"}:
                state.stop_reason = result.status
                break

        if not state.final_response:
            state.final_response = self.deps.build_final_answer(state, request)
        draft_response = state.final_response
        state.final_response = self.deps.validate_final_answer(state.final_response, state, request)
        if state.final_response != draft_response:
            append_work_trace(
                run_id=state.run_id,
                request=request,
                activity_id="final-synthesis-repair",
                phase="Finished",
                label="Rebuilt final answer",
                status="completed",
                safe_reasoning_summary="The draft answer did not match the gathered evidence, so I rebuilt it from collected files.",
                observation="Final answer repaired without storing the rejected draft text.",
                safety_note="Rejected draft text is not exposed in events.",
            )
        return self.deps.build_final_response(state, request)


def _emit_action_decision(state: AgentCoreState, request: AgentRunRequest, action: AgentAction) -> None:
    note = action.payload.get("visible_work_note") if isinstance(action.payload, dict) else None
    note = note if isinstance(note, dict) else {}
    phase = "Safety" if action.type in {"preview_command", "inspect_git_state", "run_approved_command"} else "Decision"
    safety_note = str(note.get("safety_note") or "")
    if not safety_note and action.type in {"preview_command", "inspect_git_state", "run_approved_command"}:
        safety_note = "Commands are checked through policy before any execution."
    if not safety_note and action.type == "generate_edit":
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
    if action.type == "generate_edit" and action.target_files:
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
    if action.type == "generate_edit":
        return "Preparing proposal-only edit"
    if action.type in {"inspect_repo_tree", "analyze_repository"}:
        return "Inspecting repository structure"
    if action.type == "ask_clarification":
        return "Preparing clarification"
    if action.type == "final_answer":
        return "Preparing final answer"
    return "Working on the next repository step"
