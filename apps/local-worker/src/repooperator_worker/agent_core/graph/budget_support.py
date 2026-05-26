"""Loop and continuation budgets for LangGraph agent runs."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from repooperator_worker.agent_core.planner import (
    TaskFrame,
    build_task_frame,
    edit_requested,
    likely_feature_context_files,
)
from repooperator_worker.agent_core.state import AgentCoreState
from repooperator_worker.schemas import AgentRunRequest


@dataclass(frozen=True)
class LoopBudget:
    max_loop_iterations: int
    max_file_reads: int
    max_commands: int
    max_edits: int = 6
    reason: str = "default"


HARD_MAX_LOOP_ITERATIONS = 18


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
