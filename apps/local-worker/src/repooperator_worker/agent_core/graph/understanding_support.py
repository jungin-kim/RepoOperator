"""Request-understanding setup for LangGraph agent runs."""

from __future__ import annotations

from repooperator_worker.agent_core.events import append_activity_event
from repooperator_worker.agent_core.graph.budget_support import determine_loop_budget
from repooperator_worker.agent_core.graph.context_support import refresh_context_pack_for_core
from repooperator_worker.agent_core.planner import build_task_frame
from repooperator_worker.agent_core.state import AgentCoreState
from repooperator_worker.agent_core.task_policy import ensure_subtasks
from repooperator_worker.schemas import AgentRunRequest
from repooperator_worker.services.json_safe import json_safe


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
