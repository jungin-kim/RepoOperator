"""State schema and reducers for the RepoOperator LangGraph runtime."""

from __future__ import annotations

import time
from typing import Annotated, Any, TypedDict

from repooperator_worker.agent_core.change_set import EditMode
from repooperator_worker.agent_core.graph_state import classifier_to_snapshot, request_to_snapshot
from repooperator_worker.agent_core.state import ClassifierResult
from repooperator_worker.schemas import AgentRunRequest

APPEND_REDUCER_FIELDS = {
    "messages",
    "events",
    "events_to_emit",
    "actions_taken",
    "action_results",
    "observations",
    "commands_run",
    "validation_results",
    "subtask_updates",
    "evidence_reports",
    "file_role_reports",
    "proposed_changes",
    "risk_notes",
    "worker_tasks",
    "worker_reports",
    "proposal_errors",
    "attempts",
    "visible_rationale_log",
    "evidence_basis_history",
    "understanding_history",
}

UNIQUE_APPEND_REDUCER_FIELDS = {
    "files_read",
    "files_changed",
}

def append_items(left: list[Any] | None, right: list[Any] | Any | None) -> list[Any]:
    """LangGraph reducer for append-only state channels."""
    combined = list(left or [])
    if right is None:
        return combined
    if isinstance(right, list):
        combined.extend(right)
    else:
        combined.append(right)
    return combined

def append_unique_items(left: list[Any] | None, right: list[Any] | Any | None) -> list[Any]:
    """Append while preserving first-seen order for file/path channels."""
    combined = list(left or [])
    incoming = right if isinstance(right, list) else ([] if right is None else [right])
    for item in incoming:
        if item not in combined:
            combined.append(item)
    return combined

class RepoOperatorGraphState(TypedDict, total=False):
    request_snapshot: dict[str, Any]
    run_id: str
    thread_id: str | None
    repo: str
    branch: str | None
    context_packet: dict[str, Any] | None
    capability_snapshot: dict[str, Any] | None
    model_profile_snapshot: dict[str, Any] | None
    context_pack_summary: dict[str, Any] | None
    context_pack_report: dict[str, Any] | None
    short_term_memory: dict[str, Any] | None
    request_understanding_snapshot: dict[str, Any] | None
    user_understanding_context: dict[str, Any] | None
    evidence_basis: dict[str, Any] | None
    visible_rationale_log: Annotated[list[dict[str, Any]], append_items]
    evidence_basis_history: Annotated[list[dict[str, Any]], append_items]
    understanding_history: Annotated[list[dict[str, Any]], append_items]
    classifier_snapshot: dict[str, Any]
    task_frame_snapshot: dict[str, Any] | None
    subtasks: list[dict[str, Any]]
    current_subtask_id: str | None
    plan: list[str]
    messages: Annotated[list[dict[str, Any]], append_items]
    events: Annotated[list[dict[str, Any]], append_items]
    actions_taken: Annotated[list[dict[str, Any]], append_items]
    action_results: Annotated[list[dict[str, Any]], append_items]
    observations: Annotated[list[str], append_items]
    evidence_store: dict[str, Any]
    files_read: Annotated[list[str], append_unique_items]
    files_changed: Annotated[list[str], append_unique_items]
    commands_run: Annotated[list[str], append_unique_items]
    pending_approval: dict[str, Any] | None
    change_set_proposal: dict[str, Any] | None
    validation_results: Annotated[list[dict[str, Any]], append_items]
    repair_attempts: int
    final_response: str
    response_snapshot: dict[str, Any] | None
    stop_reason: str | None
    loop_iteration: int
    budgets: dict[str, Any]
    events_to_emit: Annotated[list[dict[str, Any]], append_items]
    subtask_updates: Annotated[list[dict[str, Any]], append_items]
    evidence_reports: Annotated[list[dict[str, Any]], append_items]
    file_role_reports: Annotated[list[dict[str, Any]], append_items]
    proposed_changes: Annotated[list[dict[str, Any]], append_items]
    risk_notes: Annotated[list[str], append_items]
    worker_tasks: Annotated[list[dict[str, Any]], append_items]
    worker_reports: Annotated[list[dict[str, Any]], append_items]
    proposal_errors: Annotated[list[str], append_items]
    attempts: Annotated[list[dict[str, Any]], append_items]
    edit_mode: EditMode
    proposal_id: str | None
    proposal_status: str | None
    apply_status: str | None
    applied_change_set_id: str | None
    post_apply_validation_status: str | None
    supervisor_mode: bool
    current_worker_role: str | None
    pending_action: dict[str, Any] | None
    next_node: str | None
    routing_stage: str
    graph_started_at: float
    checkpoint_sequence: int
    skills_used: list[str]
    skills_context: str
    memories_used: list[str]
    recommendation_context: dict[str, Any] | None
    cancellation_requested: bool
    current_step: str | None
    zero_result_queries: list[str]
    failed_action_signatures: list[str]
    strategy_shifts: list[str]
    max_loop_iterations: int
    max_file_reads: int
    max_commands: int
    max_edits: int
    stream_final_answer: bool
    evidence_goal: dict[str, Any] | None
    evidence_done: bool
    analysis_done: bool
    edit_done: bool
    validation_done: bool
    approval_decision: dict[str, Any] | None
    git_workflow: dict[str, Any] | None
    routine_context: dict[str, Any] | None

def graph_config_for_request(request: AgentRunRequest, run_id: str) -> dict[str, Any]:
    stable = "|".join([run_id, request.thread_id or "", request.project_path, request.branch or ""])
    return {"configurable": {"thread_id": stable}}

def initial_graph_state(
    request: AgentRunRequest,
    *,
    run_id: str,
    stream_final_answer: bool = False,
    skills_context: str = "",
    skills_used: list[str] | None = None,
) -> RepoOperatorGraphState:
    return {
        "request_snapshot": request_to_snapshot(request),
        "run_id": run_id,
        "thread_id": request.thread_id,
        "repo": request.project_path,
        "branch": request.branch,
        "context_packet": None,
        "capability_snapshot": None,
        "model_profile_snapshot": None,
        "context_pack_summary": None,
        "context_pack_report": None,
        "short_term_memory": None,
        "request_understanding_snapshot": None,
        "user_understanding_context": None,
        "evidence_basis": None,
        "visible_rationale_log": [],
        "evidence_basis_history": [],
        "understanding_history": [],
        "classifier_snapshot": classifier_to_snapshot(ClassifierResult()),
        "task_frame_snapshot": None,
        "subtasks": [],
        "current_subtask_id": None,
        "plan": [],
        "messages": [],
        "events": [],
        "actions_taken": [],
        "action_results": [],
        "observations": [],
        "evidence_store": {},
        "files_read": [],
        "files_changed": [],
        "commands_run": [],
        "pending_approval": None,
        "change_set_proposal": None,
        "validation_results": [],
        "repair_attempts": 0,
        "final_response": "",
        "response_snapshot": None,
        "stop_reason": None,
        "loop_iteration": 0,
        "budgets": {},
        "events_to_emit": [],
        "subtask_updates": [],
        "evidence_reports": [],
        "file_role_reports": [],
        "proposed_changes": [],
        "risk_notes": [],
        "worker_tasks": [],
        "worker_reports": [],
        "proposal_errors": [],
        "attempts": [],
        "edit_mode": "explanation_only",
        "proposal_id": None,
        "proposal_status": None,
        "apply_status": None,
        "applied_change_set_id": None,
        "post_apply_validation_status": None,
        "supervisor_mode": False,
        "current_worker_role": None,
        "pending_action": None,
        "next_node": None,
        "routing_stage": "after_understanding",
        "graph_started_at": time.perf_counter(),
        "checkpoint_sequence": 0,
        "skills_used": list(skills_used or []),
        "skills_context": skills_context,
        "memories_used": [],
        "recommendation_context": None,
        "cancellation_requested": False,
        "current_step": None,
        "zero_result_queries": [],
        "failed_action_signatures": [],
        "strategy_shifts": [],
        "max_loop_iterations": 8,
        "max_file_reads": 40,
        "max_commands": 8,
        "max_edits": 6,
        "stream_final_answer": stream_final_answer,
        "evidence_goal": None,
        "evidence_done": False,
        "analysis_done": False,
        "edit_done": False,
        "validation_done": False,
        "approval_decision": None,
        "git_workflow": None,
        "routine_context": None,
    }
