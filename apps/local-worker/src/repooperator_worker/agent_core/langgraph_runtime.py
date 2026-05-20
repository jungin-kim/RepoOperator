"""LangGraph orchestration runtime for RepoOperator."""

from __future__ import annotations

import copy
import difflib
import re
import shlex
import time
from typing import Annotated, Any, Iterator, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from repooperator_worker.agent_core.actions import AgentAction, ActionResult
from repooperator_worker.agent_core.change_set import (
    ChangeSetProposal,
    change_set_from_payload,
    EditMode,
    plan_change_set,
    proposal_from_edit_result,
    validate_change_set as validate_change_set_model,
)
from repooperator_worker.agent_core.graph_checkpoints import EventServiceLangGraphSaver
from repooperator_worker.agent_core.graph_routes import choose_graph_next_action
from repooperator_worker.agent_core.graph_state import (
    action_from_snapshot,
    action_to_snapshot,
    classifier_from_snapshot,
    classifier_to_snapshot,
    request_from_snapshot,
    request_to_snapshot,
    request_understanding_from_snapshot,
    request_understanding_to_snapshot,
    response_from_snapshot,
    response_to_snapshot,
    result_from_snapshot,
    result_to_snapshot,
    subtask_from_snapshot,
    subtask_to_snapshot,
    task_frame_from_snapshot,
    task_frame_to_snapshot,
)
from repooperator_worker.agent_core.hooks import HookManager
from repooperator_worker.agent_core.planner import (
    build_task_frame,
    candidate_files_from_results,
    edit_requested,
)
from repooperator_worker.agent_core.state import AgentCoreState, ClassifierResult
from repooperator_worker.agent_core.task_policy import (
    next_evidence_gathering_action,
)
from repooperator_worker.agent_core.tool_orchestrator import ToolOrchestrator
from repooperator_worker.agent_core.tools.registry import get_default_tool_registry
from repooperator_worker.schemas import AgentRunRequest, AgentRunResponse
from repooperator_worker.services.event_service import append_run_event, list_run_events
from repooperator_worker.services.json_safe import json_safe, safe_agent_response_payload
from repooperator_worker.services.skills_service import enabled_skill_context


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
    request_understanding_snapshot: dict[str, Any] | None
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


_DEFAULT_LANGGRAPH_CHECKPOINTER = EventServiceLangGraphSaver()


def get_default_langgraph_checkpointer() -> InMemorySaver:
    return _DEFAULT_LANGGRAPH_CHECKPOINTER


def build_repooperator_state_graph() -> StateGraph:
    graph = StateGraph(RepoOperatorGraphState)
    graph.add_node("load_context", load_context_node)
    graph.add_node("understand_request", understand_request_node)
    graph.add_node("build_task_plan", build_task_plan_node)
    graph.add_node("route_next", route_next_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("gather_evidence", gather_evidence_node)
    graph.add_node("analysis_graph", analysis_graph_node)
    graph.add_node("execute_tool", execute_tool_node)
    graph.add_node("validate_result", validate_result_node)
    graph.add_node("plan_change_set", plan_change_set_node)
    graph.add_node("generate_change_set", generate_change_set_node)
    graph.add_node("validate_change_set", validate_change_set_node)
    graph.add_node("repair_change_set", repair_change_set_node)
    graph.add_node("ask_clarification", ask_clarification_node)
    graph.add_node("await_approval", await_approval_node)
    graph.add_node("await_change_approval", await_approval_node)
    graph.add_node("apply_change_set", apply_change_set_node)
    graph.add_node("post_apply_validation", post_apply_validation_node)
    graph.add_node("final_synthesis", final_synthesis_node)

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "understand_request")
    graph.add_edge("understand_request", "build_task_plan")
    graph.add_edge("build_task_plan", "route_next")
    graph.add_conditional_edges(
        "route_next",
        route_to_next_node,
        {
            "supervisor": "supervisor",
            "gather_evidence": "gather_evidence",
            "analysis_graph": "analysis_graph",
            "execute_tool": "execute_tool",
            "validate_result": "validate_result",
            "plan_change_set": "plan_change_set",
            "generate_change_set": "generate_change_set",
            "validate_change_set": "validate_change_set",
            "repair_change_set": "repair_change_set",
            "ask_clarification": "ask_clarification",
            "await_approval": "await_approval",
            "await_change_approval": "await_change_approval",
            "apply_change_set": "apply_change_set",
            "post_apply_validation": "post_apply_validation",
            "final_synthesis": "final_synthesis",
            END: END,
        },
    )
    graph.add_edge("supervisor", "route_next")
    graph.add_edge("gather_evidence", "validate_result")
    graph.add_edge("analysis_graph", "validate_result")
    graph.add_edge("execute_tool", "validate_result")
    graph.add_edge("validate_result", "route_next")
    graph.add_edge("plan_change_set", "generate_change_set")
    graph.add_edge("generate_change_set", "validate_change_set")
    graph.add_conditional_edges(
        "validate_change_set",
        route_after_change_plan,
        {
            "repair_change_set": "repair_change_set",
            "route_next": "route_next",
            "await_approval": "await_approval",
            "await_change_approval": "await_change_approval",
            "final_synthesis": "final_synthesis",
        },
    )
    graph.add_edge("repair_change_set", "route_next")
    graph.add_edge("ask_clarification", "final_synthesis")
    graph.add_edge("await_approval", "route_next")
    graph.add_edge("await_change_approval", "route_next")
    graph.add_edge("apply_change_set", "post_apply_validation")
    graph.add_edge("post_apply_validation", "final_synthesis")
    graph.add_edge("final_synthesis", END)
    return graph


def build_compiled_repooperator_graph(*, checkpoint_adapter: Any | None = None) -> Any:
    checkpointer = checkpoint_adapter if _is_langgraph_checkpointer(checkpoint_adapter) else get_default_langgraph_checkpointer()
    return build_repooperator_state_graph().compile(checkpointer=checkpointer)


def build_evidence_gathering_graph() -> StateGraph:
    graph = StateGraph(RepoOperatorGraphState)
    graph.add_node("route_evidence_next", route_evidence_next_node)
    graph.add_node("inspect_tree", evidence_inspect_tree_node)
    graph.add_node("rank_candidates", evidence_rank_candidates_node)
    graph.add_node("search_files", evidence_search_files_node)
    graph.add_node("search_text", evidence_search_text_node)
    graph.add_node("read_files", evidence_read_files_node)
    graph.add_node("update_evidence_store", update_evidence_store_node)
    graph.add_edge(START, "route_evidence_next")
    graph.add_conditional_edges(
        "route_evidence_next",
        route_evidence_next,
        {
            "inspect_tree": "inspect_tree",
            "rank_candidates": "rank_candidates",
            "search_files": "search_files",
            "search_text": "search_text",
            "read_files": "read_files",
            "update_evidence_store": "update_evidence_store",
            END: END,
        },
    )
    graph.add_edge("inspect_tree", "update_evidence_store")
    graph.add_edge("rank_candidates", "update_evidence_store")
    graph.add_edge("search_files", "update_evidence_store")
    graph.add_edge("search_text", "update_evidence_store")
    graph.add_edge("read_files", "update_evidence_store")
    graph.add_edge("update_evidence_store", END)
    return graph


def build_analysis_graph() -> StateGraph:
    graph = StateGraph(RepoOperatorGraphState)
    graph.add_node("inventory", analysis_inventory_node)
    graph.add_node("group_files", analysis_batch_files_node)
    graph.add_node("dispatch_file_role_workers", analysis_file_role_node)
    graph.add_node("reduce_file_reports", analysis_reduce_file_reports_node)
    graph.add_node("summarize_batch", analysis_summarize_batch_node)
    graph.add_node("route_batch_continue_or_end", analysis_route_batch_node)
    graph.add_edge(START, "inventory")
    graph.add_edge("inventory", "group_files")
    graph.add_edge("group_files", "dispatch_file_role_workers")
    graph.add_edge("dispatch_file_role_workers", "reduce_file_reports")
    graph.add_edge("reduce_file_reports", "summarize_batch")
    graph.add_edge("summarize_batch", "route_batch_continue_or_end")
    graph.add_edge("route_batch_continue_or_end", END)
    return graph


def build_edit_graph() -> StateGraph:
    graph = StateGraph(RepoOperatorGraphState)
    graph.add_node("route_edit_next", route_edit_next_node)
    graph.add_node("locate_targets", edit_locate_targets_node)
    graph.add_node("plan_change_set", edit_plan_change_set_node)
    graph.add_node("generate_change_set", edit_generate_change_set_node)
    graph.add_node("validate_change_set", edit_validate_change_set_node)
    graph.add_node("repair_change_set", edit_repair_change_set_node)
    graph.add_edge(START, "route_edit_next")
    graph.add_conditional_edges(
        "route_edit_next",
        route_edit_next,
        {
            "locate_targets": "locate_targets",
            "plan_change_set": "plan_change_set",
            "generate_change_set": "generate_change_set",
            "validate_change_set": "validate_change_set",
            "repair_change_set": "repair_change_set",
            END: END,
        },
    )
    graph.add_edge("locate_targets", "plan_change_set")
    graph.add_edge("plan_change_set", "generate_change_set")
    graph.add_edge("generate_change_set", "validate_change_set")
    graph.add_conditional_edges("validate_change_set", route_edit_after_validation, {"repair_change_set": "repair_change_set", END: END})
    graph.add_edge("repair_change_set", END)
    return graph


def build_validation_graph() -> StateGraph:
    graph = StateGraph(RepoOperatorGraphState)
    graph.add_node("choose_validation", validation_choose_node)
    graph.add_node("preview_command", validation_preview_command_node)
    graph.add_node("approval_interrupt_if_needed", validation_approval_interrupt_node)
    graph.add_node("run_safe_validation", validation_run_safe_node)
    graph.add_node("parse_errors", validation_parse_errors_node)
    graph.add_node("update_validation_result", validation_update_result_node)
    graph.add_node("route_validation_next", validation_route_next_node)
    graph.add_edge(START, "choose_validation")
    graph.add_edge("choose_validation", "preview_command")
    graph.add_edge("preview_command", "approval_interrupt_if_needed")
    graph.add_edge("approval_interrupt_if_needed", "run_safe_validation")
    graph.add_edge("run_safe_validation", "parse_errors")
    graph.add_edge("parse_errors", "update_validation_result")
    graph.add_edge("update_validation_result", "route_validation_next")
    graph.add_edge("route_validation_next", END)
    return graph


def build_finalization_graph() -> StateGraph:
    graph = StateGraph(RepoOperatorGraphState)
    graph.add_node("quality_guard", final_quality_guard_node)
    graph.add_node("repair_final_answer", final_repair_answer_node)
    graph.add_node("build_response", final_build_response_node)
    graph.add_node("emit_final_message", final_emit_message_node)
    graph.add_edge(START, "quality_guard")
    graph.add_edge("quality_guard", "repair_final_answer")
    graph.add_edge("repair_final_answer", "build_response")
    graph.add_edge("build_response", "emit_final_message")
    graph.add_edge("emit_final_message", END)
    return graph


def build_supervisor_graph() -> StateGraph:
    graph = StateGraph(RepoOperatorGraphState)
    graph.add_node("build_worker_tasks", supervisor_build_worker_tasks_node)
    graph.add_node("run_worker_task", supervisor_run_worker_tasks_node)
    graph.add_node("reduce_worker_reports", supervisor_reduce_worker_reports_node)
    graph.add_edge(START, "build_worker_tasks")
    graph.add_edge("build_worker_tasks", "run_worker_task")
    graph.add_edge("run_worker_task", "reduce_worker_reports")
    graph.add_edge("reduce_worker_reports", END)
    return graph


def run_langgraph_controller(
    request: AgentRunRequest,
    *,
    run_id: str | None = None,
    stream_final_answer: bool = False,
    checkpoint_adapter: Any | None = None,
) -> AgentRunResponse:
    run_id = run_id or "run_controller"
    _controller()._validate_active_repository(request)
    skills_context, skills_used = enabled_skill_context()
    initial_state = initial_graph_state(
        request,
        run_id=run_id,
        stream_final_answer=stream_final_answer,
        skills_context=skills_context,
        skills_used=skills_used,
    )
    compiled = build_compiled_repooperator_graph(checkpoint_adapter=checkpoint_adapter)
    config = graph_config_for_request(request, run_id)
    final_state = compiled.invoke(initial_state, config=config)
    if final_state.get("__interrupt__"):
        snapshot_state = dict(compiled.get_state(config).values or {})
        snapshot_state.setdefault("request_snapshot", request_to_snapshot(request))
        snapshot_state.setdefault("run_id", run_id)
        snapshot_state.setdefault("thread_id", request.thread_id)
        snapshot_state.setdefault("repo", request.project_path)
        snapshot_state.setdefault("branch", request.branch)
        return _response_from_interrupted_state(snapshot_state, request)
    response = response_from_snapshot(final_state.get("response_snapshot"))
    if isinstance(response, AgentRunResponse):
        return response
    core = _core_state_from_graph(final_state)
    if not core.final_response:
        core.final_response = final_state.get("final_response") or ""
    return _controller().build_final_response(core, request).model_copy(update={"agent_flow": "langgraph"})


def resume_langgraph_controller(
    request: AgentRunRequest,
    *,
    run_id: str,
    approval_decision: dict[str, Any],
    checkpoint_adapter: Any | None = None,
) -> AgentRunResponse:
    compiled = build_compiled_repooperator_graph(checkpoint_adapter=checkpoint_adapter)
    config = graph_config_for_request(request, run_id)
    final_state = compiled.invoke(Command(resume=json_safe(approval_decision)), config=config)
    if final_state.get("__interrupt__"):
        return _response_from_interrupted_state(dict(compiled.get_state(config).values or {}), request)
    response = response_from_snapshot(final_state.get("response_snapshot"))
    if isinstance(response, AgentRunResponse):
        return response
    core = _core_state_from_graph({**dict(final_state), "request_snapshot": request_to_snapshot(request), "run_id": run_id})
    return _controller().build_final_response(core, request).model_copy(update={"agent_flow": "langgraph"})


def stream_langgraph_controller(request: AgentRunRequest, *, run_id: str | None = None) -> Iterator[dict[str, Any]]:
    resolved_run_id = run_id or "run_controller"
    before_sequence = _latest_sequence(resolved_run_id)
    response = run_langgraph_controller(request, run_id=resolved_run_id, stream_final_answer=True)
    for event in list_run_events(resolved_run_id, after_sequence=before_sequence):
        if event.get("type") == "assistant_delta":
            before_sequence = int(event.get("sequence") or before_sequence)
            yield event
    if not any(event.get("type") == "assistant_delta" for event in list_run_events(resolved_run_id)):
        for chunk in _chunk_text(response.response):
            yield {"type": "assistant_delta", "delta": chunk, "streaming_mode": "post_hoc_chunking"}
    final = _controller()._response_json_safe(response.model_copy(update={"activity_events": []}), request)
    yield {"type": "final_message", "result": safe_agent_response_payload(final)}


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
        "request_understanding_snapshot": None,
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
    }


def load_context_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    request = _request(state)
    core = _core_state_from_graph(state)
    _controller().load_context(core, request)
    update = _updates_from_core(state, core)
    update["events_to_emit"] = [_graph_transition_event(state, "load_context", operation="load_context")]
    return _with_checkpoint_bump(update)


def understand_request_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    request = _request(state)
    core = _core_state_from_graph(state)
    _controller().classify(core, request)
    task_frame = _controller().build_task_frame(request, core)
    update = _updates_from_core(state, core)
    update["task_frame_snapshot"] = task_frame_to_snapshot(task_frame)
    update["edit_mode"] = _edit_mode_for_request(task_frame)
    update["budgets"] = {
        "max_loop_iterations": core.max_loop_iterations,
        "max_file_reads": core.max_file_reads,
        "max_commands": core.max_commands,
        "max_edits": core.max_edits,
    }
    update["events_to_emit"] = [_graph_transition_event(state, "understand_request", operation="understand_request")]
    return _with_checkpoint_bump(update)


def build_task_plan_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    request = _request(state)
    core = _core_state_from_graph(state)
    _controller().create_initial_plan(core)
    _controller().emit_plan_update(core, request, "Created initial plan")
    update = _updates_from_core(state, core)
    update["events_to_emit"] = [_graph_transition_event(state, "build_task_plan", operation="plan")]
    return _with_checkpoint_bump(update)


def route_next_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    request = _request(state)
    core = _core_state_from_graph(state)
    existing_action = _pending_action(state)
    if existing_action:
        route = route_by_stage(state)
        return _with_checkpoint_bump(
            {
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
        )
    if state.get("stop_reason") == "waiting_approval":
        return {"next_node": "await_approval", "events_to_emit": [_graph_transition_event(state, "route_next", operation="approval_gate")]}
    if state.get("stop_reason") in {"needs_clarification"}:
        return {"next_node": "ask_clarification", "events_to_emit": [_graph_transition_event(state, "route_next", operation="clarification")]}
    if state.get("stop_reason") in {"approval_denied"}:
        return {"next_node": "final_synthesis", "events_to_emit": [_graph_transition_event(state, "route_next", operation="approval_denied")]}

    should_continue = _controller().should_continue(
        core,
        request=request,
        started=float(state.get("graph_started_at") or time.perf_counter()),
        max_wall_clock_seconds=300,
    )
    update = _updates_from_core(state, core)
    if not should_continue:
        update.update({"next_node": "final_synthesis", "pending_action": None})
        update["events_to_emit"] = [_graph_transition_event(state, "route_next", operation="stop_budget")]
        return _with_checkpoint_bump(update)

    _controller().check_cancel(core, request)
    if core.cancellation_requested:
        update = _updates_from_core(state, core)
        update.update({"next_node": "final_synthesis", "pending_action": None})
        update["events_to_emit"] = [_graph_transition_event(state, "route_next", operation="cancelled")]
        return _with_checkpoint_bump(update)

    from repooperator_worker.agent_core.steering import consume_steering_for_state

    consume_steering_for_state(core, request)
    action = choose_graph_next_action(core, request)
    core.current_step = action.reason_summary
    update = _updates_from_core(state, core)
    action_snapshot = action_to_snapshot(action)
    route = route_by_stage({**dict(state), **update, "pending_action": action_snapshot})
    update.update(
        {
            "pending_action": action_snapshot,
            "next_node": route,
            "current_step": action.reason_summary,
            "task_frame_snapshot": task_frame_to_snapshot(_controller().build_task_frame(request, core)),
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
    if stage == "after_approval":
        return route_after_approval(state)
    return route_after_understanding(state)


def route_after_understanding(state: RepoOperatorGraphState) -> str:
    if _should_use_supervisor(state):
        return "supervisor"
    return _route_to_final_or_action(state)


def route_after_evidence(state: RepoOperatorGraphState) -> str:
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
    if state.get("stop_reason") in {"cancelled", "timed_out"}:
        return "final_synthesis"
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


def route_after_interrupt_resume(state: RepoOperatorGraphState) -> str:
    if _pending_action(state):
        if _pending_action(state).type == "apply_change_set":
            return "apply_change_set"
        return "execute_tool"
    if state.get("stop_reason") == "approval_denied":
        return "final_synthesis"
    return route_to_final_or_continue(state)


def route_to_final_or_continue(state: RepoOperatorGraphState) -> str:
    if state.get("stop_reason") in {"cancelled", "timed_out", "max_loop_iterations", "max_file_reads", "max_commands", "waiting_approval", "approval_denied"}:
        return "final_synthesis"
    return _route_to_final_or_action(state)


def supervisor_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    update = _invoke_subgraph_delta(build_supervisor_graph, state)
    update["supervisor_mode"] = True
    update["routing_stage"] = "after_understanding"
    update.setdefault("events_to_emit", []).append(
        _graph_transition_event(
            state,
            "supervisor",
            subgraph="supervisor",
            operation="delegate_reduce",
            status="completed",
            aggregate={"workers": ["EvidenceAgent", "AnalysisAgent", "EditPlanningAgent", "ValidationAgent", "DocumentationAgent", "TestAgent"]},
        )
    )
    return _with_checkpoint_bump(update)


def gather_evidence_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    update = _invoke_subgraph_delta(build_evidence_gathering_graph, state)
    update["routing_stage"] = "after_evidence"
    update.setdefault("events_to_emit", []).append(
        _graph_transition_event(state, "gather_evidence", subgraph="evidence_gathering_graph", operation="gather_evidence")
    )
    return _with_checkpoint_bump(update)


def analysis_graph_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    update = _invoke_subgraph_delta(build_analysis_graph, state)
    update["routing_stage"] = "after_evidence"
    update.setdefault("events_to_emit", []).append(
        _graph_transition_event(state, "analysis_graph", subgraph="analysis_graph", operation="analyze")
    )
    return _with_checkpoint_bump(update)


def execute_tool_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    update = _execute_pending_action(state, subgraph=None)
    update["routing_stage"] = "after_tool_result"
    return _with_checkpoint_bump(update)


def validate_result_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    latest = _latest_result(state)
    validation: dict[str, Any] = {
        "status": latest.status if latest else "skipped",
        "action_id": latest.action_id if latest else None,
        "errors": list(latest.errors if latest else []),
    }
    stop_reason = state.get("stop_reason")
    if latest and latest.status == "waiting_approval":
        stop_reason = "waiting_approval"
    elif latest and latest.status in {"cancelled", "timed_out"}:
        stop_reason = latest.status
    return _with_checkpoint_bump(
        {
            "validation_results": [validation],
            "stop_reason": stop_reason,
            "routing_stage": "after_validation",
            "events_to_emit": [
                _graph_transition_event(
                    state,
                    "validate_result",
                    subgraph="validation_graph",
                    operation="validate_result",
                    status=validation["status"],
                    validation_result=validation,
                )
            ],
        }
    )


def plan_change_set_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    action = _pending_action(state)
    proposal = plan_change_set(
        list(action.target_files if action else []),
        action.reason_summary if action else "Plan proposal-only change set.",
    ).model_dump()
    proposal["action_id"] = action.action_id if action else None
    return _with_checkpoint_bump(
        {
            "change_set_proposal": proposal,
            "proposal_id": proposal.get("proposal_id"),
            "proposal_status": proposal.get("status"),
            "apply_status": "not_applied",
            "proposed_changes": [proposal],
            "routing_stage": "after_change_plan",
            "events_to_emit": [
                _graph_transition_event(
                    state,
                    "plan_change_set",
                    subgraph="edit_graph",
                    operation="plan_change_set",
                    files=list((proposal.get("plan") or {}).get("target_files") or []),
                )
            ],
        }
    )


def generate_change_set_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    update = _invoke_subgraph_delta(build_edit_graph, state)
    update["routing_stage"] = "after_change_plan"
    update.setdefault("events_to_emit", []).append(
        _graph_transition_event(state, "generate_change_set", subgraph="edit_graph", operation="generate_change_set")
    )
    return _with_checkpoint_bump(update)


def validate_change_set_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    latest = _latest_result(state)
    proposal = _change_set_from_latest_result(state, latest) or state.get("change_set_proposal") or {}
    if isinstance(proposal, dict) and proposal.get("changes"):
        typed = change_set_from_payload(proposal)
        validation_model = validate_change_set_model(typed, repo=str(state.get("repo") or _request(state).project_path))
        typed.validation = validation_model
        typed.status = validation_model.status
        typed.validation_status = validation_model.status
        typed.proposal_error = "; ".join(validation_model.errors) if validation_model.errors else None
        proposal = typed.model_dump()
    validation = {
        "kind": "change_set",
        "status": (proposal.get("status") if proposal else None) or (latest.status if latest else "skipped"),
        "action_id": latest.action_id if latest else None,
        "proposal_files": [str(item.get("path")) for item in proposal.get("changes") or [] if isinstance(item, dict)],
        "errors": list((proposal.get("validation") or {}).get("errors") or []),
    }
    pending_approval = None
    stop_reason = state.get("stop_reason")
    final_response = state.get("final_response") or ""
    if validation["status"] == "valid" and proposal.get("changes") and not proposal.get("applied"):
        proposal_id = str(proposal.get("proposal_id") or "")
        pending_approval = {
            "kind": "change_set_apply",
            "proposal_id": proposal_id,
            "change_set_proposal": json_safe(proposal),
            "reason": "Applying this validated change set will modify files and requires approval.",
        }
        stop_reason = "waiting_approval"
        final_response = _final_text_for_change_set(state, proposal)
    return _with_checkpoint_bump(
        {
            "change_set_proposal": proposal,
            "pending_approval": pending_approval if pending_approval is not None else state.get("pending_approval"),
            "proposal_id": proposal.get("proposal_id") if isinstance(proposal, dict) else None,
            "proposal_status": validation["status"],
            "apply_status": "pending" if pending_approval else state.get("apply_status"),
            "stop_reason": stop_reason,
            "final_response": final_response,
            "validation_results": [validation],
            "proposal_errors": validation["errors"],
            "routing_stage": "after_change_plan",
            "events_to_emit": [
                _graph_transition_event(
                    state,
                    "validate_change_set",
                    subgraph="edit_graph",
                    operation="validate_change_set",
                    status=validation["status"],
                    files=validation["proposal_files"],
                    validation_result=validation,
                )
            ],
        }
    )


def repair_change_set_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    attempts = int(state.get("repair_attempts") or 0) + 1
    return _with_checkpoint_bump(
        {
            "repair_attempts": attempts,
            "risk_notes": ["Change-set repair requested after validation failed."],
            "routing_stage": "after_change_plan",
            "events_to_emit": [
                _graph_transition_event(state, "repair_change_set", subgraph="edit_graph", operation="repair_change_set", status="completed")
            ],
        }
    )


def ask_clarification_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    action = _pending_action(state)
    request = _request(state)
    core = _core_state_from_graph(state)
    missing = ", ".join(action.payload.get("missing_files") or []) if action else ""
    final_response = (
        action.payload.get("question")
        if action
        else None
    ) or core.classifier_result.clarification_question or (
        f"I could not find {missing}. Please confirm the repo-relative path or choose one of the candidates I found."
        if missing
        else "Could you clarify which files or workflow you want me to inspect?"
    )
    del request
    return _with_checkpoint_bump(
        {
            "stop_reason": "needs_clarification",
            "final_response": final_response,
            "events_to_emit": [
                _graph_transition_event(state, "ask_clarification", operation="clarification", action_type="ask_clarification")
            ],
        }
    )


def await_approval_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    payload = _approval_interrupt_payload(state)
    decision = interrupt(payload)
    normalized = _normalize_approval_decision(decision)
    pending = state.get("pending_approval") or {}
    if pending.get("kind") == "change_set_apply" or payload.get("kind") == "change_set_apply":
        proposal_id = str(pending.get("proposal_id") or payload.get("proposal_id") or "")
        if normalized.get("decision") == "allow":
            return _with_checkpoint_bump(
                {
                    "pending_action": action_to_snapshot(
                        AgentAction(
                            type="apply_change_set",
                            reason_summary="Apply approved ChangeSetProposal.",
                            expected_output="Files written through approved change-set apply path.",
                            payload={
                                "proposal_id": proposal_id,
                                "approval_decision": normalized,
                                "change_set_snapshot": state.get("change_set_proposal"),
                            },
                        )
                    ),
                    "pending_approval": None,
                    "stop_reason": None,
                    "routing_stage": "after_interrupt_resume",
                    "edit_mode": "apply_approved",
                    "apply_status": "pending",
                    "approval_decision": normalized,
                    "events_to_emit": [
                        _graph_transition_event(
                            state,
                            "await_change_approval",
                            operation="approval_resume",
                            status="completed",
                            files=[str(item.get("path")) for item in (state.get("change_set_proposal") or {}).get("changes") or [] if isinstance(item, dict)],
                            aggregate={"proposal_id": proposal_id, "kind": "change_set_apply"},
                        )
                    ],
                }
            )
        proposal = dict(state.get("change_set_proposal") or {})
        if proposal:
            proposal.update({"status": "rejected", "apply_status": "rejected"})
        return _with_checkpoint_bump(
            {
                "stop_reason": "approval_denied",
                "final_response": "The change-set proposal was not applied. No files were modified.",
                "pending_approval": None,
                "change_set_proposal": proposal or state.get("change_set_proposal"),
                "proposal_status": "rejected",
                "apply_status": "rejected",
                "routing_stage": "after_approval",
                "approval_decision": normalized,
                "events_to_emit": [
                    _graph_transition_event(
                        state,
                        "await_change_approval",
                        operation="approval_gate",
                        status="completed",
                        aggregate={"proposal_id": proposal_id, "decision": "deny"},
                    )
                ],
            }
        )
    if normalized.get("decision") == "allow":
        command = list((state.get("pending_approval") or {}).get("command") or payload.get("command") or [])
        approval_id = str((state.get("pending_approval") or {}).get("approval_id") or payload.get("approval_id") or "")
        return _with_checkpoint_bump(
            {
                "pending_action": action_to_snapshot(
                    AgentAction(
                        type="run_approved_command",
                        reason_summary="Run command after user approval.",
                        command=command,
                        expected_output="Command output after approval.",
                        payload={"approval_id": approval_id, "approval_decision": normalized},
                    )
                ),
                "pending_approval": None,
                "stop_reason": None,
                "routing_stage": "after_interrupt_resume",
                "approval_decision": normalized,
                "events_to_emit": [
                    _graph_transition_event(
                        state,
                        "await_approval",
                        operation="approval_resume",
                        status="completed",
                        command=command,
                    )
                ],
            }
        )
    final_response = "I did not run the command because approval was denied. No command was executed."
    return _with_checkpoint_bump(
        {
            "stop_reason": "approval_denied",
            "final_response": final_response,
            "pending_approval": None,
            "routing_stage": "after_approval",
            "approval_decision": normalized,
            "events_to_emit": [
                _graph_transition_event(
                    state,
                    "await_approval",
                    operation="approval_gate",
                    status="completed",
                    command=payload.get("command"),
                )
            ],
        }
    )


def apply_change_set_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    update = _execute_pending_action(state, subgraph=None, node_name="apply_change_set")
    result = result_from_snapshot((update.get("action_results") or [None])[-1])
    payload = result.payload if result else {}
    proposal = payload.get("change_set_proposal") if isinstance(payload.get("change_set_proposal"), dict) else state.get("change_set_proposal")
    files_changed = list(payload.get("files_modified") or []) + list(payload.get("files_created") or []) + list(payload.get("files_deleted") or [])
    for item in payload.get("files_renamed") or []:
        if isinstance(item, dict) and item.get("to"):
            files_changed.append(str(item.get("to")))
    applied = bool(payload.get("applied"))
    update.update(
        {
            "change_set_proposal": proposal,
            "files_changed": files_changed,
            "edit_mode": "applied" if applied else "blocked",
            "apply_status": "applied" if applied else "failed",
            "proposal_status": "applied" if applied else "valid",
            "applied_change_set_id": payload.get("applied_change_set_id"),
            "stop_reason": None if applied else "failed",
            "routing_stage": "after_tool_result",
            "final_response": _final_text_for_applied_change_set(state, payload) if applied else _final_text_for_failed_apply(payload),
        }
    )
    return _with_checkpoint_bump(update)


def post_apply_validation_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    status = "not_run"
    if state.get("apply_status") == "applied":
        status = "skipped_no_safe_command_selected"
    proposal = dict(state.get("change_set_proposal") or {})
    if proposal:
        proposal["post_apply_validation_status"] = status
    return _with_checkpoint_bump(
        {
            "post_apply_validation_status": status,
            "change_set_proposal": proposal or state.get("change_set_proposal"),
            "validation_results": [{"kind": "post_apply", "status": status, "errors": []}],
            "events_to_emit": [
                _graph_transition_event(
                    state,
                    "post_apply_validation",
                    operation="post_apply_validation",
                    status="completed",
                    validation_result={"status": status, "errors": []},
                )
            ],
        }
    )


def final_synthesis_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    update = _invoke_subgraph_delta(build_finalization_graph, state)
    update.setdefault("events_to_emit", []).append(
        _graph_transition_event(state, "final_synthesis", subgraph="finalization_graph", operation="final_synthesis")
    )
    return _with_checkpoint_bump(update)


def route_evidence_next_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    if _pending_action(state):
        return {
            "events_to_emit": [_graph_transition_event(state, "route_evidence_next", subgraph="evidence_gathering_graph", operation="route")],
        }
    core = _core_state_from_graph(state)
    action = next_evidence_gathering_action(core, _request(state), build_task_frame(_request(state), core))
    if action is None:
        return {
            "evidence_done": True,
            "events_to_emit": [_graph_transition_event(state, "route_evidence_next", subgraph="evidence_gathering_graph", operation="evidence_complete")],
        }
    return {
        "pending_action": action_to_snapshot(action),
        "events_to_emit": [
            _graph_transition_event(
                state,
                "route_evidence_next",
                subgraph="evidence_gathering_graph",
                operation="route",
                action_type=action.type,
            )
        ],
    }


def route_evidence_next(state: RepoOperatorGraphState) -> str:
    action = _pending_action(state)
    if not action:
        return END
    if action.type == "inspect_repo_tree":
        return "inspect_tree"
    if action.type == "search_files":
        return "search_files"
    if action.type == "search_text":
        return "search_text"
    if action.type in {"read_file", "inspect_symbol", "analyze_file"}:
        return "read_files"
    return "update_evidence_store"


def evidence_inspect_tree_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    return _execute_if_action_type(state, {"inspect_repo_tree"}, "evidence_gathering_graph", "inspect_tree")


def evidence_rank_candidates_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    core = _core_state_from_graph(state)
    candidates = candidate_files_from_results(core, edit_related=bool(edit_requested(build_task_frame(_request(state), core))))
    return {
        "evidence_store": {**dict(state.get("evidence_store") or {}), "ranked_candidates": candidates},
        "events_to_emit": [
            _graph_transition_event(
                state,
                "rank_candidates",
                subgraph="evidence_gathering_graph",
                operation="rank_candidates",
                files=candidates[:8],
            )
        ],
    }


def evidence_search_files_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    return _execute_if_action_type(state, {"search_files"}, "evidence_gathering_graph", "search_files")


def evidence_search_text_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    return _execute_if_action_type(state, {"search_text"}, "evidence_gathering_graph", "search_text")


def evidence_read_files_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    return _execute_if_action_type(state, {"read_file", "inspect_symbol", "analyze_file"}, "evidence_gathering_graph", "read_files")


def update_evidence_store_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    evidence = dict(state.get("evidence_store") or {})
    latest = _latest_result(state)
    if latest:
        evidence.setdefault("actions", []).append(latest.model_dump())
        if latest.files_read:
            evidence.setdefault("files_read", [])
            evidence["files_read"] = append_unique_items(evidence.get("files_read"), latest.files_read)
        if latest.payload.get("contents"):
            evidence.setdefault("contents", {}).update(latest.payload.get("contents") or {})
    return {
        "evidence_store": evidence,
        "evidence_reports": [_evidence_report(state, latest)] if latest else [],
        "events_to_emit": [
            _graph_transition_event(state, "update_evidence_store", subgraph="evidence_gathering_graph", operation="update_evidence_store")
        ],
    }


def analysis_inventory_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    return _execute_if_action_type(state, {"analyze_repository"}, "analysis_graph", "inventory")


def analysis_batch_files_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    groups = _supervisor_file_groups(state)
    return {
        "worker_tasks": _worker_tasks_from_groups(groups, roles=["AnalysisAgent"]),
        "events_to_emit": [_graph_transition_event(state, "group_files", subgraph="analysis_graph", operation="group_files")],
    }


def analysis_file_role_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    tasks = state.get("worker_tasks") or []
    reports = [_run_analysis_worker_task(task, state=state) for task in tasks if task.get("role") == "AnalysisAgent"]
    if not reports:
        reports = [{"worker": "AnalysisAgent", "file": path, "role": "evidence file", "files": [path]} for path in state.get("files_read") or []]
    return {
        "worker_reports": reports,
        "file_role_reports": reports,
        "events_to_emit": [_graph_transition_event(state, "dispatch_file_role_workers", subgraph="analysis_graph", operation="dispatch_file_role_workers")],
    }


def analysis_reduce_file_reports_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    reports = list(state.get("worker_reports") or state.get("file_role_reports") or [])
    return {
        "file_role_reports": reports,
        "events_to_emit": [_graph_transition_event(state, "reduce_file_reports", subgraph="analysis_graph", operation="reduce_file_reports")],
    }


def analysis_summarize_batch_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    return {
        "evidence_reports": [
            {
                "worker": "AnalysisAgent",
                "summary": f"Aggregated {len(state.get('file_role_reports') or [])} file role report(s).",
            }
        ],
        "events_to_emit": [_graph_transition_event(state, "summarize_batch", subgraph="analysis_graph", operation="summarize_batch")],
        "analysis_done": True,
    }


def analysis_route_batch_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    return {
        "events_to_emit": [_graph_transition_event(state, "route_batch_continue_or_end", subgraph="analysis_graph", operation="route_batch_continue_or_end")],
    }


def edit_locate_targets_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    action = _pending_action(state)
    return {
        "events_to_emit": [
            _graph_transition_event(
                state,
                "locate_targets",
                subgraph="edit_graph",
                operation="locate_targets",
                files=list(action.target_files if action else []),
            )
        ]
    }


def route_edit_next_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    action = _pending_action(state)
    if not action:
        return {
            "edit_done": True,
            "events_to_emit": [_graph_transition_event(state, "route_edit_next", subgraph="edit_graph", operation="edit_complete")],
        }
    return {
        "events_to_emit": [
            _graph_transition_event(
                state,
                "route_edit_next",
                subgraph="edit_graph",
                operation="route_edit_next",
                action_type=action.type,
            )
        ]
    }


def route_edit_next(state: RepoOperatorGraphState) -> str:
    action = _pending_action(state)
    if not action:
        return END
    if action.type in {"generate_change_set", "generate_edit"}:
        if not state.get("change_set_proposal"):
            return "locate_targets"
        return "generate_change_set"
    return END


def edit_plan_change_set_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    action = _pending_action(state)
    proposal = plan_change_set(
        list(action.target_files if action else []),
        action.reason_summary if action else "Plan proposal-only change set.",
    ).model_dump()
    return {
        "change_set_proposal": proposal,
        "proposed_changes": [proposal],
        "events_to_emit": [_graph_transition_event(state, "plan_change_set", subgraph="edit_graph", operation="plan_change_set")],
    }


def edit_generate_change_set_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    return _execute_if_action_type(state, {"generate_change_set", "generate_edit"}, "edit_graph", "generate_change_set")


def edit_validate_change_set_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    latest = _latest_result(state)
    proposal = _change_set_from_latest_result(state, latest)
    validation = {
        "kind": "change_set",
        "status": (proposal.get("status") if proposal else None) or (latest.status if latest else "skipped"),
        "action_id": latest.action_id if latest else None,
        "errors": list((proposal.get("validation") or {}).get("errors") or []),
    }
    return {
        "change_set_proposal": proposal or state.get("change_set_proposal"),
        "validation_results": [validation],
        "proposal_errors": validation["errors"],
        "events_to_emit": [
            _graph_transition_event(state, "validate_change_set", subgraph="edit_graph", operation="validate_change_set", validation_result=validation)
        ],
    }


def edit_repair_change_set_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    attempts = int(state.get("repair_attempts") or 0) + 1
    return {
        "repair_attempts": attempts,
        "attempts": [{"kind": "repair", "attempt": attempts, "status": "blocked" if attempts > 1 else "queued"}],
        "events_to_emit": [_graph_transition_event(state, "repair_change_set", subgraph="edit_graph", operation="repair_change_set")]
    }


def route_edit_after_validation(state: RepoOperatorGraphState) -> str:
    proposal = state.get("change_set_proposal") or {}
    errors = list((proposal.get("validation") or {}).get("errors") or state.get("proposal_errors") or [])
    if errors and int(state.get("repair_attempts") or 0) < 1:
        return "repair_change_set"
    return END


def validation_choose_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    return {"events_to_emit": [_graph_transition_event(state, "choose_validation", subgraph="validation_graph", operation="choose_validation")]}


def validation_preview_command_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    return _execute_if_action_type(state, {"preview_command", "inspect_git_state"}, "validation_graph", "preview_command")


def validation_approval_interrupt_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    if state.get("pending_approval"):
        return await_approval_node(state)
    return {
        "events_to_emit": [
            _graph_transition_event(state, "approval_interrupt_if_needed", subgraph="validation_graph", operation="approval_not_needed")
        ]
    }


def validation_run_safe_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    return _execute_if_action_type(state, {"run_approved_command"}, "validation_graph", "run_safe_validation")


def validation_parse_errors_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    latest = _latest_result(state)
    errors = list(latest.errors if latest else [])
    return {
        "validation_results": [{"kind": "command", "status": latest.status if latest else "skipped", "errors": errors}],
        "events_to_emit": [_graph_transition_event(state, "parse_errors", subgraph="validation_graph", operation="parse_errors")],
    }


def validation_update_result_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    return {
        "validation_done": True,
        "events_to_emit": [_graph_transition_event(state, "update_validation_result", subgraph="validation_graph", operation="update_validation_result")]
    }


def validation_route_next_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    return {
        "events_to_emit": [_graph_transition_event(state, "route_validation_next", subgraph="validation_graph", operation="route_validation_next")]
    }


def final_quality_guard_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    return {
        "events_to_emit": [_graph_transition_event(state, "quality_guard", subgraph="finalization_graph", operation="quality_guard")]
    }


def final_repair_answer_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    return {
        "events_to_emit": [_graph_transition_event(state, "repair_final_answer", subgraph="finalization_graph", operation="repair_final_answer")]
    }


def final_build_response_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    request = _request(state)
    core = _core_state_from_graph(state)
    proposal = state.get("change_set_proposal") if isinstance(state.get("change_set_proposal"), dict) else None
    if proposal and proposal.get("changes") and proposal.get("status") in {"invalid", "repairable", "blocked"}:
        validation = proposal.get("validation") if isinstance(proposal.get("validation"), dict) else {}
        errors = "; ".join(str(item) for item in validation.get("errors") or [proposal.get("proposal_error")] if item)
        core.final_response = (
            "I could not prepare a valid ChangeSetProposal. "
            f"Validation failed: {errors or 'unknown validation error'}. No files were modified."
        )
    if not core.final_response:
        on_delta = _stream_final_delta(core.run_id) if state.get("stream_final_answer") else None
        packet_context = ""
        if isinstance(core.context_packet, dict):
            packet_context = str(core.context_packet.get("skills_context") or "")
        core.final_response = _controller().build_final_answer_text(
            core,
            request,
            skills_context=packet_context or str(state.get("skills_context") or ""),
            on_delta=on_delta,
        )
    draft_response = core.final_response
    core.final_response = _controller().validate_or_repair_final_answer(core.final_response, core, request)
    if _is_explanation_only_edit_request(state) and not state.get("files_changed") and "no files were modified" not in core.final_response.lower():
        core.final_response = core.final_response.rstrip() + "\n\nNo files were modified."
    if core.final_response != draft_response:
        from repooperator_worker.agent_core.events import append_work_trace

        append_work_trace(
            run_id=core.run_id,
            request=request,
            activity_id="final-synthesis-repair",
            phase="Finished",
            label="Rebuilt final answer",
            status="completed",
            safe_reasoning_summary="The draft answer did not match the gathered evidence, so I rebuilt it from collected files.",
            observation="Final answer repaired without storing the rejected draft text.",
            safety_note="Rejected draft text is not exposed in events.",
        )
    return {
        "final_response": core.final_response,
        "events_to_emit": [_graph_transition_event(state, "build_response", subgraph="finalization_graph", operation="build_response")],
    }


def final_emit_message_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    request = _request(state)
    core = _core_state_from_graph(state)
    response = _response_with_change_set_payload(
        _controller().build_final_response(core, request).model_copy(update={"agent_flow": "langgraph"}),
        state,
    )
    return {
        "response_snapshot": response_to_snapshot(response),
        "events_to_emit": [_graph_transition_event(state, "emit_final_message", subgraph="finalization_graph", operation="emit_final_message")],
    }


def supervisor_build_worker_tasks_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    groups = _supervisor_file_groups(state)
    roles = ["AnalysisAgent"]
    if _frame_is_edit_like(state):
        roles = ["EvidenceAgent", "EditPlanningAgent", "ValidationAgent", "DocumentationAgent", "TestAgent"]
    tasks = _worker_tasks_from_groups(groups, roles=roles)
    return {
        "worker_tasks": tasks,
        "events_to_emit": [_graph_transition_event(state, "build_worker_tasks", subgraph="supervisor", operation="build_worker_tasks")],
    }


def supervisor_run_worker_tasks_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    reports = [_run_worker_task(task, state=state) for task in state.get("worker_tasks") or []]
    return {
        "worker_reports": reports,
        "events_to_emit": [_graph_transition_event(state, "run_worker_task", subgraph="supervisor", operation="run_worker_task")],
    }


def supervisor_reduce_worker_reports_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    reports = list(state.get("worker_reports") or [])
    file_role_reports = [report for report in reports if report.get("role") or report.get("worker") == "AnalysisAgent"]
    evidence_reports = [report for report in reports if report.get("worker") in {"EvidenceAgent", "AnalysisAgent"}]
    proposed_changes = [report for report in reports if report.get("worker") == "EditPlanningAgent"]
    risk_notes = [str(note) for report in reports for note in report.get("risk_notes") or []]
    return {
        "file_role_reports": file_role_reports,
        "evidence_reports": evidence_reports,
        "proposed_changes": proposed_changes,
        "risk_notes": risk_notes,
        "events_to_emit": [_graph_transition_event(state, "reduce_worker_reports", subgraph="supervisor", operation="reduce_worker_reports")],
    }


def _execute_if_action_type(state: RepoOperatorGraphState, action_types: set[str], subgraph: str, node_name: str) -> dict[str, Any]:
    action = _pending_action(state)
    if action and action.type in action_types:
        return _execute_pending_action(state, subgraph=subgraph, node_name=node_name)
    return {"events_to_emit": [_graph_transition_event(state, node_name, subgraph=subgraph, operation="skip")]}


def _execute_pending_action(state: RepoOperatorGraphState, *, subgraph: str | None, node_name: str | None = None) -> dict[str, Any]:
    action = _pending_action(state)
    if not action:
        return {
            "routing_stage": "after_tool_result",
            "events_to_emit": [_graph_transition_event(state, node_name or "execute_tool", subgraph=subgraph, operation="skip")],
        }
    request = _request(state)
    core = _core_state_from_graph(state)
    if action.type != "final_answer":
        from repooperator_worker.agent_core.agent_loop import _emit_action_decision

        _emit_action_decision(core, request, action)
    orchestrator = ToolOrchestrator(
        run_id=str(state.get("run_id") or "run_controller"),
        request=request,
        registry=get_default_tool_registry(),
        hook_manager=HookManager(),
    )
    result = orchestrator.execute_action(action)
    _append_action_event(str(state.get("run_id") or "run_controller"), action, result)
    core.actions_taken.append(action)
    core.action_results.append(result)
    _controller().observe_result(core, action, result, request)
    _controller().update_plan(core, action, result, request)
    _controller().check_cancel(core, request)
    update = _updates_from_core_after_action(state, core, action, result)
    operation = _action_operation(action.type)
    update["events_to_emit"] = [
        _graph_transition_event(
            state,
            node_name or "execute_tool",
            subgraph=subgraph,
            operation=operation,
            action_type=action.type,
            activity_id=f"action:{action.action_id}",
            status=result.status,
            files=list(result.files_read or result.files_changed or action.target_files),
            command=action.command,
            validation_result={"status": result.status, "errors": result.errors} if result.errors else None,
        )
    ]
    update["pending_action"] = None
    return update


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
    if action.type == "analyze_repository":
        return "analysis_graph"
    if action.type in {"generate_change_set", "generate_edit", "validate_change_set", "validate_edit"}:
        return "plan_change_set"
    if action.type == "apply_change_set":
        return "apply_change_set"
    if action.type in {"preview_command", "inspect_git_state", "run_approved_command", "request_command_approval"}:
        return "execute_tool"
    return "execute_tool"


def _should_use_supervisor(state: RepoOperatorGraphState) -> bool:
    if state.get("supervisor_mode"):
        return False
    frame = _task_frame(state)
    if frame is None:
        return False
    text = " ".join([str(getattr(frame, "user_goal", "")), *[str(item) for item in getattr(frame, "requested_outputs", [])]]).lower()
    broad = any(term in text for term in ("whole", "entire", "all files", "every file", "codebase", "source tree", "repository-wide"))
    return broad and not state.get("files_read")


def _invoke_subgraph_delta(builder: Any, state: RepoOperatorGraphState) -> dict[str, Any]:
    before = copy.copy(dict(state))
    after = builder().compile().invoke(state)
    return _delta_state(before, after)


def _delta_state(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    update: dict[str, Any] = {}
    append_fields = APPEND_REDUCER_FIELDS | UNIQUE_APPEND_REDUCER_FIELDS
    for key, value in after.items():
        if key in append_fields:
            old = before.get(key) or []
            new = value or []
            if len(new) > len(old):
                update[key] = list(new[len(old):])
            continue
        if before.get(key) != value:
            update[key] = value
    return update


def _updates_from_core(before: RepoOperatorGraphState, core: AgentCoreState) -> dict[str, Any]:
    update: dict[str, Any] = {
        "context_packet": core.context_packet,
        "request_understanding_snapshot": request_understanding_to_snapshot(core.request_understanding),
        "classifier_snapshot": classifier_to_snapshot(core.classifier_result),
        "plan": list(core.plan),
        "current_subtask_id": core.current_subtask_id,
        "pending_approval": core.pending_approval,
        "cancellation_requested": core.cancellation_requested,
        "skills_used": list(core.skills_used),
        "memories_used": list(core.memories_used),
        "recommendation_context": core.recommendation_context,
        "stop_reason": core.stop_reason,
        "final_response": core.final_response,
        "loop_iteration": core.loop_iteration,
        "max_loop_iterations": core.max_loop_iterations,
        "max_file_reads": core.max_file_reads,
        "max_commands": core.max_commands,
        "max_edits": core.max_edits,
        "subtasks": [subtask_to_snapshot(subtask) for subtask in core.subtasks],
        "current_step": core.current_step,
        "zero_result_queries": list(core.zero_result_queries),
        "failed_action_signatures": list(core.failed_action_signatures),
        "strategy_shifts": list(core.strategy_shifts),
    }
    for field_name in APPEND_REDUCER_FIELDS | UNIQUE_APPEND_REDUCER_FIELDS:
        if not hasattr(core, field_name):
            continue
        old = before.get(field_name) or []
        new = getattr(core, field_name) or []
        if len(new) > len(old):
            if field_name == "actions_taken":
                update[field_name] = [action_to_snapshot(item) for item in new[len(old):]]
            elif field_name == "action_results":
                update[field_name] = [result_to_snapshot(item) for item in new[len(old):]]
            else:
                update[field_name] = list(new[len(old):])
    return {key: value for key, value in update.items() if value is not None or key in {"pending_approval", "stop_reason"}}


def _updates_from_core_after_action(
    before: RepoOperatorGraphState,
    core: AgentCoreState,
    action: AgentAction,
    result: ActionResult,
) -> dict[str, Any]:
    update = _updates_from_core(before, core)
    update["actions_taken"] = [action_to_snapshot(action)]
    update["action_results"] = [result_to_snapshot(result)]
    if result.files_changed:
        update["files_changed"] = list(result.files_changed)
    if result.files_read:
        update["files_read"] = list(result.files_read)
    if result.status == "waiting_approval":
        update["pending_approval"] = result.command_result
    if result.status in {"cancelled", "timed_out"}:
        update["stop_reason"] = result.status
    return update


def _core_state_from_graph(state: RepoOperatorGraphState) -> AgentCoreState:
    request = _request(state)
    core = AgentCoreState(
        run_id=str(state.get("run_id") or "run_controller"),
        thread_id=state.get("thread_id"),
        repo=str(state.get("repo") or request.project_path),
        branch=state.get("branch"),
        user_task=request.task,
    )
    core.classifier_result = classifier_from_snapshot(state.get("classifier_snapshot") or state.get("classifier_result"))  # type: ignore[typeddict-item]
    core.request_understanding = request_understanding_from_snapshot(
        state.get("request_understanding_snapshot") or state.get("request_understanding")  # type: ignore[typeddict-item]
    )
    core.plan = list(state.get("plan") or [])
    core.current_step = state.get("current_step")
    core.observations = list(state.get("observations") or [])
    core.actions_taken = [action for action in (action_from_snapshot(item) for item in state.get("actions_taken") or []) if action]
    core.action_results = [result for result in (result_from_snapshot(item) for item in state.get("action_results") or []) if result]
    core.files_read = list(state.get("files_read") or [])
    core.files_changed = list(state.get("files_changed") or [])
    core.commands_run = list(state.get("commands_run") or [])
    core.pending_approval = state.get("pending_approval")
    core.cancellation_requested = bool(state.get("cancellation_requested") or False)
    core.skills_used = list(state.get("skills_used") or [])
    core.memories_used = list(state.get("memories_used") or [])
    core.recommendation_context = state.get("recommendation_context")
    core.context_packet = state.get("context_packet")
    core.stop_reason = state.get("stop_reason")
    core.final_response = str(state.get("final_response") or "")
    core.loop_iteration = int(state.get("loop_iteration") or 0)
    core.max_loop_iterations = int(state.get("max_loop_iterations") or (state.get("budgets") or {}).get("max_loop_iterations") or 8)
    core.max_file_reads = int(state.get("max_file_reads") or (state.get("budgets") or {}).get("max_file_reads") or 40)
    core.max_commands = int(state.get("max_commands") or (state.get("budgets") or {}).get("max_commands") or 8)
    core.max_edits = int(state.get("max_edits") or (state.get("budgets") or {}).get("max_edits") or 6)
    core.subtasks = [subtask_from_snapshot(item) for item in state.get("subtasks") or []]
    core.current_subtask_id = state.get("current_subtask_id")
    core.zero_result_queries = list(state.get("zero_result_queries") or [])
    core.failed_action_signatures = list(state.get("failed_action_signatures") or [])
    core.strategy_shifts = list(state.get("strategy_shifts") or [])
    return core


def _request(state: RepoOperatorGraphState) -> AgentRunRequest:
    snapshot = state.get("request_snapshot")
    if isinstance(snapshot, dict):
        return request_from_snapshot(snapshot)
    request = state.get("request")  # type: ignore[typeddict-item]
    if isinstance(request, AgentRunRequest):
        return request
    raise ValueError("RepoOperatorGraphState requires request_snapshot.")


def _pending_action(state: RepoOperatorGraphState) -> AgentAction | None:
    return action_from_snapshot(state.get("pending_action"))


def _task_frame(state: RepoOperatorGraphState) -> Any | None:
    return task_frame_from_snapshot(state.get("task_frame_snapshot") or state.get("task_frame"))  # type: ignore[typeddict-item]


def _latest_result(state: RepoOperatorGraphState) -> ActionResult | None:
    results = state.get("action_results") or []
    return result_from_snapshot(results[-1]) if results else None


def _change_set_from_latest_result(state: RepoOperatorGraphState, result: ActionResult | None) -> dict[str, Any] | None:
    if not result:
        return None
    if isinstance(result.payload.get("change_set_proposal"), dict):
        return json_safe(result.payload.get("change_set_proposal"))
    edit_proposals = result.payload.get("edit_proposals") or []
    if edit_proposals:
        plan_summary = str(((state.get("change_set_proposal") or {}).get("plan") or {}).get("summary") or "Prepare proposal-only edits.")
        return proposal_from_edit_result(edit_proposals, repo=str(state.get("repo") or _request(state).project_path), plan_summary=plan_summary).model_dump()
    if result.payload.get("proposal_error"):
        proposal = state.get("change_set_proposal") or plan_change_set([], "Prepare proposal-only edits.").model_dump()
        error = str(result.payload.get("proposal_error") or "")
        proposal.update({"status": "invalid", "proposal_error": error, "validation": {"status": "invalid", "errors": [error], "warnings": []}})
        return proposal
    proposal = state.get("change_set_proposal")
    if isinstance(proposal, dict) and proposal.get("changes"):
        typed = change_set_from_payload(proposal)
        validation = validate_change_set_model(typed, repo=str(state.get("repo") or _request(state).project_path))
        typed.validation = validation
        typed.status = validation.status
        typed.validation_status = validation.status
        proposal = typed.model_dump()
        return proposal
    return None


def _approval_interrupt_payload(state: RepoOperatorGraphState) -> dict[str, Any]:
    approval = state.get("pending_approval") or {}
    if approval.get("kind") == "change_set_apply":
        proposal = state.get("change_set_proposal") if isinstance(state.get("change_set_proposal"), dict) else approval.get("change_set_proposal")
        files = [str(item.get("path")) for item in (proposal or {}).get("changes") or [] if isinstance(item, dict)]
        proposal_id = str(approval.get("proposal_id") or (proposal or {}).get("proposal_id") or "")
        return json_safe(
            {
                "kind": "change_set_apply",
                "run_id": state.get("run_id"),
                "thread_id": state.get("thread_id"),
                "proposal_id": proposal_id,
                "change_set_proposal": proposal,
                "files": files,
                "risk": approval.get("reason") or "Applying this proposal modifies files and requires approval.",
                "resume_token": f"{state.get('run_id')}:change_set_apply:{proposal_id}",
            }
        )
    command = list(approval.get("command") or [])
    return json_safe(
        {
            "kind": "command_approval",
            "run_id": state.get("run_id"),
            "thread_id": state.get("thread_id"),
            "command": command,
            "approval_id": approval.get("approval_id"),
            "files": [],
            "risk": approval.get("reason") or approval.get("risk") or "Command requires approval before execution.",
            "resume_token": f"{state.get('run_id')}:command:{approval.get('approval_id') or shlex.join(command)}",
        }
    )


def _normalize_approval_decision(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        decision = str(value.get("decision") or value.get("approval") or value.get("action") or "").strip().lower()
        if decision in {"allow", "approved", "approve", "yes"}:
            return {**json_safe(value), "decision": "allow"}
        return {**json_safe(value), "decision": "deny"}
    if str(value).strip().lower() in {"allow", "approved", "approve", "yes", "true"}:
        return {"decision": "allow"}
    return {"decision": "deny"}


def _response_with_change_set_payload(response: AgentRunResponse, state: RepoOperatorGraphState) -> AgentRunResponse:
    proposal = state.get("change_set_proposal")
    if not isinstance(proposal, dict) or not proposal.get("changes"):
        return response
    validation = proposal.get("validation") if isinstance(proposal.get("validation"), dict) else {}
    validation_status = str(validation.get("status") or proposal.get("status") or "planned")
    errors = [str(item) for item in validation.get("errors") or proposal.get("proposal_errors") or []]
    archive_status = "applied" if proposal.get("applied") or proposal.get("status") == "applied" or state.get("apply_status") == "applied" else ("rejected" if proposal.get("status") == "rejected" else validation_status)
    archive = [_edit_archive_record_from_change(change, archive_status) for change in proposal.get("changes") or [] if isinstance(change, dict)]
    archive = [item for item in archive if item]
    first = (proposal.get("changes") or [{}])[0]
    if proposal.get("applied") or proposal.get("status") == "applied" or state.get("apply_status") == "applied":
        response_type = "edit_applied"
    elif proposal.get("status") == "rejected" or state.get("apply_status") == "rejected":
        response_type = "change_proposal"
    else:
        response_type = "change_proposal" if validation_status == "valid" else "proposal_error"
    updates: dict[str, Any] = {
        "response_type": response_type,
        "change_set_proposal": json_safe(proposal),
        "edit_archive": archive,
        "proposal_validation_status": validation_status,
        "validation_status": validation_status,
        "edit_mode": state.get("edit_mode"),
        "proposal_id": proposal.get("proposal_id"),
        "proposal_status": proposal.get("status"),
        "apply_status": state.get("apply_status") or proposal.get("apply_status"),
        "applied_change_set_id": state.get("applied_change_set_id") or proposal.get("applied_change_set_id"),
        "post_apply_validation_status": state.get("post_apply_validation_status") or proposal.get("post_apply_validation_status"),
    }
    if errors:
        updates["proposal_error_details"] = "; ".join(errors)
    if isinstance(first, dict):
        updates.update(
            {
                "proposal_relative_path": first.get("path"),
                "proposal_original_content": first.get("original_content") or "",
                "proposal_proposed_content": first.get("proposed_content") or "",
                "proposal_context_summary": ((proposal.get("plan") or {}).get("summary") if isinstance(proposal.get("plan"), dict) else None),
                "selected_target_file": first.get("path"),
            }
        )
    return response.model_copy(update=json_safe(updates))


def _edit_archive_record_from_change(change: dict[str, Any], validation_status: str) -> dict[str, Any]:
    path = str(change.get("path") or "")
    if not path:
        return {}
    operation = str(change.get("operation") or "modify")
    original = str(change.get("original_content") or "")
    proposed = "" if operation == "delete" else str(change.get("proposed_content") or "")
    if operation == "create":
        original = ""
    diff = "\n".join(
        difflib.unified_diff(
            original.splitlines(),
            proposed.splitlines(),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
        )
    )
    additions = sum(1 for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++"))
    deletions = sum(1 for line in diff.splitlines() if line.startswith("-") and not line.startswith("---"))
    return {
        "file_path": path,
        "file": path,
        "operation": operation,
        "status": "applied" if validation_status == "applied" else ("rejected" if validation_status == "rejected" else ("proposed" if validation_status == "valid" else "failed")),
        "summary": str(change.get("summary") or ""),
        "additions": additions,
        "deletions": deletions,
        "diff": diff,
        "diff_available": bool(diff.strip()),
        "proposal_id": "proposal:" + path,
        "validation_status": validation_status,
    }


def _response_from_interrupted_state(state: dict[str, Any], request: AgentRunRequest) -> AgentRunResponse:
    state = dict(state)
    state.setdefault("request_snapshot", request_to_snapshot(request))
    core = _core_state_from_graph(state)
    if not core.stop_reason:
        core.stop_reason = "waiting_approval"
    return _response_with_change_set_payload(
        _controller().build_final_response(core, request).model_copy(update={"agent_flow": "langgraph"}),
        state,
    )


def _graph_transition_event(
    state: RepoOperatorGraphState,
    node: str,
    *,
    subgraph: str | None = None,
    operation: str,
    action_type: str | None = None,
    activity_id: str | None = None,
    status: str = "completed",
    files: list[str] | None = None,
    command: list[str] | None = None,
    validation_result: dict[str, Any] | None = None,
    next_node: str | None = None,
    aggregate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    graph_metadata = {
        "graph_node": node,
        "subgraph": subgraph,
        "subtask_id": state.get("current_subtask_id"),
        "validation_result": validation_result,
        "change_set_summary": aggregate,
        "next_node": next_node,
    }
    from repooperator_worker.agent_core.events import activity_event

    event = activity_event(
        run_id=str(state.get("run_id") or "run_controller"),
        request=_request(state),
        activity_id=activity_id or f"graph:{node}",
        event_type="graph_transition",
        phase="Thinking",
        label=_graph_event_label(node, operation),
        status=status,
        visibility="debug",
        display="secondary",
        operation=operation,
        action_type=action_type,
        related_files=files or [],
        command=command,
        aggregate={key: value for key, value in graph_metadata.items() if value not in (None, [], {})},
    )
    event.update({key: value for key, value in graph_metadata.items() if value not in (None, [], {})})
    event = json_safe(event)
    _append_graph_event_safe(str(state.get("run_id") or "run_controller"), event)
    return event


def _graph_event_label(node: str, operation: str) -> str:
    labels = {
        "load_context": "Loaded runtime context",
        "understand_request": "Framed request",
        "build_task_plan": "Built task plan",
        "route_next": "Selected next step",
        "supervisor": "Delegated bounded work",
        "gather_evidence": "Gathered evidence",
        "analysis_graph": "Analyzed repository evidence",
        "execute_tool": "Ran safe tool boundary",
        "validate_result": "Validated result",
        "plan_change_set": "Planned proposal",
        "generate_change_set": "Generated proposal",
        "validate_change_set": "Validated proposal",
        "repair_change_set": "Repaired proposal",
        "ask_clarification": "Prepared clarification",
        "await_approval": "Waiting for approval",
        "final_synthesis": "Built final response",
    }
    return labels.get(node, operation.replace("_", " ").title())


def _append_graph_event_safe(run_id: str, event: dict[str, Any]) -> None:
    try:
        append_run_event(run_id, event)
    except OSError:
        return


def _append_action_event(run_id: str, action: AgentAction, result: ActionResult) -> None:
    try:
        append_run_event(
            run_id,
            {
                "type": "action_result",
                "event_type": "action_result",
                "status": result.status,
                "action": action.model_dump(),
                "result": result.model_dump(),
            },
        )
    except OSError:
        return


def _action_operation(action_type: str) -> str:
    try:
        from repooperator_worker.agent_core.task_policy import action_operation

        return action_operation(action_type)
    except Exception:
        return action_type


def _evidence_report(state: RepoOperatorGraphState, result: ActionResult | None) -> dict[str, Any]:
    action = _pending_action(state)
    return {
        "worker": "EvidenceAgent",
        "action_type": action.type if action else None,
        "status": result.status if result else "skipped",
        "files": list(result.files_read if result else []),
        "observation": result.observation if result else "",
    }


def _supervisor_file_groups(state: RepoOperatorGraphState) -> dict[str, list[str]]:
    try:
        from repooperator_worker.agent_core.task_policy import group_inventory, repository_file_inventory

        groups = group_inventory(repository_file_inventory(_request(state)))
        return {name: files[:12] for name, files in groups.items()}
    except Exception:
        return {}


def _worker_tasks_from_groups(groups: dict[str, list[str]], *, roles: list[str]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for role in roles:
        for group, files in groups.items():
            if not files:
                continue
            tasks.append(
                {
                    "id": f"{role}:{group}",
                    "task_id": f"{role}:{group}",
                    "role": role,
                    "scope": group,
                    "group": group,
                    "input_files": files[:8],
                    "files": files[:8],
                    "goal": f"Analyze {group} files for the current repository task.",
                    "status": "pending",
                }
            )
    return tasks[:12]


def _run_worker_task(task: dict[str, Any], *, state: RepoOperatorGraphState) -> dict[str, Any]:
    role = str(task.get("role") or "AnalysisAgent")
    if role == "AnalysisAgent":
        return _run_analysis_worker_task(task, state=state)
    files = [str(item) for item in task.get("input_files") or task.get("files") or []]
    if role == "EvidenceAgent":
        read = _worker_read_files(state, files[:1])
        return {
            "worker": role,
            "task_id": task.get("task_id"),
            "role": role,
            "scope": task.get("scope") or task.get("group"),
            "files_analyzed": read["files_read"],
            "files": files,
            "findings": ["Located bounded evidence candidates for the requested scope."],
            "summary": "Located bounded evidence candidates for the requested scope.",
            "status": read["status"],
        }
    if role == "EditPlanningAgent":
        return {
            "worker": role,
            "task_id": task.get("task_id"),
            "role": role,
            "scope": task.get("scope") or task.get("group"),
            "files": files,
            "files_analyzed": files,
            "findings": ["Identified files that may participate in a proposal-only change plan."],
            "recommended_next_actions": ["Generate and validate a ChangeSetProposal before presenting edits."],
            "summary": "Identified files that may participate in a proposal-only change plan.",
            "risk_notes": ["Change plan still requires proposal validation before it can be shown as valid."],
            "status": "completed",
        }
    if role == "ValidationAgent":
        return {"worker": role, "task_id": task.get("task_id"), "role": role, "files": files, "files_analyzed": [], "findings": ["Validation should run through ToolOrchestrator-backed checks."], "summary": "Validation should run through ToolOrchestrator-backed checks.", "status": "completed"}
    if role == "DocumentationAgent":
        return {"worker": role, "task_id": task.get("task_id"), "role": role, "files": files, "files_analyzed": [], "findings": ["Documentation impact should be considered if behavior changes."], "summary": "Documentation impact should be considered if behavior changes.", "status": "completed"}
    if role == "TestAgent":
        return {"worker": role, "task_id": task.get("task_id"), "role": role, "files": files, "files_analyzed": [], "findings": ["Tests or safe validation commands may be needed after a proposal."], "summary": "Tests or safe validation commands may be needed after a proposal.", "status": "completed"}
    return {"worker": role, "task_id": task.get("task_id"), "role": role, "files": files, "files_analyzed": [], "findings": ["Worker completed bounded scoped analysis."], "summary": "Worker completed bounded scoped analysis.", "status": "completed"}


def _run_analysis_worker_task(task: dict[str, Any], *, state: RepoOperatorGraphState | None = None) -> dict[str, Any]:
    files = [str(item) for item in task.get("input_files") or task.get("files") or []]
    read = _worker_read_files(state, files[:1]) if state is not None else {"files_read": [], "status": "completed"}
    return {
        "worker": "AnalysisAgent",
        "task_id": task.get("task_id"),
        "role": "AnalysisAgent",
        "scope": task.get("scope") or task.get("group"),
        "group": task.get("group"),
        "files": files,
        "files_analyzed": read["files_read"] or files,
        "file_role": f"{task.get('group') or 'files'} analysis batch",
        "findings": [f"Grouped {len(files)} file(s) for bounded file-role analysis."],
        "recommended_next_actions": ["Reduce this report into the parent graph evidence summary."],
        "summary": f"Grouped {len(files)} file(s) for bounded file-role analysis.",
        "status": read["status"],
    }


def _worker_read_files(state: RepoOperatorGraphState | None, files: list[str]) -> dict[str, Any]:
    if state is None or not files:
        return {"files_read": [], "status": "skipped"}
    action = AgentAction(
        type="read_file",
        reason_summary="Worker reads a bounded file sample through ToolOrchestrator.",
        target_files=files,
        expected_output="Bounded worker evidence sample.",
        payload={"worker": True},
    )
    orchestrator = ToolOrchestrator(
        run_id=str(state.get("run_id") or "run_controller"),
        request=_request(state),
        registry=get_default_tool_registry(),
        hook_manager=HookManager(),
    )
    result = orchestrator.execute_action(action)
    _append_action_event(str(state.get("run_id") or "run_controller"), action, result)
    return {"files_read": list(result.files_read or []), "status": result.status}


def _frame_is_edit_like(state: RepoOperatorGraphState) -> bool:
    frame = _task_frame(state)
    if frame is None:
        try:
            frame = build_task_frame(_request(state), _core_state_from_graph(state))
        except Exception:
            return False
    return edit_requested(frame)


def _edit_mode_for_request(frame: Any) -> EditMode:
    return "proposal_only" if edit_requested(frame) else "explanation_only"


def _is_explanation_only_edit_request(state: RepoOperatorGraphState) -> bool:
    frame = _task_frame(state)
    if frame is None:
        return False
    text = str(getattr(frame, "user_goal", "") or "")
    lowered = text.lower()
    asks_how = bool(re.search(r"\bhow\s+(would|do|can|should)\b", lowered)) or any(term in text for term in ("어떻게", "어떤 식으로"))
    mentions_change = bool(re.search(r"\b(change|edit|add|fix|implement|refactor|update)\b", lowered)) or any(term in text for term in ("추가", "고쳐", "구현", "수정"))
    return asks_how and mentions_change


def _final_text_for_change_set(state: RepoOperatorGraphState, proposal: dict[str, Any]) -> str:
    del state
    changes = [item for item in proposal.get("changes") or [] if isinstance(item, dict)]
    files = [f"- {str(item.get('operation') or 'modify')}: `{str(item.get('path') or '')}`" for item in changes]
    validation = proposal.get("validation") if isinstance(proposal.get("validation"), dict) else {}
    validation_status = str(validation.get("status") or proposal.get("status") or "pending")
    return "\n".join(
        [
            "I prepared a ChangeSetProposal. No files were modified.",
            "",
            "Proposed files:",
            *(files or ["- No files"]),
            "",
            f"Validation result: {validation_status}.",
            "Review the diff and approve Apply changes to write it to disk.",
        ]
    )


def _final_text_for_applied_change_set(state: RepoOperatorGraphState, payload: dict[str, Any]) -> str:
    del state
    modified = [str(item) for item in payload.get("files_modified") or []]
    created = [str(item) for item in payload.get("files_created") or []]
    deleted = [str(item) for item in payload.get("files_deleted") or []]
    renamed = [f"{item.get('from')} -> {item.get('to')}" for item in payload.get("files_renamed") or [] if isinstance(item, dict)]
    lines = ["Applied the approved ChangeSetProposal. Files were modified."]
    if modified:
        lines.append("Modified: " + ", ".join(f"`{path}`" for path in modified))
    if created:
        lines.append("Created: " + ", ".join(f"`{path}`" for path in created))
    if deleted:
        lines.append("Deleted: " + ", ".join(f"`{path}`" for path in deleted))
    if renamed:
        lines.append("Renamed: " + ", ".join(f"`{path}`" for path in renamed))
    validation = payload.get("validation_result") if isinstance(payload.get("validation_result"), dict) else {}
    lines.append(f"Validation result: {validation.get('status') or 'valid'} before apply.")
    return "\n".join(lines)


def _final_text_for_failed_apply(payload: dict[str, Any]) -> str:
    errors = "; ".join(str(item) for item in payload.get("errors") or []) or "unknown apply error"
    return f"The approved ChangeSetProposal could not be applied. No success was recorded. Error: {errors}"


def _with_checkpoint_bump(update: dict[str, Any]) -> dict[str, Any]:
    update["checkpoint_sequence"] = int(update.get("checkpoint_sequence") or 0) + 1
    return update


def _is_langgraph_checkpointer(value: Any) -> bool:
    return value is not None and hasattr(value, "put") and hasattr(value, "get_tuple")


def _stream_final_delta(run_id: str):
    def emit(delta: str) -> None:
        try:
            append_run_event(run_id, {"type": "assistant_delta", "delta": delta, "streaming_mode": "model_stream"})
        except OSError:
            return

    return emit


def _latest_sequence(run_id: str | None) -> int:
    if not run_id:
        return 0
    events = list_run_events(run_id)
    return max((int(event.get("sequence") or 0) for event in events), default=0)


def _chunk_text(text: str, *, size: int = 80) -> Iterator[str]:
    for index in range(0, len(text), size):
        yield text[index:index + size]


def _controller() -> Any:
    from repooperator_worker.agent_core import controller_graph

    return controller_graph
