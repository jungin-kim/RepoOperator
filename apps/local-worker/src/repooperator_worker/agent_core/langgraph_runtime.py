"""LangGraph orchestration seam for RepoOperator.

This module intentionally wraps the existing controller callbacks and
ToolOrchestrator boundary while introducing a real StateGraph topology. Durable
interrupt resume is represented by GraphCheckpointAdapter for this migration
patch; the next step is to back it with event_service storage and wire approved
commands back into the saved graph state instead of restarting a run.
"""

from __future__ import annotations

import copy
import json
import shlex
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, Iterator, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from repooperator_worker.agent_core.actions import AgentAction, ActionResult
from repooperator_worker.agent_core.hooks import HookManager
from repooperator_worker.agent_core.state import AgentCoreState, AgentSubtask, ClassifierResult
from repooperator_worker.agent_core.tool_orchestrator import ToolOrchestrator
from repooperator_worker.agent_core.tools.registry import get_default_tool_registry
from repooperator_worker.schemas import AgentRunRequest, AgentRunResponse
from repooperator_worker.services.event_service import append_run_event, list_run_events
from repooperator_worker.services.json_safe import json_safe, safe_agent_response_payload, safe_repr
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
    request: AgentRunRequest
    run_id: str
    thread_id: str | None
    repo: str
    branch: str | None
    context_packet: dict[str, Any] | None
    request_understanding: Any | None
    classifier_result: ClassifierResult
    task_frame: Any | None
    subtasks: list[AgentSubtask]
    current_subtask_id: str | None
    plan: list[str]
    messages: Annotated[list[dict[str, Any]], append_items]
    events: Annotated[list[dict[str, Any]], append_items]
    actions_taken: Annotated[list[AgentAction], append_items]
    action_results: Annotated[list[ActionResult], append_items]
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
    response_model: AgentRunResponse | None
    stop_reason: str | None
    loop_iteration: int
    budgets: dict[str, Any]
    events_to_emit: Annotated[list[dict[str, Any]], append_items]
    subtask_updates: Annotated[list[dict[str, Any]], append_items]
    evidence_reports: Annotated[list[dict[str, Any]], append_items]
    file_role_reports: Annotated[list[dict[str, Any]], append_items]
    proposed_changes: Annotated[list[dict[str, Any]], append_items]
    risk_notes: Annotated[list[str], append_items]
    supervisor_mode: bool
    current_worker_role: str | None
    pending_action: AgentAction | None
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


@dataclass(frozen=True)
class GraphCheckpointIdentity:
    run_id: str
    thread_id: str | None
    repo: str
    branch: str | None


@dataclass
class GraphCheckpointRecord:
    identity: GraphCheckpointIdentity
    sequence: int
    node: str
    state: dict[str, Any]
    created_at: float = field(default_factory=time.time)


class GraphCheckpointAdapter(Protocol):
    def save(self, identity: GraphCheckpointIdentity, sequence: int, node: str, state: RepoOperatorGraphState) -> GraphCheckpointRecord:
        ...

    def load_latest(self, identity: GraphCheckpointIdentity) -> GraphCheckpointRecord | None:
        ...


class InMemoryGraphCheckpointAdapter:
    """Compatibility checkpoint adapter until event-service-backed graph persistence lands."""

    def __init__(self) -> None:
        self._records: dict[tuple[str, str | None, str, str | None], list[GraphCheckpointRecord]] = {}

    def save(self, identity: GraphCheckpointIdentity, sequence: int, node: str, state: RepoOperatorGraphState) -> GraphCheckpointRecord:
        record = GraphCheckpointRecord(
            identity=identity,
            sequence=sequence,
            node=node,
            state=json_safe(_checkpointable_state(state)),
        )
        key = (identity.run_id, identity.thread_id, identity.repo, identity.branch)
        self._records.setdefault(key, []).append(record)
        return record

    def load_latest(self, identity: GraphCheckpointIdentity) -> GraphCheckpointRecord | None:
        key = (identity.run_id, identity.thread_id, identity.repo, identity.branch)
        records = self._records.get(key) or []
        return records[-1] if records else None


_DEFAULT_CHECKPOINT_ADAPTER = InMemoryGraphCheckpointAdapter()


def get_default_checkpoint_adapter() -> InMemoryGraphCheckpointAdapter:
    return _DEFAULT_CHECKPOINT_ADAPTER


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
            "final_synthesis": "final_synthesis",
        },
    )
    graph.add_edge("repair_change_set", "route_next")
    graph.add_edge("ask_clarification", "final_synthesis")
    graph.add_edge("await_approval", "final_synthesis")
    graph.add_edge("final_synthesis", END)
    return graph


def build_compiled_repooperator_graph(*, checkpoint_adapter: GraphCheckpointAdapter | None = None) -> Any:
    del checkpoint_adapter
    return build_repooperator_state_graph().compile()


def build_evidence_gathering_graph() -> StateGraph:
    graph = StateGraph(RepoOperatorGraphState)
    graph.add_node("inspect_tree", evidence_inspect_tree_node)
    graph.add_node("search_files", evidence_search_files_node)
    graph.add_node("search_text", evidence_search_text_node)
    graph.add_node("read_files", evidence_read_files_node)
    graph.add_node("update_evidence_store", update_evidence_store_node)
    graph.add_edge(START, "inspect_tree")
    graph.add_edge("inspect_tree", "search_files")
    graph.add_edge("search_files", "search_text")
    graph.add_edge("search_text", "read_files")
    graph.add_edge("read_files", "update_evidence_store")
    graph.add_edge("update_evidence_store", END)
    return graph


def build_analysis_graph() -> StateGraph:
    graph = StateGraph(RepoOperatorGraphState)
    graph.add_node("inventory", analysis_inventory_node)
    graph.add_node("batch_files", analysis_batch_files_node)
    graph.add_node("file_role_analysis", analysis_file_role_node)
    graph.add_node("summarize_batch", analysis_summarize_batch_node)
    graph.add_edge(START, "inventory")
    graph.add_edge("inventory", "batch_files")
    graph.add_edge("batch_files", "file_role_analysis")
    graph.add_edge("file_role_analysis", "summarize_batch")
    graph.add_edge("summarize_batch", END)
    return graph


def build_edit_graph() -> StateGraph:
    graph = StateGraph(RepoOperatorGraphState)
    graph.add_node("locate_targets", edit_locate_targets_node)
    graph.add_node("plan_change_set", edit_plan_change_set_node)
    graph.add_node("generate_change_set", edit_generate_change_set_node)
    graph.add_node("validate_change_set", edit_validate_change_set_node)
    graph.add_node("repair_change_set", edit_repair_change_set_node)
    graph.add_edge(START, "locate_targets")
    graph.add_edge("locate_targets", "plan_change_set")
    graph.add_edge("plan_change_set", "generate_change_set")
    graph.add_edge("generate_change_set", "validate_change_set")
    graph.add_edge("validate_change_set", "repair_change_set")
    graph.add_edge("repair_change_set", END)
    return graph


def build_validation_graph() -> StateGraph:
    graph = StateGraph(RepoOperatorGraphState)
    graph.add_node("choose_validation", validation_choose_node)
    graph.add_node("preview_command", validation_preview_command_node)
    graph.add_node("run_safe_validation", validation_run_safe_node)
    graph.add_node("parse_errors", validation_parse_errors_node)
    graph.add_node("update_validation_result", validation_update_result_node)
    graph.add_edge(START, "choose_validation")
    graph.add_edge("choose_validation", "preview_command")
    graph.add_edge("preview_command", "run_safe_validation")
    graph.add_edge("run_safe_validation", "parse_errors")
    graph.add_edge("parse_errors", "update_validation_result")
    graph.add_edge("update_validation_result", END)
    return graph


def build_finalization_graph() -> StateGraph:
    graph = StateGraph(RepoOperatorGraphState)
    graph.add_node("repair_final_answer", final_repair_answer_node)
    graph.add_node("build_response", final_build_response_node)
    graph.add_node("emit_final_message", final_emit_message_node)
    graph.add_edge(START, "repair_final_answer")
    graph.add_edge("repair_final_answer", "build_response")
    graph.add_edge("build_response", "emit_final_message")
    graph.add_edge("emit_final_message", END)
    return graph


def run_langgraph_controller(
    request: AgentRunRequest,
    *,
    run_id: str | None = None,
    stream_final_answer: bool = False,
    checkpoint_adapter: GraphCheckpointAdapter | None = None,
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
    adapter = checkpoint_adapter or get_default_checkpoint_adapter()
    _save_checkpoint(adapter, initial_state, "start")
    final_state = build_compiled_repooperator_graph(checkpoint_adapter=adapter).invoke(initial_state)
    _save_checkpoint(adapter, final_state, "end")
    response = final_state.get("response_model")
    if isinstance(response, AgentRunResponse):
        return response
    core = _core_state_from_graph(final_state)
    if not core.final_response:
        core.final_response = final_state.get("final_response") or ""
    return _controller().build_final_response(core, request)


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


def initial_graph_state(
    request: AgentRunRequest,
    *,
    run_id: str,
    stream_final_answer: bool = False,
    skills_context: str = "",
    skills_used: list[str] | None = None,
) -> RepoOperatorGraphState:
    return {
        "request": request,
        "run_id": run_id,
        "thread_id": request.thread_id,
        "repo": request.project_path,
        "branch": request.branch,
        "context_packet": None,
        "request_understanding": None,
        "classifier_result": ClassifierResult(),
        "task_frame": None,
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
        "response_model": None,
        "stop_reason": None,
        "loop_iteration": 0,
        "budgets": {},
        "events_to_emit": [],
        "subtask_updates": [],
        "evidence_reports": [],
        "file_role_reports": [],
        "proposed_changes": [],
        "risk_notes": [],
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
    update = _updates_from_core(state, core)
    update["task_frame"] = _controller().build_task_frame(request, core)
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
    if state.get("stop_reason") == "waiting_approval":
        return {"next_node": "await_approval", "events_to_emit": [_graph_transition_event(state, "route_next", operation="approval_gate")]}
    if state.get("stop_reason") in {"needs_clarification"}:
        return {"next_node": "ask_clarification", "events_to_emit": [_graph_transition_event(state, "route_next", operation="clarification")]}

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
    action = _select_next_action(core, request)
    core.current_step = action.reason_summary
    update = _updates_from_core(state, core)
    route = route_by_stage({**dict(state), **update, "pending_action": action})
    update.update(
        {
            "pending_action": action,
            "next_node": route,
            "current_step": action.reason_summary,
            "task_frame": _controller().build_task_frame(request, core),
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
    if latest and latest.status == "failed" and int(state.get("repair_attempts") or 0) < 1:
        return "repair_change_set"
    return route_to_final_or_continue(state)


def route_after_approval(state: RepoOperatorGraphState) -> str:
    if state.get("pending_approval"):
        return "final_synthesis"
    return route_to_final_or_continue(state)


def route_to_final_or_continue(state: RepoOperatorGraphState) -> str:
    if state.get("stop_reason") in {"cancelled", "timed_out", "max_loop_iterations", "max_file_reads", "max_commands", "waiting_approval"}:
        return "final_synthesis"
    return _route_to_final_or_action(state)


def supervisor_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    groups = _supervisor_file_groups(state)
    reports = [
        {
            "worker": "AnalysisAgent",
            "group": name,
            "files": files,
            "summary": f"Queued {len(files)} file(s) for bounded role analysis.",
        }
        for name, files in groups.items()
        if files
    ]
    return _with_checkpoint_bump(
        {
            "supervisor_mode": True,
            "evidence_reports": reports,
            "file_role_reports": reports,
            "routing_stage": "after_understanding",
            "events_to_emit": [
                _graph_transition_event(
                    state,
                    "supervisor",
                    subgraph="supervisor",
                    operation="delegate",
                    status="completed",
                    aggregate={"workers": ["EvidenceAgent", "AnalysisAgent", "EditAgent", "ValidationAgent", "DocumentationAgent", "TestAgent"]},
                )
            ],
        }
    )


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
    action = state.get("pending_action")
    proposal = {
        "status": "planned",
        "action_id": action.action_id if action else None,
        "files": list(action.target_files if action else []),
        "summary": action.reason_summary if action else "Plan proposal-only change set.",
    }
    return _with_checkpoint_bump(
        {
            "change_set_proposal": proposal,
            "proposed_changes": [proposal],
            "routing_stage": "after_change_plan",
            "events_to_emit": [
                _graph_transition_event(
                    state,
                    "plan_change_set",
                    subgraph="edit_graph",
                    operation="plan_change_set",
                    files=proposal["files"],
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
    proposal = state.get("change_set_proposal") or {}
    if latest and latest.payload.get("edit_proposals"):
        proposal = {
            **proposal,
            "status": latest.status,
            "edit_proposals": latest.payload.get("edit_proposals") or [],
            "proposal_error": latest.payload.get("proposal_error"),
        }
    validation = {
        "kind": "change_set",
        "status": latest.status if latest else "skipped",
        "action_id": latest.action_id if latest else None,
        "proposal_files": [str(item.get("file")) for item in proposal.get("edit_proposals") or [] if isinstance(item, dict)],
    }
    return _with_checkpoint_bump(
        {
            "change_set_proposal": proposal,
            "validation_results": [validation],
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
    action = state.get("pending_action")
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
    adapter = get_default_checkpoint_adapter()
    _save_checkpoint(adapter, state, "await_approval")
    return _with_checkpoint_bump(
        {
            "stop_reason": "waiting_approval",
            "routing_stage": "after_approval",
            "events_to_emit": [
                _graph_transition_event(
                    state,
                    "await_approval",
                    operation="approval_gate",
                    status="waiting",
                    command=(state.get("pending_approval") or {}).get("command"),
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


def evidence_inspect_tree_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    return _execute_if_action_type(state, {"inspect_repo_tree"}, "evidence_gathering_graph", "inspect_tree")


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
    return {
        "events_to_emit": [_graph_transition_event(state, "batch_files", subgraph="analysis_graph", operation="batch_files")],
    }


def analysis_file_role_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    reports = [{"file": path, "role": "evidence file"} for path in state.get("files_read") or []]
    return {
        "file_role_reports": reports,
        "events_to_emit": [_graph_transition_event(state, "file_role_analysis", subgraph="analysis_graph", operation="file_role_analysis")],
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
    }


def edit_locate_targets_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    action = state.get("pending_action")
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


def edit_plan_change_set_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    return {
        "events_to_emit": [_graph_transition_event(state, "plan_change_set", subgraph="edit_graph", operation="plan_change_set")]
    }


def edit_generate_change_set_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    return _execute_if_action_type(state, {"generate_edit"}, "edit_graph", "generate_change_set")


def edit_validate_change_set_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    latest = _latest_result(state)
    validation = {
        "kind": "change_set",
        "status": latest.status if latest else "skipped",
        "action_id": latest.action_id if latest else None,
    }
    return {
        "validation_results": [validation],
        "events_to_emit": [
            _graph_transition_event(state, "validate_change_set", subgraph="edit_graph", operation="validate_change_set", validation_result=validation)
        ],
    }


def edit_repair_change_set_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    return {
        "events_to_emit": [_graph_transition_event(state, "repair_change_set", subgraph="edit_graph", operation="repair_change_set")]
    }


def validation_choose_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    return {"events_to_emit": [_graph_transition_event(state, "choose_validation", subgraph="validation_graph", operation="choose_validation")]}


def validation_preview_command_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    return _execute_if_action_type(state, {"preview_command", "inspect_git_state"}, "validation_graph", "preview_command")


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
        "events_to_emit": [_graph_transition_event(state, "update_validation_result", subgraph="validation_graph", operation="update_validation_result")]
    }


def final_repair_answer_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    return {
        "events_to_emit": [_graph_transition_event(state, "repair_final_answer", subgraph="finalization_graph", operation="repair_final_answer")]
    }


def final_build_response_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    request = _request(state)
    core = _core_state_from_graph(state)
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
    response = _controller().build_final_response(core, request)
    return {
        "response_model": response,
        "events_to_emit": [_graph_transition_event(state, "emit_final_message", subgraph="finalization_graph", operation="emit_final_message")],
    }


def _execute_if_action_type(state: RepoOperatorGraphState, action_types: set[str], subgraph: str, node_name: str) -> dict[str, Any]:
    action = state.get("pending_action")
    if action and action.type in action_types:
        return _execute_pending_action(state, subgraph=subgraph, node_name=node_name)
    return {"events_to_emit": [_graph_transition_event(state, node_name, subgraph=subgraph, operation="skip")]}


def _execute_pending_action(state: RepoOperatorGraphState, *, subgraph: str | None, node_name: str | None = None) -> dict[str, Any]:
    action = state.get("pending_action")
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
    action = state.get("pending_action")
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
    if action.type in {"generate_edit", "validate_edit"}:
        return "plan_change_set"
    if action.type in {"preview_command", "inspect_git_state", "run_approved_command", "request_command_approval"}:
        return "execute_tool"
    return "execute_tool"


def _should_use_supervisor(state: RepoOperatorGraphState) -> bool:
    if state.get("supervisor_mode"):
        return False
    frame = state.get("task_frame")
    if frame is None:
        return False
    text = " ".join([str(getattr(frame, "user_goal", "")), *[str(item) for item in getattr(frame, "requested_outputs", [])]]).lower()
    broad = any(term in text for term in ("whole", "entire", "all files", "every file", "codebase", "source tree", "repository-wide"))
    return broad and not state.get("files_read")


def _select_next_action(core: AgentCoreState, request: AgentRunRequest) -> AgentAction:
    return _controller().controller_choose_next_action(core, request)


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
        "request_understanding": core.request_understanding,
        "classifier_result": core.classifier_result,
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
        "subtasks": list(core.subtasks),
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
            update[field_name] = list(new[len(old):])
    return {key: value for key, value in update.items() if value is not None or key in {"pending_approval", "stop_reason"}}


def _updates_from_core_after_action(
    before: RepoOperatorGraphState,
    core: AgentCoreState,
    action: AgentAction,
    result: ActionResult,
) -> dict[str, Any]:
    update = _updates_from_core(before, core)
    update["actions_taken"] = [action]
    update["action_results"] = [result]
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
    core.classifier_result = state.get("classifier_result") or ClassifierResult()
    core.request_understanding = state.get("request_understanding")
    core.plan = list(state.get("plan") or [])
    core.current_step = state.get("current_step")
    core.observations = list(state.get("observations") or [])
    core.actions_taken = list(state.get("actions_taken") or [])
    core.action_results = list(state.get("action_results") or [])
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
    core.subtasks = list(state.get("subtasks") or [])
    core.current_subtask_id = state.get("current_subtask_id")
    core.zero_result_queries = list(state.get("zero_result_queries") or [])
    core.failed_action_signatures = list(state.get("failed_action_signatures") or [])
    core.strategy_shifts = list(state.get("strategy_shifts") or [])
    return core


def _request(state: RepoOperatorGraphState) -> AgentRunRequest:
    request = state.get("request")
    if not isinstance(request, AgentRunRequest):
        raise ValueError("RepoOperatorGraphState requires an AgentRunRequest under 'request'.")
    return request


def _latest_result(state: RepoOperatorGraphState) -> ActionResult | None:
    results = state.get("action_results") or []
    return results[-1] if results else None


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
    action = state.get("pending_action")
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


def _with_checkpoint_bump(update: dict[str, Any]) -> dict[str, Any]:
    update["checkpoint_sequence"] = int(update.get("checkpoint_sequence") or 0) + 1
    return update


def _save_checkpoint(adapter: GraphCheckpointAdapter, state: RepoOperatorGraphState, node: str) -> None:
    identity = GraphCheckpointIdentity(
        run_id=str(state.get("run_id") or "run_controller"),
        thread_id=state.get("thread_id"),
        repo=str(state.get("repo") or ""),
        branch=state.get("branch"),
    )
    sequence = int(state.get("checkpoint_sequence") or 0)
    adapter.save(identity, sequence, node, state)


def _checkpointable_state(state: RepoOperatorGraphState) -> dict[str, Any]:
    payload = dict(state)
    request = payload.get("request")
    if isinstance(request, AgentRunRequest):
        payload["request"] = request.model_dump()
    response = payload.get("response_model")
    if isinstance(response, AgentRunResponse):
        payload["response_model"] = response.model_dump()
    return payload


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


def _noop_node(state: RepoOperatorGraphState, node: str, *, subgraph: str, operation: str) -> dict[str, Any]:
    return {"events_to_emit": [_graph_transition_event(state, node, subgraph=subgraph, operation=operation)]}


__all__ = [
    "GraphCheckpointAdapter",
    "GraphCheckpointIdentity",
    "GraphCheckpointRecord",
    "InMemoryGraphCheckpointAdapter",
    "RepoOperatorGraphState",
    "append_items",
    "append_unique_items",
    "build_analysis_graph",
    "build_compiled_repooperator_graph",
    "build_edit_graph",
    "build_evidence_gathering_graph",
    "build_finalization_graph",
    "build_repooperator_state_graph",
    "build_validation_graph",
    "get_default_checkpoint_adapter",
    "initial_graph_state",
    "route_after_approval",
    "route_after_change_plan",
    "route_after_evidence",
    "route_after_tool_result",
    "route_after_understanding",
    "route_after_validation",
    "route_to_final_or_continue",
    "run_langgraph_controller",
    "stream_langgraph_controller",
]
