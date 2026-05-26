"""Request-understanding nodes for RepoOperator LangGraph."""

from __future__ import annotations

from typing import Any

from repooperator_worker.agent_core.change_set import EditMode
from repooperator_worker.agent_core.graph.adapters import (
    _core_state_from_graph,
    _graph_transition_event,
    _merge_updates,
    _request,
    _updates_from_core,
    _with_checkpoint_bump,
)
from repooperator_worker.agent_core.graph.nodes.context import refresh_context_pack_update
from repooperator_worker.agent_core.graph.state import RepoOperatorGraphState
from repooperator_worker.agent_core.graph_state import task_frame_to_snapshot
from repooperator_worker.agent_core.graph.support import classify, create_initial_plan, emit_plan_update
from repooperator_worker.agent_core.planner import build_task_frame, edit_requested
from repooperator_worker.agent_core.understanding_context import (
    append_visible_rationale,
    evidence_basis_update,
    update_user_understanding_context,
)

def understand_request_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    request = _request(state)
    context_update = refresh_context_pack_update(state, kind="summary", trigger_node="understand_request")
    working_state = {**dict(state), **{key: value for key, value in context_update.items() if key != "events_to_emit"}}
    core = _core_state_from_graph(working_state)
    classify(core, request)
    task_frame = build_task_frame(request, core)
    update = _updates_from_core(working_state, core)
    update["task_frame_snapshot"] = task_frame_to_snapshot(task_frame)
    update["edit_mode"] = _edit_mode_for_request(task_frame)
    update["budgets"] = {
        "max_loop_iterations": core.max_loop_iterations,
        "max_file_reads": core.max_file_reads,
        "max_commands": core.max_commands,
        "max_edits": core.max_edits,
    }
    working_with_update = _merge_updates(working_state, update)
    update = _merge_updates(update, update_user_understanding_context(working_with_update, request, "understand_request"))
    update = _merge_updates(
        update,
        append_visible_rationale(
            working_with_update,
            node="understand_request",
            action=None,
            summary="I separated the request into expected output, constraints, mentioned files, and evidence needs before choosing actions.",
            basis_refs=[],
            safety_note="This request understanding is inspectable context, not hard routing.",
            uncertainty=[],
        ),
    )
    update["events_to_emit"] = [_graph_transition_event(state, "understand_request", operation="understand_request")]
    return _with_checkpoint_bump(_merge_updates(context_update, update))

def build_task_plan_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    request = _request(state)
    core = _core_state_from_graph(state)
    create_initial_plan(core)
    emit_plan_update(core, request, "Created initial plan")
    update = _updates_from_core(state, core)
    working_with_update = _merge_updates(dict(state), update)
    update = _merge_updates(update, update_user_understanding_context(working_with_update, request, "build_task_plan"))
    update = _merge_updates(update, evidence_basis_update(working_with_update, trigger_node="build_task_plan"))
    update["events_to_emit"] = [_graph_transition_event(state, "build_task_plan", operation="plan")]
    return _with_checkpoint_bump(update)

def _edit_mode_for_request(frame: Any) -> EditMode:
    return "proposal_only" if edit_requested(frame) else "explanation_only"
