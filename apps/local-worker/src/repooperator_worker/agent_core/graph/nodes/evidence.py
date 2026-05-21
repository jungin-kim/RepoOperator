"""Evidence-gathering nodes for RepoOperator LangGraph."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END

from repooperator_worker.agent_core.actions import ActionResult
from repooperator_worker.agent_core.graph.adapters import (
    _core_state_from_graph,
    _execute_if_action_type,
    _graph_transition_event,
    _invoke_subgraph_delta,
    _latest_result,
    _pending_action,
    _request,
    _with_checkpoint_bump,
)
from repooperator_worker.agent_core.graph.state import RepoOperatorGraphState, append_unique_items
from repooperator_worker.agent_core.planner import build_task_frame, candidate_files_from_results, edit_requested
from repooperator_worker.agent_core.task_policy import next_evidence_gathering_action

def gather_evidence_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    from repooperator_worker.agent_core.graph.builder import build_evidence_gathering_graph

    update = _invoke_subgraph_delta(build_evidence_gathering_graph, state)
    update["routing_stage"] = "after_evidence"
    update.setdefault("events_to_emit", []).append(
        _graph_transition_event(state, "gather_evidence", subgraph="evidence_gathering_graph", operation="gather_evidence")
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

def _evidence_report(state: RepoOperatorGraphState, result: ActionResult | None) -> dict[str, Any]:
    action = _pending_action(state)
    return {
        "worker": "EvidenceAgent",
        "action_type": action.type if action else None,
        "status": result.status if result else "skipped",
        "files": list(result.files_read if result else []),
        "observation": result.observation if result else "",
    }
