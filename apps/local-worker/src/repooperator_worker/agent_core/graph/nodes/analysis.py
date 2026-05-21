"""Analysis subgraph nodes for RepoOperator LangGraph."""

from __future__ import annotations

from typing import Any

from repooperator_worker.agent_core.graph.adapters import _execute_if_action_type, _graph_transition_event, _invoke_subgraph_delta, _with_checkpoint_bump
from repooperator_worker.agent_core.graph.state import RepoOperatorGraphState
from repooperator_worker.agent_core.graph.nodes.supervisor import _run_analysis_worker_task, _supervisor_file_groups, _worker_tasks_from_groups

def analysis_graph_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    from repooperator_worker.agent_core.graph.builder import build_analysis_graph

    update = _invoke_subgraph_delta(build_analysis_graph, state)
    update["routing_stage"] = "after_evidence"
    update.setdefault("events_to_emit", []).append(
        _graph_transition_event(state, "analysis_graph", subgraph="analysis_graph", operation="analyze")
    )
    return _with_checkpoint_bump(update)

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
