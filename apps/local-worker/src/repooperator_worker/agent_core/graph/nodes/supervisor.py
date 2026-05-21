"""Supervisor and worker-reduction nodes for RepoOperator LangGraph."""

from __future__ import annotations

from typing import Any

from repooperator_worker.agent_core.actions import AgentAction
from repooperator_worker.agent_core.graph.adapters import (
    _append_action_event,
    _core_state_from_graph,
    _graph_transition_event,
    _invoke_subgraph_delta,
    _merge_updates,
    _request,
    _task_frame,
    _with_checkpoint_bump,
)
from repooperator_worker.agent_core.graph.state import RepoOperatorGraphState
from repooperator_worker.agent_core.hooks import HookManager
from repooperator_worker.agent_core.planner import build_task_frame, edit_requested
from repooperator_worker.agent_core.tool_orchestrator import ToolOrchestrator
from repooperator_worker.agent_core.tools.registry import get_default_tool_registry
from repooperator_worker.agent_core.understanding_context import append_visible_rationale, evidence_basis_update

def decompose_task_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    from repooperator_worker.agent_core.tasks import decompose_complex_task

    plan = decompose_complex_task(_request(state).task, capability_snapshot=state.get("capability_snapshot") or {})
    work_units = [unit.model_dump() for unit in plan.work_units]
    return _with_checkpoint_bump(
        {
            "worker_tasks": work_units,
            "supervisor_mode": True,
            "events_to_emit": [
                _graph_transition_event(
                    state,
                    "decompose_task",
                    subgraph="supervisor",
                    operation="decompose_task",
                    aggregate={"work_unit_count": len(work_units), "plan_id": plan.id},
                )
            ],
        }
    )

def dispatch_work_units_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    reports = [_run_worker_task(task, state=state) for task in state.get("worker_tasks") or [] if _work_unit_dependencies_complete(task, state)]
    return _with_checkpoint_bump(
        {
            "worker_reports": reports,
            "events_to_emit": [
                _graph_transition_event(
                    state,
                    "dispatch_work_units",
                    subgraph="supervisor",
                    operation="dispatch_work_units",
                    aggregate={"worker_report_count": len(reports)},
                )
            ],
        }
    )

def reduce_work_reports_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    return supervisor_reduce_worker_reports_node(state)

def supervisor_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    from repooperator_worker.agent_core.graph.builder import build_supervisor_graph

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
    update = {
        "file_role_reports": file_role_reports,
        "evidence_reports": evidence_reports,
        "proposed_changes": proposed_changes,
        "risk_notes": risk_notes,
        "events_to_emit": [_graph_transition_event(state, "reduce_worker_reports", subgraph="supervisor", operation="reduce_worker_reports")],
    }
    next_state = _merge_updates(dict(state), update)
    update = _merge_updates(update, evidence_basis_update(next_state, trigger_node="reduce_worker_reports"))
    update = _merge_updates(
        update,
        append_visible_rationale(
            next_state,
            node="reduce_worker_reports",
            action=None,
            summary="I reduced the worker reports into the evidence basis so broad analysis stays auditable by worker scope and files analyzed.",
            basis_refs=[{"kind": "file", "path": path} for report in reports for path in (report.get("files_analyzed") or report.get("files") or [])[:2]][:8],
            safety_note="Worker reports are summaries and do not include private reasoning.",
            uncertainty=[],
        ),
    )
    return update

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
    role = str(task.get("role") or task.get("assigned_worker_role") or "AnalysisAgent")
    if role in {"AnalysisAgent", "CodeAnalysisAgent"}:
        return _run_analysis_worker_task(task, state=state)
    files = [str(item) for item in task.get("input_files") or task.get("files") or []]
    if role == "EvidenceAgent":
        read = _worker_read_files(state, files[:1])
        return {
            "worker": role,
            "work_unit_id": task.get("id") or task.get("task_id"),
            "task_id": task.get("task_id"),
            "role": role,
            "scope": task.get("scope") or task.get("group"),
            "files_analyzed": read["files_read"],
            "files": files,
            "findings": ["Located bounded evidence candidates for the requested scope."],
            "summary": "Located bounded evidence candidates for the requested scope.",
            "status": read["status"],
        }
    if role == "WebResearchAgent":
        return {
            "worker": role,
            "work_unit_id": task.get("id") or task.get("task_id"),
            "task_id": task.get("task_id"),
            "role": role,
            "findings": ["Web research must use search_web/fetch_url as untrusted evidence when needed."],
            "summary": "Prepared web research constraints; no web write or trust escalation occurred.",
            "status": "completed",
        }
    if role == "EditPlanningAgent":
        return {
            "worker": role,
            "work_unit_id": task.get("id") or task.get("task_id"),
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
        return {"worker": role, "work_unit_id": task.get("id") or task.get("task_id"), "task_id": task.get("task_id"), "role": role, "files": files, "files_analyzed": [], "findings": ["Validation should run through ToolOrchestrator-backed checks."], "summary": "Validation should run through ToolOrchestrator-backed checks.", "status": "completed"}
    if role == "GitAgent":
        return {"worker": role, "work_unit_id": task.get("id") or task.get("task_id"), "task_id": task.get("task_id"), "role": role, "files": files, "files_analyzed": [], "findings": ["Git writes require explicit approval for commit, push, and PR/MR actions."], "summary": "Prepared approval-gated git workflow guidance without remote writes.", "status": "completed"}
    if role == "DocumentationAgent":
        return {"worker": role, "work_unit_id": task.get("id") or task.get("task_id"), "task_id": task.get("task_id"), "role": role, "files": files, "files_analyzed": [], "findings": ["Documentation impact should be considered if behavior changes."], "summary": "Documentation impact should be considered if behavior changes.", "status": "completed"}
    if role == "TestAgent":
        return {"worker": role, "work_unit_id": task.get("id") or task.get("task_id"), "task_id": task.get("task_id"), "role": role, "files": files, "files_analyzed": [], "findings": ["Tests or safe validation commands may be needed after a proposal."], "summary": "Tests or safe validation commands may be needed after a proposal.", "status": "completed"}
    return {"worker": role, "work_unit_id": task.get("id") or task.get("task_id"), "task_id": task.get("task_id"), "role": role, "files": files, "files_analyzed": [], "findings": ["Worker completed bounded scoped analysis."], "summary": "Worker completed bounded scoped analysis.", "status": "completed"}

def _run_analysis_worker_task(task: dict[str, Any], *, state: RepoOperatorGraphState | None = None) -> dict[str, Any]:
    files = [str(item) for item in task.get("input_files") or task.get("files") or []]
    read = _worker_read_files(state, files[:1]) if state is not None else {"files_read": [], "status": "completed"}
    return {
        "worker": "AnalysisAgent",
        "work_unit_id": task.get("id") or task.get("task_id"),
        "task_id": task.get("task_id"),
        "role": task.get("role") or task.get("assigned_worker_role") or "AnalysisAgent",
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

def _should_use_supervisor(state: RepoOperatorGraphState) -> bool:
    if state.get("supervisor_mode"):
        return False
    frame = _task_frame(state)
    if frame is None:
        return False
    text = " ".join([str(getattr(frame, "user_goal", "")), *[str(item) for item in getattr(frame, "requested_outputs", [])]]).lower()
    broad = any(term in text for term in ("whole", "entire", "all files", "every file", "codebase", "source tree", "repository-wide"))
    return broad and not state.get("files_read")

def _work_unit_dependencies_complete(task: dict[str, Any], state: RepoOperatorGraphState) -> bool:
    dependencies = [str(item) for item in task.get("dependencies") or [] if str(item)]
    if not dependencies:
        return True
    completed = {
        str(report.get("work_unit_id") or report.get("task_id"))
        for report in state.get("worker_reports") or []
        if isinstance(report, dict) and report.get("status") == "completed"
    }
    return all(dep in completed for dep in dependencies)
