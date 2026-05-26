"""Context loading and packing helpers for LangGraph agent runs."""

from __future__ import annotations

from typing import Any

from repooperator_worker.agent_core.context_packer import pack_context
from repooperator_worker.agent_core.context_service import get_default_context_service
from repooperator_worker.agent_core.events import append_activity_event
from repooperator_worker.agent_core.state import AgentCoreState
from repooperator_worker.schemas import AgentRunRequest
from repooperator_worker.services.json_safe import json_safe


def load_context(state: AgentCoreState, request: AgentRunRequest) -> None:
    packet = get_default_context_service().collect(request)
    state.context_packet = packet.model_dump()
    high_signal = sorted(packet.high_signal_files)
    instructions = sorted(packet.project_instructions)
    state.observations.append("Loaded request context for the active repository.")
    append_activity_event(
        run_id=state.run_id,
        request=request,
        activity_id="langgraph-load-context",
        event_type="activity_completed",
        phase="Thinking",
        label="Loaded context",
        status="completed",
        observation="Loaded request, repository, branch, thread, and skill context.",
        aggregate={
            "repo_root_name": packet.repo_root_name,
            "branch": packet.branch,
            "high_signal_files": high_signal,
            "project_instruction_files": instructions,
            "prior_files_read": packet.prior_files_read,
            "prior_commands_run": packet.prior_commands_run,
            "git_status_available": bool(packet.git_status_summary),
            "recent_commits_available": bool(packet.recent_commits_summary),
        },
    )


def refresh_context_pack_for_core(state: AgentCoreState, request: AgentRunRequest, kind: str, trigger_node: str) -> dict[str, Any]:
    base_context = state.context_packet if isinstance(state.context_packet, dict) else {}
    packet = pack_context(kind, request, state=_core_context_pack_state(state), base_context=base_context)
    state.context_packet = json_safe({**dict(base_context or {}), **packet})
    report = packet.get("context_pack_report") if isinstance(packet.get("context_pack_report"), dict) else {}
    summary = {
        "kind": packet.get("kind") or kind,
        "trigger_node": trigger_node,
        "compression_ratio": report.get("compression_ratio"),
        "estimated_input_tokens": report.get("estimated_input_tokens"),
        "estimated_output_reserve": report.get("estimated_output_reserve"),
        "included_sections": report.get("included_sections") or [],
        "excluded_sections": report.get("excluded_sections") or [],
        "warnings": report.get("warnings") or [],
        "retained_files": report.get("retained_files") or [],
        "omitted_files": report.get("omitted_files") or [],
        "retained_web_sources": report.get("retained_web_sources") or [],
    }
    append_activity_event(
        run_id=state.run_id,
        request=request,
        activity_id=f"context-pack:{trigger_node}",
        event_type="graph_transition",
        phase="Thinking",
        label="Packed model context",
        status="completed",
        visibility="debug",
        display="secondary",
        operation="context_pack",
        aggregate=json_safe(summary),
    )
    return json_safe(packet)


def _core_context_pack_state(state: AgentCoreState) -> dict[str, Any]:
    action_results = [result.model_dump() for result in state.action_results]
    actions_taken = [action.model_dump() for action in state.actions_taken]
    proposal: dict[str, Any] | None = None
    for result in reversed(state.action_results):
        candidate = result.payload.get("change_set_proposal") if isinstance(result.payload, dict) else None
        if isinstance(candidate, dict):
            proposal = candidate
            break
    if proposal is None:
        for result in reversed(state.action_results):
            proposals = result.payload.get("edit_proposals") if isinstance(result.payload, dict) else None
            if isinstance(proposals, list) and proposals:
                proposal = {"proposal_id": result.action_id, "status": result.status, "changes": proposals}
                break
    return {
        "actions_taken": actions_taken,
        "action_results": action_results,
        "files_read": list(state.files_read),
        "files_changed": list(state.files_changed),
        "commands_run": list(state.commands_run),
        "pending_approval": state.pending_approval,
        "change_set_proposal": proposal,
        "observations": list(state.observations),
    }
