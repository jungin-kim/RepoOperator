"""Context-loading nodes for RepoOperator LangGraph."""

from __future__ import annotations

from typing import Any

from repooperator_worker.agent_core.capabilities.builtin import get_default_capability_registry
from repooperator_worker.agent_core.context_packer import pack_context
from repooperator_worker.agent_core.graph.adapters import (
    _core_state_from_graph,
    _graph_transition_event,
    _request,
    _updates_from_core,
    _with_checkpoint_bump,
)
from repooperator_worker.agent_core.graph.nodes.supervisor import _frame_is_edit_like, _should_use_supervisor
from repooperator_worker.agent_core.graph.state import RepoOperatorGraphState
from repooperator_worker.agent_core.graph.support import load_context
from repooperator_worker.agent_core.mcp import get_default_mcp_registry
from repooperator_worker.agent_core.plugins import get_default_plugin_registry
from repooperator_worker.agent_core.skills import get_default_skill_registry
from repooperator_worker.agent_core.tools.registry import get_default_tool_registry
from repooperator_worker.agent_core.understanding_context import evidence_basis_update
from repooperator_worker.services.json_safe import json_safe

def load_context_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    request = _request(state)
    core = _core_state_from_graph(state)
    load_context(core, request)
    update = _updates_from_core(state, core)
    update["events_to_emit"] = [_graph_transition_event(state, "load_context", operation="load_context")]
    return _with_checkpoint_bump(update)

def capability_discovery_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    registry = get_default_capability_registry()
    tool_registry = get_default_tool_registry()
    skill_registry = get_default_skill_registry()
    plugin_registry = get_default_plugin_registry()
    mcp_registry = get_default_mcp_registry()
    request = _request(state)
    selected_skills = skill_registry.specs_for_model(task=request.task, limit=4)
    snapshot = {
        "capabilities": registry.specs_for_model(),
        "tool_capabilities": registry.tool_map(),
        "available_tools": [item["name"] for item in tool_registry.specs_for_model()],
        "all_tools": tool_registry.allowed_action_types(),
        "skills": skill_registry.specs_for_model(),
        "selected_skills": selected_skills,
        "plugins": plugin_registry.specs_for_model(enabled_only=True),
        "plugin_tools": plugin_registry.tool_metadata(enabled_only=True),
        "mcp_servers": mcp_registry.specs_for_model(enabled_only=True),
        "mcp_tools": mcp_registry.tool_metadata(enabled_only=True),
    }
    return _with_checkpoint_bump(
        {
            "capability_snapshot": json_safe(snapshot),
            "events_to_emit": [
                _graph_transition_event(
                    state,
                    "capability_discovery",
                    operation="capability_discovery",
                    aggregate={
                        "available_capabilities": registry.selectable_names(),
                        "selected_skills": [item.get("id") for item in selected_skills],
                        "enabled_plugins": [item.get("id") for item in snapshot["plugins"]],
                        "enabled_mcp_servers": [item.get("id") for item in snapshot["mcp_servers"]],
                    },
                )
            ],
        }
    )

def context_pack_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    return _with_checkpoint_bump(refresh_context_pack_update(state, trigger_node="context_pack"))

def refresh_context_pack_update(
    state: RepoOperatorGraphState,
    *,
    kind: str | None = None,
    trigger_node: str = "manual",
) -> dict[str, Any]:
    request = _request(state)
    base_context = state.get("context_packet") if isinstance(state.get("context_packet"), dict) else {}
    pack_kind = kind or _context_kind_for_state(state)
    packet = pack_context(pack_kind, request, state=dict(state), base_context=base_context)
    merged_packet = {**dict(base_context or {}), **packet}
    report = packet.get("context_pack_report") if isinstance(packet.get("context_pack_report"), dict) else {}
    summary = {
        "kind": packet.get("kind") or pack_kind,
        "trigger_node": trigger_node,
        "compression": packet.get("compression"),
        "compression_ratio": report.get("compression_ratio"),
        "estimated_input_tokens": report.get("estimated_input_tokens"),
        "estimated_output_reserve": report.get("estimated_output_reserve"),
        "included_sections": report.get("included_sections") or [],
        "excluded_sections": report.get("excluded_sections") or [],
        "warnings": report.get("warnings") or [],
        "file_count": len(((packet.get("file_evidence") or {}).get("included_files") or {})),
        "retained_files": report.get("retained_files") or [],
        "omitted_files": report.get("omitted_files") or [],
        "retained_web_sources": report.get("retained_web_sources") or [],
    }
    update = {
        "context_packet": json_safe(merged_packet),
        "ide_context": packet.get("ide_context"),
        "model_profile_snapshot": packet.get("model_profile"),
        "context_pack_summary": json_safe(summary),
        "context_pack_report": json_safe(report),
        "short_term_memory": packet.get("short_term_memory"),
        "events_to_emit": [
            _graph_transition_event(
                state,
                "context_pack",
                operation="context_pack",
                aggregate=summary,
            )
        ],
    }
    next_state = {**dict(state), **{key: value for key, value in update.items() if key != "events_to_emit"}}
    update.update(evidence_basis_update(next_state, trigger_node=trigger_node))
    return update

def _context_kind_for_state(state: RepoOperatorGraphState) -> str:
    from repooperator_worker.agent_core.graph.nodes.web import _web_research_needed

    if state.get("repair_attempts") or state.get("proposal_errors"):
        return "repair"
    if state.get("validation_results"):
        return "validation"
    if state.get("change_set_proposal") or _frame_is_edit_like(state):
        return "edit"
    if _web_research_needed(state):
        return "web_research"
    if _should_use_supervisor(state):
        return "broad_analysis"
    if state.get("git_workflow"):
        return "git_workflow"
    return "summary"
