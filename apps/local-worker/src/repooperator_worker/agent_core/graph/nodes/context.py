"""Context-loading nodes for RepoOperator LangGraph."""

from __future__ import annotations

from typing import Any

from repooperator_worker.agent_core.capabilities.builtin import get_default_capability_registry
from repooperator_worker.agent_core.context_packer import pack_context
from repooperator_worker.agent_core.graph.adapters import (
    _controller,
    _core_state_from_graph,
    _graph_transition_event,
    _request,
    _updates_from_core,
    _with_checkpoint_bump,
)
from repooperator_worker.agent_core.graph.nodes.supervisor import _frame_is_edit_like, _should_use_supervisor
from repooperator_worker.agent_core.graph.nodes.web import _web_research_needed
from repooperator_worker.agent_core.graph.state import RepoOperatorGraphState
from repooperator_worker.agent_core.tools.registry import get_default_tool_registry
from repooperator_worker.services.json_safe import json_safe

def load_context_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    request = _request(state)
    core = _core_state_from_graph(state)
    _controller().load_context(core, request)
    update = _updates_from_core(state, core)
    update["events_to_emit"] = [_graph_transition_event(state, "load_context", operation="load_context")]
    return _with_checkpoint_bump(update)

def capability_discovery_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    registry = get_default_capability_registry()
    tool_registry = get_default_tool_registry()
    snapshot = {
        "capabilities": registry.specs_for_model(),
        "tool_capabilities": registry.tool_map(),
        "available_tools": [item["name"] for item in tool_registry.specs_for_model()],
        "all_tools": tool_registry.allowed_action_types(),
    }
    return _with_checkpoint_bump(
        {
            "capability_snapshot": json_safe(snapshot),
            "events_to_emit": [
                _graph_transition_event(
                    state,
                    "capability_discovery",
                    operation="capability_discovery",
                    aggregate={"available_capabilities": registry.selectable_names()},
                )
            ],
        }
    )

def context_pack_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    request = _request(state)
    base_context = state.get("context_packet") if isinstance(state.get("context_packet"), dict) else {}
    kind = _context_kind_for_state(state)
    packet = pack_context(kind, request, state=dict(state), base_context=base_context)
    merged_packet = {**dict(base_context or {}), **packet}
    summary = {
        "kind": kind,
        "compression": packet.get("compression"),
        "file_count": len(((packet.get("file_evidence") or {}).get("included_files") or {})),
    }
    return _with_checkpoint_bump(
        {
            "context_packet": json_safe(merged_packet),
            "model_profile_snapshot": packet.get("model_profile"),
            "context_pack_summary": json_safe(summary),
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
    )

def _context_kind_for_state(state: RepoOperatorGraphState) -> str:
    if state.get("repair_attempts") or state.get("proposal_errors"):
        return "repair_context"
    if state.get("validation_results"):
        return "validation_context"
    if state.get("change_set_proposal") or _frame_is_edit_like(state):
        return "edit_context"
    if _web_research_needed(state):
        return "web_research_context"
    if _should_use_supervisor(state):
        return "broad_analysis_context"
    return "summary_context"
