"""Routine nodes for RepoOperator LangGraph."""

from __future__ import annotations

from typing import Any

from repooperator_worker.agent_core.graph.adapters import _graph_transition_event, _merge_updates, _with_checkpoint_bump
from repooperator_worker.agent_core.graph.state import RepoOperatorGraphState
from repooperator_worker.agent_core.understanding_context import append_visible_rationale, evidence_basis_update

def routine_enqueue_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    update = {
        "routine_context": {"status": "not_enqueued", "reason": "Routine runs use normal AgentRunRequest enqueue paths."},
        "events_to_emit": [_graph_transition_event(state, "routine_enqueue_node", subgraph="routine", operation="routine_enqueue")],
    }
    next_state = _merge_updates(dict(state), update)
    update = _merge_updates(update, evidence_basis_update(next_state, trigger_node="routine_enqueue_node"))
    update = _merge_updates(
        update,
        append_visible_rationale(
            next_state,
            node="routine_enqueue_node",
            action=None,
            summary="Routine work stays on the normal run path so scheduled work cannot bypass tool or approval safety.",
            basis_refs=[],
            safety_note="Routine execution does not grant extra write permissions.",
            uncertainty=[],
        ),
    )
    return _with_checkpoint_bump(update)
