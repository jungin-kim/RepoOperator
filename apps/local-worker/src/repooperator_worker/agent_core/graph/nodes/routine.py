"""Routine nodes for RepoOperator LangGraph."""

from __future__ import annotations

from typing import Any

from repooperator_worker.agent_core.graph.adapters import _graph_transition_event, _with_checkpoint_bump
from repooperator_worker.agent_core.graph.state import RepoOperatorGraphState

def routine_enqueue_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    return _with_checkpoint_bump(
        {
            "routine_context": {"status": "not_enqueued", "reason": "Routine runs use normal AgentRunRequest enqueue paths."},
            "events_to_emit": [_graph_transition_event(state, "routine_enqueue_node", subgraph="routine", operation="routine_enqueue")],
        }
    )
