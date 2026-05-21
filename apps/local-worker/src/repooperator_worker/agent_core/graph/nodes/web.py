"""Web research nodes for RepoOperator LangGraph."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END

from repooperator_worker.agent_core.actions import AgentAction
from repooperator_worker.agent_core.graph.adapters import (
    _execute_ad_hoc_action,
    _graph_transition_event,
    _invoke_subgraph_delta,
    _latest_result,
    _merge_updates,
    _request,
    _with_checkpoint_bump,
)
from repooperator_worker.agent_core.graph.nodes.context import refresh_context_pack_update
from repooperator_worker.agent_core.graph.state import RepoOperatorGraphState
from repooperator_worker.agent_core.graph_state import result_from_snapshot
from repooperator_worker.agent_core.understanding_context import append_visible_rationale, evidence_basis_update

def web_research_graph_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    from repooperator_worker.agent_core.graph.builder import build_web_research_graph

    update = _invoke_subgraph_delta(build_web_research_graph, state)
    update["routing_stage"] = "after_tool_result"
    next_state = _merge_updates(dict(state), update)
    basis_update = evidence_basis_update(next_state, trigger_node="web_research_graph")
    update = _merge_updates(update, basis_update)
    update = _merge_updates(
        update,
        append_visible_rationale(
            next_state,
            node="web_research_graph",
            action=None,
            summary="I updated the evidence basis with the approved web source metadata and summaries.",
            basis_refs=[{"kind": "web", "url": source.get("url")} for source in (basis_update.get("evidence_basis") or {}).get("web_sources", [])[:6] if isinstance(source, dict)],
            safety_note="External web content is untrusted and used only as supporting evidence.",
            uncertainty=[],
        ),
    )
    update.setdefault("events_to_emit", []).append(
        _graph_transition_event(state, "web_research_graph", subgraph="web_research_graph", operation="web_research")
    )
    return _with_checkpoint_bump(update)

def web_decide_needed_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    needed = _web_research_needed(state) and _web_research_available(state)
    return {
        "evidence_store": {**dict(state.get("evidence_store") or {}), "web_research_needed": needed},
        "events_to_emit": [_graph_transition_event(state, "decide_web_needed", subgraph="web_research_graph", operation="decide_web_needed", aggregate={"needed": needed})],
    }

def route_web_research_next(state: RepoOperatorGraphState) -> str:
    evidence = state.get("evidence_store") if isinstance(state.get("evidence_store"), dict) else {}
    if not evidence.get("web_research_needed"):
        return END
    if _has_web_evidence(state):
        return "summarize_web_evidence"
    return "search_web"

def web_search_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    action = AgentAction(
        type="search_web",
        reason_summary="Search web for current external evidence.",
        expected_output="Untrusted web evidence records with source metadata.",
        payload={"query": _web_query_for_request(state), "max_results": 4},
    )
    return _execute_ad_hoc_action(state, action, subgraph="web_research_graph", node_name="search_web")

def web_fetch_sources_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    latest = _latest_result(state)
    sources = [item for item in (latest.payload.get("web_evidence") if latest else []) or [] if isinstance(item, dict)]
    updates: dict[str, Any] = {"events_to_emit": [_graph_transition_event(state, "fetch_sources", subgraph="web_research_graph", operation="fetch_sources")]}
    fetched: list[dict[str, Any]] = []
    for source in sources[:2]:
        url = str(source.get("url") or "")
        if not url:
            continue
        result_update = _execute_ad_hoc_action(
            {**dict(state), **updates},
            AgentAction(type="fetch_url", reason_summary="Fetch selected web source as sanitized evidence.", payload={"url": url}),
            subgraph="web_research_graph",
            node_name="fetch_sources",
        )
        latest_result = result_from_snapshot((result_update.get("action_results") or [None])[-1])
        if latest_result and latest_result.payload.get("web_evidence"):
            fetched.extend(item for item in latest_result.payload.get("web_evidence") or [] if isinstance(item, dict))
        updates = _merge_updates(updates, result_update)
    evidence = dict(state.get("evidence_store") or {})
    evidence.setdefault("web_evidence", [])
    evidence["web_evidence"] = [*evidence["web_evidence"], *sources, *fetched]
    updates["evidence_store"] = evidence
    next_state = _merge_updates(dict(state), updates)
    updates = _merge_updates(updates, evidence_basis_update(next_state, trigger_node="fetch_sources"))
    updates = _merge_updates(
        updates,
        append_visible_rationale(
            next_state,
            node="fetch_sources",
            action=None,
            summary="The fetched web sources are external and untrusted, so I am keeping only source metadata and summaries as evidence.",
            basis_refs=[{"kind": "web", "url": source.get("url")} for source in sources[:4] if source.get("url")],
            safety_note="Raw web page text is not exposed in normal chat or debug context.",
            uncertainty=[],
        ),
    )
    return updates

def web_summarize_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    context_update = refresh_context_pack_update(state, kind="web_research", trigger_node="web_research_summary")
    working_state = {**dict(state), **{key: value for key, value in context_update.items() if key != "events_to_emit"}}
    evidence = dict(state.get("evidence_store") or {})
    records = list(evidence.get("web_evidence") or [])
    action = AgentAction(
        type="summarize_web_evidence",
        reason_summary="Summarize web evidence with source metadata.",
        payload={"web_evidence": records},
    )
    return _merge_updates(
        context_update,
        _execute_ad_hoc_action(working_state, action, subgraph="web_research_graph", node_name="summarize_web_evidence"),
    )

def web_merge_evidence_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    evidence = dict(state.get("evidence_store") or {})
    latest = _latest_result(state)
    if latest and isinstance(latest.payload.get("web_evidence_summary"), dict):
        evidence["web_evidence_summary"] = latest.payload["web_evidence_summary"]
    update = {
        "evidence_store": evidence,
        "events_to_emit": [_graph_transition_event(state, "merge_web_evidence", subgraph="web_research_graph", operation="merge_web_evidence")],
    }
    next_state = _merge_updates(dict(state), update)
    return _merge_updates(update, evidence_basis_update(next_state, trigger_node="merge_web_evidence"))

def _web_research_available(state: RepoOperatorGraphState) -> bool:
    snapshot = state.get("capability_snapshot") if isinstance(state.get("capability_snapshot"), dict) else {}
    capabilities = snapshot.get("capabilities") if isinstance(snapshot.get("capabilities"), list) else []
    if not capabilities:
        return True
    return any(item.get("name") == "web_research" and item.get("available") for item in capabilities if isinstance(item, dict))

def _web_research_needed(state: RepoOperatorGraphState) -> bool:
    request = _request(state)
    text = request.task.lower()
    if any(action.get("type") in {"search_web", "fetch_url", "summarize_web_evidence"} for action in state.get("actions_taken") or [] if isinstance(action, dict)):
        return False
    explicit = any(term in text for term in ("search web", "look up", "latest", "current", "external docs", "online docs", "web research"))
    dependency = any(term in text for term in ("api version", "library version", "dependency", "release notes", "documentation for"))
    local_evidence_insufficient = bool(state.get("evidence_done")) and not state.get("files_read") and any(term in text for term in ("docs", "api", "library"))
    return explicit or dependency or local_evidence_insufficient

def _web_query_for_request(state: RepoOperatorGraphState) -> str:
    request = _request(state)
    return " ".join(request.task.split())[:280]

def _has_web_evidence(state: RepoOperatorGraphState) -> bool:
    evidence = state.get("evidence_store") if isinstance(state.get("evidence_store"), dict) else {}
    if evidence.get("web_evidence") or evidence.get("web_evidence_summary"):
        return True
    for result in state.get("action_results") or []:
        payload = result.get("payload") if isinstance(result, dict) else {}
        if isinstance(payload, dict) and (payload.get("web_evidence") or payload.get("web_evidence_summary")):
            return True
    return False

def _web_source_notes_for_final(state: RepoOperatorGraphState) -> list[str]:
    evidence = state.get("evidence_store") if isinstance(state.get("evidence_store"), dict) else {}
    sources: list[dict[str, Any]] = []
    if isinstance(evidence.get("web_evidence_summary"), dict):
        sources.extend(item for item in evidence["web_evidence_summary"].get("sources") or [] if isinstance(item, dict))
    sources.extend(item for item in evidence.get("web_evidence") or [] if isinstance(item, dict))
    for result in state.get("action_results") or []:
        payload = result.get("payload") if isinstance(result, dict) else {}
        if isinstance(payload, dict):
            sources.extend(item for item in payload.get("web_evidence") or [] if isinstance(item, dict))
            summary = payload.get("web_evidence_summary")
            if isinstance(summary, dict):
                sources.extend(item for item in summary.get("sources") or [] if isinstance(item, dict))
    notes: list[str] = []
    seen: set[str] = set()
    for source in sources:
        url = str(source.get("url") or "")
        if not url or url in seen:
            continue
        seen.add(url)
        title = str(source.get("title") or source.get("source") or url)
        notes.append(f"- {title}: {url}")
        if len(notes) >= 6:
            break
    return notes
