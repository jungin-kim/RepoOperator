"""Public-safe understanding, evidence, and rationale projections.

This module builds explicit UI/debug contracts from the existing LangGraph
state. It is intentionally projection-only: it does not route, approve tools,
or store model-private deliberation.
"""

from __future__ import annotations

import re
import time
import uuid
from typing import Any

from repooperator_worker.agent_core.actions import AgentAction
from repooperator_worker.agent_core.events import (
    EVENT_AUDIENCE_DEBUG,
    EVENT_KIND_DEBUG_RATIONALE,
    append_work_trace,
)
from repooperator_worker.agent_core.graph_state import request_from_snapshot
from repooperator_worker.schemas import AgentRunRequest
from repooperator_worker.services.json_safe import json_safe


MAX_DEBUG_ITEMS = 40


def build_user_understanding_context(request: AgentRunRequest, state: dict[str, Any]) -> dict[str, Any]:
    """Return the structured, inspectable understanding of the user request."""
    request_snapshot = _dict(state.get("request_understanding_snapshot"))
    frame_snapshot = _dict(state.get("task_frame_snapshot"))
    edit_mode = _safe_text(state.get("edit_mode"), limit=80)
    mentioned_files = _dedupe(
        [
            *_string_list(request_snapshot.get("mentioned_files"), limit=240, max_items=40),
            *_string_list(frame_snapshot.get("mentioned_files"), limit=240, max_items=40),
            *_mentioned_files_from_history(request),
        ]
    )
    mentioned_symbols = _dedupe(
        [
            *_string_list(request_snapshot.get("mentioned_symbols"), limit=160, max_items=40),
            *_string_list(frame_snapshot.get("mentioned_symbols"), limit=160, max_items=40),
        ]
    )
    requested_outputs = _requested_outputs(request, state, request_snapshot, frame_snapshot, edit_mode)
    likely_needed_tools = _dedupe(
        [
            *_string_list(request_snapshot.get("likely_needed_tools"), limit=80, max_items=30),
            *_string_list(frame_snapshot.get("likely_needed_tools"), limit=80, max_items=30),
        ]
    )
    likely_capabilities = _dedupe(
        [
            *_string_list(frame_snapshot.get("likely_capabilities"), limit=120, max_items=30),
            *_capabilities_from_tools(likely_needed_tools),
        ]
    )
    constraints = _dedupe(
        [
            *_string_list(request_snapshot.get("constraints"), limit=260, max_items=30),
            *_string_list(frame_snapshot.get("constraints"), limit=260, max_items=30),
            *_string_list(request_snapshot.get("safety_notes"), limit=260, max_items=30),
            *_string_list(frame_snapshot.get("safety_notes"), limit=260, max_items=30),
        ]
    )
    ambiguities = _dedupe(
        [
            *_string_list(request_snapshot.get("uncertainties"), limit=260, max_items=30),
            *_string_list(frame_snapshot.get("uncertainty"), limit=260, max_items=30),
        ]
    )
    normalized_goal = (
        _safe_text(request_snapshot.get("user_goal"), limit=400)
        or _safe_text(frame_snapshot.get("user_goal"), limit=400)
        or _safe_text(request.task, limit=400)
        or ""
    )
    follow_up = _follow_up_context(request, state)
    payload = {
        "raw_user_request": _safe_text(request.task, limit=2_000) or "",
        "normalized_goal": normalized_goal,
        "requested_outputs": requested_outputs,
        "edit_mode": edit_mode,
        "mentioned_files": mentioned_files,
        "mentioned_symbols": mentioned_symbols,
        "constraints": constraints,
        "ambiguities": ambiguities,
        "likely_needed_tools": likely_needed_tools,
        "likely_capabilities": likely_capabilities,
        "language": _language_for(request.task),
        "follow_up_context": follow_up,
        "source_snapshots": {
            "request_understanding_snapshot": redact_context_for_user(request_snapshot),
            "task_frame_snapshot": redact_context_for_user(frame_snapshot),
        },
    }
    return json_safe(payload)


def update_user_understanding_context(state: dict[str, Any], request: AgentRunRequest, trigger_node: str) -> dict[str, Any]:
    understanding = build_user_understanding_context(request, {**dict(state), "last_understanding_update_node": trigger_node})
    return {
        "user_understanding_context": understanding,
        "understanding_history": [{**understanding, "last_updated_by": trigger_node}],
    }


def build_evidence_basis(state: dict[str, Any], trigger_node: str) -> dict[str, Any]:
    """Return source-like public evidence metadata without raw content."""
    basis = {
        "files": _file_evidence(state),
        "web_sources": _web_sources(state),
        "commands": _commands(state),
        "validation": _validation(state),
        "active_proposal": _active_proposal(state),
        "worker_reports": _worker_reports(state),
        "target_selection": _target_selection(state),
        "memory_carryover": _memory_carryover(state),
        "missing_evidence": _missing_evidence(state),
        "uncertainty": _uncertainty(state),
        "last_updated_by": _safe_text(trigger_node, limit=120) or "unknown",
    }
    return json_safe(redact_context_for_user(basis))


def append_visible_rationale(
    state: dict[str, Any],
    *,
    node: str,
    action: AgentAction | dict[str, Any] | None = None,
    summary: str,
    basis_refs: list[dict[str, Any]] | None = None,
    safety_note: str | None = None,
    uncertainty: list[str] | None = None,
) -> dict[str, Any]:
    """Append a concise public-safe rationale entry and mirror it to work trace."""
    action_payload = _action_payload(action)
    safe_summary = _safe_text(summary, limit=360) or "Recorded a safe action rationale."
    safe_refs = [_safe_basis_ref(item) for item in basis_refs or []]
    safe_refs = [item for item in safe_refs if item]
    action_id = _safe_text(action_payload.get("action_id"), limit=160)
    action_type = _safe_text(action_payload.get("type"), limit=80)
    entry_id = f"rat_{uuid.uuid4().hex[:12]}"
    entry = {
        "id": entry_id,
        "timestamp": _utc_now(),
        "node": _safe_text(node, limit=120) or "unknown",
        "action_id": action_id,
        "action_type": action_type,
        "summary": safe_summary,
        "basis_refs": safe_refs,
        "safety_note": _safe_text(safety_note, limit=260),
        "uncertainty": _string_list(uncertainty, limit=160, max_items=8),
        "visible": True,
    }
    _emit_rationale_trace(state, entry, action_payload)
    return {"visible_rationale_log": [json_safe(entry)]}


def redact_context_for_user(payload: Any) -> Any:
    """Recursively remove raw content and non-public reasoning fields."""
    return _redact(payload)


def debug_context_payload(state: dict[str, Any]) -> dict[str, Any]:
    """Return capped, debug-safe context fields for the /debug/context payload."""
    return json_safe(
        {
            "user_understanding_context": _cap_payload(redact_context_for_user(state.get("user_understanding_context") or {})),
            "evidence_basis": _cap_payload(redact_context_for_user(state.get("evidence_basis") or {})),
            "visible_rationale_log": _cap_list(redact_context_for_user(state.get("visible_rationale_log") or []), max_items=30),
            "context_pack_report": _cap_payload(redact_context_for_user(state.get("context_pack_report") or {})),
            "context_pack_summary": _cap_payload(redact_context_for_user(state.get("context_pack_summary") or {})),
            "short_term_memory": _cap_payload(redact_context_for_user(state.get("short_term_memory") or {})),
            "target_selection": _cap_payload(redact_context_for_user(state.get("target_selection_diagnostics") or {})),
            "edit_target_candidates": _cap_list(redact_context_for_user(state.get("edit_target_candidates") or []), max_items=20),
        }
    )


def evidence_basis_update(state: dict[str, Any], *, trigger_node: str) -> dict[str, Any]:
    basis = build_evidence_basis(state, trigger_node)
    return {"evidence_basis": basis, "evidence_basis_history": [basis]}


def rationale_basis_refs_for_action(action: AgentAction | dict[str, Any] | None) -> list[dict[str, Any]]:
    payload = _action_payload(action)
    refs: list[dict[str, Any]] = []
    for path in payload.get("target_files") or []:
        if str(path).strip():
            refs.append({"kind": "file", "path": str(path).strip()})
    action_payload = _dict(payload.get("payload"))
    query = action_payload.get("query")
    if query:
        refs.append({"kind": "search", "query": _safe_text(query, limit=300)})
    for query in action_payload.get("queries") or []:
        if str(query).strip():
            refs.append({"kind": "search", "query": str(query).strip()[:300]})
    url = action_payload.get("url")
    if url:
        refs.append({"kind": "web", "url": _safe_text(url, limit=1_000)})
    return [item for item in refs if item]


def rationale_summary_for_action(action: AgentAction | dict[str, Any] | None, *, fallback: str = "") -> str:
    payload = _action_payload(action)
    note = _dict(_dict(payload.get("payload")).get("visible_work_note"))
    return (
        _safe_text(note.get("why_this_action"), limit=360)
        or _safe_text(payload.get("reason_summary"), limit=360)
        or fallback
        or "I chose the next visible action based on the current request and evidence state."
    )


def rationale_safety_note_for_action(action: AgentAction | dict[str, Any] | None) -> str | None:
    payload = _action_payload(action)
    action_type = str(payload.get("type") or "")
    note = _dict(_dict(payload.get("payload")).get("visible_work_note"))
    explicit = _safe_text(note.get("safety_note"), limit=260)
    if explicit:
        return explicit
    if action_type in {"generate_edit", "generate_change_set"}:
        return "Proposal generation does not write files."
    if action_type in {"apply_change_set", "git_commit", "git_push", "github_create_pr", "gitlab_create_mr"}:
        return "This action stays behind an explicit approval boundary."
    if action_type in {"search_web", "fetch_url", "summarize_web_evidence"}:
        return "External web content is untrusted and used only as source evidence."
    if action_type in {"preview_command", "inspect_git_state", "run_approved_command"}:
        return "Commands are checked through policy before execution."
    return None


def rationale_uncertainty_for_action(action: AgentAction | dict[str, Any] | None) -> list[str]:
    payload = _action_payload(action)
    note = _dict(_dict(payload.get("payload")).get("visible_work_note"))
    return _string_list(note.get("uncertainty"), limit=160, max_items=8)


def _requested_outputs(
    request: AgentRunRequest,
    state: dict[str, Any],
    request_snapshot: dict[str, Any],
    frame_snapshot: dict[str, Any],
    edit_mode: str | None,
) -> list[str]:
    outputs = _dedupe(
        [
            *_string_list(request_snapshot.get("requested_outputs"), limit=120, max_items=30),
            *_string_list(frame_snapshot.get("requested_outputs"), limit=120, max_items=30),
        ]
    )
    if edit_mode:
        outputs.append(edit_mode)
    if edit_mode == "proposal_only":
        outputs.extend(["change_set_proposal", "proposal_only"])
    elif edit_mode == "apply_approved":
        outputs.extend(["apply_approved", "applied_change_set"])
    elif edit_mode == "explanation_only":
        outputs.append("explanation_only")
    text = request.task.lower()
    if any(term in text for term in ("look up", "latest", "current", "web research", "search web")):
        outputs.append("web_research")
    if any(term in text for term in ("commit", "push", "pull request", "merge request")):
        outputs.append("git_workflow")
    if any(term in text for term in ("routine", "nightly", "scheduled")):
        outputs.append("routine")
    if any(term in text for term in ("whole codebase", "entire codebase", "all files", "repository-wide")):
        outputs.append("broad_analysis")
    if state.get("change_set_proposal") and "change_set_proposal" not in outputs:
        outputs.append("change_set_proposal")
    return _dedupe([item for item in outputs if item])


def _follow_up_context(request: AgentRunRequest, state: dict[str, Any]) -> dict[str, Any]:
    history_files = _mentioned_files_from_history(request)
    context_packet = _dict(state.get("context_packet"))
    base_context = _dict(context_packet.get("base_context"))
    prior_files = _dedupe(
        [
            *history_files,
            *_string_list(context_packet.get("prior_files_read"), limit=240, max_items=40),
            *_string_list(base_context.get("prior_files_read"), limit=240, max_items=40),
        ]
    )
    proposal_id = _safe_text(state.get("proposal_id"), limit=200)
    if not proposal_id:
        proposal = _dict(state.get("change_set_proposal"))
        proposal_id = _safe_text(proposal.get("proposal_id"), limit=200)
    return {
        "is_follow_up": bool(request.conversation_history or prior_files or proposal_id),
        "prior_files": prior_files[:40],
        "prior_proposal_id": proposal_id,
    }


def _file_evidence(state: dict[str, Any]) -> list[dict[str, Any]]:
    report = _dict(state.get("context_pack_report"))
    omitted_by_path: dict[str, dict[str, Any]] = {}
    for item in report.get("omitted_files") or []:
        if isinstance(item, dict):
            path = _safe_text(item.get("path"), limit=500)
            if path:
                omitted_by_path[path] = item
    summaries = _file_summary_map(state)
    entries: dict[str, dict[str, Any]] = {}

    def add(path: Any, *, reason: str, source: str, retained: bool = True, omitted_reason: str | None = None) -> None:
        cleaned = _safe_text(path, limit=500)
        if not cleaned:
            return
        omitted = omitted_by_path.get(cleaned)
        retained_value = bool(retained and not omitted)
        existing = entries.get(cleaned, {})
        entries[cleaned] = {
            "path": cleaned,
            "reason": existing.get("reason") or _safe_text(reason, limit=260) or "Evidence file.",
            "source": existing.get("source") or source,
            "retained": retained_value,
            "omitted_reason": _safe_text(omitted_reason or (omitted or {}).get("reason"), limit=260) if not retained_value else None,
            "summary": _safe_text(summaries.get(cleaned), limit=600),
        }

    for path in state.get("files_read") or []:
        add(path, reason="Read during this run.", source="read_file")
    store = _dict(state.get("evidence_store"))
    for path in store.get("files_read") or []:
        add(path, reason="Recorded by evidence store.", source="read_file")
    for path in _dict(store.get("contents")).keys():
        add(path, reason="File content was read and summarized for evidence.", source="read_file")
    for result in state.get("action_results") or []:
        result_dict = _dict(result)
        for path in result_dict.get("files_read") or []:
            add(path, reason="Tool result read this file.", source="read_file")
        for path in _dict(_dict(result_dict.get("payload")).get("contents")).keys():
            add(path, reason="Tool result included file evidence; raw contents are omitted here.", source="read_file")
    for path in report.get("retained_files") or []:
        add(path, reason="Retained by the current context pack.", source="context_memory", retained=True)
    for path, item in omitted_by_path.items():
        add(path, reason="Available evidence was omitted or compacted by the current context budget.", source="context_memory", retained=False, omitted_reason=str(item.get("reason") or "omitted"))
    for report_item in state.get("worker_reports") or []:
        report_dict = _dict(report_item)
        for path in report_dict.get("files_analyzed") or report_dict.get("files") or []:
            add(path, reason="Referenced by worker report.", source="worker_report", retained=path not in omitted_by_path)
    return _sort_files(list(entries.values()))[:MAX_DEBUG_ITEMS]


def _web_sources(state: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    store = _dict(state.get("evidence_store"))
    if isinstance(store.get("web_evidence"), list):
        records.extend(item for item in store.get("web_evidence") if isinstance(item, dict))
    summary = _dict(store.get("web_evidence_summary"))
    records.extend(item for item in summary.get("sources") or [] if isinstance(item, dict))
    for result in state.get("action_results") or []:
        payload = _dict(_dict(result).get("payload"))
        records.extend(item for item in payload.get("web_evidence") or [] if isinstance(item, dict))
        records.extend(item for item in _dict(payload.get("web_evidence_summary")).get("sources") or [] if isinstance(item, dict))
    report = _dict(state.get("context_pack_report"))
    records.extend(item for item in report.get("retained_web_sources") or [] if isinstance(item, dict))
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        url = _safe_text(record.get("url"), limit=1_000)
        key = url or _safe_text(record.get("title") or record.get("source"), limit=300) or str(len(out))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "title": _safe_text(record.get("title"), limit=240),
                "url": url or "",
                "source": _safe_text(record.get("source"), limit=240),
                "fetched_at": _safe_text(record.get("fetched_at"), limit=120),
                "reason": "Used as untrusted external web evidence.",
                "untrusted": True,
            }
        )
        if len(out) >= MAX_DEBUG_ITEMS:
            break
    return out


def _commands(state: dict[str, Any]) -> list[dict[str, Any]]:
    actions_by_id = {_safe_text(_dict(action).get("action_id"), limit=160): _dict(action) for action in state.get("actions_taken") or []}
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for result in state.get("action_results") or []:
        result_dict = _dict(result)
        action = actions_by_id.get(_safe_text(result_dict.get("action_id"), limit=160) or "") or {}
        command = action.get("command") or _dict(result_dict.get("command_result")).get("command") or _dict(result_dict.get("command_result")).get("display_command")
        if not command:
            continue
        key = _command_key(command)
        if key in seen:
            continue
        seen.add(key)
        command_result = _dict(result_dict.get("command_result"))
        out.append(
            {
                "command": command,
                "status": _safe_text(result_dict.get("status"), limit=80) or "unknown",
                "reason": _safe_text(action.get("reason_summary"), limit=260) or "Command result contributed to the run.",
                "read_only": command_result.get("read_only") if isinstance(command_result.get("read_only"), bool) else None,
            }
        )
    for command in state.get("commands_run") or []:
        key = _command_key(command)
        if key not in seen:
            seen.add(key)
            out.append({"command": command, "status": "completed", "reason": "Recorded command history.", "read_only": None})
    return json_safe(out[:MAX_DEBUG_ITEMS])


def _validation(state: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, item in enumerate(state.get("validation_results") or []):
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "id": f"validation:{index}",
                "kind": _safe_text(item.get("kind"), limit=120) or "tool_result",
                "status": _safe_text(item.get("status"), limit=80) or "unknown",
                "errors": _string_list(item.get("errors"), limit=260, max_items=12),
                "reason": "Validation result recorded for the current run.",
            }
        )
    proposal = _dict(state.get("change_set_proposal"))
    proposal_validation = _dict(proposal.get("validation"))
    if proposal_validation:
        out.append(
            {
                "id": "validation:active_proposal",
                "kind": "change_set",
                "status": _safe_text(proposal_validation.get("status") or proposal.get("status"), limit=80) or "unknown",
                "errors": _string_list(proposal_validation.get("errors"), limit=260, max_items=12),
                "reason": "Active ChangeSetProposal validation.",
            }
        )
    return out[:MAX_DEBUG_ITEMS]


def _active_proposal(state: dict[str, Any]) -> dict[str, Any] | None:
    proposal = _dict(state.get("change_set_proposal"))
    if not proposal:
        return None
    changes = [item for item in proposal.get("changes") or [] if isinstance(item, dict)]
    counts: dict[str, int] = {}
    files: list[str] = []
    for change in changes:
        operation = _safe_text(change.get("operation"), limit=80) or "modify"
        counts[operation] = counts.get(operation, 0) + 1
        path = _safe_text(change.get("path"), limit=500)
        if path:
            files.append(path)
    return {
        "proposal_id": _safe_text(proposal.get("proposal_id"), limit=240),
        "status": _safe_text(proposal.get("status") or state.get("proposal_status"), limit=80),
        "files": _dedupe(files),
        "operation_counts": counts,
    }


def _worker_reports(state: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, item in enumerate(state.get("worker_reports") or []):
        if not isinstance(item, dict):
            continue
        files = item.get("files_analyzed") or item.get("files") or item.get("input_files") or []
        out.append(
            {
                "worker_task_id": _safe_text(item.get("work_unit_id") or item.get("task_id") or item.get("id"), limit=200) or f"worker:{index}",
                "role": _safe_text(item.get("role") or item.get("worker"), limit=120) or "worker",
                "summary": _safe_text(item.get("summary") or item.get("findings"), limit=600) or "",
                "files_analyzed": _string_list(files, limit=500, max_items=30),
            }
        )
        if len(out) >= MAX_DEBUG_ITEMS:
            break
    return out


def _target_selection(state: dict[str, Any]) -> dict[str, Any]:
    diagnostics = _dict(state.get("target_selection_diagnostics"))
    candidates = diagnostics.get("candidates") if isinstance(diagnostics.get("candidates"), list) else state.get("edit_target_candidates") or []
    return json_safe(
        {
            "selected_target_files": _string_list(diagnostics.get("selected_target_files"), limit=500, max_items=12),
            "prior_evidence_reused": bool(diagnostics.get("prior_evidence_reused")),
            "fallback_attempts": diagnostics.get("fallback_attempts"),
            "failed_search_patterns": _string_list(diagnostics.get("failed_search_patterns"), limit=260, max_items=20),
            "discovery_queries": _string_list(diagnostics.get("discovery_queries"), limit=260, max_items=24),
            "discovery_file_globs": _string_list(diagnostics.get("discovery_file_globs"), limit=260, max_items=24),
            "blocked_reason": _safe_text(diagnostics.get("blocked_reason"), limit=500),
            "project_profile": _dict(diagnostics.get("project_profile")),
            "candidates": [
                {
                    "path": _safe_text(item.get("path"), limit=500),
                    "score": item.get("score"),
                    "confidence": item.get("confidence"),
                    "role": _safe_text(item.get("role"), limit=120),
                    "language": _safe_text(item.get("language"), limit=120),
                    "already_read": bool(item.get("already_read")),
                    "sources": _string_list(item.get("sources"), limit=160, max_items=12),
                    "reasons": _string_list(item.get("reasons"), limit=260, max_items=12),
                    "symbols": _string_list(item.get("symbols"), limit=160, max_items=20),
                    "matched_terms": _string_list(item.get("matched_terms"), limit=160, max_items=20),
                    "prior_reused": bool(item.get("prior_reused")),
                }
                for item in candidates
                if isinstance(item, dict)
            ][:MAX_DEBUG_ITEMS],
        }
    )


def _memory_carryover(state: dict[str, Any]) -> dict[str, Any]:
    packet = _dict(state.get("context_packet"))
    memory = _dict(state.get("short_term_memory"))
    base_context = _dict(packet.get("base_context"))
    thread_context = _dict(base_context.get("thread_context"))
    return json_safe(
        {
            "short_term_memory_entries": {
                "files_read": len(memory.get("files_read_summaries") or []),
                "target_candidates": len(memory.get("target_candidate_summaries") or []),
                "carryover": len(memory.get("carryover_summaries") or []),
                "symbols": len(memory.get("symbol_summaries") or []),
            },
            "target_candidate_summaries": list(memory.get("target_candidate_summaries") or [])[:20],
            "carryover_summaries": list(memory.get("carryover_summaries") or [])[:20],
            "thread_recent_files": list(thread_context.get("recent_files") or [])[:20],
            "thread_target_candidates": list(thread_context.get("last_target_candidates") or [])[:20],
            "last_implementation_plan": thread_context.get("last_implementation_plan"),
            "context_source": thread_context.get("context_source"),
        }
    )


def _missing_evidence(state: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    understanding = _dict(state.get("user_understanding_context"))
    read_files = {str(path) for path in state.get("files_read") or []}
    for path in understanding.get("mentioned_files") or []:
        if str(path) and str(path) not in read_files:
            missing.append(f"Explicitly mentioned file not read yet: {path}")
    if not read_files and not _web_sources(state) and not state.get("worker_reports") and not state.get("commands_run"):
        missing.append("No concrete repository, command, worker, or web evidence is recorded yet.")
    return _dedupe(missing)[:20]


def _uncertainty(state: dict[str, Any]) -> list[str]:
    understanding = _dict(state.get("user_understanding_context"))
    items = _string_list(understanding.get("ambiguities"), limit=260, max_items=20)
    report = _dict(state.get("context_pack_report"))
    items.extend(_string_list(report.get("warnings"), limit=260, max_items=20))
    for query in state.get("zero_result_queries") or []:
        if str(query).strip():
            items.append(f"No results for search query: {query}")
    return _dedupe(items)[:30]


def _file_summary_map(state: dict[str, Any]) -> dict[str, str]:
    summaries: dict[str, str] = {}
    memory = _dict(state.get("short_term_memory"))
    for item in [
        *(memory.get("files_read_summaries") or []),
        *(memory.get("file_digest_summaries") or []),
    ]:
        if isinstance(item, dict):
            path = _safe_text(item.get("path"), limit=500)
            summary = _safe_text(item.get("summary"), limit=600)
            if path and summary:
                summaries[path] = summary
    packet = _dict(state.get("context_packet"))
    evidence = _dict(packet.get("file_evidence"))
    for path, summary in _dict(evidence.get("summaries")).items():
        safe_summary = _safe_text(summary, limit=600)
        if path and safe_summary:
            summaries[str(path)] = safe_summary
    return summaries


def _capabilities_from_tools(tools: list[str]) -> list[str]:
    capabilities: list[str] = []
    if any(tool in tools for tool in ("search_web", "fetch_url", "summarize_web_evidence")):
        capabilities.append("web_research")
    if any(tool in tools for tool in ("read_file", "search_files", "search_text", "inspect_repo_tree")):
        capabilities.append("repository_evidence")
    if any(tool in tools for tool in ("generate_change_set", "generate_edit")):
        capabilities.append("edit_proposal")
    if "run_command" in tools:
        capabilities.append("command_validation")
    return capabilities


def _mentioned_files_from_history(request: AgentRunRequest) -> list[str]:
    files: list[str] = []
    for message in request.conversation_history[-8:]:
        metadata = message.metadata or {}
        for key in ("files_read", "resolved_files", "thread_context_files"):
            for path in metadata.get(key) or []:
                if isinstance(path, str) and path.strip():
                    files.append(path.strip())
    return _dedupe(files)[:40]


def _language_for(text: str) -> str | None:
    if not text:
        return None
    if re.search(r"[\uac00-\ud7af]", text):
        return "ko"
    if re.search(r"[A-Za-z]", text):
        return "en"
    return None


def _emit_rationale_trace(state: dict[str, Any], entry: dict[str, Any], action_payload: dict[str, Any]) -> None:
    try:
        request = request_from_snapshot(_dict(state.get("request_snapshot")))
    except Exception:
        return
    run_id = str(state.get("run_id") or "run_controller")
    action_type = entry.get("action_type") or action_payload.get("type")
    files = [ref.get("path") for ref in entry.get("basis_refs") or [] if isinstance(ref, dict) and ref.get("kind") == "file" and ref.get("path")]
    query = next((ref.get("query") for ref in entry.get("basis_refs") or [] if isinstance(ref, dict) and ref.get("kind") == "search" and ref.get("query")), None)
    command = action_payload.get("command")
    append_work_trace(
        run_id=run_id,
        request=request,
        activity_id=entry.get("action_id") and f"action:{entry['action_id']}" or f"rationale:{entry['id']}",
        phase=_phase_for_node(str(entry.get("node") or ""), str(action_type or "")),
        label=_label_for_rationale(str(entry.get("node") or ""), str(action_type or "")),
        kind=EVENT_KIND_DEBUG_RATIONALE,
        audience=EVENT_AUDIENCE_DEBUG,
        visibility="debug",
        display="secondary",
        status="completed",
        safe_reasoning_summary=str(entry.get("summary") or ""),
        uncertainty=[str(item) for item in entry.get("uncertainty") or []],
        safety_note=entry.get("safety_note"),
        operation=str(action_type or entry.get("node") or ""),
        action_type=str(action_type) if action_type else None,
        related_files=[str(path) for path in files if path],
        related_search_query=str(query) if query else None,
        command=command,
        aggregate={"rationale_id": entry.get("id"), "basis_refs": entry.get("basis_refs") or []},
    )


def _phase_for_node(node: str, action_type: str) -> str:
    if action_type in {"search_files", "search_text", "inspect_repo_tree"}:
        return "Searching"
    if action_type in {"read_file", "inspect_symbol", "analyze_file", "analyze_repository"}:
        return "Reading"
    if action_type in {"generate_change_set", "generate_edit", "apply_change_set"}:
        return "Editing"
    if action_type in {"search_web", "fetch_url", "summarize_web_evidence"}:
        return "Researching"
    if "approval" in node or action_type in {"git_commit", "git_push", "github_create_pr", "gitlab_create_mr"}:
        return "Safety"
    if "final" in node:
        return "Finished"
    return "Decision"


def _label_for_rationale(node: str, action_type: str) -> str:
    if action_type:
        return action_type.replace("_", " ").title()
    return node.replace("_", " ").title() or "Visible rationale"


def _safe_basis_ref(item: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    kind = _safe_text(item.get("kind"), limit=80)
    if not kind:
        return {}
    out = {"kind": kind}
    for key in ("path", "url", "id", "query", "status"):
        value = _safe_text(item.get(key), limit=1_000 if key == "url" else 500)
        if value:
            out[key] = value
    return out


def _action_payload(action: AgentAction | dict[str, Any] | None) -> dict[str, Any]:
    if action is None:
        return {}
    if isinstance(action, AgentAction):
        return action.model_dump()
    if isinstance(action, dict):
        return json_safe(action)
    return {}


def _redact(value: Any, *, depth: int = 0) -> Any:
    if depth > 12:
        return "[redacted-depth]"
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _forbidden_key(key_text):
                continue
            if _raw_content_key(key_text):
                out[key_text] = _raw_content_placeholder(item)
                continue
            out[key_text] = _redact(item, depth=depth + 1)
        return out
    if isinstance(value, list):
        return [_redact(item, depth=depth + 1) for item in value[:MAX_DEBUG_ITEMS]]
    if isinstance(value, str):
        if _contains_nonpublic_marker(value):
            return "[redacted]"
        return value if len(value) <= 2_000 else value[:1_997].rstrip() + "..."
    return json_safe(value)


def _forbidden_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    if normalized.startswith("safe_reasoning"):
        return False
    compact = normalized.replace("_", "")
    return (
        normalized == "reasoning"
        or "private" in normalized and "reasoning" in normalized
        or "hidden" in normalized and "reasoning" in normalized
        or "raw" in normalized and "reasoning" in normalized
        or "chain" in compact and "thought" in compact
    )


def _raw_content_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    return normalized in {
        "content",
        "contents",
        "text",
        "raw_text",
        "body",
        "html",
        "original_content",
        "proposed_content",
        "diff",
        "context_packet",
    }


def _raw_content_placeholder(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {"redacted": True, "keys": list(value.keys())[:20], "reason": "raw content omitted"}
    if isinstance(value, list):
        return {"redacted": True, "items": len(value), "reason": "raw content omitted"}
    text = str(value or "")
    return {"redacted": True, "chars": len(text), "reason": "raw content omitted"}


def _contains_nonpublic_marker(text: str) -> bool:
    lowered = text.lower()
    markers = (
        "<think>",
        "chain-" + "of-thought",
        "chain " + "of thought",
        "private " + "reasoning",
        "hidden " + "reasoning",
        "raw " + "reasoning",
    )
    return any(marker in lowered for marker in markers)


def _cap_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: _cap_payload(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return _cap_list(payload)
    return payload


def _cap_list(values: Any, *, max_items: int = MAX_DEBUG_ITEMS) -> list[Any]:
    if not isinstance(values, list):
        return []
    return values[:max_items]


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string_list(value: Any, *, limit: int, max_items: int) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = _safe_text(item, limit=limit)
        if text:
            out.append(text)
        if len(out) >= max_items:
            break
    return out


def _safe_text(value: Any, *, limit: int) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    if not text or _contains_nonpublic_marker(text):
        return None
    return text if len(text) <= limit else text[: max(0, limit - 3)].rstrip() + "..."


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def _sort_files(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=lambda item: (0 if item.get("retained") else 1, str(item.get("path") or "")))


def _command_key(command: Any) -> str:
    if isinstance(command, list):
        return " ".join(str(item) for item in command)
    return str(command)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
