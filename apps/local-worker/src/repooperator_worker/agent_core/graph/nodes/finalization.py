"""Finalization nodes and response adapters for RepoOperator LangGraph."""

from __future__ import annotations

import difflib
import re
from typing import Any

from repooperator_worker.agent_core.graph.adapters import (
    _controller,
    _core_state_from_graph,
    _graph_transition_event,
    _invoke_subgraph_delta,
    _pending_action,
    _request,
    _task_frame,
    _with_checkpoint_bump,
)
from repooperator_worker.agent_core.graph.state import RepoOperatorGraphState
from repooperator_worker.agent_core.graph_state import response_to_snapshot
from repooperator_worker.agent_core.graph.nodes.web import _web_source_notes_for_final
from repooperator_worker.schemas import AgentRunResponse
from repooperator_worker.services.event_service import append_run_event
from repooperator_worker.services.json_safe import json_safe

def ask_clarification_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    action = _pending_action(state)
    request = _request(state)
    core = _core_state_from_graph(state)
    missing = ", ".join(action.payload.get("missing_files") or []) if action else ""
    final_response = (
        action.payload.get("question")
        if action
        else None
    ) or core.classifier_result.clarification_question or (
        f"I could not find {missing}. Please confirm the repo-relative path or choose one of the candidates I found."
        if missing
        else "Could you clarify which files or workflow you want me to inspect?"
    )
    del request
    return _with_checkpoint_bump(
        {
            "stop_reason": "needs_clarification",
            "final_response": final_response,
            "events_to_emit": [
                _graph_transition_event(state, "ask_clarification", operation="clarification", action_type="ask_clarification")
            ],
        }
    )

def final_synthesis_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    from repooperator_worker.agent_core.graph.builder import build_finalization_graph

    update = _invoke_subgraph_delta(build_finalization_graph, state)
    update.setdefault("events_to_emit", []).append(
        _graph_transition_event(state, "final_synthesis", subgraph="finalization_graph", operation="final_synthesis")
    )
    return _with_checkpoint_bump(update)

def final_quality_guard_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    return {
        "events_to_emit": [_graph_transition_event(state, "quality_guard", subgraph="finalization_graph", operation="quality_guard")]
    }

def final_repair_answer_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    return {
        "events_to_emit": [_graph_transition_event(state, "repair_final_answer", subgraph="finalization_graph", operation="repair_final_answer")]
    }

def final_build_response_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    request = _request(state)
    core = _core_state_from_graph(state)
    proposal = state.get("change_set_proposal") if isinstance(state.get("change_set_proposal"), dict) else None
    if proposal and proposal.get("changes") and proposal.get("status") in {"invalid", "repairable", "blocked"}:
        validation = proposal.get("validation") if isinstance(proposal.get("validation"), dict) else {}
        errors = "; ".join(str(item) for item in validation.get("errors") or [proposal.get("proposal_error")] if item)
        core.final_response = (
            "I could not prepare a valid ChangeSetProposal. "
            f"Validation failed: {errors or 'unknown validation error'}. No files were modified."
        )
    if not core.final_response:
        on_delta = _stream_final_delta(core.run_id) if state.get("stream_final_answer") else None
        packet_context = ""
        if isinstance(core.context_packet, dict):
            packet_context = str(core.context_packet.get("skills_context") or "")
        core.final_response = _controller().build_final_answer_text(
            core,
            request,
            skills_context=packet_context or str(state.get("skills_context") or ""),
            on_delta=on_delta,
        )
    draft_response = core.final_response
    core.final_response = _controller().validate_or_repair_final_answer(core.final_response, core, request)
    if _is_explanation_only_edit_request(state) and not state.get("files_changed") and "no files were modified" not in core.final_response.lower():
        core.final_response = core.final_response.rstrip() + "\n\nNo files were modified."
    source_notes = _web_source_notes_for_final(state)
    if source_notes and "Source notes:" not in core.final_response:
        core.final_response = core.final_response.rstrip() + "\n\nSource notes:\n" + "\n".join(source_notes)
    if core.final_response != draft_response:
        from repooperator_worker.agent_core.events import append_work_trace

        append_work_trace(
            run_id=core.run_id,
            request=request,
            activity_id="final-synthesis-repair",
            phase="Finished",
            label="Rebuilt final answer",
            status="completed",
            safe_reasoning_summary="The draft answer did not match the gathered evidence, so I rebuilt it from collected files.",
            observation="Final answer repaired without storing the rejected draft text.",
            safety_note="Rejected draft text is not exposed in events.",
        )
    return {
        "final_response": core.final_response,
        "events_to_emit": [_graph_transition_event(state, "build_response", subgraph="finalization_graph", operation="build_response")],
    }

def final_emit_message_node(state: RepoOperatorGraphState) -> dict[str, Any]:
    request = _request(state)
    core = _core_state_from_graph(state)
    response = _response_with_change_set_payload(
        _controller().build_final_response(core, request).model_copy(update={"agent_flow": "langgraph"}),
        state,
    )
    return {
        "response_snapshot": response_to_snapshot(response),
        "events_to_emit": [_graph_transition_event(state, "emit_final_message", subgraph="finalization_graph", operation="emit_final_message")],
    }

def _response_with_change_set_payload(response: AgentRunResponse, state: RepoOperatorGraphState) -> AgentRunResponse:
    proposal = state.get("change_set_proposal")
    if not isinstance(proposal, dict) or not proposal.get("changes"):
        return response
    validation = proposal.get("validation") if isinstance(proposal.get("validation"), dict) else {}
    validation_status = str(validation.get("status") or proposal.get("status") or "planned")
    errors = [str(item) for item in validation.get("errors") or proposal.get("proposal_errors") or []]
    archive_status = "applied" if proposal.get("applied") or proposal.get("status") == "applied" or state.get("apply_status") == "applied" else ("rejected" if proposal.get("status") == "rejected" else validation_status)
    archive = [_edit_archive_record_from_change(change, archive_status) for change in proposal.get("changes") or [] if isinstance(change, dict)]
    archive = [item for item in archive if item]
    first = (proposal.get("changes") or [{}])[0]
    if proposal.get("applied") or proposal.get("status") == "applied" or state.get("apply_status") == "applied":
        response_type = "edit_applied"
    elif proposal.get("status") == "rejected" or state.get("apply_status") == "rejected":
        response_type = "change_proposal"
    else:
        response_type = "change_proposal" if validation_status == "valid" else "proposal_error"
    updates: dict[str, Any] = {
        "response_type": response_type,
        "change_set_proposal": json_safe(proposal),
        "edit_archive": archive,
        "proposal_validation_status": validation_status,
        "validation_status": validation_status,
        "edit_mode": state.get("edit_mode"),
        "proposal_id": proposal.get("proposal_id"),
        "proposal_status": proposal.get("status"),
        "apply_status": state.get("apply_status") or proposal.get("apply_status"),
        "applied_change_set_id": state.get("applied_change_set_id") or proposal.get("applied_change_set_id"),
        "post_apply_validation_status": state.get("post_apply_validation_status") or proposal.get("post_apply_validation_status"),
    }
    if errors:
        updates["proposal_error_details"] = "; ".join(errors)
    if isinstance(first, dict):
        updates.update(
            {
                "proposal_relative_path": first.get("path"),
                "proposal_original_content": first.get("original_content") or "",
                "proposal_proposed_content": first.get("proposed_content") or "",
                "proposal_context_summary": ((proposal.get("plan") or {}).get("summary") if isinstance(proposal.get("plan"), dict) else None),
                "selected_target_file": first.get("path"),
            }
        )
    return response.model_copy(update=json_safe(updates))

def _edit_archive_record_from_change(change: dict[str, Any], validation_status: str) -> dict[str, Any]:
    path = str(change.get("path") or "")
    if not path:
        return {}
    operation = str(change.get("operation") or "modify")
    original = str(change.get("original_content") or "")
    proposed = "" if operation == "delete" else str(change.get("proposed_content") or "")
    if operation == "create":
        original = ""
    diff = "\n".join(
        difflib.unified_diff(
            original.splitlines(),
            proposed.splitlines(),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
        )
    )
    additions = sum(1 for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++"))
    deletions = sum(1 for line in diff.splitlines() if line.startswith("-") and not line.startswith("---"))
    return {
        "file_path": path,
        "file": path,
        "operation": operation,
        "status": "applied" if validation_status == "applied" else ("rejected" if validation_status == "rejected" else ("proposed" if validation_status == "valid" else "failed")),
        "summary": str(change.get("summary") or ""),
        "additions": additions,
        "deletions": deletions,
        "diff": diff,
        "diff_available": bool(diff.strip()),
        "proposal_id": "proposal:" + path,
        "validation_status": validation_status,
    }

def _is_explanation_only_edit_request(state: RepoOperatorGraphState) -> bool:
    frame = _task_frame(state)
    if frame is None:
        return False
    text = str(getattr(frame, "user_goal", "") or "")
    lowered = text.lower()
    asks_how = bool(re.search(r"\bhow\s+(would|do|can|should)\b", lowered)) or any(term in text for term in ("어떻게", "어떤 식으로"))
    mentions_change = bool(re.search(r"\b(change|edit|add|fix|implement|refactor|update)\b", lowered)) or any(term in text for term in ("추가", "고쳐", "구현", "수정"))
    return asks_how and mentions_change

def _stream_final_delta(run_id: str):
    def emit(delta: str) -> None:
        try:
            append_run_event(run_id, {"type": "assistant_delta", "delta": delta, "streaming_mode": "model_stream"})
        except OSError:
            return

    return emit
