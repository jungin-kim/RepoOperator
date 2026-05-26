"""User-visible action trace helpers for LangGraph agent runs."""

from __future__ import annotations

from repooperator_worker.agent_core.actions import AgentAction
from repooperator_worker.agent_core.events import append_work_trace
from repooperator_worker.agent_core.state import AgentCoreState
from repooperator_worker.schemas import AgentRunRequest


def emit_action_decision(state: AgentCoreState, request: AgentRunRequest, action: AgentAction) -> None:
    note = action.payload.get("visible_work_note") if isinstance(action.payload, dict) else None
    note = note if isinstance(note, dict) else {}
    phase = "Safety" if action.type in {"preview_command", "inspect_git_state", "run_approved_command"} else "Decision"
    safety_note = str(note.get("safety_note") or "")
    if not safety_note and action.type in {"preview_command", "inspect_git_state", "run_approved_command"}:
        safety_note = "Commands are checked through policy before any execution."
    if not safety_note and action.type in {"generate_edit", "generate_change_set"}:
        safety_note = "Edit generation is proposal-only; no files are written by this action."
    append_work_trace(
        run_id=state.run_id,
        request=request,
        activity_id=f"action:{action.action_id}",
        phase=phase,
        label=_action_label(action),
        status="completed",
        safe_reasoning_summary=str(note.get("why_this_action") or action.reason_summary),
        current_action=_action_current_text(action),
        next_action=action.expected_output,
        evidence_needed=[str(item) for item in note.get("evidence_needed") or []],
        uncertainty=[str(item) for item in note.get("uncertainty") or []],
        safety_note=safety_note or None,
        related_files=list(action.target_files),
        command=action.command,
        aggregate={"action_type": action.type},
    )


def _action_current_text(action: AgentAction) -> str:
    if action.type == "read_file" and action.target_files:
        return "Read " + ", ".join(f"`{path}`" for path in action.target_files[:6]) + "."
    if action.type in {"search_files", "search_text"}:
        queries = []
        if isinstance(action.payload, dict):
            queries = [str(item) for item in action.payload.get("queries") or []]
            query = action.payload.get("query")
            if query:
                queries.append(str(query))
        return "Search repository evidence" + (f" for {', '.join(queries[:4])}." if queries else ".")
    if action.type in {"generate_edit", "generate_change_set"} and action.target_files:
        return "Prepare a proposal-only patch for " + ", ".join(f"`{path}`" for path in action.target_files[:4]) + "."
    if action.command:
        return "Preview or run command through policy: `" + " ".join(action.command) + "`."
    if action.type == "ask_clarification":
        return "Ask for the missing information needed to proceed safely."
    if action.type == "final_answer":
        return "Prepare the final answer from gathered evidence."
    return action.reason_summary


def _action_label(action: AgentAction) -> str:
    if action.type == "read_file" and action.target_files:
        first = action.target_files[0]
        return f"Reading {first}" if len(action.target_files) == 1 else "Reading repository files"
    if action.type in {"search_files", "search_text"}:
        return "Searching for repository evidence"
    if action.type in {"preview_command", "inspect_git_state", "run_approved_command"}:
        return "Checking command safety"
    if action.type in {"generate_edit", "generate_change_set"}:
        return "Preparing proposal-only edit"
    if action.type in {"inspect_repo_tree", "analyze_repository"}:
        return "Inspecting repository structure"
    if action.type == "ask_clarification":
        return "Preparing clarification"
    if action.type == "final_answer":
        return "Preparing final answer"
    return "Working on the next repository step"
