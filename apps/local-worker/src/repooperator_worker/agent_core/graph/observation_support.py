"""Observation and plan updates for LangGraph agent runs."""

from __future__ import annotations

from repooperator_worker.agent_core.actions import AgentAction, ActionResult
from repooperator_worker.agent_core.events import append_activity_event
from repooperator_worker.agent_core.state import AgentCoreState
from repooperator_worker.agent_core.task_policy import action_operation, update_subtasks_after_action
from repooperator_worker.schemas import AgentRunRequest, AgentRunResponse


def create_initial_plan(state: AgentCoreState) -> None:
    if state.subtasks:
        state.plan = [f"{item.title}: {item.status}" for item in state.subtasks]
        return
    state.plan = [
        "Frame the user's goal",
        "Resolve missing evidence",
        "Use safe primitive actions",
        "Answer only from gathered evidence or ask a precise clarification",
    ]


def observe_result(state: AgentCoreState, action: AgentAction, result: ActionResult, request: AgentRunRequest) -> None:
    from repooperator_worker.agent_core.task_policy import record_ineffective_action

    record_ineffective_action(state, action, result)
    if result.files_read:
        for path in result.files_read:
            if path not in state.files_read:
                state.files_read.append(path)
    if result.files_changed:
        for path in result.files_changed:
            if path not in state.files_changed:
                state.files_changed.append(path)
    if result.command_result and result.command_result.get("display_command"):
        command = str(result.command_result.get("display_command"))
        if result.status == "success" and result.command_result.get("exit_code") is not None:
            state.commands_run.append(command)
    if result.status == "waiting_approval":
        if result.command_result:
            state.pending_approval = result.command_result
        else:
            decision = result.payload.get("permission_decision") if isinstance(result.payload, dict) else {}
            metadata = decision.get("metadata") if isinstance(decision, dict) and isinstance(decision.get("metadata"), dict) else {}
            state.pending_approval = {
                "kind": action.type,
                "reason": result.observation,
                "approval_payload": decision.get("approval_payload") or metadata.get("approval_payload") or action.payload,
                "tool_name": action.type,
            }
    observation = _safe_observation(action, result)
    if observation:
        state.observations.append(observation)
        append_activity_event(
            run_id=state.run_id,
            request=request,
            activity_id=f"langgraph-observe:{action.action_id}",
            event_type="activity_completed",
            phase="Observing",
            label="Recorded observation",
            status="completed",
            observation=observation,
            related_files=result.files_read,
            related_command=action.command,
        )
    if action.type == "analyze_repository":
        response = result.payload.get("response")
        if isinstance(response, AgentRunResponse):
            state.final_response = str(response.response)
        elif isinstance(response, dict):
            state.final_response = str(response.get("response") or "")
    update_subtasks_after_action(state, action, result, action_operation(action.type))


def update_plan(state: AgentCoreState, action: AgentAction, result: ActionResult, request: AgentRunRequest) -> None:
    if state.subtasks:
        state.plan = [f"{item.title}: {item.status}" for item in state.subtasks]
    if result.status == "waiting_approval":
        state.plan.append("Wait for user approval before running the command")
    elif result.next_recommended_action:
        state.plan.append(f"Consider next safe action: {result.next_recommended_action}")
    elif result.status == "success":
        state.plan.append(f"Completed: {action.type}")
    emit_plan_update(state, request, "Updated plan")


def emit_plan_update(state: AgentCoreState, request: AgentRunRequest, label: str) -> None:
    append_activity_event(
        run_id=state.run_id,
        request=request,
        activity_id="langgraph-plan",
        event_type="activity_updated",
        phase="Planning",
        label=label,
        status="running",
        observation="; ".join(state.plan[-4:]),
        aggregate={"plan_steps": list(state.plan), "loop_iteration": state.loop_iteration},
    )


def _safe_observation(action: AgentAction, result: ActionResult) -> str:
    if result.status == "waiting_approval":
        return "Command preview requires approval before execution."
    if result.files_read:
        files = ", ".join(result.files_read)
        return f"Read {files}."
    if action.type == "generate_edit" and result.status == "success":
        proposals = result.payload.get("edit_proposals") or []
        files = ", ".join(str(item.get("file")) for item in proposals if isinstance(item, dict))
        return f"Prepared proposed edit for {files}. No files were written."
    if action.type == "search_files" and result.status == "success":
        candidates = result.payload.get("candidates") or []
        return f"Found candidate files: {', '.join(candidates[:8])}."
    if action.type == "analyze_repository" and result.status == "success":
        return "Completed repository-wide review and collected per-file evidence."
    if result.command_result and result.command_result.get("exit_code") is not None:
        return f"Ran `{result.command_result.get('display_command')}` with exit code {result.command_result.get('exit_code')}."
    if result.observation:
        return " ".join(str(result.observation).split())[:500]
    return ""
