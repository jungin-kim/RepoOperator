from __future__ import annotations

import json
import time
from dataclasses import replace
from typing import Any

from repooperator_worker.agent_core.artifacts import ArtifactStore, get_default_artifact_store
from repooperator_worker.agent_core.actions import AgentAction, ActionResult
from repooperator_worker.agent_core.hooks import HookManager
from repooperator_worker.agent_core.permissions import PermissionMode, PermissionPolicy, ToolPermissionContext, permission_mode_from_value
from repooperator_worker.agent_core.secret_scanner import redact_json_payload
from repooperator_worker.agent_core.tools.base import (
    ToolExecutionContext,
    ToolResult,
    agent_action_to_tool_payload,
    tool_result_to_action_result,
)
from repooperator_worker.agent_core.events import append_work_trace
from repooperator_worker.agent_core.tools.registry import ToolRegistry, get_default_tool_registry
from repooperator_worker.schemas import AgentRunRequest
from repooperator_worker.services.active_repository import get_active_repository
from repooperator_worker.services.event_service import get_run
from repooperator_worker.services.json_safe import json_safe, safe_repr


class ToolOrchestrator:
    def __init__(
        self,
        *,
        run_id: str,
        request: AgentRunRequest,
        registry: ToolRegistry | None = None,
        hook_manager: HookManager | None = None,
        permission_mode: PermissionMode | str | None = None,
        permission_policy: PermissionPolicy | None = None,
        artifact_store: ArtifactStore | None = None,
    ) -> None:
        self.run_id = run_id
        self.request = request
        self.registry = registry or get_default_tool_registry()
        self.hook_manager = hook_manager or HookManager()
        self.permission_mode = permission_mode_from_value(permission_mode)
        self.permission_policy = permission_policy or PermissionPolicy()
        self.artifact_store = artifact_store
        self.prior_denials: list[dict[str, Any]] = []

    def execute_action(self, action: AgentAction) -> ActionResult:
        started = time.perf_counter()
        tool = None
        try:
            tool = self.registry.get(action.type)
        except Exception:
            tool = None
        self._emit_tool_trace(action, status="running", tool_name=action.type)
        try:
            result = self.execute_tool(action.type, agent_action_to_tool_payload(action))
        except Exception as exc:  # noqa: BLE001
            result = ToolResult(
                tool_name=action.type,
                status="failed",
                observation="Action failed.",
                payload={"errors": [safe_repr(exc, limit=500)]},
            )
        result.duration_ms = int((time.perf_counter() - started) * 1000)
        summary = tool.summarize_result(result) if tool else result.observation
        self._emit_tool_trace(action, status=_trace_status(result.status), tool_name=result.tool_name, observation=summary, result=result)
        action_result = tool_result_to_action_result(action, result)
        if "errors" in result.payload and action_result.status == "failed":
            action_result.errors = [str(item) for item in result.payload.get("errors") or []]
        return action_result

    def execute_tool(self, tool_name: str, payload: dict[str, Any]) -> ToolResult:
        tool = self.registry.get(tool_name)
        validated = json_safe(tool.validate_input(dict(payload), self.request))
        if self._is_cancelled():
            return ToolResult(tool_name=tool_name, status="cancelled", observation="Run was cancelled before tool execution.")

        pre_hook = self.hook_manager.run_pre_tool(tool_name=tool_name, payload=validated, run_id=self.run_id, request=self.request)
        hook_metadata: dict[str, Any] = {}
        if pre_hook.updated_input is not None:
            if not isinstance(pre_hook.updated_input, dict):
                return ToolResult(
                    tool_name=tool_name,
                    status="failed",
                    observation="Pre-tool hook returned invalid updated input; expected an object.",
                    payload={
                        "hook_updated_input": True,
                        "hook_revalidated": False,
                        "hook_source": pre_hook.source,
                        "hook_reason": pre_hook.reason,
                    },
                )
            try:
                validated = json_safe(tool.validate_input(dict(pre_hook.updated_input), self.request))
            except Exception as exc:  # noqa: BLE001
                return ToolResult(
                    tool_name=tool_name,
                    status="failed",
                    observation="Pre-tool hook updated input failed tool validation.",
                    payload={
                        "hook_updated_input": True,
                        "hook_revalidated": False,
                        "hook_source": pre_hook.source,
                        "hook_reason": pre_hook.reason,
                        "errors": [safe_repr(exc, limit=500)],
                    },
                )
            hook_metadata = {
                "hook_updated_input": True,
                "hook_revalidated": True,
                "hook_source": pre_hook.source,
                "hook_reason": pre_hook.reason,
            }
        if not pre_hook.continue_ or pre_hook.decision == "deny":
            return ToolResult(
                tool_name=tool_name,
                status="skipped",
                observation=pre_hook.reason or "Tool blocked by pre-tool hook.",
                payload={"hook_decision": pre_hook.decision, "hook_reason": pre_hook.reason, **hook_metadata},
            )

        permission_context = ToolPermissionContext(
            request=self.request,
            run_id=self.run_id,
            permission_mode=self.permission_mode,
            active_repository=self._active_repository_path(),
            prior_denials=list(self.prior_denials),
            reason=str(validated.get("reason_summary") or ""),
        )
        base_decision = tool.check_permission(validated, permission_context)
        decision, audit = self.permission_policy.evaluate(
            tool_name=tool_name,
            payload=validated,
            context=permission_context,
            base_decision=base_decision,
        )
        if decision.decision == "ask":
            self.hook_manager.run_permission_request(tool_name=tool_name, payload=validated, run_id=self.run_id, request=self.request)
            metadata = json_safe(decision.metadata)
            return ToolResult(
                tool_name=tool_name,
                status="waiting_approval",
                observation=decision.reason or "Tool requires approval.",
                command_result=metadata.get("command_preview") if isinstance(metadata, dict) else None,
                payload={"permission_decision": json_safe(decision), "permission_audit": audit.model_dump(), **hook_metadata},
                next_recommended_action="request_approval",
            )
        if decision.decision == "deny":
            self.prior_denials.append({"tool": tool_name, "reason": decision.reason, "metadata": json_safe(decision.metadata)})
            return ToolResult(
                tool_name=tool_name,
                status="failed",
                observation=decision.reason or "Tool denied by permission policy.",
                payload={"permission_decision": json_safe(decision), "permission_audit": audit.model_dump(), **hook_metadata},
            )

        context = ToolExecutionContext(
            request=self.request,
            run_id=self.run_id,
            permission_mode=self.permission_mode,
            active_repository=permission_context.active_repository,
        )
        try:
            result = tool.call(validated, context)
        except Exception as exc:  # noqa: BLE001
            result = ToolResult(tool_name=tool_name, status="failed", observation="Tool execution failed.", payload={"errors": [safe_repr(exc, limit=500)]})
            self.hook_manager.run_post_tool_failure(tool_name=tool_name, payload=validated, result=result, run_id=self.run_id, request=self.request)
            return result

        if hook_metadata or audit:
            result = replace(
                result,
                payload={
                    **json_safe(result.payload),
                    **hook_metadata,
                    "permission_audit": audit.model_dump(),
                },
            )
        result = self._cap_result(result, max_chars=tool.spec.max_result_chars, tool_name=tool_name)
        post_hook = self.hook_manager.run_post_tool(tool_name=tool_name, payload=validated, result=result, run_id=self.run_id, request=self.request)
        if not post_hook.continue_ or post_hook.decision == "deny":
            return ToolResult(
                tool_name=tool_name,
                status="skipped",
                observation=post_hook.reason or "Tool result blocked by post-tool hook.",
                payload={"hook_decision": post_hook.decision, "hook_reason": post_hook.reason, "original_result": result.model_dump()},
            )
        return result

    def _is_cancelled(self) -> bool:
        try:
            run = get_run(self.run_id) or {}
        except OSError:
            run = {}
        return run.get("status") in {"cancelled", "cancelling"}

    def _active_repository_path(self) -> str | None:
        try:
            active = get_active_repository()
        except Exception:
            active = None
        return str(active.project_path) if active else None

    def _cap_result(self, result: ToolResult, *, max_chars: int, tool_name: str) -> ToolResult:
        updated = result
        payload = json_safe(result.payload)
        metadata: dict[str, Any] = {}
        if len(result.observation or "") > max_chars:
            updated = replace(updated, observation=(result.observation or "")[:max_chars] + "\n[truncated]")
            metadata["observation_truncated"] = True
        try:
            payload_chars = len(json.dumps(payload, ensure_ascii=False))
        except TypeError:
            payload = json_safe(payload)
            payload_chars = len(json.dumps(payload, ensure_ascii=False))
        if payload_chars > max_chars:
            store = self.artifact_store or get_default_artifact_store()
            record = store.write(self.run_id, f"tool_result:{tool_name}", payload)
            redacted_payload, secret_findings = redact_json_payload(payload)
            payload = _truncate_payload(redacted_payload, max_chars=max_chars)
            record_payload = record.record_dump()
            metadata.update(
                {
                    "payload_truncated": True,
                    "artifact_id": record.artifact_id,
                    "artifact_store": "local",
                    "byte_size": record.byte_size,
                    "sha256": record.sha256,
                    "preview": record.preview,
                    "redacted": record.redacted,
                    "blocked": record.blocked,
                    "secret_findings": [item.model_dump() for item in secret_findings],
                    "original_payload_chars": payload_chars,
                    "record": record_payload,
                }
            )
        if metadata:
            payload = {**payload, "_artifact": metadata}
            updated = replace(updated, payload=payload)
        return updated

    def _emit_tool_trace(
        self,
        action: AgentAction,
        *,
        status: str,
        tool_name: str,
        observation: str | None = None,
        result: ToolResult | None = None,
    ) -> None:
        phase = _tool_phase(action.type, result.status if result else None)
        files = list((result.files_read if result else None) or action.target_files or [])
        command = action.command or (result.command_result.get("command") if result and result.command_result else None)
        aggregate: dict[str, Any] = {"tool": tool_name, "action_type": action.type}
        if result and result.command_result:
            if result.command_result.get("exit_code") is not None:
                aggregate["exit_code"] = result.command_result.get("exit_code")
            if result.command_result.get("display_command"):
                aggregate["display_command"] = result.command_result.get("display_command")
        append_work_trace(
            run_id=self.run_id,
            request=self.request,
            activity_id=f"action:{action.action_id}",
            phase=phase,
            label=_tool_label(action.type),
            status=status,
            safe_reasoning_summary=None,
            current_action=_tool_current_action(action, tool_name),
            observation=observation,
            next_action=result.next_recommended_action if result else action.expected_output,
            safety_note=_tool_safety_note(action.type, result),
            related_files=files,
            command=command,
            aggregate=aggregate,
        )


def _truncate_payload(value: Any, *, max_chars: int) -> Any:
    if isinstance(value, str):
        return value if len(value) <= max_chars else value[:max_chars] + "\n[truncated]"
    if isinstance(value, list):
        result = []
        remaining = max_chars
        for item in value:
            if remaining <= 0:
                result.append("[truncated]")
                break
            truncated = _truncate_payload(item, max_chars=_child_limit(remaining, max_chars))
            result.append(truncated)
            remaining -= len(safe_repr(truncated, limit=max_chars))
        return result
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        remaining = max_chars
        for key, item in value.items():
            if remaining <= 0:
                result[str(key)] = "[truncated]"
                break
            truncated = _truncate_payload(item, max_chars=_child_limit(remaining, max_chars))
            result[str(key)] = truncated
            remaining -= len(str(key)) + len(safe_repr(truncated, limit=max_chars))
        return result
    return json_safe(value)


def _child_limit(remaining: int, max_chars: int) -> int:
    return max(32, min(max_chars, max(1, remaining // 2)))


def _trace_status(status: str) -> str:
    if status == "waiting_approval":
        return "waiting"
    if status in {"success", "skipped"}:
        return "completed"
    return status


def _tool_phase(action_type: str, result_status: str | None = None) -> str:
    if result_status == "waiting_approval" or action_type in {"preview_command", "inspect_git_state", "run_approved_command"}:
        return "Safety"
    if action_type in {"search_files", "search_text", "inspect_repo_tree"}:
        return "Searching"
    if action_type == "read_file":
        return "Reading files"
    if action_type == "generate_edit":
        return "Editing"
    if action_type == "final_answer":
        return "Finished"
    return "Observing"


def _tool_label(action_type: str) -> str:
    labels = {
        "inspect_repo_tree": "Inspecting repository structure",
        "search_files": "Searching file names",
        "search_text": "Searching file contents",
        "read_file": "Reading repository files",
        "generate_edit": "Preparing proposal-only edit",
        "preview_command": "Checking command safety",
        "inspect_git_state": "Checking command safety",
        "run_approved_command": "Running approved command",
        "analyze_repository": "Reviewing repository evidence",
        "ask_clarification": "Preparing clarification",
        "final_answer": "Preparing final answer",
    }
    return labels.get(action_type, action_type.replace("_", " ").title())


def _tool_current_action(action: AgentAction, tool_name: str) -> str:
    if action.type == "read_file" and action.target_files:
        return "Reading " + ", ".join(action.target_files[:6]) + "."
    if action.type == "generate_edit" and action.target_files:
        return "Preparing a proposal-only patch for " + ", ".join(action.target_files[:4]) + "."
    if action.type in {"preview_command", "inspect_git_state", "run_approved_command"} and action.command:
        return "Checking command through policy: " + " ".join(action.command)
    return f"Running `{tool_name}`."


def _tool_safety_note(action_type: str, result: ToolResult | None) -> str | None:
    if result and result.status == "waiting_approval":
        return "This command may change repository state, so approval is required before running it."
    if action_type == "generate_edit":
        return "This action only creates a proposed patch and does not write files."
    if action_type in {"preview_command", "inspect_git_state", "run_approved_command"}:
        return "Command safety is enforced by policy before execution."
    return None
