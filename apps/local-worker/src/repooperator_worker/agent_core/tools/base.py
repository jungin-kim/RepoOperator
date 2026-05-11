from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from repooperator_worker.agent_core.actions import AgentAction, ActionResult
from repooperator_worker.agent_core.permissions import PermissionDecision, PermissionMode, ToolPermissionContext
from repooperator_worker.schemas import AgentRunRequest
from repooperator_worker.services.json_safe import json_safe


ToolStatus = Literal["success", "skipped", "failed", "waiting_approval", "cancelled", "timed_out"]
ToolOperation = Literal[
    "list_files",
    "search",
    "read_file",
    "analyze_repository",
    "command",
    "edit",
    "final_answer",
    "clarification",
    "custom",
]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    operation: ToolOperation
    input_schema: dict[str, Any]
    read_only: bool
    concurrency_safe: bool
    requires_approval_by_default: bool = False
    max_result_chars: int = 100_000

    def model_dump(self) -> dict[str, Any]:
        return json_safe(
            {
                "name": self.name,
                "description": self.description,
                "operation": self.operation,
                "input_schema": self.input_schema,
                "read_only": self.read_only,
                "concurrency_safe": self.concurrency_safe,
                "requires_approval_by_default": self.requires_approval_by_default,
                "max_result_chars": self.max_result_chars,
            }
        )


@dataclass
class ToolResult:
    tool_name: str
    status: ToolStatus
    observation: str
    payload: dict[str, Any] = field(default_factory=dict)
    files_read: list[str] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)
    command_result: dict[str, Any] | None = None
    next_recommended_action: str | None = None
    duration_ms: int | None = None

    def model_dump(self) -> dict[str, Any]:
        return json_safe(
            {
                "tool_name": self.tool_name,
                "status": self.status,
                "observation": self.observation,
                "payload": self.payload,
                "files_read": self.files_read,
                "files_changed": self.files_changed,
                "command_result": self.command_result,
                "next_recommended_action": self.next_recommended_action,
                "duration_ms": self.duration_ms,
            }
        )


@dataclass(frozen=True)
class ToolExecutionContext:
    request: AgentRunRequest
    run_id: str
    permission_mode: PermissionMode = PermissionMode.DEFAULT
    active_repository: str | None = None


class Tool(Protocol):
    spec: ToolSpec

    def validate_input(self, payload: dict[str, Any], request: AgentRunRequest) -> dict[str, Any]:
        ...

    def check_permission(self, payload: dict[str, Any], context: ToolPermissionContext) -> PermissionDecision:
        ...

    def call(self, payload: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        ...

    def summarize_result(self, result: ToolResult) -> str:
        ...


class BaseTool:
    spec: ToolSpec

    def validate_input(self, payload: dict[str, Any], request: AgentRunRequest) -> dict[str, Any]:
        return dict(payload)

    def check_permission(self, payload: dict[str, Any], context: ToolPermissionContext) -> PermissionDecision:
        if self.spec.read_only:
            return PermissionDecision.allow("Read-only tool.")
        if self.spec.name == "generate_edit":
            return PermissionDecision.allow("Edit generation is proposal-only and writes no files.")
        if self.spec.requires_approval_by_default:
            return PermissionDecision.ask("Tool requires approval by default.")
        return PermissionDecision.allow("Tool allowed by default policy.")

    def summarize_result(self, result: ToolResult) -> str:
        return result.observation


def agent_action_to_tool_payload(action: AgentAction) -> dict[str, Any]:
    return json_safe(
        {
            "action_id": action.action_id,
            "reason_summary": action.reason_summary,
            "target_files": action.target_files,
            "target_symbols": action.target_symbols,
            "command": action.command,
            "expected_output": action.expected_output,
            "safety_requirements": action.safety_requirements,
            "requires_approval": action.requires_approval,
            **(action.payload or {}),
        }
    )


def tool_result_to_action_result(action: AgentAction, result: ToolResult) -> ActionResult:
    return ActionResult(
        action_id=action.action_id,
        status=result.status,
        observation=result.observation,
        files_read=list(result.files_read),
        files_changed=list(result.files_changed),
        command_result=result.command_result,
        next_recommended_action=result.next_recommended_action,
        duration_ms=int(result.duration_ms or 0),
        payload=json_safe(result.payload),
    )
