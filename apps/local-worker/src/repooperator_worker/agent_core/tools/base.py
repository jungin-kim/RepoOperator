from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from repooperator_worker.agent_core.actions import AgentAction, ActionResult
from repooperator_worker.agent_core.permissions import PermissionDecision, PermissionMode, ToolPermissionContext
from repooperator_worker.schemas import AgentRunRequest
from repooperator_worker.services.json_safe import json_safe


ToolStatus = Literal["success", "skipped", "failed", "waiting_approval", "cancelled", "timed_out"]
ToolSideEffectLevel = Literal["none", "read", "write", "command", "network", "remote_write"]
ToolInterruptBehavior = Literal["none", "approval", "cancellable", "background"]
OversizedResultStrategy = Literal["inline_preview", "artifact_ref", "reject"]
ToolOperation = Literal[
    "list_files",
    "search",
    "read_file",
    "analyze_repository",
    "web_search",
    "web_fetch",
    "command",
    "git_status",
    "git_diff",
    "git_log",
    "git_branch",
    "git_commit",
    "git_push",
    "git_provider_request",
    "routine",
    "edit",
    "validation",
    "write",
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
    side_effect_level: ToolSideEffectLevel = "none"
    is_destructive: bool = False
    is_open_world: bool = False
    workspace_bound: bool = True
    network_access: bool = False
    interrupt_behavior: ToolInterruptBehavior = "none"
    can_be_retried: bool = True
    idempotent: bool = True
    should_defer: bool = False
    always_load: bool = False
    tool_search_keywords: tuple[str, ...] = ()
    capability_names: tuple[str, ...] = ()
    prompt_summary: str = ""
    input_schema_summary: str = ""
    output_schema_summary: str = ""
    max_result_size_chars: int = 100_000
    oversized_result_strategy: OversizedResultStrategy = "artifact_ref"
    produces_artifact: bool = False
    produces_evidence: bool = False
    evidence_kind: str | None = None
    progress_kind: str = "none"
    ui_renderer_kind: str = "text"
    grouped_display_key: str = ""
    compact_label_template: str = ""
    rejected_message_template: str = ""
    error_message_template: str = ""
    permission_matcher_kind: str = "none"
    required_permissions: tuple[str, ...] = ()
    denial_recovery_hint: str = ""
    max_result_chars: int = 100_000
    permission_required: bool = False
    parallel_safe: bool = True
    capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.max_result_chars != 100_000 and self.max_result_size_chars == 100_000:
            max_result_size = int(self.max_result_chars)
        else:
            max_result_size = int(self.max_result_size_chars or self.max_result_chars or 100_000)
        object.__setattr__(self, "max_result_size_chars", max_result_size)
        object.__setattr__(self, "max_result_chars", max_result_size)
        if not self.prompt_summary:
            object.__setattr__(self, "prompt_summary", self.description)
        if not self.input_schema_summary:
            object.__setattr__(self, "input_schema_summary", summarize_input_schema(self.input_schema))
        if not self.output_schema_summary:
            object.__setattr__(
                self,
                "output_schema_summary",
                "Returns status, observation, payload metadata, files read/changed, and optional command result.",
            )
        if not self.compact_label_template:
            object.__setattr__(self, "compact_label_template", self.name.replace("_", " "))
        if not self.grouped_display_key:
            object.__setattr__(self, "grouped_display_key", self.operation)
        if not self.rejected_message_template:
            object.__setattr__(self, "rejected_message_template", "{tool_name} was not approved.")
        if not self.error_message_template:
            object.__setattr__(self, "error_message_template", "{tool_name} failed.")
        if not self.denial_recovery_hint:
            object.__setattr__(self, "denial_recovery_hint", "Adjust the request or ask for explicit approval before retrying.")
        if not self.capability_names and self.capabilities:
            object.__setattr__(self, "capability_names", tuple(self.capabilities))
        if not self.capabilities and self.capability_names:
            object.__setattr__(self, "capabilities", tuple(self.capability_names))

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
                "side_effect_level": self.side_effect_level,
                "is_destructive": self.is_destructive,
                "is_open_world": self.is_open_world,
                "workspace_bound": self.workspace_bound,
                "network_access": self.network_access,
                "interrupt_behavior": self.interrupt_behavior,
                "can_be_retried": self.can_be_retried,
                "idempotent": self.idempotent,
                "should_defer": self.should_defer,
                "always_load": self.always_load,
                "tool_search_keywords": list(self.tool_search_keywords),
                "capability_names": list(self.capability_names),
                "prompt_summary": self.prompt_summary,
                "input_schema_summary": self.input_schema_summary,
                "output_schema_summary": self.output_schema_summary,
                "max_result_size_chars": self.max_result_size_chars,
                "oversized_result_strategy": self.oversized_result_strategy,
                "produces_artifact": self.produces_artifact,
                "produces_evidence": self.produces_evidence,
                "evidence_kind": self.evidence_kind,
                "progress_kind": self.progress_kind,
                "ui_renderer_kind": self.ui_renderer_kind,
                "grouped_display_key": self.grouped_display_key,
                "compact_label_template": self.compact_label_template,
                "rejected_message_template": self.rejected_message_template,
                "error_message_template": self.error_message_template,
                "permission_matcher_kind": self.permission_matcher_kind,
                "required_permissions": list(self.required_permissions),
                "denial_recovery_hint": self.denial_recovery_hint,
                "max_result_chars": self.max_result_chars,
                "permission_required": self.permission_required,
                "parallel_safe": self.parallel_safe,
                "capabilities": list(self.capabilities),
            }
        )


def summarize_input_schema(schema: dict[str, Any]) -> str:
    if not isinstance(schema, dict):
        return "Accepts a JSON object."
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    required = [str(item) for item in schema.get("required") or [] if str(item)]
    parts: list[str] = []
    if required:
        parts.append("required: " + ", ".join(required[:8]))
    optional = [str(name) for name in properties if str(name) not in required]
    if optional:
        parts.append("optional: " + ", ".join(optional[:10]))
    if not parts:
        return "Accepts a JSON object."
    return "; ".join(parts) + "."


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
        if self.spec.name in {"generate_edit", "generate_change_set", "validate_change_set"}:
            return PermissionDecision.allow("Change-set proposal tools are non-mutating and write no files.")
        if self.spec.requires_approval_by_default or self.spec.permission_required:
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
