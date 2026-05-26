from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from typing import Any, Iterable

from repooperator_worker.agent_core.capabilities.builtin import get_default_capability_registry
from repooperator_worker.agent_core.capabilities.registry import CapabilityRegistry
from repooperator_worker.agent_core.permissions import permission_matcher_kind_for_tool
from repooperator_worker.agent_core.tools.base import Tool, ToolSpec
from repooperator_worker.agent_core.tools.builtin import (
    AnalyzeRepositoryTool,
    ApplyChangeSetTool,
    AskClarificationTool,
    CompactThreadContextTool,
    CreateFileTool,
    DeleteFileTool,
    FinalAnswerTool,
    GenerateChangeSetTool,
    GenerateEditTool,
    GitBranchCreateTool,
    GitCommitTool,
    GitDiffTool,
    GitHubCreatePrTool,
    GitLabCreateMrTool,
    GitLogTool,
    GitPushTool,
    GitStatusTool,
    InspectGitStateTool,
    InspectRepoTreeTool,
    ModifyFileTool,
    PreviewCommandTool,
    ReadFileTool,
    ReadManyFilesTool,
    RenameFileTool,
    RefreshContextPackTool,
    FetchUrlTool,
    RunValidationCommandTool,
    RunApprovedCommandTool,
    SearchFilesTool,
    SearchTextTool,
    SearchWebTool,
    SummarizeWebEvidenceTool,
    ValidateChangeSetTool,
)
from repooperator_worker.services.json_safe import json_safe


ALWAYS_LOAD_TOOL_NAMES = {
    "inspect_repo_tree",
    "search_files",
    "search_text",
    "read_file",
    "read_many_files",
    "generate_change_set",
    "validate_change_set",
    "final_answer",
    # These are always-load so compaction and refresh remain explicit model-visible
    # recovery options. The tool payloads return summaries/reports, never raw packs.
    "refresh_context_pack",
    "compact_thread_context",
}

DEFERRED_TOOL_NAMES = {
    "analyze_repository",
    "preview_command",
    "inspect_git_state",
    "run_approved_command",
    "run_validation_command",
    "search_web",
    "fetch_url",
    "summarize_web_evidence",
    "apply_change_set",
    "git_status",
    "git_diff",
    "git_log",
    "git_branch_create",
    "git_commit",
    "git_push",
    "github_create_pr",
    "gitlab_create_mr",
    "create_file",
    "modify_file",
    "delete_file",
    "rename_file",
}

NON_RETRYABLE_DESTRUCTIVE_TOOLS = {
    "run_approved_command",
    "run_validation_command",
    "apply_change_set",
    "create_file",
    "modify_file",
    "delete_file",
    "rename_file",
    "git_branch_create",
    "git_commit",
    "git_push",
    "github_create_pr",
    "gitlab_create_mr",
}

MODEL_SPEC_FIELDS = {
    "name",
    "operation",
    "read_only",
    "concurrency_safe",
    "requires_approval_by_default",
    "side_effect_level",
    "is_destructive",
    "is_open_world",
    "workspace_bound",
    "network_access",
    "interrupt_behavior",
    "can_be_retried",
    "idempotent",
    "should_defer",
    "always_load",
    "tool_search_keywords",
    "capability_names",
    "prompt_summary",
    "input_schema_summary",
    "output_schema_summary",
    "max_result_size_chars",
    "produces_artifact",
    "produces_evidence",
    "evidence_kind",
}


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] | None = None, *, capability_registry: CapabilityRegistry | None = None) -> None:
        self._tools: OrderedDict[str, Tool] = OrderedDict()
        self.capability_registry = capability_registry or get_default_capability_registry()
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        name = tool.spec.name
        if name in self._tools:
            raise ValueError(f"Tool {name!r} is already registered.")
        if not getattr(tool.spec, "operation", None):
            raise ValueError(f"Tool {name!r} must declare an operation.")
        tool.spec = self._normalized_spec(tool.spec)
        self._tools[name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    def internal_specs(self) -> list[dict]:
        """Return full JSON-safe specs for orchestration, UI, and audit layers."""
        return json_safe([spec.model_dump() for spec in self.specs()])

    def internal_spec(self, tool_name: str) -> dict:
        return json_safe(self.get(tool_name).spec.model_dump())

    def specs_for_model(
        self,
        *,
        capabilities: Iterable[str] | None = None,
        tool_names: Iterable[str] | None = None,
        include_deferred: bool = False,
        include_default: bool = True,
    ) -> list[dict]:
        requested_capabilities = {str(item).strip() for item in capabilities or [] if str(item).strip()}
        requested_tools = {str(item).strip() for item in tool_names or [] if str(item).strip()}
        include_all = include_deferred and not requested_capabilities and not requested_tools
        return json_safe(
            [
                _model_spec(spec.model_dump())
                for spec in self.specs()
                if include_all
                or (include_default and (spec.always_load or not spec.should_defer))
                or spec.name in requested_tools
                or bool(requested_capabilities.intersection(spec.capability_names))
            ]
        )

    def search_tools(
        self,
        *,
        query: str | None = None,
        capability: str | None = None,
        capabilities: Iterable[str] | None = None,
        names: Iterable[str] | None = None,
        keywords: Iterable[str] | None = None,
        limit: int = 12,
        model_specs: bool = True,
        include_external: bool = False,
    ) -> list[dict]:
        from repooperator_worker.agent_core.tools.tool_search import ToolSearch

        return ToolSearch(self).search(
            query=query,
            capability=capability,
            capabilities=capabilities,
            names=names,
            keywords=keywords,
            limit=limit,
            model_specs=model_specs,
            include_external=include_external,
        )

    def capabilities_for_tool(self, tool_name: str, *, available_only: bool = False) -> list[str]:
        return self.capability_registry.capability_names_for_tool(tool_name, available_only=available_only)

    def capability_specs_for_model(self) -> list[dict]:
        return self.capability_registry.specs_for_model()

    def allowed_action_types(self) -> list[str]:
        return list(self._tools.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def _normalized_spec(self, spec: ToolSpec) -> ToolSpec:
        capability_names = tuple(
            self.capability_registry.capability_names_for_tool(spec.name, available_only=True)
            or spec.capability_names
            or spec.capabilities
        )
        side_effect_level = _side_effect_level(spec)
        network_access = bool(spec.network_access)
        is_open_world = bool(
            spec.is_open_world
            or network_access
            or side_effect_level in {"command", "network", "remote_write"}
            or not spec.workspace_bound
        )
        is_destructive = bool(spec.is_destructive or _is_destructive(spec.name, side_effect_level))
        requires_approval = bool(spec.requires_approval_by_default or (is_destructive and not _is_proposal_only(spec.name)))
        permission_required = bool(spec.permission_required or (is_destructive and not _is_proposal_only(spec.name)) or network_access)
        should_defer = bool(spec.should_defer or spec.name in DEFERRED_TOOL_NAMES or is_destructive or network_access)
        always_load = bool(spec.always_load or spec.name in ALWAYS_LOAD_TOOL_NAMES)
        if always_load:
            should_defer = False
        required_permissions = tuple(spec.required_permissions or _required_permissions(spec.name, side_effect_level, network_access))
        keywords = tuple(_dedupe([*spec.tool_search_keywords, *capability_names, spec.name, spec.operation, *_split_identifier(spec.name)]))
        produces_evidence = bool(spec.produces_evidence or _produces_evidence(spec.name, spec.operation, side_effect_level))
        interrupt_behavior = spec.interrupt_behavior
        if interrupt_behavior == "none":
            if requires_approval:
                interrupt_behavior = "approval"
            elif not spec.concurrency_safe or side_effect_level in {"command", "network", "remote_write"}:
                interrupt_behavior = "cancellable"
        progress_kind = spec.progress_kind if spec.progress_kind != "none" else _progress_kind(spec.name, spec.operation, side_effect_level)
        ui_renderer_kind = spec.ui_renderer_kind if spec.ui_renderer_kind != "text" else _ui_renderer_kind(spec.name, spec.operation, side_effect_level)
        evidence_kind = spec.evidence_kind or _evidence_kind(spec.name, spec.operation, side_effect_level, produces_evidence)
        idempotent = bool(spec.idempotent and not is_destructive and side_effect_level not in {"command", "network", "remote_write"} and not _is_non_deterministic(spec.name))
        return replace(
            spec,
            side_effect_level=side_effect_level,
            is_destructive=is_destructive,
            is_open_world=is_open_world,
            network_access=network_access,
            requires_approval_by_default=requires_approval,
            permission_required=permission_required,
            interrupt_behavior=interrupt_behavior,
            can_be_retried=bool(spec.can_be_retried and not (is_destructive and spec.name in NON_RETRYABLE_DESTRUCTIVE_TOOLS)),
            idempotent=idempotent,
            should_defer=should_defer,
            always_load=always_load,
            tool_search_keywords=keywords,
            capability_names=capability_names,
            capabilities=capability_names,
            required_permissions=required_permissions,
            permission_matcher_kind=spec.permission_matcher_kind if spec.permission_matcher_kind != "none" else _permission_matcher_kind(spec.name, side_effect_level, network_access),
            denial_recovery_hint=_denial_recovery_hint(spec, side_effect_level, network_access, is_destructive),
            progress_kind=progress_kind,
            ui_renderer_kind=ui_renderer_kind,
            grouped_display_key=(
                spec.grouped_display_key
                if spec.grouped_display_key and spec.grouped_display_key != spec.operation
                else _grouped_display_key(spec.name, spec.operation, side_effect_level)
            ),
            compact_label_template=spec.compact_label_template or spec.name.replace("_", " "),
            rejected_message_template=(
                spec.rejected_message_template
                if spec.rejected_message_template != "{tool_name} was not approved."
                else _rejected_message_template(spec.name)
            ),
            error_message_template=(
                spec.error_message_template
                if spec.error_message_template != "{tool_name} failed."
                else _error_message_template(spec.name)
            ),
            produces_evidence=produces_evidence,
            evidence_kind=evidence_kind,
        )


def _model_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return {key: spec.get(key) for key in sorted(MODEL_SPEC_FIELDS) if key in spec}


def _side_effect_level(spec: ToolSpec) -> str:
    if spec.operation in {"git_push", "git_provider_request"}:
        return "remote_write"
    if spec.operation in {"git_branch", "git_commit", "write"}:
        return "write"
    if spec.operation == "command" and not spec.read_only:
        return "command"
    if spec.network_access:
        return "network"
    if spec.side_effect_level != "none":
        return str(spec.side_effect_level)
    if spec.operation in {"list_files", "search", "read_file", "analyze_repository", "git_status", "git_diff", "git_log", "web_search", "web_fetch"}:
        return "read"
    return "none"


def _is_destructive(name: str, side_effect_level: str) -> bool:
    if name in {"generate_change_set", "generate_edit", "validate_change_set", "preview_command", "inspect_git_state"}:
        return False
    return side_effect_level in {"write", "command", "remote_write"}


def _is_proposal_only(name: str) -> bool:
    return name in {"generate_change_set", "generate_edit", "validate_change_set"}


def _is_non_deterministic(name: str) -> bool:
    return name in {"generate_change_set", "generate_edit", "search_web", "fetch_url", "summarize_web_evidence", "analyze_repository"}


def _required_permissions(name: str, side_effect_level: str, network_access: bool) -> tuple[str, ...]:
    if name in {"inspect_repo_tree", "search_files", "search_text", "read_file", "read_many_files", "analyze_repository"}:
        return ("repository_read",)
    if name in {"generate_change_set", "validate_change_set", "apply_change_set", "create_file", "modify_file", "delete_file", "rename_file", "generate_edit"}:
        return ("repository_write",)
    if name in {"preview_command", "inspect_git_state", "run_approved_command"}:
        return ("command_execution",)
    if name == "run_validation_command":
        return ("command_execution", "validation")
    if name in {"search_web", "fetch_url", "summarize_web_evidence"}:
        return ("network",)
    if name in {"git_status", "git_diff", "git_log"}:
        return ("git_read",)
    if name in {"git_branch_create", "git_commit"}:
        return ("git_local_write",)
    if name in {"git_push", "github_create_pr", "gitlab_create_mr"}:
        return ("git_remote_write",)
    if network_access:
        return ("network",)
    if side_effect_level == "write":
        return ("repository_write",)
    if side_effect_level == "command":
        return ("command_execution",)
    return ()


def _permission_matcher_kind(name: str, side_effect_level: str, network_access: bool) -> str:
    del side_effect_level, network_access
    return permission_matcher_kind_for_tool(name).value


def _denial_recovery_hint(spec: ToolSpec, side_effect_level: str, network_access: bool, is_destructive: bool) -> str:
    if spec.denial_recovery_hint and spec.denial_recovery_hint != "Adjust the request or ask for explicit approval before retrying.":
        return spec.denial_recovery_hint
    if network_access:
        return "Ask for network approval or continue with local repository evidence only."
    if side_effect_level == "command":
        return "Preview the command, narrow it to an allowlisted read-only command, or request approval."
    if is_destructive:
        return "Use a proposal-only tool first, then ask for explicit approval before applying changes."
    return "Use a narrower repository-bounded request and retry."


def _progress_kind(name: str, operation: str, side_effect_level: str) -> str:
    if name in {"search_web", "fetch_url", "summarize_web_evidence"}:
        return "web_research"
    if name.startswith("git_") or name.startswith("github_") or name.startswith("gitlab_"):
        return "git"
    if operation in {"search", "list_files"}:
        return "search"
    if operation == "read_file":
        return "read_file"
    if operation == "command" or side_effect_level == "command":
        return "command"
    if operation in {"edit", "write", "validation"}:
        return "change_set"
    if operation == "final_answer":
        return "final_answer"
    return "generic"


def _ui_renderer_kind(name: str, operation: str, side_effect_level: str) -> str:
    if name in {"search_web", "fetch_url", "summarize_web_evidence"}:
        return "web_evidence"
    if name.startswith("git_") or name.startswith("github_") or name.startswith("gitlab_"):
        return "git"
    if operation in {"search", "list_files"}:
        return "search_results"
    if operation == "read_file":
        return "file"
    if operation == "command" or side_effect_level == "command":
        return "command"
    if operation in {"edit", "write", "validation"}:
        return "change_set"
    if operation == "final_answer":
        return "final_answer"
    return "text"


def _evidence_kind(name: str, operation: str, side_effect_level: str, produces_evidence: bool) -> str | None:
    if name in {"generate_change_set", "generate_edit", "apply_change_set"}:
        return "change_set"
    if name in {"search_web", "fetch_url", "summarize_web_evidence"}:
        return "web_source"
    if name.startswith("git_"):
        return "git"
    if operation == "validation":
        return "validation"
    if operation == "command" or side_effect_level == "command":
        return "command"
    if produces_evidence or operation in {"list_files", "search", "read_file", "analyze_repository"}:
        return "repository"
    return None


def _produces_evidence(name: str, operation: str, side_effect_level: str) -> bool:
    if name in {"ask_clarification", "final_answer"}:
        return False
    return operation in {
        "list_files",
        "search",
        "read_file",
        "analyze_repository",
        "web_search",
        "web_fetch",
        "git_status",
        "git_diff",
        "git_log",
        "edit",
        "validation",
    } or side_effect_level in {"read", "network", "remote_write", "command"}


def _grouped_display_key(name: str, operation: str, side_effect_level: str) -> str:
    if name in {"search_web", "fetch_url", "summarize_web_evidence"}:
        return "web_research"
    if name.startswith("git_") or name.startswith("github_") or name.startswith("gitlab_"):
        return "git"
    if operation == "command" or side_effect_level == "command":
        return "command"
    if operation in {"edit", "write", "validation"}:
        return "change_set"
    return operation


def _rejected_message_template(name: str) -> str:
    return f"{name} was blocked or not approved."


def _error_message_template(name: str) -> str:
    return f"{name} failed. Review the tool result for details."


def _split_identifier(value: str) -> list[str]:
    return [part for part in value.replace("-", "_").split("_") if part]


def _dedupe(items: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item).strip().lower()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def get_default_tool_registry() -> ToolRegistry:
    try:
        from repooperator_worker.agent_core.mcp import configured_mcp_tool_adapters

        mcp_tools = configured_mcp_tool_adapters()
    except Exception:
        mcp_tools = []
    return ToolRegistry(
        [
            InspectRepoTreeTool(),
            SearchFilesTool(),
            SearchTextTool(),
            ReadFileTool(),
            ReadManyFilesTool(),
            AnalyzeRepositoryTool(),
            PreviewCommandTool(),
            InspectGitStateTool(),
            RunApprovedCommandTool(),
            RunValidationCommandTool(),
            SearchWebTool(),
            FetchUrlTool(),
            SummarizeWebEvidenceTool(),
            RefreshContextPackTool(),
            CompactThreadContextTool(),
            GenerateChangeSetTool(),
            ValidateChangeSetTool(),
            ApplyChangeSetTool(),
            GitStatusTool(),
            GitDiffTool(),
            GitLogTool(),
            GitBranchCreateTool(),
            GitCommitTool(),
            GitPushTool(),
            GitHubCreatePrTool(),
            GitLabCreateMrTool(),
            CreateFileTool(),
            ModifyFileTool(),
            DeleteFileTool(),
            RenameFileTool(),
            GenerateEditTool(),
            AskClarificationTool(),
            FinalAnswerTool(),
            *mcp_tools,
        ]
    )
