from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

from repooperator_worker.agent_core.capabilities.builtin import get_default_capability_registry
from repooperator_worker.agent_core.capabilities.registry import CapabilityRegistry
from repooperator_worker.agent_core.tools.base import Tool, ToolSpec
from repooperator_worker.agent_core.tools.builtin import (
    AnalyzeRepositoryTool,
    ApplyChangeSetTool,
    AskClarificationTool,
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
        self._tools[name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    def specs_for_model(self) -> list[dict]:
        return json_safe(
            [
                {
                    "name": spec.name,
                    "description": spec.description,
                    "operation": spec.operation,
                    "input_schema": spec.input_schema,
                    "read_only": spec.read_only,
                    "concurrency_safe": spec.concurrency_safe,
                    "requires_approval_by_default": spec.requires_approval_by_default,
                    "side_effect_level": spec.side_effect_level,
                    "permission_required": spec.permission_required,
                    "parallel_safe": spec.parallel_safe,
                    "workspace_bound": spec.workspace_bound,
                    "network_access": spec.network_access,
                    "produces_artifact": spec.produces_artifact,
                    "produces_evidence": spec.produces_evidence,
                    "can_be_retried": spec.can_be_retried,
                    "capabilities": self.capability_registry.capability_names_for_tool(spec.name, available_only=True),
                }
                for spec in self.specs()
            ]
        )

    def capabilities_for_tool(self, tool_name: str, *, available_only: bool = False) -> list[str]:
        return self.capability_registry.capability_names_for_tool(tool_name, available_only=available_only)

    def capability_specs_for_model(self) -> list[dict]:
        return self.capability_registry.specs_for_model()

    def allowed_action_types(self) -> list[str]:
        return list(self._tools.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._tools


def get_default_tool_registry() -> ToolRegistry:
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
        ]
    )
