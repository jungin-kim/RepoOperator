from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

from repooperator_worker.agent_core.tools.base import Tool, ToolSpec
from repooperator_worker.agent_core.tools.builtin import (
    AnalyzeRepositoryTool,
    AskClarificationTool,
    FinalAnswerTool,
    GenerateEditTool,
    InspectGitStateTool,
    InspectRepoTreeTool,
    PreviewCommandTool,
    ReadFileTool,
    RunApprovedCommandTool,
    SearchFilesTool,
    SearchTextTool,
)
from repooperator_worker.services.json_safe import json_safe


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] | None = None) -> None:
        self._tools: OrderedDict[str, Tool] = OrderedDict()
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
                }
                for spec in self.specs()
            ]
        )

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
            AnalyzeRepositoryTool(),
            PreviewCommandTool(),
            InspectGitStateTool(),
            RunApprovedCommandTool(),
            GenerateEditTool(),
            AskClarificationTool(),
            FinalAnswerTool(),
        ]
    )
