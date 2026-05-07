from __future__ import annotations

from repooperator_worker.agent_core.actions import AgentAction, ActionResult
from repooperator_worker.agent_core.tool_orchestrator import ToolOrchestrator
from repooperator_worker.agent_core.tools.registry import ToolRegistry, get_default_tool_registry
from repooperator_worker.schemas import AgentRunRequest


class ActionExecutor:
    """Compatibility shim for the historical action executor API."""

    def __init__(self, *, run_id: str, request: AgentRunRequest, registry: ToolRegistry | None = None) -> None:
        self.run_id = run_id
        self.request = request
        self.registry = registry or get_default_tool_registry()
        self.orchestrator = ToolOrchestrator(run_id=run_id, request=request, registry=self.registry)

    def execute(self, action: AgentAction) -> ActionResult:
        return self.orchestrator.execute_action(action)


__all__ = [
    "ActionExecutor",
]
