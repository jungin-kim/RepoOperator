from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

from repooperator_worker.agent_core.capabilities.base import CapabilitySpec, capability_is_selectable
from repooperator_worker.services.json_safe import json_safe


class CapabilityRegistry:
    """Registry for reusable agent capabilities.

    Capabilities describe availability, permissions, and safety surfaces for
    groups of tools. They intentionally do not classify user intent or force a
    controller workflow.
    """

    def __init__(self, capabilities: Iterable[CapabilitySpec] | None = None) -> None:
        self._capabilities: OrderedDict[str, CapabilitySpec] = OrderedDict()
        self._tools_to_capabilities: dict[str, list[str]] = {}
        for capability in capabilities or []:
            self.register(capability)

    def register(self, capability: CapabilitySpec) -> None:
        if capability.name in self._capabilities:
            raise ValueError(f"Capability {capability.name!r} is already registered.")
        self._capabilities[capability.name] = capability
        for tool in capability.tools:
            self._tools_to_capabilities.setdefault(tool, [])
            if capability.name not in self._tools_to_capabilities[tool]:
                self._tools_to_capabilities[tool].append(capability.name)

    def get(self, name: str) -> CapabilitySpec:
        try:
            return self._capabilities[name]
        except KeyError as exc:
            raise KeyError(f"Unknown capability: {name}") from exc

    def specs(self) -> list[CapabilitySpec]:
        return list(self._capabilities.values())

    def specs_for_model(self, *, available_only: bool = True) -> list[dict]:
        specs = self.available_specs() if available_only else self.specs()
        return json_safe([spec.model_dump() for spec in specs])

    def available_specs(self) -> list[CapabilitySpec]:
        return [spec for spec in self.specs() if capability_is_selectable(spec)]

    def selectable_names(self) -> list[str]:
        return [spec.name for spec in self.available_specs()]

    def select_available(self, names: Iterable[str]) -> list[CapabilitySpec]:
        selected: list[CapabilitySpec] = []
        for name in names:
            if name not in self._capabilities:
                continue
            spec = self._capabilities[name]
            if capability_is_selectable(spec):
                selected.append(spec)
        return selected

    def capabilities_for_tool(self, tool_name: str, *, available_only: bool = False) -> list[CapabilitySpec]:
        names = list(self._tools_to_capabilities.get(tool_name) or [])
        specs = [self._capabilities[name] for name in names if name in self._capabilities]
        if available_only:
            specs = [spec for spec in specs if capability_is_selectable(spec)]
        return specs

    def capability_names_for_tool(self, tool_name: str, *, available_only: bool = False) -> list[str]:
        return [spec.name for spec in self.capabilities_for_tool(tool_name, available_only=available_only)]

    def tool_map(self) -> dict[str, list[str]]:
        return json_safe({tool: list(names) for tool, names in sorted(self._tools_to_capabilities.items())})

    def __contains__(self, name: str) -> bool:
        return name in self._capabilities
