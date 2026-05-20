from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from repooperator_worker.services.json_safe import json_safe


CapabilityCategory = Literal[
    "repository_read",
    "repository_write",
    "command_execution",
    "web_research",
    "git_provider",
    "routine",
    "context_memory",
    "validation",
    "multi_agent",
]

CapabilitySideEffectLevel = Literal["none", "read", "write", "command", "network", "remote_write"]


@dataclass(frozen=True)
class CapabilitySpec:
    name: str
    category: CapabilityCategory
    description: str
    tools: list[str] = field(default_factory=list)
    required_permissions: list[str] = field(default_factory=list)
    side_effect_level: CapabilitySideEffectLevel = "none"
    network_access: bool = False
    requires_approval: bool = False
    available: bool = True
    provider: str | None = None
    config_requirements: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return json_safe(
            {
                "name": self.name,
                "category": self.category,
                "description": self.description,
                "tools": list(self.tools),
                "required_permissions": list(self.required_permissions),
                "side_effect_level": self.side_effect_level,
                "network_access": self.network_access,
                "requires_approval": self.requires_approval,
                "available": self.available,
                "provider": self.provider,
                "config_requirements": list(self.config_requirements),
            }
        )


def capability_is_selectable(spec: CapabilitySpec) -> bool:
    """Return whether graph planning may consider this capability.

    This is intentionally only availability gating. It is not an intent
    classifier and does not choose a workflow by itself.
    """
    return bool(spec.available)
