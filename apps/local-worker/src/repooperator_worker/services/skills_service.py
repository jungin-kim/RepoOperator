from __future__ import annotations

from typing import Any, Iterable

from repooperator_worker.agent_core.skills import SkillRegistry, get_default_skill_registry
from repooperator_worker.services.json_safe import json_safe


def discover_skills() -> dict[str, Any]:
    registry = get_default_skill_registry()
    skills = [spec.model_dump() for spec in registry.specs()]
    enabled = [spec.model_dump() for spec in registry.enabled_specs()]
    return {"skills": json_safe(skills), "effective_skills": json_safe(enabled)}


def enabled_skill_context(
    max_chars: int = 4_000,
    *,
    task: str | None = None,
    kind: str | None = None,
    capabilities: Iterable[str] | None = None,
    tool_names: Iterable[str] | None = None,
    registry: SkillRegistry | None = None,
) -> tuple[str, list[str]]:
    selected_registry = registry or get_default_skill_registry()
    return selected_registry.context_for_task(
        task,
        kind=kind,
        capabilities=capabilities,
        tool_names=tool_names,
        max_chars=max_chars,
    )


def resolve_effective_skills(skills: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    if skills is None:
        return discover_skills()["effective_skills"]
    registry = SkillRegistry(skills)
    return json_safe([spec.model_dump() for spec in registry.enabled_specs()])
