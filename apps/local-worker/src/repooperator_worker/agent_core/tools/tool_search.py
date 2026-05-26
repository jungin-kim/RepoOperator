from __future__ import annotations

from typing import Any, Iterable

from repooperator_worker.agent_core.tools.base import ToolSpec
from repooperator_worker.services.json_safe import json_safe


class ToolSearch:
    """Search registered tool contracts without executing tools."""

    def __init__(self, registry, *, skill_registry=None, plugin_registry=None, mcp_registry=None) -> None:
        self.registry = registry
        self.skill_registry = skill_registry if skill_registry is not None else _safe_default_skill_registry()
        self.plugin_registry = plugin_registry if plugin_registry is not None else _safe_default_plugin_registry()
        self.mcp_registry = mcp_registry if mcp_registry is not None else _safe_default_mcp_registry()

    def search(
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
        requested_capabilities = _normalized_terms([capability, *(capabilities or [])])
        requested_names = _normalized_terms(names or [])
        requested_keywords = _normalized_terms(keywords or [])
        query_terms = _normalized_terms((query or "").replace("-", "_").split())
        scored: list[tuple[int, int, str, Any]] = []
        for index, spec in enumerate(self.registry.specs()):
            score = self._score(
                spec,
                query_terms=query_terms,
                requested_capabilities=requested_capabilities,
                requested_names=requested_names,
                requested_keywords=requested_keywords,
            )
            if score > 0:
                scored.append((score, -index, "tool", spec))
        if include_external:
            offset = len(scored) + 1
            for ext_index, item in enumerate(_external_search_items(self.skill_registry, self.plugin_registry, self.mcp_registry)):
                score = self._score_external(
                    item,
                    query_terms=query_terms,
                    requested_capabilities=requested_capabilities,
                    requested_names=requested_names,
                    requested_keywords=requested_keywords,
                )
                if score > 0:
                    scored.append((score, -(offset + ext_index), "external", item))
        scored.sort(reverse=True)
        selected = scored[: max(1, int(limit or 12))]
        if model_specs:
            names_for_model = [item.name for _, _, kind, item in selected if kind == "tool"]
            tool_specs = self.registry.specs_for_model(tool_names=names_for_model, include_default=False)
            external_specs = [json_safe(item) for _, _, kind, item in selected if kind == "external"]
            return json_safe([*tool_specs, *external_specs])
        return json_safe([item.model_dump() if kind == "tool" else item for _, _, kind, item in selected])

    def _score(
        self,
        spec: ToolSpec,
        *,
        query_terms: set[str],
        requested_capabilities: set[str],
        requested_names: set[str],
        requested_keywords: set[str],
    ) -> int:
        haystack = {
            spec.name.lower(),
            spec.operation.lower(),
            *[item.lower() for item in spec.capability_names],
            *[item.lower() for item in spec.tool_search_keywords],
            *_normalized_terms(spec.description.replace("-", "_").split()),
            *_normalized_terms(spec.prompt_summary.replace("-", "_").split()),
        }
        score = 0
        if requested_names and spec.name.lower() in requested_names:
            score += 100
        capability_matches = requested_capabilities.intersection({item.lower() for item in spec.capability_names})
        if capability_matches:
            score += 80 + len(capability_matches) * 5
        keyword_matches = requested_keywords.intersection(haystack)
        if keyword_matches:
            score += 40 + len(keyword_matches) * 3
        query_matches = query_terms.intersection(haystack)
        if query_matches:
            score += 20 + len(query_matches) * 2
        if not (requested_names or requested_capabilities or requested_keywords or query_terms):
            score = 1 if spec.always_load else 0
        return score

    def _score_external(
        self,
        item: dict[str, Any],
        *,
        query_terms: set[str],
        requested_capabilities: set[str],
        requested_names: set[str],
        requested_keywords: set[str],
    ) -> int:
        haystack = _normalized_terms(
            [
                item.get("id"),
                item.get("name"),
                item.get("tool_name"),
                item.get("plugin_id"),
                item.get("server_id"),
                item.get("kind"),
                item.get("operation"),
                *[str(value) for value in item.get("capability_names") or []],
                *[str(value) for value in item.get("tool_search_keywords") or []],
                *_normalized_terms(str(item.get("description") or item.get("prompt_summary") or "").replace("-", "_").split()),
            ]
        )
        score = 0
        names = {str(item.get("id") or "").lower(), str(item.get("name") or "").lower(), str(item.get("tool_name") or "").lower()}
        if requested_names and requested_names.intersection(names):
            score += 90
        capability_matches = requested_capabilities.intersection({value.lower() for value in item.get("capability_names") or []})
        if capability_matches:
            score += 70 + len(capability_matches) * 5
        keyword_matches = requested_keywords.intersection(haystack)
        if keyword_matches:
            score += 35 + len(keyword_matches) * 3
        query_matches = query_terms.intersection(haystack)
        if query_matches:
            score += 18 + len(query_matches) * 2
        if not (requested_names or requested_capabilities or requested_keywords or query_terms):
            score = 0
        return score


def _normalized_terms(items: Iterable[str | None]) -> set[str]:
    result: set[str] = set()
    for item in items:
        text = str(item or "").strip().lower()
        if text:
            result.add(text)
    return result


def _external_search_items(skill_registry, plugin_registry, mcp_registry) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if skill_registry is not None:
        for spec in skill_registry.enabled_specs():
            items.append(
                {
                    "kind": "skill",
                    "id": spec.id,
                    "name": spec.name,
                    "operation": "skill_instruction",
                    "read_only": True,
                    "side_effect_level": "none",
                    "requires_approval_by_default": False,
                    "capability_names": list(spec.required_capabilities),
                    "tool_search_keywords": [
                        spec.id,
                        spec.name,
                        spec.when_to_use,
                        *str(spec.id).replace("_", " ").split(),
                        *str(spec.name).replace("_", " ").split(),
                        *str(spec.when_to_use).replace("_", " ").split(),
                        *spec.required_tools,
                        *spec.required_capabilities,
                    ],
                    "prompt_summary": spec.description or spec.when_to_use,
                    "input_schema_summary": "No direct execution; selected skills are context instructions only.",
                    "output_schema_summary": spec.output_contract,
                    "enabled": spec.enabled,
                    "executable": False,
                }
            )
    if plugin_registry is not None:
        for tool in plugin_registry.tool_metadata(enabled_only=True):
            items.append(_external_tool_item(tool, kind="plugin_tool"))
    if mcp_registry is not None:
        for tool in mcp_registry.tool_metadata(enabled_only=True):
            items.append(_external_tool_item(tool, kind="mcp_tool"))
    return items


def _external_tool_item(tool: dict[str, Any], *, kind: str) -> dict[str, Any]:
    capability_names = [str(item) for item in tool.get("required_capabilities") or [] if str(item)]
    return json_safe(
        {
            "kind": kind,
            "id": tool.get("id"),
            "name": tool.get("name"),
            "tool_name": tool.get("name"),
            "operation": "external_tool_metadata",
            "description": tool.get("description") or "",
            "read_only": bool(tool.get("read_only")),
            "side_effect_level": "network" if tool.get("network_access") else "none",
            "requires_approval_by_default": True,
            "capability_names": capability_names,
            "tool_search_keywords": [
                str(tool.get("id") or ""),
                str(tool.get("name") or ""),
                str(tool.get("plugin_id") or ""),
                str(tool.get("server_id") or ""),
                *capability_names,
            ],
            "prompt_summary": tool.get("description") or "External tool metadata.",
            "input_schema_summary": "Metadata only; execution is unavailable unless routed through an approved adapter.",
            "output_schema_summary": "External tools must remain permission-gated.",
            "plugin_id": tool.get("plugin_id"),
            "server_id": tool.get("server_id"),
            "provider": tool.get("provider"),
            "enabled": bool(tool.get("enabled", True)),
            "source": tool.get("source"),
            "executable": False,
        }
    )


def _safe_default_skill_registry():
    try:
        from repooperator_worker.agent_core.skills import get_default_skill_registry

        return get_default_skill_registry()
    except Exception:
        return None


def _safe_default_plugin_registry():
    try:
        from repooperator_worker.agent_core.plugins import get_default_plugin_registry

        return get_default_plugin_registry()
    except Exception:
        return None


def _safe_default_mcp_registry():
    try:
        from repooperator_worker.agent_core.mcp import get_default_mcp_registry

        return get_default_mcp_registry()
    except Exception:
        return None
