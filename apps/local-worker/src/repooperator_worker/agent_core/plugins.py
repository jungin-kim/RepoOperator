from __future__ import annotations

import json
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from repooperator_worker.config import get_settings
from repooperator_worker.services.active_repository import get_active_repository
from repooperator_worker.services.common import resolve_project_path
from repooperator_worker.services.json_safe import json_safe


@dataclass(frozen=True)
class PluginSpec:
    id: str
    name: str
    provider: str = "local"
    tools: list[dict[str, Any]] = field(default_factory=list)
    skills: list[dict[str, Any]] = field(default_factory=list)
    hooks: list[dict[str, Any]] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)
    enabled: bool = False
    config_schema: dict[str, Any] = field(default_factory=dict)
    source_type: str = "config"
    source_path: str = ""

    def __post_init__(self) -> None:
        plugin_id = _slug(self.id or self.name)
        object.__setattr__(self, "id", plugin_id)
        object.__setattr__(self, "tools", [_normalize_tool_metadata(tool, plugin_id=plugin_id, provider=self.provider) for tool in self.tools or []])
        object.__setattr__(self, "skills", [json_safe(skill) for skill in self.skills or [] if isinstance(skill, (dict, str))])
        object.__setattr__(self, "hooks", [json_safe(hook) for hook in self.hooks or [] if isinstance(hook, (dict, str))])
        object.__setattr__(self, "permissions", [str(item) for item in self.permissions or [] if str(item).strip()])
        object.__setattr__(self, "enabled", bool(self.enabled))

    def model_dump(self) -> dict[str, Any]:
        return json_safe(
            {
                "id": self.id,
                "name": self.name,
                "provider": self.provider,
                "tools": list(self.tools),
                "skills": list(self.skills),
                "hooks": list(self.hooks),
                "permissions": list(self.permissions),
                "enabled": self.enabled,
                "config_schema": self.config_schema,
                "source_type": self.source_type,
                "source_path": self.source_path,
            }
        )

    def model_hint(self) -> dict[str, Any]:
        return json_safe(
            {
                "id": self.id,
                "name": self.name,
                "provider": self.provider,
                "tools": [_tool_hint(tool) for tool in self.tools],
                "skills": list(self.skills),
                "hooks": list(self.hooks),
                "permissions": list(self.permissions),
                "enabled": self.enabled,
            }
        )


class PluginRegistry:
    """Metadata registry for plugins.

    Plugin entries describe tools, skills, and hooks, but this foundation does
    not execute plugin code or auto-enable anything.
    """

    def __init__(self, plugins: Iterable[PluginSpec | dict[str, Any]] | None = None) -> None:
        self._plugins: OrderedDict[str, PluginSpec] = OrderedDict()
        for plugin in plugins or []:
            self.register(plugin)

    def register(self, plugin: PluginSpec | dict[str, Any]) -> None:
        spec = plugin if isinstance(plugin, PluginSpec) else plugin_spec_from_dict(plugin)
        if not spec.id:
            raise ValueError("Plugin id is required.")
        self._plugins[spec.id] = spec

    def get(self, plugin_id: str) -> PluginSpec:
        key = _slug(plugin_id)
        try:
            return self._plugins[key]
        except KeyError as exc:
            raise KeyError(f"Unknown plugin: {plugin_id}") from exc

    def specs(self) -> list[PluginSpec]:
        return list(self._plugins.values())

    def enabled_specs(self) -> list[PluginSpec]:
        return [spec for spec in self.specs() if spec.enabled]

    def specs_for_model(self, *, enabled_only: bool = True) -> list[dict[str, Any]]:
        specs = self.enabled_specs() if enabled_only else self.specs()
        return json_safe([spec.model_hint() for spec in specs])

    def tool_metadata(self, *, enabled_only: bool = True) -> list[dict[str, Any]]:
        plugins = self.enabled_specs() if enabled_only else self.specs()
        tools: list[dict[str, Any]] = []
        for plugin in plugins:
            for tool in plugin.tools:
                tools.append(
                    json_safe(
                        {
                            **tool,
                            "plugin_id": plugin.id,
                            "plugin_name": plugin.name,
                            "provider": plugin.provider,
                            "permissions": list(tool.get("permissions") or plugin.permissions),
                            "enabled": plugin.enabled,
                            "source": "plugin",
                        }
                    )
                )
        return tools

    def search_tools(self, query: str | None = None, *, enabled_only: bool = True, limit: int = 12) -> list[dict[str, Any]]:
        terms = _terms(query or "")
        scored: list[tuple[int, int, dict[str, Any]]] = []
        for index, tool in enumerate(self.tool_metadata(enabled_only=enabled_only)):
            score = _tool_score(tool, terms)
            if score > 0:
                scored.append((score, -index, tool))
        scored.sort(reverse=True)
        if not terms:
            return self.tool_metadata(enabled_only=enabled_only)[: max(1, int(limit or 12))]
        return [tool for _, _, tool in scored[: max(1, int(limit or 12))]]

    def model_dump(self) -> dict[str, Any]:
        return json_safe({"plugins": [spec.model_dump() for spec in self.specs()]})


def get_default_plugin_registry() -> PluginRegistry:
    return load_plugin_registry_from_default_configs()


def load_plugin_registry_from_default_configs() -> PluginRegistry:
    registry = PluginRegistry()
    for spec in _load_user_plugin_specs():
        registry.register(spec)
    for spec in _load_project_plugin_specs():
        registry.register(spec)
    return registry


def plugin_spec_from_dict(item: dict[str, Any]) -> PluginSpec:
    return PluginSpec(
        id=str(item.get("id") or item.get("name") or ""),
        name=str(item.get("name") or item.get("id") or ""),
        provider=str(item.get("provider") or "local"),
        tools=_list_of_dicts(item.get("tools") or []),
        skills=_list_of_dicts(item.get("skills") or []),
        hooks=_list_of_dicts(item.get("hooks") or []),
        permissions=[str(value) for value in item.get("permissions") or [] if str(value).strip()],
        enabled=bool(item.get("enabled", False)),
        config_schema=item.get("config_schema") if isinstance(item.get("config_schema"), dict) else {},
        source_type=str(item.get("source_type") or "config"),
        source_path=str(item.get("source_path") or ""),
    )


def _load_project_plugin_specs() -> list[PluginSpec]:
    try:
        active = get_active_repository()
    except Exception:
        active = None
    if not active:
        return []
    try:
        repo_root = resolve_project_path(active.project_path)
    except Exception:
        return []
    return [
        *_load_config_plugins(repo_root / ".repooperator" / "config.json", source_type="repo"),
        *_load_config_plugins(repo_root / ".repooperator" / "plugins.json", source_type="repo"),
    ]


def _load_user_plugin_specs() -> list[PluginSpec]:
    try:
        settings = get_settings()
    except Exception:
        return []
    home = settings.repooperator_home_dir
    return [
        *_load_config_plugins(settings.repooperator_config_path, source_type="user"),
        *_load_config_plugins(home / "plugins.json", source_type="user"),
    ]


def _load_config_plugins(path: Path, *, source_type: str) -> list[PluginSpec]:
    payload = _read_json(path)
    if not payload:
        return []
    raw_plugins = _extract_plugin_items(payload)
    specs: list[PluginSpec] = []
    for index, item in enumerate(raw_plugins):
        if isinstance(item, str):
            item = {"id": item, "name": item}
        if not isinstance(item, dict):
            continue
        item = {**item, "source_type": source_type, "source_path": str(path)}
        if not item.get("id"):
            item["id"] = item.get("name") or f"{path.stem}_{index}"
        specs.append(plugin_spec_from_dict(item))
    return specs


def _extract_plugin_items(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    candidates = [
        payload.get("plugins"),
        (payload.get("agent") or {}).get("plugins") if isinstance(payload.get("agent"), dict) else None,
        (payload.get("repooperator") or {}).get("plugins") if isinstance(payload.get("repooperator"), dict) else None,
    ]
    for candidate in candidates:
        if isinstance(candidate, list):
            return candidate
        if isinstance(candidate, dict):
            return [{"id": key, **(value if isinstance(value, dict) else {"name": str(value)})} for key, value in candidate.items()]
    return []


def _normalize_tool_metadata(tool: Any, *, plugin_id: str, provider: str) -> dict[str, Any]:
    if isinstance(tool, str):
        payload = {"name": tool}
    elif isinstance(tool, dict):
        payload = dict(tool)
    else:
        payload = {"name": str(tool)}
    name = str(payload.get("name") or payload.get("id") or "").strip()
    tool_id = str(payload.get("id") or name)
    return json_safe(
        {
            "id": _slug(tool_id),
            "name": name or _slug(tool_id),
            "description": str(payload.get("description") or ""),
            "input_schema": payload.get("input_schema") if isinstance(payload.get("input_schema"), dict) else {},
            "permissions": [str(value) for value in payload.get("permissions") or [] if str(value).strip()],
            "required_capabilities": [str(value) for value in payload.get("required_capabilities") or [] if str(value).strip()],
            "read_only": bool(payload.get("read_only", False)),
            "network_access": bool(payload.get("network_access", False)),
            "plugin_id": plugin_id,
            "provider": provider,
            "source": "plugin",
            "executable": False,
        }
    )


def _tool_hint(tool: dict[str, Any]) -> dict[str, Any]:
    return json_safe(
        {
            "id": tool.get("id"),
            "name": tool.get("name"),
            "description": tool.get("description"),
            "permissions": tool.get("permissions") or [],
            "required_capabilities": tool.get("required_capabilities") or [],
            "read_only": bool(tool.get("read_only")),
            "network_access": bool(tool.get("network_access")),
            "executable": False,
        }
    )


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [{"id": key, **(item if isinstance(item, dict) else {"description": str(item)})} for key, item in value.items()]
    if isinstance(value, list):
        result: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict):
                result.append(json_safe(item))
            elif isinstance(item, str):
                result.append({"name": item})
        return result
    return []


def _tool_score(tool: dict[str, Any], query_terms: set[str]) -> int:
    if not query_terms:
        return 1
    haystack = _terms(
        " ".join(
            [
                str(tool.get("id") or ""),
                str(tool.get("name") or ""),
                str(tool.get("description") or ""),
                " ".join(str(item) for item in tool.get("required_capabilities") or []),
                " ".join(str(item) for item in tool.get("permissions") or []),
            ]
        )
    )
    matches = query_terms.intersection(haystack)
    return 20 + len(matches) * 3 if matches else 0


def _read_json(path: Path) -> Any:
    if not path.exists() or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _terms(text: str) -> set[str]:
    normalized = re.sub(r"[^A-Za-z0-9_./-]+", " ", str(text or "").lower().replace("-", "_"))
    terms: set[str] = set()
    for part in normalized.split():
        if not part:
            continue
        terms.add(part)
        for subpart in part.replace("/", "_").replace(".", "_").split("_"):
            if subpart:
                terms.add(subpart)
    return terms


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", str(value or "").strip().lower().replace("-", "_"))
    return re.sub(r"_+", "_", text).strip("_")
