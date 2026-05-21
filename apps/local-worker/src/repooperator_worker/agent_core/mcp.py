from __future__ import annotations

import json
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from repooperator_worker.agent_core.permissions import PermissionDecision, ToolPermissionContext
from repooperator_worker.agent_core.tools.base import BaseTool, ToolExecutionContext, ToolResult, ToolSpec
from repooperator_worker.config import get_settings
from repooperator_worker.services.active_repository import get_active_repository
from repooperator_worker.services.common import resolve_project_path
from repooperator_worker.services.json_safe import json_safe


@dataclass(frozen=True)
class MCPServerSpec:
    id: str
    name: str
    transport: str = "stdio"
    command: str | None = None
    args: list[str] = field(default_factory=list)
    url: str | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)
    enabled: bool = False
    config: dict[str, Any] = field(default_factory=dict)
    source_type: str = "config"
    source_path: str = ""

    def __post_init__(self) -> None:
        server_id = _slug(self.id or self.name)
        object.__setattr__(self, "id", server_id)
        object.__setattr__(self, "transport", str(self.transport or "stdio"))
        object.__setattr__(self, "args", [str(arg) for arg in self.args or []])
        object.__setattr__(self, "permissions", [str(item) for item in self.permissions or [] if str(item).strip()])
        object.__setattr__(self, "tools", [_normalize_tool_metadata(tool, server_id=server_id) for tool in self.tools or []])
        object.__setattr__(self, "enabled", bool(self.enabled))

    def model_dump(self) -> dict[str, Any]:
        return json_safe(
            {
                "id": self.id,
                "name": self.name,
                "transport": self.transport,
                "command": self.command,
                "args": list(self.args),
                "url": self.url,
                "tools": list(self.tools),
                "permissions": list(self.permissions),
                "enabled": self.enabled,
                "config": _redacted_config(self.config),
                "source_type": self.source_type,
                "source_path": self.source_path,
            }
        )

    def model_hint(self) -> dict[str, Any]:
        return json_safe(
            {
                "id": self.id,
                "name": self.name,
                "transport": self.transport,
                "tools": [_tool_hint(tool) for tool in self.tools],
                "permissions": list(self.permissions),
                "enabled": self.enabled,
            }
        )


class MCPToolAdapter(BaseTool):
    """Metadata-backed MCP tool adapter.

    This foundation does not start MCP servers or execute external tool code.
    The adapter exists so execution attempts still pass through ToolOrchestrator
    and permission policy before any future connector is attached.
    """

    def __init__(self, *, server: MCPServerSpec, tool_metadata: dict[str, Any]) -> None:
        self.server = server
        self.tool_metadata = json_safe(tool_metadata)
        self.spec = ToolSpec(
            name=_adapter_tool_name(server.id, str(tool_metadata.get("name") or tool_metadata.get("id") or "")),
            description=str(tool_metadata.get("description") or f"MCP tool metadata for {server.name}."),
            operation="custom",
            input_schema=tool_metadata.get("input_schema") if isinstance(tool_metadata.get("input_schema"), dict) else {},
            read_only=bool(tool_metadata.get("read_only", False)),
            concurrency_safe=False,
            requires_approval_by_default=True,
            side_effect_level="network" if tool_metadata.get("network_access") or server.transport in {"http", "sse"} else "command",
            is_destructive=not bool(tool_metadata.get("read_only", False)),
            is_open_world=True,
            workspace_bound=False,
            network_access=bool(tool_metadata.get("network_access") or server.transport in {"http", "sse"}),
            interrupt_behavior="approval",
            can_be_retried=False,
            idempotent=False,
            should_defer=True,
            always_load=False,
            tool_search_keywords=(
                "mcp",
                server.id,
                server.name,
                str(tool_metadata.get("name") or ""),
                *[str(item) for item in tool_metadata.get("required_capabilities") or []],
            ),
            capability_names=tuple(str(item) for item in tool_metadata.get("required_capabilities") or ["external_tool"]),
            prompt_summary=str(tool_metadata.get("description") or "External MCP tool metadata; execution requires permission."),
            input_schema_summary="External MCP input schema metadata only.",
            output_schema_summary="Execution is permission-gated and requires a configured MCP runtime adapter.",
            required_permissions=tuple(str(item) for item in tool_metadata.get("permissions") or server.permissions or ["external_tool"]),
            permission_required=True,
            parallel_safe=False,
        )

    def check_permission(self, payload: dict[str, Any], context: ToolPermissionContext) -> PermissionDecision:
        del context
        return PermissionDecision.ask(
            "MCP tool execution requires explicit permission and a configured runtime adapter.",
            approval_payload={
                "tool_name": self.spec.name,
                "server_id": self.server.id,
                "server_name": self.server.name,
                "tool_metadata": self.tool_metadata,
                "payload": json_safe(payload),
            },
            external_tool=True,
            mcp_server_id=self.server.id,
        )

    def call(self, payload: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        del payload, context
        return ToolResult(
            tool_name=self.spec.name,
            status="failed",
            observation="MCP execution is not implemented in this foundation; only metadata is loaded.",
            payload={"server": self.server.model_hint(), "tool_metadata": self.tool_metadata, "executed": False},
        )


class MCPRegistry:
    def __init__(self, servers: Iterable[MCPServerSpec | dict[str, Any]] | None = None) -> None:
        self._servers: OrderedDict[str, MCPServerSpec] = OrderedDict()
        for server in servers or []:
            self.register(server)

    def register(self, server: MCPServerSpec | dict[str, Any]) -> None:
        spec = server if isinstance(server, MCPServerSpec) else mcp_server_spec_from_dict(server)
        if not spec.id:
            raise ValueError("MCP server id is required.")
        self._servers[spec.id] = spec

    def get(self, server_id: str) -> MCPServerSpec:
        key = _slug(server_id)
        try:
            return self._servers[key]
        except KeyError as exc:
            raise KeyError(f"Unknown MCP server: {server_id}") from exc

    def servers(self) -> list[MCPServerSpec]:
        return list(self._servers.values())

    def enabled_servers(self) -> list[MCPServerSpec]:
        return [server for server in self.servers() if server.enabled]

    def list_configured_servers(self, *, enabled_only: bool = False) -> list[dict[str, Any]]:
        servers = self.enabled_servers() if enabled_only else self.servers()
        return json_safe([server.model_dump() for server in servers])

    def specs_for_model(self, *, enabled_only: bool = True) -> list[dict[str, Any]]:
        servers = self.enabled_servers() if enabled_only else self.servers()
        return json_safe([server.model_hint() for server in servers])

    def tool_metadata(self, *, enabled_only: bool = True) -> list[dict[str, Any]]:
        servers = self.enabled_servers() if enabled_only else self.servers()
        tools: list[dict[str, Any]] = []
        for server in servers:
            for tool in server.tools:
                tools.append(
                    json_safe(
                        {
                            **tool,
                            "server_id": server.id,
                            "server_name": server.name,
                            "server_transport": server.transport,
                            "permissions": list(tool.get("permissions") or server.permissions),
                            "enabled": server.enabled,
                            "source": "mcp",
                        }
                    )
                )
        return tools

    def tool_adapters(self, *, enabled_only: bool = True) -> list[MCPToolAdapter]:
        servers = self.enabled_servers() if enabled_only else self.servers()
        return [MCPToolAdapter(server=server, tool_metadata=tool) for server in servers for tool in server.tools]

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


def get_default_mcp_registry() -> MCPRegistry:
    return load_mcp_registry_from_default_configs()


def load_mcp_registry_from_default_configs() -> MCPRegistry:
    registry = MCPRegistry()
    for spec in _load_user_mcp_specs():
        registry.register(spec)
    for spec in _load_project_mcp_specs():
        registry.register(spec)
    return registry


def configured_mcp_tool_adapters() -> list[MCPToolAdapter]:
    return get_default_mcp_registry().tool_adapters(enabled_only=True)


def list_configured_mcp_servers(*, enabled_only: bool = False) -> list[dict[str, Any]]:
    return get_default_mcp_registry().list_configured_servers(enabled_only=enabled_only)


def mcp_server_spec_from_dict(item: dict[str, Any]) -> MCPServerSpec:
    return MCPServerSpec(
        id=str(item.get("id") or item.get("name") or ""),
        name=str(item.get("name") or item.get("id") or ""),
        transport=str(item.get("transport") or "stdio"),
        command=str(item.get("command")) if item.get("command") is not None else None,
        args=[str(arg) for arg in item.get("args") or []],
        url=str(item.get("url") or item.get("endpoint")) if (item.get("url") or item.get("endpoint")) else None,
        tools=_list_of_dicts(item.get("tools") or []),
        permissions=[str(value) for value in item.get("permissions") or [] if str(value).strip()],
        enabled=bool(item.get("enabled", False)),
        config=item.get("config") if isinstance(item.get("config"), dict) else {},
        source_type=str(item.get("source_type") or "config"),
        source_path=str(item.get("source_path") or ""),
    )


def _load_project_mcp_specs() -> list[MCPServerSpec]:
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
        *_load_config_mcp_servers(repo_root / ".repooperator" / "config.json", source_type="repo"),
        *_load_config_mcp_servers(repo_root / ".repooperator" / "mcp.json", source_type="repo"),
    ]


def _load_user_mcp_specs() -> list[MCPServerSpec]:
    try:
        settings = get_settings()
    except Exception:
        return []
    home = settings.repooperator_home_dir
    return [
        *_load_config_mcp_servers(settings.repooperator_config_path, source_type="user"),
        *_load_config_mcp_servers(home / "mcp.json", source_type="user"),
    ]


def _load_config_mcp_servers(path: Path, *, source_type: str) -> list[MCPServerSpec]:
    payload = _read_json(path)
    if not payload:
        return []
    raw_servers = _extract_mcp_items(payload)
    specs: list[MCPServerSpec] = []
    for index, item in enumerate(raw_servers):
        if isinstance(item, str):
            item = {"id": item, "name": item}
        if not isinstance(item, dict):
            continue
        item = {**item, "source_type": source_type, "source_path": str(path)}
        if not item.get("id"):
            item["id"] = item.get("name") or f"{path.stem}_{index}"
        specs.append(mcp_server_spec_from_dict(item))
    return specs


def _extract_mcp_items(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    mcp = payload.get("mcp")
    candidates = [
        payload.get("mcpServers"),
        payload.get("mcp_servers"),
        (mcp or {}).get("servers") if isinstance(mcp, dict) else None,
        (payload.get("agent") or {}).get("mcp_servers") if isinstance(payload.get("agent"), dict) else None,
    ]
    for candidate in candidates:
        if isinstance(candidate, list):
            return candidate
        if isinstance(candidate, dict):
            return [{"id": key, **(value if isinstance(value, dict) else {"name": str(value)})} for key, value in candidate.items()]
    return []


def _normalize_tool_metadata(tool: Any, *, server_id: str) -> dict[str, Any]:
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
            "server_id": server_id,
            "source": "mcp",
        }
    )


def _adapter_tool_name(server_id: str, tool_name: str) -> str:
    return f"mcp_{_slug(server_id)}_{_slug(tool_name)}"


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
                str(tool.get("server_id") or ""),
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


def _redacted_config(config: dict[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in json_safe(config).items():
        lowered = str(key).lower()
        if any(marker in lowered for marker in ("token", "secret", "password", "key")):
            redacted[str(key)] = "<redacted>"
        else:
            redacted[str(key)] = value
    return redacted


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
