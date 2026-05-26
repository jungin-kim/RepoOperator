from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from repooperator_worker.services.common import resolve_project_path
from repooperator_worker.services.json_safe import json_safe


IDE_CONTEXT_TTL_SECONDS = 300


@dataclass
class IDEBridgeState:
    active_file: str | None = None
    selected_text: str | None = None
    open_files: list[str] = field(default_factory=list)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    cursor_position: dict[str, Any] | None = None
    workspace_root: str | None = None
    branch: str | None = None
    editor: str | None = None
    timestamp: float = field(default_factory=time.time)

    def model_dump(self) -> dict[str, Any]:
        return json_safe(
            {
                "active_file": self.active_file,
                "selected_text": self.selected_text,
                "open_files": list(self.open_files),
                "diagnostics": list(self.diagnostics),
                "cursor_position": self.cursor_position,
                "workspace_root": self.workspace_root,
                "branch": self.branch,
                "editor": self.editor,
                "timestamp": self.timestamp,
            }
        )


_IDE_CONTEXTS: dict[tuple[str, str], IDEBridgeState] = {}


def update_ide_context(payload: IDEBridgeState | dict[str, Any]) -> dict[str, Any]:
    state = payload if isinstance(payload, IDEBridgeState) else _state_from_payload(payload)
    workspace_root = _normalise_workspace_root(state.workspace_root)
    if not workspace_root:
        raise ValueError("workspace_root is required for IDE bridge context.")
    state.workspace_root = workspace_root
    key = (workspace_root, state.branch or "")
    _IDE_CONTEXTS[key] = state
    return state.model_dump()


def get_ide_context(
    *,
    project_path: str | None = None,
    branch: str | None = None,
    now: float | None = None,
    ttl_seconds: int = IDE_CONTEXT_TTL_SECONDS,
) -> dict[str, Any] | None:
    state = _find_context(project_path=project_path, branch=branch)
    if state is None:
        return None
    current_time = time.time() if now is None else now
    if ttl_seconds >= 0 and current_time - float(state.timestamp or 0) > ttl_seconds:
        _IDE_CONTEXTS.pop((state.workspace_root or "", state.branch or ""), None)
        return None
    return _context_for_project(state, project_path)


def clear_ide_context(*, project_path: str | None = None, branch: str | None = None) -> dict[str, Any]:
    if project_path is None:
        count = len(_IDE_CONTEXTS)
        _IDE_CONTEXTS.clear()
        return {"cleared": count}
    workspace_root = _normalise_workspace_root(project_path)
    removed = 0
    for key in list(_IDE_CONTEXTS):
        key_root, key_branch = key
        if key_root != workspace_root:
            continue
        if branch is not None and key_branch != (branch or ""):
            continue
        _IDE_CONTEXTS.pop(key, None)
        removed += 1
    return {"cleared": removed}


def _state_from_payload(payload: dict[str, Any]) -> IDEBridgeState:
    workspace_root = payload.get("workspace_root") or payload.get("project_path")
    return IDEBridgeState(
        active_file=_clean_optional_text(payload.get("active_file"), limit=4_000),
        selected_text=_clean_optional_text(payload.get("selected_text"), limit=40_000),
        open_files=_clean_string_list(payload.get("open_files"), limit=80, item_limit=4_000),
        diagnostics=_clean_diagnostics(payload.get("diagnostics")),
        cursor_position=payload.get("cursor_position") if isinstance(payload.get("cursor_position"), dict) else None,
        workspace_root=_clean_optional_text(workspace_root, limit=4_000),
        branch=_clean_optional_text(payload.get("branch"), limit=300),
        editor=_clean_optional_text(payload.get("editor"), limit=120),
        timestamp=_coerce_timestamp(payload.get("timestamp")),
    )


def _find_context(*, project_path: str | None, branch: str | None) -> IDEBridgeState | None:
    if project_path is None:
        return next(reversed(_IDE_CONTEXTS.values()), None) if _IDE_CONTEXTS else None
    workspace_root = _normalise_workspace_root(project_path)
    if not workspace_root:
        return None
    exact = _IDE_CONTEXTS.get((workspace_root, branch or ""))
    if exact is not None:
        return exact
    unbranched = _IDE_CONTEXTS.get((workspace_root, ""))
    if unbranched is not None:
        return unbranched
    for (stored_root, _stored_branch), state in reversed(_IDE_CONTEXTS.items()):
        if stored_root == workspace_root:
            return state
    return None


def _context_for_project(state: IDEBridgeState, project_path: str | None) -> dict[str, Any]:
    payload = state.model_dump()
    repo_root = _safe_project_root(project_path or state.workspace_root)
    if repo_root is None:
        return payload
    active_file = _repo_relative(repo_root, state.active_file)
    open_files = [_repo_relative(repo_root, item) for item in state.open_files]
    payload["active_file"] = active_file
    payload["open_files"] = [item for item in open_files if item]
    payload["diagnostics"] = [_diagnostic_for_repo(repo_root, item) for item in state.diagnostics]
    payload["diagnostics"] = [item for item in payload["diagnostics"] if item]
    payload["workspace_root"] = str(repo_root)
    return json_safe(payload)


def _diagnostic_for_repo(repo_root: Path, item: dict[str, Any]) -> dict[str, Any]:
    diagnostic = {
        key: value
        for key, value in item.items()
        if key in {"path", "file", "message", "severity", "source", "code", "line", "column", "end_line", "end_column"}
    }
    path_value = diagnostic.get("path") or diagnostic.get("file")
    relative_path = _repo_relative(repo_root, str(path_value or ""))
    if relative_path:
        diagnostic["path"] = relative_path
        diagnostic.pop("file", None)
    return json_safe(diagnostic)


def _normalise_workspace_root(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return str(resolve_project_path(text).resolve())
    except ValueError:
        return str(Path(text).expanduser().resolve())


def _safe_project_root(value: Any) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return resolve_project_path(text).resolve()
    except ValueError:
        path = Path(text).expanduser()
        return path.resolve() if path.exists() and path.is_dir() else None


def _repo_relative(repo_root: Path, value: str | None) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    try:
        resolved = candidate.resolve()
        resolved.relative_to(repo_root)
    except ValueError:
        return None
    return str(resolved.relative_to(repo_root))


def _coerce_timestamp(value: Any) -> float:
    if value is None or value == "":
        return time.time()
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        pass
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except ValueError:
        return time.time()


def _clean_optional_text(value: Any, *, limit: int) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text[:limit]


def _clean_string_list(value: Any, *, limit: int, item_limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text[:item_limit])
        if len(result) >= limit:
            break
    return result


def _clean_diagnostics(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    diagnostics: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        diagnostics.append(json_safe(item))
        if len(diagnostics) >= 100:
            break
    return diagnostics
