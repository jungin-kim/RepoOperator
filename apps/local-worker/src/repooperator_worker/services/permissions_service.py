from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from repooperator_worker.config import (
    AVAILABLE_PERMISSION_MODES,
    PERMISSION_MODE_ACCEPT_EDITS,
    PERMISSION_MODE_AUTO_REVIEW,
    PERMISSION_MODE_AUTO_READONLY,
    PERMISSION_MODE_BASIC,
    PERMISSION_MODE_DEFAULT,
    PERMISSION_MODE_FULL_ACCESS,
    PERMISSION_MODE_HEADLESS_SAFE,
    PERMISSION_MODE_PLAN_ONLY,
    PERMISSION_MODE_PROPOSAL_ONLY,
    PERMISSION_MODE_ROUTINE_SAFE,
    WRITE_MODE_READ_ONLY,
    WRITE_MODE_AUTO_APPLY,
    get_settings,
)
from repooperator_worker.schemas import PermissionModeResponse


def permission_profile(mode: str | None = None) -> dict[str, Any]:
    selected = _normalize_mode(mode or get_settings().permission_mode)
    if selected == PERMISSION_MODE_FULL_ACCESS:
        return {
            "mode": PERMISSION_MODE_FULL_ACCESS,
            "coreMode": PERMISSION_MODE_FULL_ACCESS,
            "write_mode": WRITE_MODE_AUTO_APPLY,
            "sandbox": {
                "scope": "computer",
                "allowFileRead": True,
                "allowFileWrite": True,
                "allowCommandRun": True,
                "allowNetwork": True,
                "allowOutsideRepo": True,
            },
            "approval": {
                "requireForElevatedActions": False,
                "requireForNetwork": False,
                "requireForOutsideRepo": False,
                "requireForGitCommitPush": True,
                "requireForDestructiveCommands": True,
            },
            "tools": {
                "git": "read-write-with-guardrails",
                "glab": "approval-for-mutating-actions",
                "shell": "approved-local-commands",
            },
        }
    if selected in {PERMISSION_MODE_AUTO_REVIEW, PERMISSION_MODE_ACCEPT_EDITS, PERMISSION_MODE_ROUTINE_SAFE}:
        return {
            "mode": selected,
            "coreMode": PERMISSION_MODE_ACCEPT_EDITS if selected != PERMISSION_MODE_ROUTINE_SAFE else PERMISSION_MODE_ROUTINE_SAFE,
            "write_mode": WRITE_MODE_AUTO_APPLY,
            "sandbox": {
                "scope": "repository",
                "allowFileRead": True,
                "allowFileWrite": True,
                "allowCommandRun": True,
                "allowNetwork": False,
                "allowOutsideRepo": False,
            },
            "approval": {
                "requireForElevatedActions": True,
                "requireForNetwork": True,
                "requireForOutsideRepo": True,
                "requireForGitCommitPush": True,
                "requireForDestructiveCommands": True,
            },
            "tools": {
                "git": "safe-read-automatic-mutating-approval",
                "glab": "read-only-or-approval-required",
                "shell": "sandboxed-with-review",
            },
        }
    if selected in {PERMISSION_MODE_PLAN_ONLY, PERMISSION_MODE_PROPOSAL_ONLY, PERMISSION_MODE_AUTO_READONLY, PERMISSION_MODE_HEADLESS_SAFE}:
        return {
            "mode": selected,
            "coreMode": selected,
            "write_mode": WRITE_MODE_READ_ONLY,
            "sandbox": {
                "scope": "repository",
                "allowFileRead": True,
                "allowFileWrite": False,
                "allowCommandRun": selected != PERMISSION_MODE_PLAN_ONLY,
                "allowNetwork": False,
                "allowOutsideRepo": False,
            },
            "approval": {
                "requireForElevatedActions": True,
                "requireForNetwork": True,
                "requireForOutsideRepo": True,
                "requireForGitCommitPush": True,
                "requireForDestructiveCommands": True,
                "interactivePrompts": False if selected == PERMISSION_MODE_HEADLESS_SAFE else True,
            },
            "tools": {
                "git": "read-only" if selected == PERMISSION_MODE_AUTO_READONLY else "approval-gated",
                "glab": "approval-gated",
                "shell": "disabled" if selected in {PERMISSION_MODE_PLAN_ONLY, PERMISSION_MODE_HEADLESS_SAFE} else "approval-gated",
            },
        }
    return {
        "mode": selected if selected == PERMISSION_MODE_DEFAULT else PERMISSION_MODE_BASIC,
        "coreMode": PERMISSION_MODE_DEFAULT,
        "write_mode": WRITE_MODE_AUTO_APPLY,
        "sandbox": {
            "scope": "repository",
            "allowFileRead": True,
            "allowFileWrite": True,
            "allowCommandRun": True,
            "allowNetwork": False,
            "allowOutsideRepo": False,
        },
        "approval": {
            "requireForElevatedActions": True,
            "requireForNetwork": True,
            "requireForOutsideRepo": True,
            "requireForGitCommitPush": True,
            "requireForDestructiveCommands": True,
        },
        "tools": {
            "git": "safe-read-automatic-mutating-approval",
            "glab": "read-only-or-approval-required",
            "shell": "repository-sandboxed",
        },
    }


def get_permission_mode() -> PermissionModeResponse:
    profile = permission_profile()
    return PermissionModeResponse(
        mode=profile["mode"],
        write_mode=profile["write_mode"],
        available_modes=AVAILABLE_PERMISSION_MODES,
        unsupported_modes=[],
        sandbox=profile["sandbox"],
        approval=profile["approval"],
        tools=profile["tools"],
        profile=profile,
    )


def update_permission_mode(mode: str) -> PermissionModeResponse:
    selected = _normalize_mode(mode)
    settings = get_settings()
    config = _read_config(settings.repooperator_config_path)
    permissions = config.get("permissions")
    if not isinstance(permissions, dict):
        permissions = {}
    profile = permission_profile(selected)
    permissions["mode"] = selected
    permissions["writeMode"] = profile["write_mode"]
    config["permissions"] = permissions
    _write_config(settings.repooperator_config_path, config)
    return PermissionModeResponse(
        mode=selected,
        write_mode=profile["write_mode"],
        available_modes=AVAILABLE_PERMISSION_MODES,
        unsupported_modes=[],
        sandbox=profile["sandbox"],
        approval=profile["approval"],
        tools=profile["tools"],
        profile=profile,
    )


def _normalize_mode(value: str | None) -> str:
    if not value:
        return PERMISSION_MODE_DEFAULT
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "basic": PERMISSION_MODE_BASIC,
        "default": PERMISSION_MODE_DEFAULT,
        "read_only": PERMISSION_MODE_BASIC,
        "auto_readonly": PERMISSION_MODE_AUTO_READONLY,
        "auto_read_only": PERMISSION_MODE_AUTO_READONLY,
        "readonly": PERMISSION_MODE_AUTO_READONLY,
        "plan": PERMISSION_MODE_PLAN_ONLY,
        "plan_only": PERMISSION_MODE_PLAN_ONLY,
        "proposal": PERMISSION_MODE_PROPOSAL_ONLY,
        "proposal_only": PERMISSION_MODE_PROPOSAL_ONLY,
        "auto_review": PERMISSION_MODE_AUTO_REVIEW,
        "accept_edits": PERMISSION_MODE_ACCEPT_EDITS,
        "write_with_approval": PERMISSION_MODE_AUTO_REVIEW,
        "full_access": PERMISSION_MODE_FULL_ACCESS,
        "auto_apply": PERMISSION_MODE_FULL_ACCESS,
        "routine_safe": PERMISSION_MODE_ROUTINE_SAFE,
        "headless_safe": PERMISSION_MODE_HEADLESS_SAFE,
    }
    if normalized not in aliases:
        raise ValueError("Unsupported permission mode.")
    return aliases[normalized]


def _read_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_config(config_path: Path, config: dict) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = config_path.with_name(f"{config_path.name}.tmp")
    tmp_path.write_text(json.dumps(config, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    tmp_path.replace(config_path)
