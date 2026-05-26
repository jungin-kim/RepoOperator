from __future__ import annotations

import shlex
import subprocess
import time
import uuid
from dataclasses import dataclass
from typing import Any

from repooperator_worker.config import PERMISSION_MODE_FULL_ACCESS
from repooperator_worker.services.active_repository import get_active_repository
from repooperator_worker.services.common import resolve_project_path
from repooperator_worker.services.event_service import record_event
from repooperator_worker.services.permissions_service import permission_profile

_SESSION_APPROVALS: dict[str, dict[str, Any]] = {}

READ_ONLY_PREFIXES = {
    ("pwd",),
    ("ls",),
    ("find",),
    ("git", "status"),
    ("git", "branch"),
    ("git", "diff"),
    ("git", "log"),
    ("git", "remote"),
    ("git", "rev-parse"),
    ("glab", "mr", "list"),
    ("glab", "mr", "view"),
    ("glab", "pipeline", "list"),
    ("glab", "auth", "status"),
}

APPROVAL_PREFIXES = {
    ("npm", "install"),
    ("pip", "install"),
    ("brew", "install"),
    ("curl",),
    ("wget",),
    ("git", "checkout"),
    ("git", "add"),
    ("git", "commit"),
    ("git", "push"),
    ("git", "branch"),
    ("glab", "mr", "create"),
    ("glab", "mr", "update"),
}

DANGEROUS_PREFIXES = {
    ("sudo",),
    ("rm", "-rf"),
    ("chmod",),
    ("chown",),
    ("git", "reset"),
    ("git", "clean"),
}


@dataclass(frozen=True)
class CommandPreview:
    approval_id: str
    command: list[str]
    display_command: str
    cwd: str | None
    risk: str
    read_only: bool
    needs_network: bool
    touches_outside_repo: bool
    needs_approval: bool
    blocked: bool
    reason: str
    pattern: str


def preview_command(
    argv: list[str] | str,
    *,
    reason: str | None = None,
    project_path: str | None = None,
) -> dict[str, Any]:
    preview = _classify_command(_parse_argv(argv), reason=reason, project_path=project_path)
    record_event(
        event_type="command_preview",
        repo=preview.cwd,
        status="blocked" if preview.blocked else "ok",
        summary=preview.display_command,
        command=preview.command,
        tool=preview.command[0] if preview.command else None,
        error=preview.reason if preview.blocked else None,
    )
    return _preview_payload(preview)


def run_command_with_policy(
    argv: list[str] | str,
    *,
    approval_id: str | None = None,
    remember_for_session: bool = False,
    reason: str | None = None,
    project_path: str | None = None,
) -> dict[str, Any]:
    preview = _classify_command(_parse_argv(argv), reason=reason, project_path=project_path)
    if preview.blocked:
        record_event(
            event_type="command_denied",
            repo=preview.cwd,
            status="blocked",
            summary=preview.display_command,
            command=preview.command,
            tool=preview.command[0] if preview.command else None,
            error=preview.reason,
        )
        raise ValueError(preview.reason)

    approval = _find_session_approval(preview)
    if preview.needs_approval and approval_id != preview.approval_id and approval is None:
        raise PermissionError("Command requires approval before it can run.")

    if remember_for_session:
        _SESSION_APPROVALS[preview.approval_id] = {
            "id": preview.approval_id,
            "repo": preview.cwd,
            "pattern": preview.pattern,
            "risk": preview.risk,
            "created_at": int(time.time()),
        }

    started = time.perf_counter()
    result = subprocess.run(
        preview.command,
        cwd=preview.cwd,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    latency_ms = int((time.perf_counter() - started) * 1000)
    payload = {
        **_preview_payload(preview),
        "status": "ok" if result.returncode == 0 else "error",
        "exit_code": result.returncode,
        "stdout": _redact(result.stdout[-16_000:]),
        "stderr": _redact(result.stderr[-8_000:]),
        "duration_ms": latency_ms,
    }
    record_event(
        event_type="command_run",
        repo=preview.cwd,
        status=payload["status"],
        summary=preview.display_command,
        command=preview.command,
        tool=preview.command[0] if preview.command else None,
        error=payload["stderr"][:500] if result.returncode else None,
    )
    return payload


def list_command_approvals() -> dict[str, Any]:
    return {"approvals": list(_SESSION_APPROVALS.values())}


def revoke_command_approval(approval_id: str) -> dict[str, Any]:
    removed = _SESSION_APPROVALS.pop(approval_id, None)
    return {"revoked": bool(removed), "approval_id": approval_id}


def _classify_command(
    argv: list[str],
    *,
    reason: str | None = None,
    project_path: str | None = None,
) -> CommandPreview:
    try:
        active = get_active_repository()
    except Exception:
        active = None
    repo_path = resolve_project_path(project_path) if project_path else (resolve_project_path(active.project_path) if active else None)
    display = shlex.join(argv)
    profile = permission_profile()
    pattern = _pattern_for(argv)
    approval_id = "cmd_" + uuid.uuid5(uuid.NAMESPACE_URL, f"{repo_path}:{pattern}").hex[:12]

    if not argv:
        return _preview(approval_id, argv, display, repo_path, "high", False, False, False, True, True, "Command is empty.", pattern)
    if repo_path is None:
        return _preview(approval_id, argv, display, repo_path, "high", False, False, False, True, True, "Open a repository before running local commands.", pattern)
    outside_repo = _touches_outside_repo(argv, str(repo_path))
    sandbox = profile.get("sandbox") if isinstance(profile.get("sandbox"), dict) else {}
    if not sandbox.get("allowCommandRun", True):
        return _preview(approval_id, argv, display, repo_path, "high", False, False, outside_repo, True, True, "Command execution is disabled by the current permission profile.", pattern)
    if outside_repo and profile["mode"] != PERMISSION_MODE_FULL_ACCESS:
        return _preview(approval_id, argv, display, repo_path, "high", False, False, True, True, True, "Commands that access paths outside the active repository are blocked unless full_access is active.", pattern)
    if _has_secret_dump(argv):
        return _preview(approval_id, argv, display, repo_path, "high", False, False, False, True, True, "Commands that may expose secrets are blocked.", pattern)
    policy_argv = _policy_argv(argv)
    if _matches(policy_argv, DANGEROUS_PREFIXES):
        return _preview(approval_id, argv, display, repo_path, "high", False, False, False, True, True, "This destructive command is blocked by default.", pattern)

    needs_network = policy_argv[0] in {"curl", "wget", "brew"} or tuple(policy_argv[:2]) in {("npm", "install"), ("pip", "install")}
    read_only = _is_read_only_command(policy_argv)
    needs_approval = not read_only or _matches(policy_argv, APPROVAL_PREFIXES) or needs_network
    if read_only:
        needs_approval = False

    risk = "low" if read_only else "medium"
    if needs_network or tuple(argv[:2]) in {("git", "push"), ("glab", "mr")}:
        risk = "medium"
    reason_text = reason or (
        "Safe repository command." if read_only else "This command can modify state or require elevated access."
    )
    return _preview(approval_id, argv, display, repo_path, risk, read_only, needs_network, outside_repo, needs_approval, False, reason_text, pattern)


def _preview(
    approval_id: str,
    argv: list[str],
    display: str,
    repo_path,
    risk: str,
    read_only: bool,
    needs_network: bool,
    outside_repo: bool,
    needs_approval: bool,
    blocked: bool,
    reason: str,
    pattern: str,
) -> CommandPreview:
    return CommandPreview(
        approval_id=approval_id,
        command=argv,
        display_command=display,
        cwd=str(repo_path) if repo_path else None,
        risk=risk,
        read_only=read_only,
        needs_network=needs_network,
        touches_outside_repo=outside_repo,
        needs_approval=needs_approval,
        blocked=blocked,
        reason=reason,
        pattern=pattern,
    )


def _preview_payload(preview: CommandPreview) -> dict[str, Any]:
    return {
        "type": "command_approval",
        "approval_id": preview.approval_id,
        "command": preview.command,
        "display_command": preview.display_command,
        "cwd": preview.cwd,
        "risk": preview.risk,
        "read_only": preview.read_only,
        "needs_network": preview.needs_network,
        "touches_outside_repo": preview.touches_outside_repo,
        "needs_approval": preview.needs_approval,
        "blocked": preview.blocked,
        "reason": preview.reason,
        "pattern": preview.pattern,
        "options": ["yes", "yes_session", "no_explain"],
    }


def _parse_argv(argv: list[str] | str) -> list[str]:
    if isinstance(argv, str):
        return shlex.split(argv)
    return [str(part) for part in argv]


def _matches(argv: list[str], prefixes: set[tuple[str, ...]]) -> bool:
    lowered = [part.lower() for part in argv]
    return any(tuple(lowered[: len(prefix)]) == prefix for prefix in prefixes)


def _is_read_only_command(argv: list[str]) -> bool:
    lowered = [part.lower() for part in argv]
    if not lowered:
        return False
    if tuple(lowered[:1]) in {("pwd",), ("ls",), ("find",)}:
        return True
    if tuple(lowered[:2]) in {
        ("git", "status"),
        ("git", "diff"),
        ("git", "log"),
        ("git", "remote"),
        ("git", "rev-parse"),
    }:
        return True
    if tuple(lowered[:2]) == ("node", "--check") and len(lowered) >= 3:
        return True
    if tuple(lowered[:3]) == ("npm", "run", "typecheck"):
        return True
    if tuple(lowered[:3]) == ("glab", "auth", "status"):
        return True
    if tuple(lowered[:3]) in {
        ("glab", "mr", "list"),
        ("glab", "mr", "view"),
        ("glab", "pipeline", "list"),
    }:
        return True
    if tuple(lowered[:2]) == ("git", "branch"):
        if len(lowered) == 2:
            return True
        safe_flags = {"--show-current", "--list", "--format=%(refname:short)"}
        return all(part in safe_flags or part.startswith("--format=") for part in lowered[2:])
    return False


def _pattern_for(argv: list[str]) -> str:
    argv = _policy_argv(argv)
    if not argv:
        return ""
    if argv[0] in {"git", "glab"}:
        return " ".join(argv[:3])
    return " ".join(argv[:2])


def _find_session_approval(preview: CommandPreview) -> dict[str, Any] | None:
    for approval in _SESSION_APPROVALS.values():
        if (
            approval.get("repo") == preview.cwd
            and approval.get("pattern") == preview.pattern
            and approval.get("risk") == preview.risk
        ):
            return approval
    return None


def _has_secret_dump(argv: list[str]) -> bool:
    joined = " ".join(part.lower() for part in argv)
    return any(token in joined for token in {"printenv", "env", "token", "secret", "api_key", "apikey"})


def _touches_outside_repo(argv: list[str], repo_path: str) -> bool:
    for part in argv[1:]:
        if part.startswith("-"):
            continue
        if part.startswith("/") and not part.startswith(repo_path.rstrip("/") + "/") and part != repo_path:
            return True
        if part.startswith("..") or "/../" in part:
            return True
    return False


def _policy_argv(argv: list[str]) -> list[str]:
    if not argv or argv[0] != "git":
        return argv
    cleaned = ["git"]
    index = 1
    while index < len(argv):
        if argv[index] == "-c" and index + 1 < len(argv):
            index += 2
            continue
        cleaned.extend(argv[index:])
        break
    return cleaned


def _redact(text: str) -> str:
    redacted = text
    for marker in ("token", "api_key", "apikey", "secret"):
        redacted = redacted.replace(marker, "[redacted]")
        redacted = redacted.replace(marker.upper(), "[redacted]")
    return redacted
