from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from repooperator_worker.agent_core.policies import command_policy_preview
from repooperator_worker.services.common import resolve_project_path
from repooperator_worker.services.json_safe import json_safe


@dataclass(frozen=True)
class ValidationCommandCandidate:
    command: list[str]
    reason: str
    safety_classification: str
    language: str
    project_type: str
    requires_approval: bool
    blocked: bool = False
    display_command: str = ""
    preview: dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        return json_safe(
            {
                "command": self.command,
                "display_command": self.display_command or shlex.join(self.command),
                "reason": self.reason,
                "safety_classification": self.safety_classification,
                "language": self.language,
                "project_type": self.project_type,
                "requires_approval": self.requires_approval,
                "blocked": self.blocked,
                "preview": self.preview,
            }
        )


@dataclass(frozen=True)
class ValidationCommandSelection:
    changed_files: list[str]
    repo_files: list[str]
    language: str
    project_type: str
    user_request: str
    available_scripts: dict[str, str]
    permission_mode: str
    candidates: list[ValidationCommandCandidate]
    selected_index: int | None
    reason: str

    @property
    def selected(self) -> ValidationCommandCandidate | None:
        if self.selected_index is None:
            return None
        if self.selected_index < 0 or self.selected_index >= len(self.candidates):
            return None
        return self.candidates[self.selected_index]

    def model_dump(self) -> dict[str, Any]:
        return json_safe(
            {
                "changed_files": self.changed_files,
                "repo_files": self.repo_files,
                "language": self.language,
                "project_type": self.project_type,
                "user_request": self.user_request,
                "available_scripts": self.available_scripts,
                "permission_mode": self.permission_mode,
                "candidates": [candidate.model_dump() for candidate in self.candidates],
                "selected_index": self.selected_index,
                "selected": self.selected.model_dump() if self.selected else None,
                "reason": self.reason,
            }
        )


class ValidationCommandSelector:
    """Select post-apply validation commands from repository facts and policy previews."""

    def select(
        self,
        *,
        project_path: str,
        changed_files: list[str],
        user_request: str,
        permission_mode: str,
    ) -> ValidationCommandSelection:
        repo = resolve_project_path(project_path)
        normalized_changed = _dedupe_paths(changed_files)
        repo_files = _repo_file_inventory(repo)
        scripts = _package_scripts(repo)
        language = _language_for(normalized_changed, repo_files)
        project_type = _project_type_for(repo, normalized_changed, repo_files, scripts)
        candidates = self._candidate_commands(
            repo=repo,
            changed_files=normalized_changed,
            repo_files=repo_files,
            scripts=scripts,
            language=language,
            project_type=project_type,
        )
        selected_index = next((index for index, candidate in enumerate(candidates) if not candidate.blocked), None)
        if selected_index is None:
            reason = "No runnable validation command was selected from the changed files and repository configuration."
        else:
            selected = candidates[selected_index]
            reason = f"Selected `{selected.display_command or shlex.join(selected.command)}` because {selected.reason}"
        return ValidationCommandSelection(
            changed_files=normalized_changed,
            repo_files=repo_files,
            language=language,
            project_type=project_type,
            user_request=user_request,
            available_scripts=scripts,
            permission_mode=permission_mode,
            candidates=candidates,
            selected_index=selected_index,
            reason=reason,
        )

    def _candidate_commands(
        self,
        *,
        repo: Path,
        changed_files: list[str],
        repo_files: list[str],
        scripts: dict[str, str],
        language: str,
        project_type: str,
    ) -> list[ValidationCommandCandidate]:
        candidates: list[tuple[list[str], str, str, str]] = []
        py_files = [path for path in changed_files if path.endswith(".py")]
        js_files = [path for path in changed_files if path.endswith(".js")]
        ts_files = [path for path in changed_files if path.endswith((".ts", ".tsx"))]

        if py_files:
            candidates.append(
                (
                    ["python", "-m", "py_compile", *py_files],
                    "changed Python files can be syntax-checked before broader tests.",
                    "syntax_only",
                    "python",
                )
            )
            if _python_tests_exist(repo, repo_files):
                candidates.append((["pytest"], "Python tests are present in the repository.", "full_test_suite", "python"))

        for path in js_files:
            candidates.append(
                (
                    ["node", "--check", path],
                    f"`{path}` is a changed JavaScript file and supports syntax-only checking.",
                    "syntax_only",
                    "node",
                )
            )

        if scripts:
            if ts_files and "typecheck" in scripts:
                candidates.append((["npm", "run", "typecheck"], "TypeScript changed and a typecheck script exists.", "typecheck", "typescript"))
            if (ts_files or js_files or project_type in {"node", "typescript"}) and "build" in scripts:
                candidates.append((["npm", "run", "build"], "A build script exists for this Node project.", "build", "node"))
            if (ts_files or js_files or project_type in {"node", "typescript"}) and "test" in scripts:
                candidates.append((["npm", "test"], "A test script exists for this Node project.", "full_test_suite", "node"))

        candidates.append((["git", "diff", "--check"], "generic whitespace validation is read-only and applies to any repository.", "read_only", "generic"))

        materialized: list[ValidationCommandCandidate] = []
        seen: set[tuple[str, ...]] = set()
        for command, reason, safety, candidate_language in candidates:
            key = tuple(command)
            if key in seen:
                continue
            seen.add(key)
            materialized.append(
                _candidate_with_policy(
                    repo=repo,
                    command=command,
                    reason=reason,
                    safety=safety,
                    language=candidate_language if candidate_language != "generic" else language,
                    project_type=project_type,
                )
            )
        return materialized


def _candidate_with_policy(
    *,
    repo: Path,
    command: list[str],
    reason: str,
    safety: str,
    language: str,
    project_type: str,
) -> ValidationCommandCandidate:
    try:
        preview = command_policy_preview(command, project_path=str(repo), reason=reason)
    except Exception as exc:  # noqa: BLE001
        preview = {"blocked": True, "needs_approval": True, "reason": str(exc), "command": command, "display_command": shlex.join(command)}
    return ValidationCommandCandidate(
        command=[str(part) for part in command],
        display_command=str(preview.get("display_command") or shlex.join(command)),
        reason=str(preview.get("reason") or reason),
        safety_classification=safety if not preview.get("blocked") else "blocked",
        language=language,
        project_type=project_type,
        requires_approval=bool(preview.get("needs_approval")),
        blocked=bool(preview.get("blocked")),
        preview=json_safe(preview),
    )


def _repo_file_inventory(repo: Path) -> list[str]:
    files: list[str] = []
    for path in sorted(repo.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(repo))
        if relative.startswith(".git/") or "/.git/" in relative:
            continue
        if any(part in {"node_modules", ".venv", "__pycache__", ".next", "dist", "build"} for part in path.relative_to(repo).parts):
            continue
        files.append(relative)
        if len(files) >= 500:
            break
    return files


def _package_scripts(repo: Path) -> dict[str, str]:
    package_json = repo / "package.json"
    if not package_json.exists():
        return {}
    try:
        payload = json.loads(package_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    scripts = payload.get("scripts") if isinstance(payload, dict) else {}
    if not isinstance(scripts, dict):
        return {}
    return {str(key): str(value) for key, value in scripts.items() if str(key)}


def _python_tests_exist(repo: Path, repo_files: list[str]) -> bool:
    return (repo / "tests").is_dir() or any(Path(path).name.startswith("test_") and path.endswith(".py") for path in repo_files)


def _language_for(changed_files: list[str], repo_files: list[str]) -> str:
    paths = [*changed_files, *repo_files[:80]]
    if any(path.endswith((".ts", ".tsx")) for path in paths):
        return "typescript"
    if any(path.endswith((".js", ".jsx")) for path in paths):
        return "node"
    if any(path.endswith(".py") for path in paths):
        return "python"
    return "generic"


def _project_type_for(repo: Path, changed_files: list[str], repo_files: list[str], scripts: dict[str, str]) -> str:
    if (repo / "tsconfig.json").exists() or any(path.endswith((".ts", ".tsx")) for path in [*changed_files, *repo_files]):
        return "typescript"
    if scripts or (repo / "package.json").exists() or any(path.endswith((".js", ".jsx")) for path in [*changed_files, *repo_files]):
        return "node"
    if any((repo / name).exists() for name in ("pyproject.toml", "setup.py", "setup.cfg", "requirements.txt")) or any(
        path.endswith(".py") for path in [*changed_files, *repo_files]
    ):
        return "python"
    return "generic"


def _dedupe_paths(paths: list[str]) -> list[str]:
    result: list[str] = []
    for path in paths:
        normalized = str(path).strip().lstrip("/")
        if normalized and normalized not in result:
            result.append(normalized)
    return result
