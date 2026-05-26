from __future__ import annotations

import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from repooperator_worker.agent_core.change_set import ChangeSetProposal, ProposedFileChange, change_set_from_payload, validate_change_set
from repooperator_worker.agent_core.tools.builtin import is_supported_text_file
from repooperator_worker.services.command_service import preview_command
from repooperator_worker.services.common import ensure_git_repository, get_repooperator_home_dir, resolve_project_path
from repooperator_worker.services.json_safe import json_safe
from repooperator_worker.services.subprocess_utils import run_subprocess


@dataclass
class WorktreeSandbox:
    project_path: str
    worktree_path: str
    sandbox_root: str
    base_ref: str
    created_at: float = field(default_factory=time.time)

    def model_dump(self) -> dict[str, Any]:
        return json_safe(self.__dict__)


@dataclass
class SandboxValidationResult:
    status: str
    worktree_path: str
    base_ref: str
    diff: str = ""
    commands: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return json_safe(self.__dict__)


class WorktreeSandboxService:
    def __init__(self, sandbox_root: str | Path | None = None) -> None:
        root = Path(sandbox_root).expanduser() if sandbox_root is not None else get_repooperator_home_dir() / "worktree-sandboxes"
        self.sandbox_root = root.resolve()

    def create_temp_worktree(self, project_path: str, *, base_ref: str = "HEAD") -> WorktreeSandbox:
        repo = resolve_project_path(project_path).resolve()
        ensure_git_repository(repo)
        self.sandbox_root.mkdir(parents=True, exist_ok=True)
        worktree_path = (self.sandbox_root / f"{repo.name}-{uuid.uuid4().hex[:12]}").resolve()
        self._ensure_safe_sandbox_path(worktree_path)
        rev = run_subprocess(command=["git", "rev-parse", "--verify", base_ref], cwd=repo, timeout_seconds=30)
        if rev.returncode != 0:
            raise RuntimeError((rev.stderr or rev.stdout or "Unable to resolve base ref.").strip())
        resolved_ref = rev.stdout.strip()
        add = run_subprocess(command=["git", "worktree", "add", "--detach", str(worktree_path), resolved_ref], cwd=repo, timeout_seconds=60)
        if add.returncode != 0:
            raise RuntimeError((add.stderr or add.stdout or "Unable to create temporary worktree.").strip())
        return WorktreeSandbox(project_path=str(repo), worktree_path=str(worktree_path), sandbox_root=str(self.sandbox_root), base_ref=resolved_ref)

    def apply_change_set_to_worktree(
        self,
        sandbox: WorktreeSandbox | dict[str, Any],
        proposal: ChangeSetProposal | dict[str, Any],
    ) -> SandboxValidationResult:
        sandbox_state = _sandbox_from_any(sandbox)
        worktree_path = self._safe_worktree_path(sandbox_state)
        proposal_model = _proposal_from_any(proposal)
        validation = validate_change_set(proposal_model, repo=str(worktree_path))
        if validation.status != "valid":
            return SandboxValidationResult(
                status=validation.status,
                worktree_path=str(worktree_path),
                base_ref=sandbox_state.base_ref,
                errors=list(validation.errors),
                warnings=list(validation.warnings),
            )
        try:
            for change in _deterministic_changes(proposal_model.changes):
                _apply_sandbox_change(worktree_path, change)
        except Exception as exc:  # noqa: BLE001
            return SandboxValidationResult(
                status="failed",
                worktree_path=str(worktree_path),
                base_ref=sandbox_state.base_ref,
                errors=[str(exc)],
                warnings=["Sandbox apply failed; the proposal remains viewable but should be treated as risky."],
            )
        return SandboxValidationResult(status="valid", worktree_path=str(worktree_path), base_ref=sandbox_state.base_ref)

    def run_validation_in_worktree(
        self,
        sandbox: WorktreeSandbox | dict[str, Any],
        commands: list[list[str]],
    ) -> SandboxValidationResult:
        sandbox_state = _sandbox_from_any(sandbox)
        worktree_path = self._safe_worktree_path(sandbox_state)
        command_results: list[dict[str, Any]] = []
        errors: list[str] = []
        warnings: list[str] = []
        for command in commands:
            argv = [str(part) for part in command if str(part)]
            if not argv:
                continue
            policy = preview_command(argv, project_path=str(worktree_path), reason="Worktree sandbox validation command.")
            if policy.get("blocked") or policy.get("needs_network") or policy.get("touches_outside_repo"):
                message = str(policy.get("reason") or "Validation command blocked by policy.")
                errors.append(message)
                command_results.append({"command": argv, "status": "blocked", "policy": policy, "stdout": "", "stderr": message, "exit_code": None})
                continue
            if policy.get("needs_approval"):
                message = "Validation command requires approval by command policy and was not run in the sandbox."
                errors.append(message)
                command_results.append({"command": argv, "status": "blocked", "policy": policy, "stdout": "", "stderr": message, "exit_code": None})
                continue
            result = run_subprocess(command=argv, cwd=worktree_path, timeout_seconds=120)
            status = "success" if result.returncode == 0 else "failed"
            if result.returncode != 0:
                errors.append((result.stderr or result.stdout or f"{argv[0]} exited with {result.returncode}").strip())
            command_results.append(
                {
                    "command": argv,
                    "status": status,
                    "policy": policy,
                    "stdout": result.stdout[-16_000:],
                    "stderr": result.stderr[-8_000:],
                    "exit_code": result.returncode,
                }
            )
        overall = "valid" if not errors else "failed"
        if any(item.get("status") == "blocked" for item in command_results):
            overall = "blocked"
        return SandboxValidationResult(
            status=overall,
            worktree_path=str(worktree_path),
            base_ref=sandbox_state.base_ref,
            commands=command_results,
            errors=errors,
            warnings=warnings,
        )

    def compute_diff_from_worktree(self, sandbox: WorktreeSandbox | dict[str, Any]) -> str:
        sandbox_state = _sandbox_from_any(sandbox)
        worktree_path = self._safe_worktree_path(sandbox_state)
        result = run_subprocess(command=["git", "diff", "--no-ext-diff", "--"], cwd=worktree_path, timeout_seconds=30)
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout or "Unable to compute sandbox diff.").strip())
        return result.stdout

    def cleanup_worktree(self, sandbox: WorktreeSandbox | dict[str, Any]) -> dict[str, Any]:
        sandbox_state = _sandbox_from_any(sandbox)
        worktree_path = self._safe_worktree_path(sandbox_state)
        repo = resolve_project_path(sandbox_state.project_path).resolve()
        removed = False
        if worktree_path.exists():
            remove = run_subprocess(command=["git", "worktree", "remove", "--force", str(worktree_path)], cwd=repo, timeout_seconds=60)
            if remove.returncode == 0:
                removed = True
            elif worktree_path.exists():
                shutil.rmtree(worktree_path)
                removed = True
        return {"removed": removed, "worktree_path": str(worktree_path)}

    def validate_proposal_in_sandbox(
        self,
        *,
        project_path: str,
        proposal: ChangeSetProposal | dict[str, Any],
        commands: list[list[str]] | None = None,
    ) -> dict[str, Any]:
        proposal_model = _proposal_from_any(proposal)
        sandbox = self.create_temp_worktree(project_path)
        result = SandboxValidationResult(status="failed", worktree_path=sandbox.worktree_path, base_ref=sandbox.base_ref)
        try:
            result = self.apply_change_set_to_worktree(sandbox, proposal_model)
            if result.status == "valid":
                diff = self.compute_diff_from_worktree(sandbox)
                command_result = self.run_validation_in_worktree(sandbox, commands or [])
                result = SandboxValidationResult(
                    status=command_result.status,
                    worktree_path=sandbox.worktree_path,
                    base_ref=sandbox.base_ref,
                    diff=diff,
                    commands=command_result.commands,
                    errors=command_result.errors,
                    warnings=command_result.warnings,
                )
            proposal_model.sandbox_validation = result.model_dump()
            if result.status != "valid":
                _mark_sandbox_risk(proposal_model, result)
            return proposal_model.model_dump()
        finally:
            self.cleanup_worktree(sandbox)

    def _safe_worktree_path(self, sandbox: WorktreeSandbox) -> Path:
        root = Path(sandbox.sandbox_root).expanduser().resolve()
        if root != self.sandbox_root:
            raise ValueError("Sandbox root does not match this service instance.")
        path = Path(sandbox.worktree_path).expanduser().resolve()
        self._ensure_safe_sandbox_path(path)
        return path

    def _ensure_safe_sandbox_path(self, path: Path) -> None:
        try:
            path.relative_to(self.sandbox_root)
        except ValueError as exc:
            raise ValueError("Worktree sandbox path must stay under the controlled sandbox root.") from exc
        if path == self.sandbox_root or not path.name:
            raise ValueError("Refusing to operate on the sandbox root itself.")


def _proposal_from_any(value: ChangeSetProposal | dict[str, Any]) -> ChangeSetProposal:
    if isinstance(value, ChangeSetProposal):
        return value
    return change_set_from_payload(value)


def _sandbox_from_any(value: WorktreeSandbox | dict[str, Any]) -> WorktreeSandbox:
    if isinstance(value, WorktreeSandbox):
        return value
    return WorktreeSandbox(
        project_path=str(value.get("project_path") or ""),
        worktree_path=str(value.get("worktree_path") or ""),
        sandbox_root=str(value.get("sandbox_root") or ""),
        base_ref=str(value.get("base_ref") or "HEAD"),
        created_at=float(value.get("created_at") or time.time()),
    )


def _deterministic_changes(changes: list[ProposedFileChange]) -> list[ProposedFileChange]:
    order = {"rename": 0, "delete": 1, "modify": 2, "create": 3}
    return sorted(changes, key=lambda change: (order.get(change.operation, 99), change.path))


def _apply_sandbox_change(worktree_path: Path, change: ProposedFileChange) -> None:
    target = _safe_change_target(worktree_path, change.path)
    if change.operation in {"modify", "delete", "rename"}:
        if target.is_symlink():
            raise RuntimeError(f"{change.path}: symlink targets are not writable through sandbox apply")
        if not target.is_file() or not is_supported_text_file(target):
            raise RuntimeError(f"{change.path}: sandbox target is missing or unsupported")
    if change.operation == "modify":
        target.write_text(str(change.proposed_content or ""), encoding="utf-8")
        return
    if change.operation == "create":
        if target.exists():
            raise RuntimeError(f"{change.path}: create target already exists in sandbox")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(change.proposed_content or ""), encoding="utf-8")
        return
    if change.operation == "delete":
        target.unlink()
        return
    if change.operation == "rename":
        if not change.rename_to:
            raise RuntimeError(f"{change.path}: rename target is missing")
        destination = _safe_change_target(worktree_path, change.rename_to)
        if destination.exists():
            raise RuntimeError(f"{change.rename_to}: rename target already exists in sandbox")
        destination.parent.mkdir(parents=True, exist_ok=True)
        target.rename(destination)
        return
    raise RuntimeError(f"{change.path}: unsupported change operation {change.operation}")


def _safe_change_target(worktree_path: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts or ".git" in path.parts:
        raise RuntimeError(f"{relative_path}: path must stay inside the sandbox worktree")
    target = (worktree_path / relative_path).resolve()
    try:
        target.relative_to(worktree_path)
    except ValueError as exc:
        raise RuntimeError(f"{relative_path}: path escapes sandbox worktree") from exc
    return target


def _mark_sandbox_risk(proposal: ChangeSetProposal, result: SandboxValidationResult) -> None:
    note = "Sandbox validation did not pass; review this proposal before applying."
    if result.errors:
        note = f"{note} {'; '.join(result.errors[:2])}"
    for change in proposal.changes:
        if note not in change.risk_notes:
            change.risk_notes.append(note)
    if proposal.validation:
        proposal.validation.warnings.append(note)
