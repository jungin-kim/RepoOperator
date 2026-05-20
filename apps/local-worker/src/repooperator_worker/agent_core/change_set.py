from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from repooperator_worker.agent_core.tools.builtin import is_supported_text_file
from repooperator_worker.services.common import resolve_project_path
from repooperator_worker.services.json_safe import json_safe


ChangeOperation = Literal["modify", "create", "delete", "rename"]


@dataclass
class ProposedFileChange:
    path: str
    operation: ChangeOperation
    summary: str
    original_content: str | None = None
    proposed_content: str | None = None
    risk_notes: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return json_safe(self.__dict__)


@dataclass
class ChangePlan:
    summary: str
    target_files: list[str] = field(default_factory=list)
    operations: list[ChangeOperation] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return json_safe(self.__dict__)


@dataclass
class ValidationResult:
    status: Literal["valid", "invalid"]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return json_safe(self.__dict__)


@dataclass
class ChangeSetProposal:
    plan: ChangePlan
    changes: list[ProposedFileChange] = field(default_factory=list)
    status: Literal["planned", "valid", "invalid", "blocked"] = "planned"
    validation: ValidationResult | None = None
    proposal_error: str | None = None

    def model_dump(self) -> dict[str, Any]:
        return json_safe(
            {
                "plan": self.plan.model_dump(),
                "changes": [change.model_dump() for change in self.changes],
                "status": self.status,
                "validation": self.validation.model_dump() if self.validation else None,
                "proposal_error": self.proposal_error,
            }
        )


def plan_change_set(target_files: list[str], summary: str) -> ChangeSetProposal:
    return ChangeSetProposal(
        plan=ChangePlan(
            summary=summary or "Prepare a proposal-only change set.",
            target_files=list(target_files),
            operations=["modify"] if target_files else [],
        )
    )


def proposal_from_edit_result(
    edit_proposals: list[dict[str, Any]],
    *,
    repo: str,
    plan_summary: str = "",
) -> ChangeSetProposal:
    changes: list[ProposedFileChange] = []
    repo_path = resolve_project_path(repo).resolve()
    for item in edit_proposals:
        if not isinstance(item, dict):
            continue
        relative_path = str(item.get("file") or "").strip().lstrip("/")
        original_content = _read_original(repo_path, relative_path)
        changes.append(
            ProposedFileChange(
                path=relative_path,
                operation="modify",
                summary=str(item.get("summary") or "Modify file content."),
                original_content=original_content,
                proposed_content=str(item.get("proposed_content") or ""),
                risk_notes=[str(note) for note in item.get("risk_notes") or []],
            )
        )
    proposal = ChangeSetProposal(
        plan=ChangePlan(
            summary=plan_summary or "Prepare validated proposal-only edits.",
            target_files=[change.path for change in changes],
            operations=["modify"] if changes else [],
        ),
        changes=changes,
    )
    validation = validate_change_set(proposal, repo=repo)
    proposal.validation = validation
    proposal.status = "valid" if validation.status == "valid" else "invalid"
    proposal.proposal_error = "; ".join(validation.errors) if validation.errors else None
    return proposal


def validate_change_set(proposal: ChangeSetProposal, *, repo: str) -> ValidationResult:
    errors: list[str] = []
    repo_path = resolve_project_path(repo).resolve()
    if not proposal.changes:
        errors.append("change set has no file changes")
    seen: set[str] = set()
    for change in proposal.changes:
        if not change.path or Path(change.path).is_absolute() or ".." in Path(change.path).parts:
            errors.append(f"{change.path or '<missing>'}: path must stay inside the repository")
            continue
        if change.path in seen:
            errors.append(f"{change.path}: duplicate file change")
        seen.add(change.path)
        if ".git" in Path(change.path).parts:
            errors.append(f"{change.path}: .git edits are not allowed")
        target = (repo_path / change.path).resolve()
        try:
            target.relative_to(repo_path)
        except ValueError:
            errors.append(f"{change.path}: resolved path escapes repository")
            continue
        if change.operation == "delete":
            errors.append(f"{change.path}: delete proposals require explicit protected-delete approval")
            continue
        if change.operation == "rename":
            errors.append(f"{change.path}: rename proposals are deferred in this graph path")
            continue
        if change.operation == "modify":
            if not target.is_file():
                errors.append(f"{change.path}: modify target does not exist")
                continue
            if not is_supported_text_file(target):
                errors.append(f"{change.path}: binary or unsupported file edits are not allowed")
                continue
            if change.proposed_content is None:
                errors.append(f"{change.path}: proposed content is missing")
            if change.original_content is not None and change.proposed_content == change.original_content:
                errors.append(f"{change.path}: proposed content is unchanged")
        if change.operation == "create":
            if target.exists():
                errors.append(f"{change.path}: create target already exists")
            if change.proposed_content is None:
                errors.append(f"{change.path}: proposed content is missing")
    return ValidationResult(status="invalid" if errors else "valid", errors=errors)


def _read_original(repo_path: Path, relative_path: str) -> str | None:
    if not relative_path:
        return None
    target = (repo_path / relative_path).resolve()
    try:
        target.relative_to(repo_path)
    except ValueError:
        return None
    if not target.is_file() or not is_supported_text_file(target):
        return None
    return target.read_text(encoding="utf-8", errors="replace")
