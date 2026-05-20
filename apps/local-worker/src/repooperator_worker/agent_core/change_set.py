from __future__ import annotations

import ast
import difflib
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from repooperator_worker.agent_core.tools.builtin import (
    BINARY_OR_CACHE_SUFFIXES,
    TEXT_FILE_BASENAMES,
    TEXT_FILE_SUFFIXES,
    is_supported_text_file,
)
from repooperator_worker.services.common import resolve_project_path
from repooperator_worker.services.json_safe import json_safe


ChangeOperation = Literal["modify", "create", "delete", "rename"]
EditMode = Literal["explanation_only", "proposal_only", "approval_required", "apply_approved", "applied", "blocked"]
ProposalStatus = Literal["planned", "valid", "invalid", "repairable", "blocked", "applied", "rejected"]
ValidationStatus = Literal["valid", "invalid", "repairable", "blocked"]


@dataclass
class ProposedFileChange:
    path: str
    operation: ChangeOperation
    summary: str
    original_content: str | None = None
    proposed_content: str | None = None
    rename_to: str | None = None
    delete_justification: str | None = None
    risk_notes: list[str] = field(default_factory=list)
    additions: int = 0
    deletions: int = 0
    validation_status: str | None = None

    def model_dump(self) -> dict[str, Any]:
        return json_safe(self.__dict__)


@dataclass
class ChangePlan:
    summary: str
    target_files: list[str] = field(default_factory=list)
    operations: list[ChangeOperation] = field(default_factory=list)
    evidence_files: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    validation_requirements: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return json_safe(self.__dict__)


@dataclass
class ValidationResult:
    status: ValidationStatus
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    repairable: bool = False

    def model_dump(self) -> dict[str, Any]:
        return json_safe(self.__dict__)


@dataclass
class ChangeSetProposal:
    plan: ChangePlan
    proposal_id: str = ""
    changes: list[ProposedFileChange] = field(default_factory=list)
    status: ProposalStatus = "planned"
    validation: ValidationResult | None = None
    proposal_error: str | None = None
    validation_status: str = "pending"
    applied: bool = False
    applied_change_set_id: str | None = None
    apply_status: str | None = None
    post_apply_validation_status: str | None = None
    applied_at: str | None = None

    def model_dump(self) -> dict[str, Any]:
        proposal_id = self.proposal_id or stable_proposal_id(self.plan.summary, self.plan.target_files)
        return json_safe(
            {
                "proposal_id": proposal_id,
                "plan": self.plan.model_dump(),
                "changes": [change.model_dump() for change in self.changes],
                "status": self.status,
                "validation": self.validation.model_dump() if self.validation else None,
                "proposal_error": self.proposal_error,
                "validation_status": self.validation_status,
                "applied": self.applied,
                "applied_change_set_id": self.applied_change_set_id,
                "apply_status": self.apply_status,
                "post_apply_validation_status": self.post_apply_validation_status,
                "applied_at": self.applied_at,
            }
        )


def stable_proposal_id(summary: str, paths: list[str]) -> str:
    digest = hashlib.sha256(json.dumps({"summary": summary, "paths": paths}, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"proposal:{digest}"


def plan_change_set(target_files: list[str], summary: str) -> ChangeSetProposal:
    cleaned_targets = [str(item).strip().lstrip("/") for item in target_files if str(item).strip()]
    return ChangeSetProposal(
        proposal_id=stable_proposal_id(summary or "change-set", cleaned_targets),
        plan=ChangePlan(
            summary=summary or "Prepare a proposal-only change set.",
            target_files=cleaned_targets,
            operations=["modify"] if cleaned_targets else [],
        ),
    )


def change_set_from_payload(payload: dict[str, Any]) -> ChangeSetProposal:
    plan_payload = payload.get("plan") if isinstance(payload.get("plan"), dict) else {}
    changes_payload = payload.get("changes") if isinstance(payload.get("changes"), list) else []
    plan = ChangePlan(
        summary=str(plan_payload.get("summary") or ""),
        target_files=[str(item) for item in plan_payload.get("target_files") or []],
        operations=[item for item in plan_payload.get("operations") or [] if item in {"modify", "create", "delete", "rename"}],
        evidence_files=[str(item) for item in plan_payload.get("evidence_files") or []],
        constraints=[str(item) for item in plan_payload.get("constraints") or []],
        validation_requirements=[str(item) for item in plan_payload.get("validation_requirements") or []],
    )
    changes = [
        ProposedFileChange(
            path=str(item.get("path") or ""),
            operation=item.get("operation") if item.get("operation") in {"modify", "create", "delete", "rename"} else "modify",
            summary=str(item.get("summary") or ""),
            original_content=item.get("original_content") if item.get("original_content") is not None else None,
            proposed_content=item.get("proposed_content") if item.get("proposed_content") is not None else None,
            rename_to=item.get("rename_to") if item.get("rename_to") is not None else None,
            delete_justification=item.get("delete_justification") if item.get("delete_justification") is not None else None,
            risk_notes=[str(note) for note in item.get("risk_notes") or []],
            additions=int(item.get("additions") or 0),
            deletions=int(item.get("deletions") or 0),
            validation_status=item.get("validation_status"),
        )
        for item in changes_payload
        if isinstance(item, dict)
    ]
    validation_payload = payload.get("validation") if isinstance(payload.get("validation"), dict) else None
    validation = None
    if validation_payload:
        validation_status = validation_payload.get("status") if validation_payload.get("status") in {"valid", "invalid", "repairable", "blocked"} else "invalid"
        validation = ValidationResult(
            status=validation_status,
            errors=[str(item) for item in validation_payload.get("errors") or []],
            warnings=[str(item) for item in validation_payload.get("warnings") or []],
            repairable=bool(validation_payload.get("repairable")),
        )
    status = payload.get("status") if payload.get("status") in {"planned", "valid", "invalid", "repairable", "blocked", "applied", "rejected"} else "planned"
    return ChangeSetProposal(
        proposal_id=str(payload.get("proposal_id") or stable_proposal_id(plan.summary, plan.target_files)),
        plan=plan,
        changes=changes,
        status=status,
        validation=validation,
        proposal_error=payload.get("proposal_error"),
        validation_status=str(payload.get("validation_status") or (validation.status if validation else "pending")),
        applied=bool(payload.get("applied")),
        applied_change_set_id=payload.get("applied_change_set_id"),
        apply_status=payload.get("apply_status"),
        post_apply_validation_status=payload.get("post_apply_validation_status"),
        applied_at=payload.get("applied_at"),
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
        change = ProposedFileChange(
            path=relative_path,
            operation="modify",
            summary=str(item.get("summary") or "Modify file content."),
            original_content=original_content,
            proposed_content=str(item.get("proposed_content") or ""),
            risk_notes=[str(note) for note in item.get("risk_notes") or []],
        )
        changes.append(_with_diff_counts(change))
    proposal = ChangeSetProposal(
        proposal_id=stable_proposal_id(plan_summary or "compat-edit", [change.path for change in changes]),
        plan=ChangePlan(
            summary=plan_summary or "Prepare validated proposal-only edits.",
            target_files=[change.path for change in changes],
            operations=["modify"] if changes else [],
        ),
        changes=changes,
    )
    validation = validate_change_set(proposal, repo=repo)
    proposal.validation = validation
    proposal.status = validation.status
    proposal.validation_status = validation.status
    proposal.proposal_error = "; ".join(validation.errors) if validation.errors else None
    return proposal


def validate_change_set(proposal: ChangeSetProposal, *, repo: str) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    repo_path = resolve_project_path(repo).resolve()
    if not proposal.changes:
        errors.append("change set has no file changes")
    seen: set[str] = set()
    created_paths: set[str] = set()
    deleted_paths: set[str] = set()
    renamed_from_to: dict[str, str] = {}
    for change in proposal.changes:
        if not change.path or _unsafe_relative_path(change.path):
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
        if _is_binary_or_cache_path(target):
            errors.append(f"{change.path}: binary or cache file edits are not allowed")
            continue
        if change.operation == "delete":
            if not target.exists():
                errors.append(f"{change.path}: delete target does not exist")
            if target.exists() and not target.is_file():
                errors.append(f"{change.path}: delete target must be a file")
            if not _delete_is_explicitly_justified(change, proposal):
                errors.append(f"{change.path}: delete proposals require explicit justification")
            if _is_protected_path(change.path):
                errors.append(f"{change.path}: protected delete is blocked without extra approval")
            deleted_paths.add(change.path)
            continue
        if change.operation == "rename":
            if not target.is_file():
                errors.append(f"{change.path}: rename source does not exist")
            if not change.rename_to or _unsafe_relative_path(change.rename_to):
                errors.append(f"{change.path}: rename target must stay inside the repository")
            else:
                destination = (repo_path / change.rename_to).resolve()
                try:
                    destination.relative_to(repo_path)
                except ValueError:
                    errors.append(f"{change.path}: rename target escapes repository")
                if destination.exists():
                    errors.append(f"{change.rename_to}: rename target already exists")
                if not _is_supported_new_text_path(destination):
                    errors.append(f"{change.rename_to}: binary or unsupported rename target is not allowed")
                renamed_from_to[change.path] = change.rename_to
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
            if not str(change.proposed_content or "").strip():
                errors.append(f"{change.path}: proposed content must not be empty")
            if change.original_content is not None and change.proposed_content == change.original_content:
                errors.append(f"{change.path}: proposed content is unchanged")
            _validate_proposed_text(change, target=target, errors=errors)
        if change.operation == "create":
            create_target_exists = target.exists()
            if create_target_exists:
                errors.append(f"{change.path}: create target already exists")
            if change.proposed_content is None:
                errors.append(f"{change.path}: proposed content is missing")
            if not str(change.proposed_content or "").strip():
                errors.append(f"{change.path}: proposed content must not be empty")
            if not _is_supported_new_text_path(target):
                errors.append(f"{change.path}: binary or unsupported file edits are not allowed")
            _validate_proposed_text(change, target=target, errors=errors)
            if not create_target_exists:
                created_paths.add(change.path)
        if _is_protected_path(change.path):
            warnings.append(f"{change.path}: package/config files require extra review.")

    proposed_text_by_path = {
        change.path: str(change.proposed_content or "")
        for change in proposal.changes
        if change.operation in {"modify", "create", "rename"}
    }
    for created in created_paths:
        if _created_file_unreferenced(created, proposal, proposed_text_by_path):
            errors.append(f"{created}: created helper file is not referenced by another proposed file or plan rationale")
    for deleted in deleted_paths:
        basename = Path(deleted).name
        if basename and any(basename in text for path, text in proposed_text_by_path.items() if path != deleted):
            errors.append(f"{deleted}: deleted file is still referenced in proposed content")
    for old, new in renamed_from_to.items():
        if old and any(old in text for text in proposed_text_by_path.values()):
            errors.append(f"{old}: rename references were not updated to {new}")

    if errors:
        blocked = any("protected delete" in error or ".git edits" in error for error in errors)
        repairable = not blocked and any(_is_repairable_error(error) for error in errors)
        return ValidationResult(
            status="blocked" if blocked else ("repairable" if repairable else "invalid"),
            errors=errors,
            warnings=warnings,
            repairable=repairable,
        )
    return ValidationResult(status="valid", errors=[], warnings=warnings)


def _with_diff_counts(change: ProposedFileChange) -> ProposedFileChange:
    original = "" if change.operation == "create" else str(change.original_content or "")
    proposed = "" if change.operation == "delete" else str(change.proposed_content or "")
    diff = difflib.unified_diff(original.splitlines(), proposed.splitlines(), lineterm="")
    lines = list(diff)
    change.additions = sum(1 for line in lines if line.startswith("+") and not line.startswith("+++"))
    change.deletions = sum(1 for line in lines if line.startswith("-") and not line.startswith("---"))
    return change


def _unsafe_relative_path(value: str) -> bool:
    path = Path(value)
    return not value.strip() or path.is_absolute() or ".." in path.parts


def _delete_is_explicitly_justified(change: ProposedFileChange, proposal: ChangeSetProposal) -> bool:
    text = " ".join(
        [
            change.summary,
            change.delete_justification or "",
            " ".join(change.risk_notes),
            proposal.plan.summary,
        ]
    ).lower()
    return "delete" in text and any(term in text for term in ("explicit", "requested", "obsolete", "remove"))


def _is_binary_or_cache_path(target: Path) -> bool:
    return target.suffix.lower() in BINARY_OR_CACHE_SUFFIXES or any(
        part in {"__pycache__", ".pytest_cache", ".mypy_cache", "node_modules", ".next"} for part in target.parts
    )


def _is_protected_path(path: str) -> bool:
    name = Path(path).name.lower()
    return name in {
        "package.json",
        "package-lock.json",
        "pyproject.toml",
        "poetry.lock",
        "requirements.txt",
        "pnpm-lock.yaml",
        "yarn.lock",
    } or path.endswith((".lock", ".toml"))


def _is_supported_new_text_path(target: Path) -> bool:
    suffix = target.suffix.lower()
    if suffix in BINARY_OR_CACHE_SUFFIXES:
        return False
    return suffix in TEXT_FILE_SUFFIXES or target.name.lower() in TEXT_FILE_BASENAMES


def _created_file_unreferenced(created: str, proposal: ChangeSetProposal, proposed_text_by_path: dict[str, str]) -> bool:
    basename = Path(created).stem
    plan_text = " ".join([proposal.plan.summary, *proposal.plan.constraints]).lower()
    if Path(created).name.lower() in plan_text or basename.lower() in plan_text:
        return False
    return not any((basename and basename in text) or created in text for path, text in proposed_text_by_path.items() if path != created)


def _is_repairable_error(error: str) -> bool:
    lowered = error.lower()
    return any(term in lowered for term in ("syntax", "markdown fences", "truncated", "unchanged", "empty", "referenced"))


def _validate_proposed_text(change: ProposedFileChange, *, target: Path, errors: list[str]) -> None:
    content = change.proposed_content
    if content is None:
        return
    suffix = target.suffix.lower()
    if suffix not in {".md", ".markdown", ".rst", ".txt"} and "```" in content:
        errors.append(f"{change.path}: proposed source content must not include markdown fences")
    if suffix == ".py":
        try:
            ast.parse(content or "\n", filename=change.path)
        except SyntaxError as exc:
            errors.append(f"{change.path}: Python syntax is invalid: line {exc.lineno}")
    if suffix == ".json":
        try:
            json.loads(content)
        except json.JSONDecodeError as exc:
            errors.append(f"{change.path}: JSON syntax is invalid: line {exc.lineno}")
    if suffix in {".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".kt", ".go", ".rs", ".cs", ".c", ".cpp", ".h", ".hpp"}:
        bracket_error = _basic_bracket_error(content)
        if bracket_error:
            errors.append(f"{change.path}: {bracket_error}")
    original = change.original_content or ""
    if original and len(original) > 240 and len((content or "").strip()) < max(40, int(len(original.strip()) * 0.2)):
        errors.append(f"{change.path}: proposed content appears truncated")


def _basic_bracket_error(content: str) -> str | None:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    in_string: str | None = None
    escaped = False
    for char in content:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == in_string:
                in_string = None
            continue
        if char in {"'", '"'}:
            in_string = char
        elif char in pairs.values():
            stack.append(char)
        elif char in pairs:
            if not stack or stack[-1] != pairs[char]:
                return "bracket/brace structure is unbalanced"
            stack.pop()
    return "bracket/brace structure is unbalanced" if stack else None


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
