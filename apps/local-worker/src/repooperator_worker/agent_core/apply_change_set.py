from __future__ import annotations

import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from repooperator_worker.agent_core.change_set import (
    ChangeSetProposal,
    ProposedFileChange,
    change_set_from_payload,
    validate_change_set,
)
from repooperator_worker.agent_core.tools.builtin import is_supported_text_file
from repooperator_worker.services.common import get_repooperator_home_dir, resolve_project_path
from repooperator_worker.services.event_service import append_run_event, get_run
from repooperator_worker.services.json_safe import json_safe


@dataclass
class ApplyChangeSetResult:
    applied: bool
    proposal_id: str
    applied_change_set_id: str | None = None
    files_modified: list[str] = field(default_factory=list)
    files_created: list[str] = field(default_factory=list)
    files_deleted: list[str] = field(default_factory=list)
    files_renamed: list[dict[str, str]] = field(default_factory=list)
    validation_result: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)
    change_set_proposal: dict[str, Any] | None = None
    archive: list[dict[str, Any]] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return json_safe(self.__dict__)


def apply_change_set_for_run(
    *,
    run_id: str,
    project_path: str,
    proposal_id: str,
    approval_decision: dict[str, Any] | None = None,
    fallback_change_set: dict[str, Any] | None = None,
) -> ApplyChangeSetResult:
    decision = str((approval_decision or {}).get("decision") or "").strip().lower()
    if decision not in {"allow", "approved", "approve", "yes"}:
        return ApplyChangeSetResult(applied=False, proposal_id=proposal_id, errors=["Change set apply requires explicit approval."])

    proposal_payload = _load_persisted_proposal(run_id, proposal_id)
    if proposal_payload is None and fallback_change_set is not None:
        proposal_payload = fallback_change_set
    if not isinstance(proposal_payload, dict):
        return ApplyChangeSetResult(applied=False, proposal_id=proposal_id, errors=["Approved proposal was not found in the persisted run store."])

    proposal = change_set_from_payload(proposal_payload)
    if proposal.proposal_id != proposal_id:
        return ApplyChangeSetResult(applied=False, proposal_id=proposal_id, errors=["Approved proposal id does not match the persisted proposal."])
    if proposal.applied or proposal.apply_status == "applied":
        return ApplyChangeSetResult(applied=False, proposal_id=proposal_id, errors=["This proposal has already been applied."])

    validation = validate_change_set(proposal, repo=project_path)
    proposal.validation = validation
    proposal.status = validation.status
    proposal.validation_status = validation.status
    if validation.status != "valid":
        proposal.apply_status = "failed"
        return ApplyChangeSetResult(
            applied=False,
            proposal_id=proposal.proposal_id,
            validation_result=validation.model_dump(),
            errors=["Proposal validation failed before apply.", *validation.errors],
            change_set_proposal=proposal.model_dump(),
        )

    repo_path = resolve_project_path(project_path).resolve()
    applied_change_set_id = f"changeset_{uuid.uuid4().hex[:12]}"
    backup_dir = get_repooperator_home_dir() / "apply-backups" / applied_change_set_id
    backup_dir.mkdir(parents=True, exist_ok=True)
    result = ApplyChangeSetResult(
        applied=False,
        proposal_id=proposal.proposal_id,
        applied_change_set_id=applied_change_set_id,
        validation_result=validation.model_dump(),
    )
    try:
        for change in _deterministic_changes(proposal.changes):
            _apply_one_change(repo_path, backup_dir, change, result)
    except Exception as exc:  # noqa: BLE001
        result.errors.append(str(exc))
        result.change_set_proposal = proposal.model_dump()
        append_run_event(
            run_id,
            {
                "type": "progress_delta",
                "event_type": "change_set_apply_failed",
                "phase": "Editing",
                "label": "Apply failed",
                "status": "failed",
                "detail": str(exc),
                "aggregate": result.model_dump(),
            },
        )
        return result

    proposal.applied = True
    proposal.status = "applied"
    proposal.apply_status = "applied"
    proposal.applied_change_set_id = applied_change_set_id
    proposal.applied_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    result.applied = True
    result.change_set_proposal = proposal.model_dump()
    append_run_event(
        run_id,
        {
            "type": "progress_delta",
            "event_type": "change_set_applied",
            "phase": "Editing",
            "label": "Applied change set",
            "status": "completed",
            "files": result.files_modified + result.files_created + result.files_deleted + [item["from"] for item in result.files_renamed],
            "aggregate": result.model_dump(),
        },
    )
    return result


def _load_persisted_proposal(run_id: str, proposal_id: str) -> dict[str, Any] | None:
    run = get_run(run_id) or {}
    pending = run.get("pending_approval") if isinstance(run.get("pending_approval"), dict) else {}
    for source in (
        pending.get("change_set_proposal"),
        (run.get("final_result") or {}).get("change_set_proposal") if isinstance(run.get("final_result"), dict) else None,
    ):
        if isinstance(source, dict) and source.get("proposal_id") == proposal_id:
            return source
    return None


def _deterministic_changes(changes: list[ProposedFileChange]) -> list[ProposedFileChange]:
    order = {"rename": 0, "delete": 1, "modify": 2, "create": 3}
    return sorted(changes, key=lambda change: (order.get(change.operation, 99), change.path))


def _apply_one_change(repo_path: Path, backup_dir: Path, change: ProposedFileChange, result: ApplyChangeSetResult) -> None:
    target = _safe_target(repo_path, change.path)
    if change.operation in {"modify", "delete", "rename"}:
        if target.is_symlink():
            raise RuntimeError(f"{change.path}: symlink targets are not writable through change-set apply")
        if not target.is_file() or not is_supported_text_file(target):
            raise RuntimeError(f"{change.path}: apply target is missing or unsupported")
        backup_target = backup_dir / change.path
        backup_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, backup_target)

    if change.operation == "modify":
        target.write_text(str(change.proposed_content or ""), encoding="utf-8")
        result.files_modified.append(change.path)
        return
    if change.operation == "create":
        if target.exists():
            raise RuntimeError(f"{change.path}: create target already exists")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(change.proposed_content or ""), encoding="utf-8")
        result.files_created.append(change.path)
        return
    if change.operation == "delete":
        target.unlink()
        result.files_deleted.append(change.path)
        return
    if change.operation == "rename":
        if not change.rename_to:
            raise RuntimeError(f"{change.path}: rename target is missing")
        destination = _safe_target(repo_path, change.rename_to)
        if destination.exists():
            raise RuntimeError(f"{change.rename_to}: rename target already exists")
        destination.parent.mkdir(parents=True, exist_ok=True)
        target.rename(destination)
        result.files_renamed.append({"from": change.path, "to": change.rename_to})
        return
    raise RuntimeError(f"{change.path}: unsupported change operation {change.operation}")


def _safe_target(repo_path: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts or ".git" in path.parts:
        raise RuntimeError(f"{relative_path}: path must stay inside the repository")
    target = (repo_path / relative_path).resolve()
    try:
        target.relative_to(repo_path)
    except ValueError as exc:
        raise RuntimeError(f"{relative_path}: path escapes repository") from exc
    return target
