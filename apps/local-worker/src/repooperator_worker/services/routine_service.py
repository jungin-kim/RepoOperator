from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Literal

from repooperator_worker.schemas import AgentRunRequest
from repooperator_worker.services.agent_run_coordinator import cancel_run, enqueue_message, start_run
from repooperator_worker.services.common import get_repooperator_home_dir
from repooperator_worker.services.event_service import record_event
from repooperator_worker.services.json_safe import json_safe


RoutineTriggerType = Literal["manual", "cron", "interval", "git_event", "webhook"]


@dataclass(frozen=True)
class RoutineTrigger:
    type: RoutineTriggerType = "manual"
    expression: str | None = None
    interval_seconds: int | None = None

    def model_dump(self) -> dict[str, Any]:
        return json_safe(self)


@dataclass(frozen=True)
class RoutinePermission:
    profile: str = "basic"
    allow_writes_without_approval: bool = False
    allow_git_remote_write_without_approval: bool = False

    def model_dump(self) -> dict[str, Any]:
        return json_safe(self)


@dataclass
class RoutineDefinition:
    id: str
    name: str
    enabled: bool
    trigger: RoutineTrigger
    repo_identity: str
    branch: str | None
    task_prompt: str
    runtime: str = "langgraph"
    permission_profile: RoutinePermission = field(default_factory=RoutinePermission)
    max_duration: int = 900
    requires_approval_for_writes: bool = True
    last_run_at: str | None = None
    next_run_at: str | None = None
    thread_id: str | None = None

    def model_dump(self) -> dict[str, Any]:
        return json_safe(self)


@dataclass
class RoutineRun:
    id: str
    routine_id: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    run_id: str | None = None
    queued_message_id: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None

    def model_dump(self) -> dict[str, Any]:
        return json_safe(self)


class RoutineStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or (get_repooperator_home_dir() / "routines")
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def list(self) -> list[RoutineDefinition]:
        return [self._definition_from_payload(item) for item in self._read_definitions()]

    def create(self, payload: dict[str, Any]) -> RoutineDefinition:
        name = str(payload.get("name") or "Routine").strip()
        task_prompt = str(payload.get("task_prompt") or payload.get("task") or "").strip()
        repo_identity = str(payload.get("repo_identity") or payload.get("project_path") or "").strip()
        if not task_prompt:
            raise ValueError("task_prompt is required.")
        if not repo_identity:
            raise ValueError("repo_identity is required.")
        trigger_payload = payload.get("trigger") if isinstance(payload.get("trigger"), dict) else {}
        trigger = RoutineTrigger(
            type=_trigger_type(trigger_payload.get("type") or payload.get("trigger_type") or "manual"),
            expression=trigger_payload.get("expression"),
            interval_seconds=_optional_int(trigger_payload.get("interval_seconds") or payload.get("interval_seconds")),
        )
        routine = RoutineDefinition(
            id=f"routine_{uuid.uuid4().hex[:12]}",
            name=name,
            enabled=bool(payload.get("enabled", True)),
            trigger=trigger,
            repo_identity=repo_identity,
            branch=payload.get("branch"),
            task_prompt=task_prompt,
            runtime=str(payload.get("runtime") or "langgraph"),
            permission_profile=RoutinePermission(),
            max_duration=int(payload.get("max_duration") or 900),
            requires_approval_for_writes=bool(payload.get("requires_approval_for_writes", True)),
            next_run_at=_next_run_at(trigger),
            thread_id=payload.get("thread_id"),
        )
        with self._lock:
            definitions = self._read_definitions()
            definitions.append(routine.model_dump())
            self._write_definitions(definitions)
        return routine

    def update_enabled(self, routine_id: str, enabled: bool) -> RoutineDefinition:
        with self._lock:
            definitions = self._read_definitions()
            for item in definitions:
                if item.get("id") == routine_id:
                    item["enabled"] = enabled
                    if enabled and not item.get("next_run_at"):
                        item["next_run_at"] = _next_run_at(_trigger_from_payload(item.get("trigger") or {}))
                    self._write_definitions(definitions)
                    return self._definition_from_payload(item)
        raise ValueError("Routine not found.")

    def run_now(self, routine_id: str) -> RoutineRun:
        routine = self.get(routine_id)
        run = RoutineRun(id=f"routine_run_{uuid.uuid4().hex[:12]}", routine_id=routine.id, status="queued", started_at=_now_iso())
        if routine.thread_id:
            queued = enqueue_message(routine.thread_id, routine.repo_identity, routine.branch, routine.task_prompt)
            run.queued_message_id = queued.get("id")
        else:
            request = AgentRunRequest(project_path=routine.repo_identity, git_provider="local", branch=routine.branch, thread_id=routine.thread_id, task=routine.task_prompt)
            response = start_run(request, stream=False)
            run.run_id = response.run_id
            run.status = "running" if response.stop_reason == "waiting_approval" else "completed"
            run.completed_at = _now_iso() if run.status == "completed" else None
        self._append_run(run)
        self._mark_ran(routine.id)
        record_event(event_type="routine_run_enqueued", repo=routine.repo_identity, branch=routine.branch, summary=routine.name)
        return run

    def enqueue_due(self, now: datetime | None = None) -> list[RoutineRun]:
        now = now or datetime.now(timezone.utc)
        runs: list[RoutineRun] = []
        for routine in self.list():
            if not routine.enabled or not routine.next_run_at:
                continue
            due = _parse_iso(routine.next_run_at)
            if due and due <= now:
                runs.append(self.run_now(routine.id))
        return runs

    def list_runs(self, routine_id: str | None = None) -> list[RoutineRun]:
        runs = [self._run_from_payload(item) for item in self._read_runs()]
        if routine_id:
            runs = [run for run in runs if run.routine_id == routine_id]
        return runs

    def cancel_routine_run(self, routine_run_id: str) -> RoutineRun:
        with self._lock:
            runs = self._read_runs()
            for item in runs:
                if item.get("id") != routine_run_id:
                    continue
                if item.get("run_id"):
                    cancel_run(str(item.get("run_id")))
                item["status"] = "cancelled"
                item["completed_at"] = _now_iso()
                self._write_runs(runs)
                return self._run_from_payload(item)
        raise ValueError("Routine run not found.")

    def get(self, routine_id: str) -> RoutineDefinition:
        for routine in self.list():
            if routine.id == routine_id:
                return routine
        raise ValueError("Routine not found.")

    def _mark_ran(self, routine_id: str) -> None:
        with self._lock:
            definitions = self._read_definitions()
            for item in definitions:
                if item.get("id") == routine_id:
                    item["last_run_at"] = _now_iso()
                    item["next_run_at"] = _next_run_at(_trigger_from_payload(item.get("trigger") or {}))
            self._write_definitions(definitions)

    def _append_run(self, run: RoutineRun) -> None:
        with self._lock:
            runs = self._read_runs()
            runs.append(run.model_dump())
            self._write_runs(runs)

    def _definitions_path(self) -> Path:
        return self.root / "definitions.json"

    def _runs_path(self) -> Path:
        return self.root / "runs.json"

    def _read_definitions(self) -> list[dict[str, Any]]:
        return _read_json_list(self._definitions_path())

    def _write_definitions(self, definitions: list[dict[str, Any]]) -> None:
        self._definitions_path().write_text(json.dumps(json_safe(definitions), ensure_ascii=False, indent=2), encoding="utf-8")

    def _read_runs(self) -> list[dict[str, Any]]:
        return _read_json_list(self._runs_path())

    def _write_runs(self, runs: list[dict[str, Any]]) -> None:
        self._runs_path().write_text(json.dumps(json_safe(runs), ensure_ascii=False, indent=2), encoding="utf-8")

    def _definition_from_payload(self, payload: dict[str, Any]) -> RoutineDefinition:
        return RoutineDefinition(
            id=str(payload.get("id")),
            name=str(payload.get("name") or ""),
            enabled=bool(payload.get("enabled")),
            trigger=_trigger_from_payload(payload.get("trigger") or {}),
            repo_identity=str(payload.get("repo_identity") or ""),
            branch=payload.get("branch"),
            task_prompt=str(payload.get("task_prompt") or ""),
            runtime=str(payload.get("runtime") or "langgraph"),
            permission_profile=RoutinePermission(**dict(payload.get("permission_profile") or {})),
            max_duration=int(payload.get("max_duration") or 900),
            requires_approval_for_writes=bool(payload.get("requires_approval_for_writes", True)),
            last_run_at=payload.get("last_run_at"),
            next_run_at=payload.get("next_run_at"),
            thread_id=payload.get("thread_id"),
        )

    def _run_from_payload(self, payload: dict[str, Any]) -> RoutineRun:
        return RoutineRun(
            id=str(payload.get("id")),
            routine_id=str(payload.get("routine_id")),
            status=payload.get("status") or "queued",
            run_id=payload.get("run_id"),
            queued_message_id=payload.get("queued_message_id"),
            started_at=payload.get("started_at"),
            completed_at=payload.get("completed_at"),
            error=payload.get("error"),
        )


def get_default_routine_store() -> RoutineStore:
    return RoutineStore()


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return []
    return payload if isinstance(payload, list) else []


def _trigger_type(value: str) -> RoutineTriggerType:
    normalized = str(value or "manual").strip().lower()
    if normalized in {"manual", "cron", "interval", "git_event", "webhook"}:
        return normalized  # type: ignore[return-value]
    raise ValueError("Unsupported routine trigger type.")


def _trigger_from_payload(payload: dict[str, Any]) -> RoutineTrigger:
    return RoutineTrigger(type=_trigger_type(payload.get("type") or "manual"), expression=payload.get("expression"), interval_seconds=_optional_int(payload.get("interval_seconds")))


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _next_run_at(trigger: RoutineTrigger) -> str | None:
    if trigger.type == "interval" and trigger.interval_seconds:
        return (datetime.now(timezone.utc) + timedelta(seconds=max(1, trigger.interval_seconds))).strftime("%Y-%m-%dT%H:%M:%SZ")
    if trigger.type == "cron":
        return None
    return None


def _parse_iso(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
