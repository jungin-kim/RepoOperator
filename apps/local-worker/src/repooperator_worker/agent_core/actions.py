from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from repooperator_worker.services.json_safe import json_safe

ActionType = Literal[
    "inspect_repo_tree",
    "search_files",
    "read_file",
    "inspect_symbol",
    "analyze_file",
    "analyze_repository",
    "create_plan",
    "update_plan",
    "generate_recommendation",
    "generate_change_set",
    "validate_change_set",
    "apply_change_set",
    "read_many_files",
    "create_file",
    "modify_file",
    "delete_file",
    "rename_file",
    "run_validation_command",
    "generate_edit",
    "validate_edit",
    "write_file",
    "preview_command",
    "request_command_approval",
    "run_approved_command",
    "inspect_git_state",
    "git_status",
    "git_diff",
    "git_log",
    "git_branch_create",
    "git_commit",
    "git_push",
    "github_create_pr",
    "gitlab_create_mr",
    "search_web",
    "fetch_url",
    "summarize_web_evidence",
    "compact_thread_context",
    "refresh_context_pack",
    "inspect_gitlab_mr",
    "ask_clarification",
    "final_answer",
]

ActionStatus = Literal[
    "success",
    "failed",
    "skipped",
    "waiting_approval",
    "cancelled",
    "timed_out",
]


def new_action_id() -> str:
    return f"act_{uuid.uuid4().hex[:12]}"


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class AgentAction:
    type: ActionType
    reason_summary: str
    action_id: str = field(default_factory=new_action_id)
    target_files: list[str] = field(default_factory=list)
    target_symbols: list[str] = field(default_factory=list)
    command: list[str] | None = None
    expected_output: str | None = None
    safety_requirements: list[str] = field(default_factory=list)
    requires_approval: bool = False
    created_at: str = field(default_factory=now_iso)
    payload: dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        return json_safe({
            "action_id": self.action_id,
            "type": self.type,
            "reason_summary": self.reason_summary,
            "target_files": self.target_files,
            "target_symbols": self.target_symbols,
            "command": self.command,
            "expected_output": self.expected_output,
            "safety_requirements": self.safety_requirements,
            "requires_approval": self.requires_approval,
            "created_at": self.created_at,
            "payload": self.payload,
        })


@dataclass
class ActionResult:
    action_id: str
    status: ActionStatus
    observation: str = ""
    files_read: list[str] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)
    command_result: dict[str, Any] | None = None
    edit_records: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    next_recommended_action: str | None = None
    duration_ms: int = 0
    payload: dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        return json_safe({
            "action_id": self.action_id,
            "status": self.status,
            "observation": self.observation,
            "files_read": self.files_read,
            "files_changed": self.files_changed,
            "command_result": self.command_result,
            "edit_records": self.edit_records,
            "errors": self.errors,
            "next_recommended_action": self.next_recommended_action,
            "duration_ms": self.duration_ms,
            "payload": self.payload,
        })
