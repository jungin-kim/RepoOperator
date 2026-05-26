from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from repooperator_worker.agent_core.policies import command_policy_preview
from repooperator_worker.agent_core.tools.builtin import is_supported_text_file
from repooperator_worker.schemas import AgentRunRequest
from repooperator_worker.services.command_service import run_command_with_policy
from repooperator_worker.services.common import resolve_project_path
from repooperator_worker.services.json_safe import json_safe
from repooperator_worker.services.skills_service import enabled_skill_context
from repooperator_worker.services.active_repository import get_active_repository


@dataclass
class ContextPacket:
    repo_root_name: str
    repo_path: str
    branch: str | None
    git_status_summary: str | None = None
    recent_commits_summary: str | None = None
    project_instructions: dict[str, str] = field(default_factory=dict)
    high_signal_files: dict[str, str] = field(default_factory=dict)
    prior_files_read: list[str] = field(default_factory=list)
    prior_commands_run: list[str] = field(default_factory=list)
    thread_context: dict[str, Any] = field(default_factory=dict)
    prior_target_candidates: list[dict[str, Any]] = field(default_factory=list)
    skills_context: str = ""
    created_at: str = ""
    cache_key: str = ""
    cache_hit: bool = False
    invalidation_reason: str | None = None
    high_signal_fingerprint: str | None = None

    def model_dump(self) -> dict[str, Any]:
        return json_safe(self)


class ContextService:
    def __init__(self, *, max_file_chars: int = 20_000, ttl_seconds: int = 300) -> None:
        self.max_file_chars = max_file_chars
        self.ttl_seconds = ttl_seconds
        self._cache: dict[tuple[str, str | None, str | None, str | None], tuple[float, ContextPacket]] = {}
        self._last_active_repository: str | None = None

    def collect(self, request: AgentRunRequest, *, force_refresh: bool = False) -> ContextPacket:
        repo = resolve_project_path(request.project_path).resolve()
        branch = request.branch or self._git_branch(repo, request)
        fingerprint = self._high_signal_fingerprint(repo)
        key = (str(repo), branch, request.thread_id, fingerprint)
        cache_key = "|".join(str(item or "") for item in key)
        request_refresh = self._request_refresh_requested(request)
        active_changed = self._active_repository_changed(str(repo))
        invalidation_reason = (
            "force_refresh" if force_refresh
            else "request_refresh" if request_refresh
            else "active_repository_changed" if active_changed
            else None
        )
        now = time.time()
        cached = self._cache.get(key)
        if cached and now - cached[0] <= self.ttl_seconds and not (force_refresh or request_refresh or active_changed):
            packet = self._with_fresh_thread_scope(cached[1], request)
            self._cache[key] = (cached[0], replace(packet, cache_hit=False, invalidation_reason=cached[1].invalidation_reason))
            return replace(packet, cache_hit=True, invalidation_reason=None)

        skills_context, _skills_used = enabled_skill_context(task=request.task)
        thread_context = self._thread_context(request)
        packet = ContextPacket(
            repo_root_name=repo.name,
            repo_path=str(repo),
            branch=branch,
            git_status_summary=self._git_command(repo, request, ["git", "status", "--short"]),
            recent_commits_summary=self._git_command(repo, request, ["git", "log", "--oneline", "-n", "5"]),
            project_instructions=self._read_named_files(
                repo,
                ["CLAUDE.md", "AGENTS.md", "REPOOPERATOR.md", ".repooperator/instructions.md"],
            ),
            high_signal_files=self._read_named_files(
                repo,
                [
                    "README.md",
                    "readme.md",
                    "package.json",
                    "pyproject.toml",
                    "Cargo.toml",
                    "go.mod",
                    "manifest.json",
                    "apps/web/package.json",
                    "apps/local-worker/pyproject.toml",
                ],
            ),
            prior_files_read=self._prior_metadata(request, keys=("files_read", "resolved_files")),
            prior_commands_run=self._prior_metadata(request, keys=("commands_run", "commands_planned")),
            thread_context=thread_context,
            prior_target_candidates=list(thread_context.get("last_target_candidates") or thread_context.get("target_candidates") or []),
            skills_context=skills_context,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            cache_key=cache_key,
            cache_hit=False,
            invalidation_reason=invalidation_reason,
            high_signal_fingerprint=fingerprint,
        )
        self._cache[key] = (now, packet)
        return packet

    def _with_fresh_thread_scope(self, packet: ContextPacket, request: AgentRunRequest) -> ContextPacket:
        thread_context = self._thread_context(request)
        return replace(
            packet,
            prior_files_read=self._prior_metadata(request, keys=("files_read", "resolved_files")),
            prior_commands_run=self._prior_metadata(request, keys=("commands_run", "commands_planned")),
            thread_context=thread_context,
            prior_target_candidates=list(thread_context.get("last_target_candidates") or thread_context.get("target_candidates") or []),
        )

    def clear(self) -> None:
        self._cache.clear()

    def invalidate(self, repo_path: str | None = None, branch: str | None = None, thread_id: str | None = None) -> None:
        if repo_path is None and branch is None and thread_id is None:
            self._cache.clear()
            return
        resolved_repo = None
        if repo_path:
            try:
                resolved_repo = str(resolve_project_path(repo_path).resolve())
            except Exception:
                resolved_repo = str(repo_path)
        for key in list(self._cache):
            key_repo, key_branch, key_thread, _fingerprint = key
            if resolved_repo is not None and key_repo != resolved_repo:
                continue
            if branch is not None and key_branch != branch:
                continue
            if thread_id is not None and key_thread != thread_id:
                continue
            self._cache.pop(key, None)

    def _read_named_files(self, repo: Path, relative_paths: list[str]) -> dict[str, str]:
        result: dict[str, str] = {}
        seen: set[str] = set()
        for rel in relative_paths:
            target = repo / rel
            try:
                marker = str(target.resolve()).lower()
            except OSError:
                marker = str(target).lower()
            if marker in seen or not target.is_file() or not is_supported_text_file(target):
                continue
            seen.add(marker)
            try:
                result[rel] = target.read_text(encoding="utf-8", errors="replace")[: self.max_file_chars]
            except OSError:
                continue
        return result

    def _prior_metadata(self, request: AgentRunRequest, *, keys: tuple[str, ...]) -> list[str]:
        values: list[str] = []
        for item in request.conversation_history[-12:]:
            metadata = item.metadata or {}
            for key in keys:
                raw = metadata.get(key) or []
                if isinstance(raw, str):
                    raw = [raw]
                for value in raw:
                    text = str(value).strip()
                    if text and text not in values:
                        values.append(text)
        return values[:40]

    def _thread_context(self, request: AgentRunRequest) -> dict[str, Any]:
        try:
            from repooperator_worker.services.thread_context_service import build_thread_context

            context = build_thread_context(request)
        except Exception:
            return {}
        return json_safe(
            {
                "active_repo": context.active_repo,
                "branch": context.branch,
                "recent_files": context.recent_files,
                "symbols": context.symbols,
                "last_analyzed_file": context.last_analyzed_file,
                "last_proposed_target_file": context.last_proposed_target_file,
                "last_candidate_files": context.last_candidate_files,
                "last_proposal_id": context.last_proposal_id,
                "last_answer_summary": context.last_answer_summary,
                "last_implementation_plan": context.last_implementation_plan,
                "last_target_candidates": context.last_target_candidates,
                "target_candidates": context.last_target_candidates,
                "last_evidence_basis": context.last_evidence_basis,
                "last_user_understanding_context": context.last_user_understanding_context,
                "context_source": context.context_source,
            }
        )

    def _git_branch(self, repo: Path, request: AgentRunRequest) -> str | None:
        summary = self._git_command(repo, request, ["git", "rev-parse", "--abbrev-ref", "HEAD"])
        return summary.splitlines()[0].strip() if summary else None

    def _git_command(self, repo: Path, request: AgentRunRequest, command: list[str]) -> str | None:
        if not (repo / ".git").exists():
            return None
        try:
            preview = command_policy_preview(command, project_path=request.project_path, reason="Collect bounded repository context.")
        except Exception:
            return None
        if preview.get("blocked") or preview.get("needs_approval") or not preview.get("read_only"):
            return None
        try:
            result = run_command_with_policy(command, project_path=request.project_path, reason="Collect bounded repository context.")
        except Exception:
            return None
        text = str(result.get("stdout") or "").strip()
        return text[:4_000] or None

    def _request_refresh_requested(self, request: AgentRunRequest) -> bool:
        metadata = getattr(request, "metadata", None) or {}
        if bool(metadata.get("refresh_context") or metadata.get("context_refresh")):
            return True
        for item in request.conversation_history[-2:]:
            item_metadata = item.metadata or {}
            if bool(item_metadata.get("refresh_context") or item_metadata.get("context_refresh")):
                return True
        return False

    def _active_repository_changed(self, repo_path: str) -> bool:
        try:
            active = get_active_repository()
        except Exception:
            active = None
        active_path = str(active.project_path) if active else repo_path
        changed = self._last_active_repository is not None and self._last_active_repository != active_path
        self._last_active_repository = active_path
        return changed

    def _high_signal_fingerprint(self, repo: Path) -> str:
        parts: list[str] = []
        for rel in [
            "README.md",
            "readme.md",
            "package.json",
            "pyproject.toml",
            "Cargo.toml",
            "go.mod",
            "manifest.json",
            "CLAUDE.md",
            "AGENTS.md",
            "REPOOPERATOR.md",
            ".repooperator/instructions.md",
            "apps/web/package.json",
            "apps/local-worker/pyproject.toml",
        ]:
            target = repo / rel
            try:
                stat = target.stat()
            except OSError:
                continue
            if target.is_file():
                parts.append(f"{rel}:{stat.st_mtime_ns}:{stat.st_size}")
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16] if parts else None


_DEFAULT_CONTEXT_SERVICE = ContextService()


def get_default_context_service() -> ContextService:
    return _DEFAULT_CONTEXT_SERVICE
