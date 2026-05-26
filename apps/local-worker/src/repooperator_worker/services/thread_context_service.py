"""Thread context extraction and symbol tracking.

Public API
----------
- build_thread_context  — build ThreadContext from conversation history
- extract_symbols_from_text — extract defined symbols from source text

Context references are resolved by context_reference_service. This module only
extracts durable thread state and validates exact symbol/file carry-over.
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from repooperator_worker.schemas import AgentRunRequest
from repooperator_worker.services.common import ensure_relative_to_repo, get_repooperator_home_dir, resolve_project_path

SYMBOL_RE = re.compile(
    r"^\s*(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)\b|"
    r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)\b|"
    r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=",
    re.MULTILINE,
)


@dataclass
class ThreadContext:
    active_repo: str
    branch: str | None
    recent_files: list[str] = field(default_factory=list)
    symbols: dict[str, str] = field(default_factory=dict)
    last_analyzed_file: str | None = None
    last_proposed_target_file: str | None = None
    last_candidate_files: list[str] = field(default_factory=list)
    last_proposal_id: str | None = None
    last_answer_summary: str | None = None
    last_implementation_plan: dict[str, Any] | None = None
    last_target_candidates: list[dict[str, Any]] = field(default_factory=list)
    last_evidence_basis: list[dict[str, Any]] = field(default_factory=list)
    last_user_understanding_context: dict[str, Any] | None = None
    context_source: str = "retrieval"

    @property
    def symbol_names(self) -> list[str]:
        return sorted(self.symbols)


def build_thread_context(request: AgentRunRequest) -> ThreadContext:
    context = _load_durable_context(request) or ThreadContext(active_repo=request.project_path, branch=request.branch)
    for message in reversed(request.conversation_history):
        metadata = message.metadata if isinstance(message.metadata, dict) else {}
        for file_path in metadata.get("files_read") or []:
            _add_recent_file(context, str(file_path))
        selected = metadata.get("selected_target_file") or metadata.get("proposal_relative_path")
        if selected:
            context.last_proposed_target_file = str(selected)
            _add_recent_file(context, str(selected))
        _merge_target_candidates(context, metadata.get("edit_target_candidates") or metadata.get("target_candidates") or [])
        proposal = metadata.get("change_set_proposal") if isinstance(metadata.get("change_set_proposal"), dict) else None
        if proposal:
            _merge_proposal_context(context, proposal)
        if isinstance(metadata.get("implementation_plan"), dict):
            context.last_implementation_plan = metadata.get("implementation_plan")
        if isinstance(metadata.get("user_understanding_context"), dict):
            context.last_user_understanding_context = metadata.get("user_understanding_context")
        if isinstance(metadata.get("evidence_basis"), dict):
            context.last_evidence_basis = _evidence_basis_summaries(metadata.get("evidence_basis"))
        candidates = metadata.get("clarification_candidates") or []
        if candidates and not context.last_candidate_files:
            context.last_candidate_files = [str(candidate) for candidate in candidates]
        if metadata.get("proposal_relative_path") and not context.last_proposal_id:
            context.last_proposal_id = str(metadata.get("proposal_relative_path"))
        for symbol in metadata.get("thread_context_symbols") or []:
            if context.recent_files:
                context.symbols.setdefault(str(symbol), context.recent_files[0])
        if message.role == "assistant" and not context.last_answer_summary:
            context.last_answer_summary = _summarize(message.content)

    for relative_path in list(context.recent_files):
        _load_file_symbols(request.project_path, relative_path, context)

    if not context.last_analyzed_file and context.recent_files:
        context.last_analyzed_file = context.recent_files[0]
    return context


def update_thread_context(request: AgentRunRequest, response: Any) -> None:
    if not request.thread_id:
        return
    context = build_thread_context(request)
    for file_path in getattr(response, "files_read", []) or []:
        _add_recent_file(context, str(file_path))
    for file_path in getattr(response, "resolved_files", []) or []:
        _add_recent_file(context, str(file_path))
    selected = getattr(response, "selected_target_file", None) or getattr(response, "proposal_relative_path", None)
    if selected:
        context.last_proposed_target_file = str(selected)
        _add_recent_file(context, str(selected))
    if getattr(response, "proposal_relative_path", None):
        context.last_proposal_id = str(getattr(response, "proposal_relative_path"))
    proposal = getattr(response, "change_set_proposal", None)
    if isinstance(proposal, dict):
        _merge_proposal_context(context, proposal)
    if getattr(response, "recommendation_context", None):
        context.last_answer_summary = "Stored structured repository recommendations."
        recommendation = getattr(response, "recommendation_context", None)
        if isinstance(recommendation, dict):
            target_selection = recommendation.get("target_selection") if isinstance(recommendation.get("target_selection"), dict) else {}
            _merge_target_candidates(context, target_selection.get("candidates") or recommendation.get("edit_target_candidates") or [])
            if target_selection:
                context.last_evidence_basis = [
                    {
                        "path": item.get("path"),
                        "score": item.get("score"),
                        "reasons": item.get("reasons"),
                        "sources": item.get("sources"),
                    }
                    for item in target_selection.get("candidates") or []
                    if isinstance(item, dict)
                ][:20]
            if isinstance(recommendation.get("implementation_plan"), dict):
                context.last_implementation_plan = recommendation["implementation_plan"]
            if isinstance(recommendation.get("user_understanding_context"), dict):
                context.last_user_understanding_context = recommendation["user_understanding_context"]
            if isinstance(recommendation.get("evidence_basis"), dict):
                context.last_evidence_basis = _evidence_basis_summaries(recommendation["evidence_basis"])
    if getattr(response, "response", None):
        context.last_answer_summary = _summarize(str(getattr(response, "response")))
    for symbol in getattr(response, "resolved_symbols", []) or []:
        if context.recent_files:
            context.symbols.setdefault(str(symbol), context.recent_files[0])
    context.context_source = "durable_thread"
    _save_durable_context(request.thread_id, context)


def resolve_followup_file(request: AgentRunRequest, context: ThreadContext) -> tuple[str | None, str]:
    """Deprecated exact-symbol fallback; use context_reference_service for references."""
    lowered = request.task.lower()
    for symbol, relative_path in context.symbols.items():
        if symbol.lower() in lowered:
            return relative_path, "recent_thread"
    return None, "retrieval"


def extract_symbols_from_text(content: str) -> list[str]:
    symbols: list[str] = []
    for match in SYMBOL_RE.finditer(content):
        symbol = next((group for group in match.groups() if group), None)
        if symbol and symbol not in symbols:
            symbols.append(symbol)
    return symbols


def _add_recent_file(context: ThreadContext, relative_path: str) -> None:
    if relative_path and relative_path not in context.recent_files:
        context.recent_files.insert(0, relative_path)
    context.recent_files = context.recent_files[:12]


def _load_file_symbols(project_path: str, relative_path: str, context: ThreadContext) -> None:
    try:
        repo_path = resolve_project_path(project_path)
        target = ensure_relative_to_repo(repo_path, relative_path)
        if not target.is_file():
            return
        content = target.read_text(encoding="utf-8", errors="replace")[:80_000]
    except (OSError, ValueError):
        return
    for symbol in extract_symbols_from_text(content):
        context.symbols.setdefault(symbol, relative_path)


def _summarize(text: str, max_len: int = 300) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) > max_len:
        return cleaned[: max_len - 1].rstrip() + "..."
    return cleaned


def _thread_context_path(thread_id: str) -> Path:
    safe = "".join(ch for ch in thread_id if ch.isalnum() or ch in {"_", "-"})
    path = get_repooperator_home_dir() / "threads"
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{safe}.context.json"


def _load_durable_context(request: AgentRunRequest) -> ThreadContext | None:
    if not request.thread_id:
        return None
    path = _thread_context_path(request.thread_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("active_repo") != request.project_path:
        return None
    evidence_basis = payload.get("last_evidence_basis")
    if isinstance(evidence_basis, dict):
        evidence_basis = _evidence_basis_summaries(evidence_basis)
    return ThreadContext(
        active_repo=request.project_path,
        branch=request.branch or payload.get("branch"),
        recent_files=[str(item) for item in payload.get("recent_files", []) if isinstance(item, str)],
        symbols={str(key): str(value) for key, value in (payload.get("symbols") or {}).items()},
        last_analyzed_file=payload.get("last_analyzed_file"),
        last_proposed_target_file=payload.get("last_proposed_target_file"),
        last_candidate_files=[str(item) for item in payload.get("last_candidate_files", []) if isinstance(item, str)],
        last_proposal_id=payload.get("last_proposal_id"),
        last_answer_summary=payload.get("last_answer_summary"),
        last_implementation_plan=payload.get("last_implementation_plan") if isinstance(payload.get("last_implementation_plan"), dict) else None,
        last_target_candidates=[item for item in payload.get("last_target_candidates", []) if isinstance(item, dict)],
        last_evidence_basis=[item for item in evidence_basis or [] if isinstance(item, dict)],
        last_user_understanding_context=payload.get("last_user_understanding_context") if isinstance(payload.get("last_user_understanding_context"), dict) else None,
        context_source="durable_thread",
    )


def _save_durable_context(thread_id: str, context: ThreadContext) -> None:
    payload = {
        "active_repo": context.active_repo,
        "branch": context.branch,
        "recent_files": context.recent_files[:20],
        "symbols": context.symbols,
        "last_analyzed_file": context.last_analyzed_file,
        "last_proposed_target_file": context.last_proposed_target_file,
        "last_candidate_files": context.last_candidate_files[:20],
        "last_proposal_id": context.last_proposal_id,
        "last_answer_summary": context.last_answer_summary,
        "last_implementation_plan": context.last_implementation_plan,
        "last_target_candidates": context.last_target_candidates[:20],
        "last_evidence_basis": context.last_evidence_basis[:20],
        "last_user_understanding_context": context.last_user_understanding_context,
    }
    _thread_context_path(thread_id).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _merge_proposal_context(context: ThreadContext, proposal: dict[str, Any]) -> None:
    plan = proposal.get("plan") if isinstance(proposal.get("plan"), dict) else {}
    paths: list[str] = []
    paths.extend(str(item) for item in plan.get("target_files") or [] if str(item))
    for change in proposal.get("changes") or []:
        if isinstance(change, dict) and change.get("path"):
            paths.append(str(change["path"]))
    paths = list(dict.fromkeys(paths))
    if paths:
        context.last_proposed_target_file = paths[0]
        context.last_candidate_files = list(dict.fromkeys([*paths, *context.last_candidate_files]))[:20]
        for path in paths:
            _add_recent_file(context, path)
    proposal_id = proposal.get("proposal_id")
    if proposal_id:
        context.last_proposal_id = str(proposal_id)
    if plan:
        context.last_implementation_plan = {
            "summary": plan.get("summary"),
            "target_files": paths,
            "operations": plan.get("operations") or [],
            "evidence_files": plan.get("evidence_files") or [],
        }
    _merge_target_candidates(
        context,
        [
            {"path": path, "score": 85.0, "sources": ["prior_change_set"], "reasons": ["target from prior ChangeSetProposal"]}
            for path in paths
        ],
    )


def _merge_target_candidates(context: ThreadContext, candidates: Any) -> None:
    if not isinstance(candidates, list):
        return
    by_path = {str(item.get("path")): dict(item) for item in context.last_target_candidates if isinstance(item, dict) and item.get("path")}
    for item in candidates:
        if not isinstance(item, dict) or not item.get("path"):
            continue
        path = str(item.get("path"))
        current = by_path.get(path, {})
        score = max(_score_value(current.get("score"), default=0.0), _score_value(item.get("score"), default=0.0))
        by_path[path] = {
            "path": path,
            "score": score,
            "role": item.get("role") or current.get("role"),
            "language": item.get("language") or current.get("language"),
            "sources": list(dict.fromkeys([*(current.get("sources") or []), *(item.get("sources") or [])]))[:12],
            "reasons": list(dict.fromkeys([*(current.get("reasons") or []), *(item.get("reasons") or [])]))[:12],
            "symbols": list(dict.fromkeys([*(current.get("symbols") or []), *(item.get("symbols") or [])]))[:20],
        }
    context.last_target_candidates = sorted(by_path.values(), key=lambda item: (-_score_value(item.get("score"), default=0.0), str(item.get("path") or "")))[:20]


def _evidence_basis_summaries(evidence_basis: dict[str, Any]) -> list[dict[str, Any]]:
    target_selection = evidence_basis.get("target_selection") if isinstance(evidence_basis.get("target_selection"), dict) else {}
    files = evidence_basis.get("files") if isinstance(evidence_basis.get("files"), list) else []
    return [
        {
            "kind": "target_selection",
            "selected_target_files": target_selection.get("selected_target_files") or [],
            "prior_evidence_reused": bool(target_selection.get("prior_evidence_reused")),
            "candidate_count": len(target_selection.get("candidates") or []),
        },
        {
            "kind": "files",
            "paths": [item.get("path") for item in files if isinstance(item, dict) and item.get("path")][:20],
        },
    ]


def _score_value(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def list_thread_context_items(limit: int = 100) -> dict[str, Any]:
    directory = get_repooperator_home_dir() / "threads"
    items: list[dict[str, Any]] = []
    if not directory.exists():
        return {"items": []}
    for path in sorted(directory.glob("*.context.json"), key=lambda item: item.stat().st_mtime if item.exists() else 0):
        try:
            payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            items.append({"thread_id": path.name.removesuffix(".context.json"), **payload})
    return {"items": list(reversed(items[-limit:]))}
