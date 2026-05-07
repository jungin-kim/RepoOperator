from __future__ import annotations

import json
import re
import shlex
from difflib import unified_diff
from pathlib import Path
from typing import Any

from repooperator_worker.agent_core.command_security import validate_argv_shape
from repooperator_worker.agent_core.events import append_activity_event
from repooperator_worker.agent_core.policies import command_policy_preview, validate_repo_file
from repooperator_worker.agent_core.repository_review import run_repository_review
from repooperator_worker.agent_core.secret_scanner import redact_secrets
from repooperator_worker.agent_core.tools.base import BaseTool, ToolExecutionContext, ToolResult, ToolSpec
from repooperator_worker.agent_core.permissions import PermissionDecision, ToolPermissionContext
from repooperator_worker.services.command_service import run_command_with_policy
from repooperator_worker.services.common import resolve_project_path
from repooperator_worker.services.json_safe import json_safe, safe_agent_response_payload
from repooperator_worker.services.model_client import ModelGenerationRequest, OpenAICompatibleModelClient


TEXT_FILE_SUFFIXES = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".cs", ".java", ".kt", ".go", ".rs", ".rb", ".php",
    ".c", ".cpp", ".h", ".hpp", ".md", ".txt", ".rst", ".json", ".toml", ".yaml", ".yml",
    ".ini", ".cfg", ".gradle", ".xml", ".html", ".css", ".sh",
}
TEXT_FILE_BASENAMES = {"readme", "makefile", "dockerfile", "license"}
BINARY_OR_CACHE_SUFFIXES = {
    ".sqlite", ".sqlite3", ".db", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".zip",
    ".tar", ".gz", ".7z", ".dll", ".exe", ".so", ".dylib", ".class", ".jar", ".bin",
}


class InspectRepoTreeTool(BaseTool):
    spec = ToolSpec(
        name="inspect_repo_tree",
        description="List top-level repository entries to orient repository inspection.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": True},
        read_only=True,
        concurrency_safe=True,
    )

    def call(self, payload: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        repo = resolve_project_path(context.request.project_path)
        append_activity_event(
            run_id=context.run_id,
            request=context.request,
            activity_id="inspect-repo-tree",
            event_type="activity_started",
            phase="Searching",
            label="Inspect repository tree",
            status="running",
            current_action="Listing top-level repository entries.",
            next_action="Use the listing to choose files or answer from inventory.",
        )
        try:
            entries = sorted(path.name for path in repo.iterdir())[:80]
        except OSError as exc:
            return ToolResult(tool_name=self.spec.name, status="failed", observation=str(exc), payload={"errors": [str(exc)]})
        append_activity_event(
            run_id=context.run_id,
            request=context.request,
            activity_id="inspect-repo-tree",
            event_type="activity_completed",
            phase="Searching",
            label="Inspect repository tree",
            status="completed",
            observation=f"Found {len(entries)} top-level entr{'y' if len(entries) == 1 else 'ies'}.",
            next_action="Prepare the answer or inspect targeted files.",
            aggregate={"entries_count": len(entries)},
        )
        return ToolResult(tool_name=self.spec.name, status="success", observation=", ".join(entries), payload={"entries": entries})


class ReadFileTool(BaseTool):
    spec = ToolSpec(
        name="read_file",
        description="Read supported text files inside the repository, skipping binary/cache files.",
        input_schema={
            "type": "object",
            "properties": {"target_files": {"type": "array", "items": {"type": "string"}, "maxItems": 8}},
            "required": ["target_files"],
            "additionalProperties": True,
        },
        read_only=True,
        concurrency_safe=True,
        max_result_chars=400_000,
    )

    def validate_input(self, payload: dict[str, Any], request) -> dict[str, Any]:
        cleaned = dict(payload)
        cleaned["target_files"] = [str(item).strip().lstrip("/") for item in payload.get("target_files") or [] if str(item).strip()]
        return cleaned

    def call(self, payload: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        target_files = list(payload.get("target_files") or [])
        if not target_files:
            return ToolResult(tool_name=self.spec.name, status="skipped", observation="No target file was provided.")
        files_read: list[str] = []
        contents: dict[str, str] = {}
        skipped: list[str] = []
        for relative_path in target_files[:8]:
            target = validate_repo_file(context.request.project_path, relative_path)
            activity_id = f"read-file:{relative_path}"
            if not is_supported_text_file(target):
                skipped.append(relative_path)
                append_activity_event(
                    run_id=context.run_id,
                    request=context.request,
                    activity_id=activity_id,
                    event_type="activity_completed",
                    phase="Reading files",
                    label=Path(relative_path).name,
                    status="completed",
                    observation=f"Skipped unsupported or binary file `{relative_path}`.",
                    next_action="Use supported source, config, or documentation files as evidence.",
                    related_files=[relative_path],
                    aggregate={"skip_reason": "unsupported_or_binary"},
                )
                continue
            append_activity_event(
                run_id=context.run_id,
                request=context.request,
                activity_id=activity_id,
                event_type="activity_started",
                phase="Reading files",
                label=Path(relative_path).name,
                status="running",
                current_action=f"Reading `{relative_path}`.",
                related_files=[relative_path],
            )
            raw = target.read_text(encoding="utf-8", errors="replace")
            files_read.append(relative_path)
            contents[relative_path] = raw[:100_000]
            append_activity_event(
                run_id=context.run_id,
                request=context.request,
                activity_id=activity_id,
                event_type="activity_completed",
                phase="Reading files",
                label=Path(relative_path).name,
                status="completed",
                observation=f"Read {len(raw.splitlines())} line(s).",
                next_action="Use the file content as evidence.",
                related_files=[relative_path],
            )
        status = "success" if files_read else "skipped"
        observation = f"Read {len(files_read)} file(s)." if files_read else "No supported text files were read."
        return ToolResult(
            tool_name=self.spec.name,
            status=status,
            files_read=files_read,
            observation=observation,
            payload={"contents": contents, "skipped_files": skipped},
        )


class SearchFilesTool(BaseTool):
    spec = ToolSpec(
        name="search_files",
        description="Find ranked repo-contained text files by path, basename, extension, symbol, or text evidence.",
        input_schema={
            "type": "object",
            "properties": {
                "queries": {"type": "array", "items": {"type": "string"}},
                "target_symbols": {"type": "array", "items": {"type": "string"}},
                "text_queries": {"type": "array", "items": {"type": "string"}},
                "file_globs": {"type": "array", "items": {"type": "string"}},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "additionalProperties": True,
        },
        read_only=True,
        concurrency_safe=True,
    )

    def call(self, payload: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        repo = resolve_project_path(context.request.project_path).resolve()
        raw_queries = [*(payload.get("queries") or payload.get("target_files") or []), *(payload.get("target_symbols") or [])]
        raw_queries.extend(payload.get("file_globs") or [])
        queries: list[str] = []
        for item in raw_queries:
            text = str(item).strip()
            if text and text not in queries:
                queries.append(text)
        text_queries = [str(item).strip() for item in payload.get("text_queries") or [] if str(item).strip()]
        max_results = int(payload.get("max_results") or 8)
        append_activity_event(
            run_id=context.run_id,
            request=context.request,
            activity_id="search-files:" + "-".join(queries)[:120],
            event_type="activity_started",
            phase="Searching",
            label="Resolving target files",
            status="running",
            current_action="Searching repository files by path, basename, extension, or symbol.",
            next_action="Read the best matching repo-contained file.",
            aggregate={"queries": queries, "text_queries": text_queries},
        )
        candidate_details = find_file_candidates(repo, queries, text_queries=text_queries, max_results=max_results)
        candidates = [item["path"] for item in candidate_details]
        append_activity_event(
            run_id=context.run_id,
            request=context.request,
            activity_id="search-files:" + "-".join(queries)[:120],
            event_type="activity_completed",
            phase="Searching",
            label="Resolved target files",
            status="completed",
            observation=f"Found {len(candidates)} candidate file(s).",
            related_files=candidates,
            aggregate={"queries": queries, "text_queries": text_queries, "candidates": candidates, "candidate_details": candidate_details},
        )
        return ToolResult(
            tool_name=self.spec.name,
            status="success",
            observation=f"Found {len(candidates)} candidate file(s).",
            payload={"queries": queries, "text_queries": text_queries, "candidates": candidates, "candidate_details": candidate_details},
        )


class SearchTextTool(BaseTool):
    spec = ToolSpec(
        name="search_text",
        description="Search text content inside supported repo files without invoking shell grep or rg.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "path_globs": {"type": "array", "items": {"type": "string"}},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 200},
                "case_sensitive": {"type": "boolean"},
                "regex": {"type": "boolean"},
                "context_lines": {"type": "integer", "minimum": 0, "maximum": 3},
            },
            "required": ["query"],
            "additionalProperties": True,
        },
        read_only=True,
        concurrency_safe=True,
        max_result_chars=200_000,
    )

    def validate_input(self, payload: dict[str, Any], request) -> dict[str, Any]:
        query = str(payload.get("query") or "").strip()
        globs = [str(item).strip().lstrip("/") for item in payload.get("path_globs") or [] if str(item).strip()]
        return {
            **dict(payload),
            "query": query[:500],
            "path_globs": globs[:20],
            "max_results": max(1, min(int(payload.get("max_results") or 50), 200)),
            "case_sensitive": bool(payload.get("case_sensitive")),
            "regex": bool(payload.get("regex")),
            "context_lines": max(0, min(int(payload.get("context_lines") or 0), 3)),
        }

    def call(self, payload: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        query = str(payload.get("query") or "")
        if not query:
            return ToolResult(tool_name=self.spec.name, status="skipped", observation="No search query was provided.")
        repo = resolve_project_path(context.request.project_path).resolve()
        matches, files_searched, truncated = search_text_matches(
            repo,
            query=query,
            path_globs=list(payload.get("path_globs") or []),
            max_results=int(payload.get("max_results") or 50),
            case_sensitive=bool(payload.get("case_sensitive")),
            regex=bool(payload.get("regex")),
            context_lines=int(payload.get("context_lines") or 0),
        )
        files_with_matches = sorted({str(item["path"]) for item in matches})
        append_activity_event(
            run_id=context.run_id,
            request=context.request,
            activity_id="search-text:" + query[:80],
            event_type="activity_completed",
            phase="Searching",
            label="Searched text",
            status="completed",
            observation=f"Found {len(matches)} text match(es) in {len(files_with_matches)} file(s).",
            related_files=files_with_matches[:20],
            aggregate={"query": query, "files_searched": files_searched, "files_with_matches": files_with_matches, "truncated": truncated},
        )
        return ToolResult(
            tool_name=self.spec.name,
            status="success",
            observation=f"Found {len(matches)} text match(es).",
            payload={
                "query": query,
                "matches": matches,
                "files_searched": files_searched,
                "files_with_matches": files_with_matches,
                "truncated": truncated,
            },
        )


class AnalyzeRepositoryTool(BaseTool):
    spec = ToolSpec(
        name="analyze_repository",
        description="Run the repository-wide review pipeline and return summarized evidence.",
        input_schema={"type": "object", "properties": {"classifier": {"type": "object"}}, "additionalProperties": True},
        read_only=True,
        concurrency_safe=False,
        max_result_chars=500_000,
    )

    def call(self, payload: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        response = run_repository_review(
            request=context.request,
            run_id=context.run_id,
            classifier=payload.get("classifier"),
        )
        return ToolResult(
            tool_name=self.spec.name,
            status="success",
            observation="Repository review completed.",
            files_read=response.files_read,
            payload={"response": safe_agent_response_payload(response)},
        )


class PreviewCommandTool(BaseTool):
    spec = ToolSpec(
        name="preview_command",
        description="Classify a local command through the command policy without executing it.",
        input_schema={
            "type": "object",
            "properties": {"command": {"type": "array", "items": {"type": "string"}}},
            "required": ["command"],
            "additionalProperties": True,
        },
        read_only=True,
        concurrency_safe=True,
    )

    def call(self, payload: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        raw_command = _command_from_payload(payload, default=["git", "status", "--short"])
        reason = str(payload.get("reason_summary") or "")
        shape = validate_argv_shape(raw_command)
        if not shape.allowed:
            return ToolResult(
                tool_name=self.spec.name,
                status="failed",
                observation=shape.reason,
                payload={"command_security": shape.model_dump()},
            )
        command = list(raw_command)
        activity_id = "command-preview:" + shlex.join(command)
        append_activity_event(
            run_id=context.run_id,
            request=context.request,
            activity_id=activity_id,
            event_type="activity_started",
            phase="Commands",
            label="Preview command",
            status="running",
            current_action=f"Classifying `{shlex.join(command)}` through command policy.",
            related_command=command,
        )
        preview = command_policy_preview(command, project_path=context.request.project_path, reason=reason)
        status = "waiting_approval" if preview.get("needs_approval") else "success"
        append_activity_event(
            run_id=context.run_id,
            request=context.request,
            activity_id=activity_id,
            event_type="activity_completed" if status == "success" else "activity_updated",
            phase="Commands",
            label="Preview command",
            status="waiting" if status == "waiting_approval" else "completed",
            observation="Command requires approval." if status == "waiting_approval" else "Command is allowed by policy.",
            next_action="Request approval before running." if status == "waiting_approval" else "Run the command if needed.",
            related_command=command,
        )
        return ToolResult(
            tool_name=self.spec.name,
            status=status,
            observation=str(preview.get("reason") or ""),
            command_result=preview,
            next_recommended_action="request_command_approval" if status == "waiting_approval" else "run_approved_command",
        )


class InspectGitStateTool(PreviewCommandTool):
    spec = ToolSpec(
        name="inspect_git_state",
        description="Preview or classify a Git state/history command through command policy before execution.",
        input_schema=PreviewCommandTool.spec.input_schema,
        read_only=True,
        concurrency_safe=True,
    )


class RunApprovedCommandTool(BaseTool):
    spec = ToolSpec(
        name="run_approved_command",
        description="Execute a command only after command policy proves it is read-only or has explicit approval.",
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "array", "items": {"type": "string"}},
                "approval_id": {"type": "string"},
                "remember_for_session": {"type": "boolean"},
            },
            "required": ["command"],
            "additionalProperties": True,
        },
        read_only=False,
        concurrency_safe=False,
        requires_approval_by_default=True,
    )

    def check_permission(self, payload: dict[str, Any], context: ToolPermissionContext) -> PermissionDecision:
        raw_command = _command_from_payload(payload)
        shape = validate_argv_shape(raw_command)
        if not shape.allowed:
            return PermissionDecision.deny(shape.reason, command_security=shape.model_dump())
        command = list(raw_command)
        preview = command_policy_preview(command, project_path=context.request.project_path, reason=context.reason)
        if preview.get("blocked"):
            return PermissionDecision.deny(str(preview.get("reason") or "Command is blocked by policy."), command_preview=preview)
        if preview.get("read_only") and not preview.get("needs_approval"):
            return PermissionDecision.allow("Read-only command allowed by command policy.", command_preview=preview)
        approval_id = str(payload.get("approval_id") or "")
        if approval_id and approval_id == preview.get("approval_id"):
            return PermissionDecision.allow("Command approval id supplied; command service remains authoritative.", command_preview=preview)
        return PermissionDecision.ask(
            str(preview.get("reason") or "Command requires approval before execution."),
            approval_id=str(preview.get("approval_id") or ""),
            command_preview=preview,
        )

    def call(self, payload: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        raw_command = _command_from_payload(payload)
        shape = validate_argv_shape(raw_command)
        if not shape.allowed:
            return ToolResult(
                tool_name=self.spec.name,
                status="failed",
                observation=shape.reason,
                payload={"command_security": shape.model_dump()},
            )
        command = list(raw_command)
        result = run_command_with_policy(
            command,
            project_path=context.request.project_path,
            reason=str(payload.get("reason_summary") or ""),
            approval_id=payload.get("approval_id"),
            remember_for_session=bool(payload.get("remember_for_session")),
        )
        return ToolResult(
            tool_name=self.spec.name,
            status="success" if result.get("exit_code") == 0 else "failed",
            observation=str(result.get("stdout") or result.get("stderr") or ""),
            command_result=result,
        )


class GenerateEditTool(BaseTool):
    spec = ToolSpec(
        name="generate_edit",
        description="Prepare a validated proposal-only patch for already identified text files without writing files.",
        input_schema={
            "type": "object",
            "properties": {"target_files": {"type": "array", "items": {"type": "string"}, "maxItems": 4}},
            "required": ["target_files"],
            "additionalProperties": True,
        },
        read_only=False,
        concurrency_safe=False,
        max_result_chars=500_000,
    )

    def check_permission(self, payload: dict[str, Any], context: ToolPermissionContext) -> PermissionDecision:
        return PermissionDecision.allow("Edit generation is proposal-only and writes no files.")

    def call(self, payload: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        target_files = [str(item) for item in payload.get("target_files") or []]
        proposals: list[dict[str, Any]] = []
        for relative_path in target_files[:4]:
            target = validate_repo_file(context.request.project_path, relative_path)
            if not is_supported_text_file(target):
                continue
            content = target.read_text(encoding="utf-8", errors="replace")
            proposal = model_generate_edit_proposal(relative_path, content, context.request.task, payload)
            if proposal is None:
                proposed = propose_content_update(relative_path, content, context.request.task)
                proposal = build_fallback_edit_proposal(relative_path, content, proposed, context.request.task)
            if proposal and proposal.get("proposed_content") != content:
                proposals.append(
                    {
                        "file": relative_path,
                        "summary": str(proposal.get("summary") or "Prepare a safe minimal edit proposal."),
                        "before_summary": summarize_code_change(content),
                        "after_summary": summarize_code_change(str(proposal.get("proposed_content") or "")),
                        "proposed_content": str(proposal.get("proposed_content") or ""),
                        "diff_summary": str(proposal.get("unified_diff") or summarize_diff(content, str(proposal.get("proposed_content") or ""))),
                        "risk_notes": list(proposal.get("risk_notes") or []),
                        "preserves_existing_behavior": bool(proposal.get("preserves_existing_behavior")),
                    }
                )
        if proposals:
            append_activity_event(
                run_id=context.run_id,
                request=context.request,
                activity_id="generate-edit:" + ",".join(item["file"] for item in proposals)[:120],
                event_type="activity_completed",
                phase="Editing",
                label="Prepared patch",
                status="completed",
                observation="Prepared a proposed patch without writing files.",
                current_action="Built a minimal proposal from the file contents already read.",
                next_action="Report the proposal honestly as not applied.",
                related_files=[item["file"] for item in proposals],
                aggregate={"applied": False, "proposal_count": len(proposals)},
            )
        status = "success" if proposals else "skipped"
        return ToolResult(
            tool_name=self.spec.name,
            status=status,
            observation="Prepared a proposed edit. No file was written." if proposals else "No safe edit proposal could be generated.",
            files_read=target_files,
            payload={"edit_proposals": proposals, "applied": False},
            next_recommended_action="write_file" if proposals else None,
        )


class AskClarificationTool(BaseTool):
    spec = ToolSpec(
        name="ask_clarification",
        description="Stop the loop and ask the user for a precise missing file, scope, approval, or workflow detail.",
        input_schema={"type": "object", "properties": {"question": {"type": "string"}}, "additionalProperties": True},
        read_only=True,
        concurrency_safe=True,
    )

    def call(self, payload: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        return ToolResult(
            tool_name=self.spec.name,
            status="success",
            observation=str(payload.get("question") or payload.get("reason_summary") or "Clarification needed."),
            payload=payload,
        )


class FinalAnswerTool(BaseTool):
    spec = ToolSpec(
        name="final_answer",
        description="Stop tool execution so final synthesis can answer from gathered evidence.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": True},
        read_only=True,
        concurrency_safe=True,
    )

    def call(self, payload: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        return ToolResult(tool_name=self.spec.name, status="success", observation="Ready to prepare the final answer.")


def find_file_candidates(repo: Path, queries: list[str], *, text_queries: list[str] | None = None, max_results: int = 8) -> list[dict[str, Any]]:
    skip_dirs = {".git", ".claude", "node_modules", "runtime", ".next", "dist", "build", "out", "coverage", ".venv", "venv", "__pycache__"}
    files: list[Path] = []
    for path in repo.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(repo)
        if any(part.lower() in skip_dirs for part in rel.parts):
            continue
        if is_stale_duplicate_copy(rel):
            continue
        if not is_supported_text_file(path):
            continue
        files.append(path)
    text_queries = text_queries or []
    scored: dict[str, dict[str, Any]] = {}
    for path in files:
        rel = path.relative_to(repo)
        rel_text = str(rel)
        path_lower = rel_text.lower()
        name_lower = path.name.lower()
        stem_lower = path.stem.lower()
        score = 0.0
        reasons: list[str] = []
        matched: list[str] = []
        for query in queries:
            lowered = query.lower()
            query_name = Path(query).name.lower()
            if lowered.startswith("*.") and path.suffix.lower() == lowered[1:]:
                score += 4.0
                reasons.append(f"extension: {lowered}")
                matched.append(query)
            elif path_lower == lowered:
                score += 120.0
                reasons.append(f"exact path: {query}")
                matched.append(query)
            elif name_lower == query_name:
                score += 90.0
                reasons.append(f"basename: {query_name}")
                matched.append(query)
            elif query_name and query_name.rstrip(".cs").lower() in stem_lower:
                score += 35.0
                reasons.append(f"name contains: {query}")
                matched.append(query)
        text = ""
        if text_queries or any("." not in query and not query.startswith("*.") for query in queries):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")[:120_000]
            except OSError:
                text = ""
        for query in queries:
            if "." in query or query.startswith("*.") or not text:
                continue
            if re.search(rf"\b(class|struct|interface|enum|def|function|const|let|var)\s+{re.escape(query)}\b", text):
                score += 70.0
                reasons.append(f"symbol: {query}")
                matched.append(query)
        for query in text_queries:
            if not text:
                continue
            count = text.lower().count(query.lower())
            if count:
                score += min(60.0, 14.0 * count)
                reasons.append(f"contains: {query}")
                matched.append(query)
        if score > 0:
            source_rank = candidate_priority(rel)
            score += max(0.0, 5.0 - source_rank[0] - source_rank[1])
            scored[rel_text] = {
                "path": rel_text,
                "score": round(score, 2),
                "reasons": _dedupe_strings(reasons),
                "matched_queries": _dedupe_strings(matched),
            }
    return sorted(scored.values(), key=lambda item: (-float(item["score"]), item["path"]))[:max_results]


def search_text_matches(
    repo: Path,
    *,
    query: str,
    path_globs: list[str],
    max_results: int,
    case_sensitive: bool,
    regex: bool,
    context_lines: int,
) -> tuple[list[dict[str, Any]], int, bool]:
    skip_dirs = {".git", ".claude", "node_modules", "runtime", ".next", "dist", "build", "out", "coverage", ".venv", "venv", "__pycache__"}
    patterns = path_globs or ["**/*"]
    matches: list[dict[str, Any]] = []
    files_searched = 0
    truncated = False
    flags = 0 if case_sensitive else re.IGNORECASE
    compiled: re.Pattern[str] | None = None
    if regex:
        try:
            compiled = re.compile(query, flags=flags)
        except re.error as exc:
            return ([{"path": "", "line": 0, "column": 0, "preview": f"Invalid regex: {exc}"}], 0, False)
    needle = query if case_sensitive else query.lower()
    for path in repo.rglob("*"):
        if len(matches) >= max_results:
            truncated = True
            break
        if not path.is_file():
            continue
        rel = path.relative_to(repo)
        rel_text = str(rel)
        if any(part.lower() in skip_dirs for part in rel.parts):
            continue
        if not any(Path(rel_text).match(pattern) for pattern in patterns):
            continue
        if is_stale_duplicate_copy(rel) or not is_supported_text_file(path):
            continue
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")[:240_000]
        except OSError:
            continue
        files_searched += 1
        lines = raw.splitlines()
        for line_number, line in enumerate(lines, start=1):
            if len(matches) >= max_results:
                truncated = True
                break
            if compiled:
                found = next(compiled.finditer(line), None)
                if not found:
                    continue
                column = found.start() + 1
            else:
                haystack = line if case_sensitive else line.lower()
                index = haystack.find(needle)
                if index < 0:
                    continue
                column = index + 1
            start_context = max(0, line_number - 1 - context_lines)
            end_context = min(len(lines), line_number + context_lines)
            before = [_redact_preview(item) for item in lines[start_context : line_number - 1]]
            after = [_redact_preview(item) for item in lines[line_number:end_context]]
            matches.append(
                {
                    "path": rel_text,
                    "line": line_number,
                    "column": column,
                    "preview": _redact_preview(line),
                    "before": before,
                    "after": after,
                }
            )
    return matches, files_searched, truncated


def candidate_priority(relative_path: Path) -> tuple[int, int, str]:
    parts = [part.lower() for part in relative_path.parts]
    source = 0 if relative_path.suffix.lower() in {".cs", ".py", ".js", ".ts", ".tsx"} else 1
    source_dir = 0 if any(part in {"assets", "scripts", "src", "app", "apps"} for part in parts) else 1
    return (source + source_dir, len(relative_path.parts), str(relative_path).lower())


EDIT_PROPOSAL_PROMPT = """\
You are RepoOperator's edit proposal generator. Return JSON only.
Create a proposal for the single provided file. Do not claim the change was applied.
Schema:
{
  "file": "repo-relative path",
  "summary": "short summary",
  "proposed_content": "complete replacement content for this file",
  "unified_diff": "unified diff from original to proposed",
  "risk_notes": [],
  "preserves_existing_behavior": true
}
Preserve existing class structure and lifecycle methods unless the requested change requires otherwise.
"""


def model_generate_edit_proposal(relative_path: str, content: str, task: str, context: dict[str, Any] | None = None) -> dict[str, Any] | None:
    try:
        raw = OpenAICompatibleModelClient().generate_text(
            ModelGenerationRequest(
                system_prompt=EDIT_PROPOSAL_PROMPT,
                user_prompt=json.dumps(
                    {
                        "task": task,
                        "file": relative_path,
                        "content": content[:80_000],
                        "context": json_safe(context or {}),
                    },
                    ensure_ascii=False,
                ),
            )
        )
        payload = parse_json_object(raw)
    except Exception:
        return None
    return validate_edit_proposal(relative_path, content, payload, task)


def validate_edit_proposal(relative_path: str, original: str, payload: dict[str, Any], task: str) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    if str(payload.get("file") or relative_path) != relative_path:
        return None
    proposed = str(payload.get("proposed_content") or "")
    if not proposed.strip() or proposed == original or len(proposed) > max(200_000, len(original) * 5):
        return None
    original_structure = extract_source_structure(relative_path, original)
    proposed_structure = extract_source_structure(relative_path, proposed)
    risk_notes = [str(item) for item in payload.get("risk_notes") or []]
    risk_text = " ".join(risk_notes).lower()
    missing_classes = sorted(set(original_structure["classes"]) - set(proposed_structure["classes"]))
    if missing_classes and not mentions_removal_justification(task, risk_text, missing_classes):
        return None
    missing_public = sorted(set(original_structure["public_members"]) - set(proposed_structure["public_members"]))
    if missing_public and not mentions_removal_justification(task, risk_text, missing_public):
        return None
    missing_fields = sorted(set(original_structure["serialized_or_public_fields"]) - set(proposed_structure["serialized_or_public_fields"]))
    if missing_fields and not mentions_removal_justification(task, risk_text, missing_fields):
        return None
    missing_lifecycle = sorted(set(original_structure["unity_lifecycle_methods"]) - set(proposed_structure["unity_lifecycle_methods"]))
    for method in missing_lifecycle:
        if method == "Update" and method_body_is_empty(original, method):
            continue
        if not mentions_removal_justification(task, risk_text, [method]):
            return None
    if relative_path.lower().endswith(".cs") and not csharp_roughly_valid(proposed):
        return None
    hardening = "BinaryFormatter" in original or "binaryformatter" in task.lower()
    if hardening and "BinaryFormatter" in proposed:
        return None
    if hardening and not ("JsonUtility" in proposed or "System.Text.Json" in proposed or "Newtonsoft.Json" in proposed):
        return None
    if hardening and not ("File.Exists" in proposed and ("catch" in proposed or "try" in proposed)):
        return None
    if unsafe_bitwise_change(original, proposed):
        return None
    diff = str(payload.get("unified_diff") or summarize_diff(original, proposed))
    return {
        "file": relative_path,
        "summary": str(payload.get("summary") or "Prepared edit proposal."),
        "proposed_content": proposed,
        "unified_diff": diff,
        "risk_notes": risk_notes,
        "preserves_existing_behavior": bool(payload.get("preserves_existing_behavior", True)),
        "removed_members": {
            "classes": missing_classes,
            "public_members": missing_public,
            "serialized_or_public_fields": missing_fields,
            "unity_lifecycle_methods": missing_lifecycle,
        },
        "preserved_members": proposed_structure,
    }


def build_fallback_edit_proposal(relative_path: str, original: str, proposed: str, task: str) -> dict[str, Any] | None:
    if proposed == original:
        return None
    payload = {
        "file": relative_path,
        "summary": fallback_summary(relative_path, original, proposed, task),
        "proposed_content": proposed,
        "unified_diff": summarize_diff(original, proposed),
        "risk_notes": fallback_risk_notes(original, proposed),
        "preserves_existing_behavior": preserves_named_members(original, proposed),
    }
    return validate_edit_proposal(relative_path, original, payload, task)


def propose_content_update(relative_path: str, content: str, task: str) -> str:
    name = Path(relative_path).name.lower()
    task_text = task or ""
    proposed = content
    explicit_border_edit = Path(relative_path).name in task_text or "&&" in task_text or "&" in task_text
    if name == "border.cs" and explicit_border_edit:
        proposed = replace_boolean_ampersands(proposed)
        proposed = re.sub(
            r"\n\s*(?:private|public|protected|internal)?\s*void\s+Update\s*\(\s*\)\s*\{\s*\}",
            "",
            proposed,
            flags=re.MULTILINE,
        )
    if name == "datahandler.cs" or "BinaryFormatter" in content:
        proposed = propose_json_save_handler(content)
    return proposed


def propose_json_save_handler(content: str) -> str:
    without_binary_formatter = re.sub(r"^\s*using\s+System\.Runtime\.Serialization\.Formatters\.Binary;\s*\n", "", content, flags=re.MULTILINE)
    without_file_stream = re.sub(r"^\s*using\s+System\.Runtime\.Serialization;\s*\n", "", without_binary_formatter, flags=re.MULTILINE)
    if "BinaryFormatter" not in without_file_stream and ".dat" not in without_file_stream:
        return without_file_stream
    if "class DataHandler" not in without_file_stream:
        return without_file_stream.replace("BinaryFormatter", "JsonUtility")
    preserved_methods = "\n\n".join(extract_methods(without_file_stream, ["Awake", "Start"]))
    preserved_members = extract_class_member_lines(without_file_stream)
    preserved_block = (preserved_members + "\n\n" + preserved_methods).strip()
    preserved_block = f"\n{preserved_block}\n\n" if preserved_block else "\n"
    return f"""using System;
using System.IO;
using UnityEngine;

public class DataHandler : MonoBehaviour
{{{preserved_block}
    private string SavePath => Path.Combine(Application.persistentDataPath, "playerData.json");

    public void Save(PlayerData data)
    {{
        string json = JsonUtility.ToJson(data);
        File.WriteAllText(SavePath, json);
    }}

    public PlayerData Load()
    {{
        if (!File.Exists(SavePath))
        {{
            return new PlayerData();
        }}

        try
        {{
            string json = File.ReadAllText(SavePath);
            PlayerData data = JsonUtility.FromJson<PlayerData>(json);
            return data ?? new PlayerData();
        }}
        catch (Exception)
        {{
            return new PlayerData();
        }}
    }}
}}
"""


def summarize_code_change(content: str) -> str:
    markers: list[str] = []
    if "BinaryFormatter" in content:
        markers.append("uses BinaryFormatter")
    if "JsonUtility" in content:
        markers.append("uses JsonUtility")
    if re.search(r"(?<!&)&(?!&)", content):
        markers.append("contains single ampersand boolean checks")
    if re.search(r"void\s+Update\s*\(\s*\)\s*\{\s*\}", content, flags=re.MULTILINE):
        markers.append("has an empty Update method")
    return ", ".join(markers) if markers else f"{len(content.splitlines())} line(s)"


def summarize_diff(before: str, after: str, *, limit: int = 4000) -> str:
    diff = "\n".join(
        unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile="before",
            tofile="after",
            lineterm="",
        )
    )
    return diff[:limit]


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines)
    try:
        payload = json.loads(stripped)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        return {}


def csharp_roughly_valid(content: str) -> bool:
    balance = 0
    for char in content:
        if char == "{":
            balance += 1
        elif char == "}":
            balance -= 1
        if balance < 0:
            return False
    return balance == 0


def extract_source_structure(relative_path: str, content: str) -> dict[str, list[str]]:
    suffix = Path(relative_path).suffix.lower()
    if suffix != ".cs":
        return {
            "classes": [],
            "methods": [],
            "fields": [],
            "unity_lifecycle_methods": [],
            "public_members": [],
            "serialized_or_public_fields": [],
        }
    classes = re.findall(r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)\b", content)
    methods = re.findall(
        r"\b(?:public|private|protected|internal)?\s*(?:static\s+)?(?:void|bool|int|string|float|double|PlayerData|[A-Za-z_][A-Za-z0-9_<>,\[\]]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        content,
    )
    field_pattern = re.compile(
        r"^\s*(?:\[SerializeField\]\s*)?(?:(public|private|protected|internal)\s+)?(?:static\s+)?(?:readonly\s+)?[A-Za-z_][A-Za-z0-9_<>,\[\]]*\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:=|;)",
        re.MULTILINE,
    )
    fields: list[str] = []
    serialized_or_public: list[str] = []
    for match in field_pattern.finditer(content):
        visibility = match.group(1) or ""
        name = match.group(2)
        fields.append(name)
        prefix = content[max(0, match.start() - 80):match.start()]
        if visibility == "public" or "[SerializeField]" in prefix:
            serialized_or_public.append(name)
    lifecycle_names = {"Awake", "Start", "Update", "FixedUpdate", "LateUpdate", "OnEnable", "OnDisable", "OnDestroy"}
    lifecycle = [name for name in methods if name in lifecycle_names]
    public_methods = re.findall(
        r"\bpublic\s+(?:static\s+)?(?:void|bool|int|string|float|double|PlayerData|[A-Za-z_][A-Za-z0-9_<>,\[\]]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        content,
    )
    public_fields = [
        match.group(2)
        for match in field_pattern.finditer(content)
        if (match.group(1) or "") == "public"
    ]
    return {
        "classes": _dedupe_strings(classes),
        "methods": _dedupe_strings(methods),
        "fields": _dedupe_strings(fields),
        "unity_lifecycle_methods": _dedupe_strings(lifecycle),
        "public_members": _dedupe_strings([*public_methods, *public_fields]),
        "serialized_or_public_fields": _dedupe_strings(serialized_or_public),
    }


def mentions_removal_justification(task: str, risk_text: str, names: list[str]) -> bool:
    lowered_task = (task or "").lower()
    if "remove" in lowered_task or "delete" in lowered_task or "\uc81c\uac70" in task:
        return True
    return any(name.lower() in risk_text for name in names) and any(word in risk_text for word in ("remove", "rename", "delete", "drop"))


def method_body_is_empty(content: str, method_name: str) -> bool:
    match = re.search(rf"\b{re.escape(method_name)}\s*\([^)]*\)\s*\{{", content)
    if not match:
        return False
    start = content.find("{", match.start())
    end = find_matching_brace(content, start)
    if end == -1:
        return False
    body = re.sub(r"//.*|/\*.*?\*/", "", content[start + 1:end], flags=re.DOTALL).strip()
    return not body


def unsafe_bitwise_change(original: str, proposed: str) -> bool:
    original_lines = original.splitlines()
    proposed_text = proposed
    for line in original_lines:
        stripped = line.strip()
        if not re.search(r"(?<!&)&(?!&)", stripped):
            continue
        bitwise_like = re.search(r"\b(int|long|uint|ulong|short|byte)\b", stripped) or re.search(r"\b(mask|flag|flags|bits?)\b", stripped, re.IGNORECASE)
        if not bitwise_like:
            continue
        if stripped not in proposed_text and re.sub(r"(?<!&)&(?!&)", "&&", stripped) in proposed_text:
            return True
    return False


def replace_boolean_ampersands(content: str) -> str:
    updated: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        likely_boolean_context = stripped.startswith(("if ", "if(", "while ", "while(", "return ")) or re.search(r"\bbool\b", stripped)
        likely_bitwise_context = re.search(r"\b(int|long|uint|ulong|short|byte)\b", stripped) or re.search(r"\b(mask|flag|flags|bits?)\b", stripped, re.IGNORECASE)
        if likely_boolean_context and not likely_bitwise_context:
            line = re.sub(r"(?<!&)&(?!&)", "&&", line)
        updated.append(line)
    return "\n".join(updated) + ("\n" if content.endswith("\n") else "")


def extract_methods(content: str, names: list[str]) -> list[str]:
    methods: list[str] = []
    for name in names:
        match = re.search(
            rf"((?:public|private|protected|internal)?\s*void\s+{re.escape(name)}\s*\([^)]*\)\s*\{{)",
            content,
        )
        if not match:
            continue
        start = match.start(1)
        brace = content.find("{", match.start(1))
        end = find_matching_brace(content, brace)
        if end != -1:
            methods.append(content[start : end + 1].strip())
    return methods


def find_matching_brace(content: str, start: int) -> int:
    if start < 0:
        return -1
    balance = 0
    for index in range(start, len(content)):
        char = content[index]
        if char == "{":
            balance += 1
        elif char == "}":
            balance -= 1
            if balance == 0:
                return index
    return -1


def extract_class_member_lines(content: str) -> str:
    class_match = re.search(r"\bclass\s+DataHandler\b[^{]*\{", content)
    if not class_match:
        return ""
    body_start = class_match.end()
    lines = []
    for line in content[body_start:].splitlines():
        stripped = line.strip()
        if not stripped or stripped == "}":
            continue
        if "(" in stripped or "{" in stripped or "}" in stripped:
            continue
        if any(skip in stripped for skip in ("BinaryFormatter", "FileStream", "formatter")):
            continue
        if stripped.endswith(";"):
            lines.append(line.rstrip())
    return "\n".join(lines[:12])


def preserves_named_members(original: str, proposed: str) -> bool:
    names = re.findall(r"\b(?:void|PlayerData|public|private|protected|internal)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", original)
    essential = [name for name in names if name not in {"Save", "Load"}]
    return all(re.search(rf"\b{name}\s*\(", proposed) for name in essential)


def fallback_summary(relative_path: str, original: str, proposed: str, task: str) -> str:
    if "BinaryFormatter" in original and "BinaryFormatter" not in proposed:
        return "Replace BinaryFormatter persistence with JsonUtility JSON persistence and corrupt-file fallback."
    if Path(relative_path).name.lower() == "border.cs":
        return "Use short-circuit boolean checks where safe and remove an empty Unity Update method."
    return "Prepared a validated proposal from the requested file content."


def fallback_risk_notes(original: str, proposed: str) -> list[str]:
    notes: list[str] = []
    if "BinaryFormatter" in original and "BinaryFormatter" not in proposed:
        notes.append("Existing binary save files will not be migrated by this proposal.")
    if not preserves_named_members(original, proposed):
        notes.append("Some existing methods may need manual review before applying.")
    return notes


def is_stale_duplicate_copy(relative_path: Path) -> bool:
    return bool(re.search(r" 2\.(py|tsx|js|json|cs)$", str(relative_path), flags=re.IGNORECASE))


def is_supported_text_file(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in BINARY_OR_CACHE_SUFFIXES:
        return False
    if suffix not in TEXT_FILE_SUFFIXES and path.name.lower() not in TEXT_FILE_BASENAMES:
        return False
    try:
        sample = path.read_bytes()[:4096]
    except OSError:
        return False
    if b"\x00" in sample:
        return False
    if not sample:
        return True
    controlish = sum(1 for byte in sample if byte < 9 or (13 < byte < 32))
    return (controlish / max(1, len(sample))) < 0.05


def _dedupe_strings(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item not in result:
            result.append(item)
    return result


def _redact_preview(text: str, *, limit: int = 240) -> str:
    redacted, _findings = redact_secrets(str(text or "")[:limit])
    return redacted


def _command_from_payload(payload: dict[str, Any], default: list[str] | None = None) -> list[str] | str:
    command = payload.get("command")
    if command is None:
        return list(default or [])
    if isinstance(command, str):
        return command
    if isinstance(command, list):
        return [str(item) for item in command]
    return str(command)
