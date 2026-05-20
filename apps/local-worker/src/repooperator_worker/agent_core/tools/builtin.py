from __future__ import annotations

import ast
import json
import re
import shlex
import tomllib
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
from repooperator_worker.services.common import ensure_git_repository, resolve_project_path
from repooperator_worker.services.permissions_service import permission_profile
from repooperator_worker.services.json_safe import json_safe, safe_agent_response_payload
from repooperator_worker.services.model_client import ModelGenerationRequest, OpenAICompatibleModelClient
from repooperator_worker.services.subprocess_utils import run_subprocess


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
        operation="list_files",
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
        operation="read_file",
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
        operation="search",
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
        operation="search",
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
        operation="analyze_repository",
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
        operation="command",
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
        operation="command",
        input_schema=PreviewCommandTool.spec.input_schema,
        read_only=True,
        concurrency_safe=True,
    )


class RunApprovedCommandTool(BaseTool):
    spec = ToolSpec(
        name="run_approved_command",
        description="Execute a command only after command policy proves it is read-only or has explicit approval.",
        operation="command",
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


class RunValidationCommandTool(RunApprovedCommandTool):
    spec = ToolSpec(
        name="run_validation_command",
        description="Run an approved validation command through the command permission path.",
        operation="command",
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
        side_effect_level="command",
        permission_required=True,
        parallel_safe=False,
        produces_evidence=True,
    )


class _NetworkEvidenceTool(BaseTool):
    def check_permission(self, payload: dict[str, Any], context: ToolPermissionContext) -> PermissionDecision:
        del context
        decision = payload.get("approval_decision") if isinstance(payload.get("approval_decision"), dict) else {}
        if str(decision.get("decision") or "").lower() == "allow":
            return PermissionDecision.allow("User approved network access for this web evidence tool.")
        try:
            profile = permission_profile()
        except Exception:
            profile = {}
        approval = profile.get("approval") if isinstance(profile.get("approval"), dict) else {}
        sandbox = profile.get("sandbox") if isinstance(profile.get("sandbox"), dict) else {}
        if sandbox.get("allowNetwork") and not approval.get("requireForNetwork", True):
            return PermissionDecision.allow("Network access is allowed by the active permission profile.")
        return PermissionDecision.ask("Web research uses network access and requires approval by the active permission profile.")


class SearchWebTool(_NetworkEvidenceTool):
    spec = ToolSpec(
        name="search_web",
        description="Search the public web and return untrusted evidence records with source metadata.",
        operation="web_search",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 8},
            },
            "required": ["query"],
            "additionalProperties": True,
        },
        read_only=True,
        concurrency_safe=True,
        requires_approval_by_default=True,
        side_effect_level="none",
        permission_required=True,
        parallel_safe=True,
        workspace_bound=False,
        network_access=True,
        produces_evidence=True,
        max_result_chars=200_000,
    )

    def validate_input(self, payload: dict[str, Any], request) -> dict[str, Any]:
        del request
        query = " ".join(str(payload.get("query") or "").split())[:300]
        return {**dict(payload), "query": query, "max_results": max(1, min(int(payload.get("max_results") or 5), 8))}

    def call(self, payload: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        from repooperator_worker.agent_core.web_research import search_web

        records = search_web(str(payload.get("query") or ""), run_id=context.run_id, max_results=int(payload.get("max_results") or 5))
        evidence = [record.model_dump() for record in records]
        append_activity_event(
            run_id=context.run_id,
            request=context.request,
            activity_id="web-search:" + str(payload.get("query") or "")[:80],
            event_type="activity_completed",
            phase="Research",
            label="Searched web",
            status="completed",
            observation=f"Searched web and found {len(evidence)} source(s).",
            related_search_query=str(payload.get("query") or ""),
            aggregate={"operation": "web_search", "query": payload.get("query"), "source_count": len(evidence), "sources": _source_notes(evidence)},
        )
        return ToolResult(
            tool_name=self.spec.name,
            status="success",
            observation=f"Searched web for {payload.get('query')}; found {len(evidence)} source(s).",
            payload={"query": payload.get("query"), "web_evidence": evidence, "untrusted": True},
        )


class FetchUrlTool(_NetworkEvidenceTool):
    spec = ToolSpec(
        name="fetch_url",
        description="Fetch a public URL as sanitized, untrusted evidence with citation metadata.",
        operation="web_fetch",
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "max_bytes": {"type": "integer", "minimum": 1, "maximum": 750000},
            },
            "required": ["url"],
            "additionalProperties": True,
        },
        read_only=True,
        concurrency_safe=True,
        requires_approval_by_default=True,
        side_effect_level="none",
        permission_required=True,
        parallel_safe=True,
        workspace_bound=False,
        network_access=True,
        produces_evidence=True,
        max_result_chars=300_000,
    )

    def call(self, payload: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        from repooperator_worker.agent_core.web_research import fetch_url

        record = fetch_url(str(payload.get("url") or ""), run_id=context.run_id, max_bytes=int(payload.get("max_bytes") or 750_000))
        evidence = record.model_dump()
        append_activity_event(
            run_id=context.run_id,
            request=context.request,
            activity_id="web-fetch:" + str(payload.get("url") or "")[:100],
            event_type="activity_completed",
            phase="Research",
            label="Read docs page",
            status="completed",
            observation=f"Fetched and sanitized web evidence from {record.source}.",
            aggregate={"operation": "web_fetch", "source_count": 1, "sources": _source_notes([evidence])},
        )
        return ToolResult(
            tool_name=self.spec.name,
            status="success",
            observation=f"Fetched sanitized web evidence from {record.url}.",
            payload={"web_evidence": [evidence], "untrusted": True},
        )


class SummarizeWebEvidenceTool(BaseTool):
    spec = ToolSpec(
        name="summarize_web_evidence",
        description="Summarize already fetched web evidence with source metadata and an untrusted-content safety note.",
        operation="web_fetch",
        input_schema={"type": "object", "properties": {"web_evidence": {"type": "array"}}, "additionalProperties": True},
        read_only=True,
        concurrency_safe=True,
        side_effect_level="none",
        parallel_safe=True,
        workspace_bound=False,
        produces_evidence=True,
        max_result_chars=200_000,
    )

    def call(self, payload: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        from repooperator_worker.agent_core.web_research import summarize_web_evidence

        summary = summarize_web_evidence(list(payload.get("web_evidence") or payload.get("records") or []))
        return ToolResult(
            tool_name=self.spec.name,
            status="success",
            observation=f"Summarized {summary.get('source_count', 0)} web source(s).",
            payload={"web_evidence_summary": summary},
        )


class _GitTool(BaseTool):
    permission_level = "git_read"

    def _repo(self, context: ToolExecutionContext):
        repo = resolve_project_path(context.request.project_path)
        ensure_git_repository(repo)
        return repo


class GitStatusTool(_GitTool):
    spec = ToolSpec(
        name="git_status",
        description="Read local git status without changing repository state.",
        operation="git_status",
        input_schema={"type": "object", "properties": {}, "additionalProperties": True},
        read_only=True,
        concurrency_safe=True,
        side_effect_level="read",
        produces_evidence=True,
    )

    def call(self, payload: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        del payload
        repo = self._repo(context)
        result = run_subprocess(command=["git", "status", "--short", "--branch"], cwd=repo, timeout_seconds=30)
        return ToolResult(
            tool_name=self.spec.name,
            status="success" if result.returncode == 0 else "failed",
            observation=result.stdout.strip() or result.stderr.strip(),
            payload={"stdout": result.stdout, "stderr": result.stderr, "exit_code": result.returncode},
        )


class GitDiffTool(_GitTool):
    spec = ToolSpec(
        name="git_diff",
        description="Read local git diff without changing repository state.",
        operation="git_diff",
        input_schema={"type": "object", "properties": {"staged": {"type": "boolean"}, "relative_paths": {"type": "array", "items": {"type": "string"}}}, "additionalProperties": True},
        read_only=True,
        concurrency_safe=True,
        side_effect_level="read",
        produces_evidence=True,
        max_result_chars=400_000,
    )

    def call(self, payload: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        repo = self._repo(context)
        command = ["git", "diff"]
        if payload.get("staged"):
            command.append("--cached")
        relative_paths = [str(item).strip().lstrip("/") for item in payload.get("relative_paths") or [] if str(item).strip()]
        if relative_paths:
            command.extend(["--", *relative_paths])
        result = run_subprocess(command=command, cwd=repo, timeout_seconds=30)
        return ToolResult(
            tool_name=self.spec.name,
            status="success" if result.returncode == 0 else "failed",
            observation=result.stdout,
            payload={"diff": result.stdout, "stderr": result.stderr, "exit_code": result.returncode, "relative_paths": relative_paths},
        )


class GitLogTool(_GitTool):
    spec = ToolSpec(
        name="git_log",
        description="Read recent local git history without changing repository state.",
        operation="git_log",
        input_schema={"type": "object", "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 50}}, "additionalProperties": True},
        read_only=True,
        concurrency_safe=True,
        side_effect_level="read",
        produces_evidence=True,
    )

    def call(self, payload: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        repo = self._repo(context)
        limit = max(1, min(int(payload.get("limit") or 10), 50))
        result = run_subprocess(command=["git", "log", "--oneline", "-n", str(limit)], cwd=repo, timeout_seconds=30)
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        return ToolResult(
            tool_name=self.spec.name,
            status="success" if result.returncode == 0 else "failed",
            observation="\n".join(lines),
            payload={"commits": lines, "stderr": result.stderr, "exit_code": result.returncode},
        )


class _GitWriteTool(_GitTool):
    def check_permission(self, payload: dict[str, Any], context: ToolPermissionContext) -> PermissionDecision:
        decision = payload.get("approval_decision") if isinstance(payload.get("approval_decision"), dict) else {}
        if str(decision.get("decision") or "").lower() == "allow":
            return PermissionDecision.allow("User approved this git write action.", git_permission_level=self.permission_level)
        return PermissionDecision.ask(self._approval_reason(payload), git_permission_level=self.permission_level, approval_payload=self._approval_payload(payload, context))

    def _approval_reason(self, payload: dict[str, Any]) -> str:
        del payload
        return "Git write action requires explicit approval."

    def _approval_payload(self, payload: dict[str, Any], context: ToolPermissionContext) -> dict[str, Any]:
        del context
        return json_safe(payload)


class GitBranchCreateTool(_GitWriteTool):
    permission_level = "git_local_write"
    spec = ToolSpec(
        name="git_branch_create",
        description="Create a local branch only after explicit approval.",
        operation="git_branch",
        input_schema={"type": "object", "properties": {"branch": {"type": "string"}, "from_ref": {"type": "string"}, "checkout": {"type": "boolean"}, "approval_decision": {"type": "object"}}, "required": ["branch"], "additionalProperties": True},
        read_only=False,
        concurrency_safe=False,
        requires_approval_by_default=True,
        side_effect_level="write",
        permission_required=True,
        parallel_safe=False,
    )

    def _approval_reason(self, payload: dict[str, Any]) -> str:
        return f"Creating branch {payload.get('branch') or ''} changes local git state and requires approval."

    def call(self, payload: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        repo = self._repo(context)
        branch = str(payload.get("branch") or "").strip()
        from_ref = str(payload.get("from_ref") or "HEAD").strip()
        command = ["git", "checkout", "-b", branch, from_ref] if payload.get("checkout", True) else ["git", "branch", branch, from_ref]
        result = run_subprocess(command=command, cwd=repo, timeout_seconds=30)
        return ToolResult(tool_name=self.spec.name, status="success" if result.returncode == 0 else "failed", observation=result.stdout or result.stderr, payload={"command": command, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr, "branch": branch})


class GitCommitTool(_GitWriteTool):
    permission_level = "git_local_write"
    spec = ToolSpec(
        name="git_commit",
        description="Create a local commit only after explicit approval; the message and files must be visible before approval.",
        operation="git_commit",
        input_schema={"type": "object", "properties": {"message": {"type": "string"}, "stage_all": {"type": "boolean"}, "files": {"type": "array", "items": {"type": "string"}}, "approval_decision": {"type": "object"}}, "required": ["message"], "additionalProperties": True},
        read_only=False,
        concurrency_safe=False,
        requires_approval_by_default=True,
        side_effect_level="write",
        permission_required=True,
        parallel_safe=False,
        produces_evidence=True,
    )

    def _approval_reason(self, payload: dict[str, Any]) -> str:
        return f"Creating a commit requires approval. Proposed message: {payload.get('message') or ''}"

    def _approval_payload(self, payload: dict[str, Any], context: ToolPermissionContext) -> dict[str, Any]:
        return {"message": payload.get("message"), "files": payload.get("files") or [], "repo": getattr(context.request, "project_path", None)}

    def call(self, payload: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        repo = self._repo(context)
        if payload.get("stage_all", True):
            add = run_subprocess(command=["git", "add", "--all"], cwd=repo, timeout_seconds=30)
            if add.returncode != 0:
                return ToolResult(tool_name=self.spec.name, status="failed", observation=add.stderr, payload={"exit_code": add.returncode, "stderr": add.stderr})
        message = str(payload.get("message") or "").strip()
        result = run_subprocess(command=["git", "commit", "-m", message], cwd=repo, timeout_seconds=60)
        sha = ""
        if result.returncode == 0:
            rev = run_subprocess(command=["git", "rev-parse", "HEAD"], cwd=repo, timeout_seconds=30)
            sha = rev.stdout.strip()
        return ToolResult(tool_name=self.spec.name, status="success" if result.returncode == 0 else "failed", observation=result.stdout or result.stderr, payload={"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr, "commit_sha": sha, "message": message})


class GitPushTool(_GitWriteTool):
    permission_level = "git_remote_write"
    spec = ToolSpec(
        name="git_push",
        description="Push a branch to a remote only after explicit approval.",
        operation="git_push",
        input_schema={"type": "object", "properties": {"remote": {"type": "string"}, "branch": {"type": "string"}, "set_upstream": {"type": "boolean"}, "approval_decision": {"type": "object"}}, "required": ["branch"], "additionalProperties": True},
        read_only=False,
        concurrency_safe=False,
        requires_approval_by_default=True,
        side_effect_level="command",
        permission_required=True,
        parallel_safe=False,
        network_access=True,
    )

    def _approval_reason(self, payload: dict[str, Any]) -> str:
        return f"Pushing {payload.get('branch') or ''} to {payload.get('remote') or 'origin'} contacts a remote and requires approval."

    def call(self, payload: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        repo = self._repo(context)
        remote = str(payload.get("remote") or "origin")
        branch = str(payload.get("branch") or "").strip()
        command = ["git", "push"]
        if payload.get("set_upstream", True):
            command.append("--set-upstream")
        command.extend([remote, branch])
        result = run_subprocess(command=command, cwd=repo, timeout_seconds=180)
        return ToolResult(tool_name=self.spec.name, status="success" if result.returncode == 0 else "failed", observation=result.stdout or result.stderr, payload={"command": command, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr, "remote": remote, "branch": branch})


class _ProviderReviewTool(_GitWriteTool):
    permission_level = "git_remote_write"

    def call(self, payload: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        del context
        return ToolResult(
            tool_name=self.spec.name,
            status="failed",
            observation="Provider review creation is approval-gated; configure the provider client before executing this remote write.",
            payload={"errors": ["provider client not configured"], "requested": json_safe(payload)},
        )


class GitHubCreatePrTool(_ProviderReviewTool):
    spec = ToolSpec(
        name="github_create_pr",
        description="Create a GitHub pull request only after explicit approval.",
        operation="git_provider_request",
        input_schema={"type": "object", "properties": {"source_branch": {"type": "string"}, "target_branch": {"type": "string"}, "title": {"type": "string"}, "body": {"type": "string"}, "approval_decision": {"type": "object"}}, "required": ["source_branch", "target_branch", "title"], "additionalProperties": True},
        read_only=False,
        concurrency_safe=False,
        requires_approval_by_default=True,
        side_effect_level="command",
        permission_required=True,
        parallel_safe=False,
        network_access=True,
    )


class GitLabCreateMrTool(_ProviderReviewTool):
    spec = ToolSpec(
        name="gitlab_create_mr",
        description="Create a GitLab merge request only after explicit approval.",
        operation="git_provider_request",
        input_schema={"type": "object", "properties": {"source_branch": {"type": "string"}, "target_branch": {"type": "string"}, "title": {"type": "string"}, "description": {"type": "string"}, "approval_decision": {"type": "object"}}, "required": ["source_branch", "target_branch", "title"], "additionalProperties": True},
        read_only=False,
        concurrency_safe=False,
        requires_approval_by_default=True,
        side_effect_level="command",
        permission_required=True,
        parallel_safe=False,
        network_access=True,
    )


class ReadManyFilesTool(ReadFileTool):
    spec = ToolSpec(
        name="read_many_files",
        description="Read a bounded batch of supported text files inside the repository.",
        operation="read_file",
        input_schema={
            "type": "object",
            "properties": {"target_files": {"type": "array", "items": {"type": "string"}, "maxItems": 20}},
            "required": ["target_files"],
            "additionalProperties": True,
        },
        read_only=True,
        concurrency_safe=True,
        max_result_chars=800_000,
        side_effect_level="read",
        produces_evidence=True,
    )

    def call(self, payload: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        updated = {**payload, "target_files": list(payload.get("target_files") or [])[:20]}
        return super().call(updated, context)


class GenerateChangeSetTool(BaseTool):
    spec = ToolSpec(
        name="generate_change_set",
        description="Generate a multi-file ChangeSetProposal without writing files.",
        operation="edit",
        input_schema={
            "type": "object",
            "properties": {
                "target_files": {"type": "array", "items": {"type": "string"}},
                "new_file_paths": {"type": "array", "items": {"type": "string"}},
                "delete_plan": {"type": "array", "items": {"type": "string"}},
                "rename_plan": {"type": "array", "items": {"type": "object"}},
                "change_plan": {"type": "object"},
                "constraints": {"type": "array", "items": {"type": "string"}},
                "validation_requirements": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": True,
        },
        read_only=False,
        concurrency_safe=False,
        max_result_chars=1_200_000,
        side_effect_level="none",
        permission_required=False,
        parallel_safe=False,
        produces_artifact=True,
        produces_evidence=True,
    )

    def check_permission(self, payload: dict[str, Any], context: ToolPermissionContext) -> PermissionDecision:
        return PermissionDecision.allow("Change-set generation is proposal-only and writes no files.")

    def call(self, payload: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        from repooperator_worker.agent_core.change_set import (
            ChangePlan,
            ChangeSetProposal,
            ProposedFileChange,
            stable_proposal_id,
            validate_change_set,
        )

        repo = resolve_project_path(context.request.project_path).resolve()
        target_files = [str(item).strip().lstrip("/") for item in payload.get("target_files") or [] if str(item).strip()]
        evidence_contents: dict[str, str] = {}
        for relative_path in target_files[:12]:
            target = validate_repo_file(context.request.project_path, relative_path)
            if target.is_file() and is_supported_text_file(target):
                evidence_contents[relative_path] = target.read_text(encoding="utf-8", errors="replace")

        raw_proposal = model_generate_change_set_proposal(
            task=context.request.task,
            repo=str(repo),
            evidence_contents=evidence_contents,
            payload=payload,
        )
        if raw_proposal is None:
            return ToolResult(
                tool_name=self.spec.name,
                status="failed",
                observation="No safe change-set proposal could be generated.",
                files_read=list(evidence_contents),
                payload={"proposal_error": "model returned no change-set proposal", "applied": False},
            )

        proposal = change_set_from_model_payload(
            raw_proposal,
            task=context.request.task,
            evidence_contents=evidence_contents,
            payload=payload,
        )
        validation = validate_change_set(proposal, repo=context.request.project_path)
        if validation.status != "valid":
            repaired = repair_change_set_proposal(
                raw_proposal,
                task=context.request.task,
                evidence_contents=evidence_contents,
                validation_errors=validation.errors,
                payload=payload,
            )
            if repaired is not None:
                repaired_proposal = change_set_from_model_payload(
                    repaired,
                    task=context.request.task,
                    evidence_contents=evidence_contents,
                    payload=payload,
                )
                repaired_validation = validate_change_set(repaired_proposal, repo=context.request.project_path)
                if repaired_validation.status == "valid":
                    proposal = repaired_proposal
                    validation = repaired_validation

        proposal.validation = validation
        proposal.status = validation.status
        proposal.validation_status = validation.status
        proposal.proposal_error = "; ".join(validation.errors) if validation.errors else None
        status = "success" if validation.status == "valid" else "failed"
        observation = (
            "Prepared a validated change-set proposal. No file was written."
            if status == "success"
            else "Change-set proposal validation failed. No file was written."
        )
        return ToolResult(
            tool_name=self.spec.name,
            status=status,
            observation=observation,
            files_read=list(evidence_contents),
            payload={"change_set_proposal": proposal.model_dump(), "applied": False, **({"proposal_error": proposal.proposal_error} if proposal.proposal_error else {})},
            next_recommended_action="await_change_approval" if status == "success" else None,
        )


class ValidateChangeSetTool(BaseTool):
    spec = ToolSpec(
        name="validate_change_set",
        description="Validate a ChangeSetProposal without writing files.",
        operation="validation",
        input_schema={"type": "object", "properties": {"change_set_proposal": {"type": "object"}}, "required": ["change_set_proposal"], "additionalProperties": True},
        read_only=True,
        concurrency_safe=True,
        side_effect_level="none",
        produces_evidence=True,
    )

    def call(self, payload: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        from repooperator_worker.agent_core.change_set import change_set_from_payload, validate_change_set

        proposal_payload = payload.get("change_set_proposal") if isinstance(payload.get("change_set_proposal"), dict) else payload
        proposal = change_set_from_payload(proposal_payload)
        validation = validate_change_set(proposal, repo=context.request.project_path)
        proposal.validation = validation
        proposal.status = validation.status
        proposal.validation_status = validation.status
        proposal.proposal_error = "; ".join(validation.errors) if validation.errors else None
        return ToolResult(
            tool_name=self.spec.name,
            status="success" if validation.status == "valid" else "failed",
            observation=f"Change-set validation status: {validation.status}.",
            payload={"change_set_proposal": proposal.model_dump(), "validation": validation.model_dump()},
        )


class ApplyChangeSetTool(BaseTool):
    spec = ToolSpec(
        name="apply_change_set",
        description="Apply an approved persisted ChangeSetProposal to disk through the safe apply path.",
        operation="write",
        input_schema={
            "type": "object",
            "properties": {
                "proposal_id": {"type": "string"},
                "approval_decision": {"type": "object"},
                "change_set_snapshot": {"type": "object"},
            },
            "required": ["proposal_id"],
            "additionalProperties": True,
        },
        read_only=False,
        concurrency_safe=False,
        requires_approval_by_default=True,
        side_effect_level="write",
        permission_required=True,
        parallel_safe=False,
        workspace_bound=True,
        produces_artifact=True,
        produces_evidence=True,
        can_be_retried=False,
        max_result_chars=1_200_000,
    )

    def check_permission(self, payload: dict[str, Any], context: ToolPermissionContext) -> PermissionDecision:
        decision = payload.get("approval_decision") if isinstance(payload.get("approval_decision"), dict) else {}
        if str(decision.get("decision") or "").lower() == "allow":
            return PermissionDecision.allow("User approved applying this change set.")
        return PermissionDecision.ask("Applying a change set writes files and requires approval.")

    def call(self, payload: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        from repooperator_worker.agent_core.apply_change_set import apply_change_set_for_run

        result = apply_change_set_for_run(
            run_id=context.run_id,
            project_path=context.request.project_path,
            proposal_id=str(payload.get("proposal_id") or ""),
            approval_decision=payload.get("approval_decision") if isinstance(payload.get("approval_decision"), dict) else {},
            fallback_change_set=payload.get("change_set_snapshot") if isinstance(payload.get("change_set_snapshot"), dict) else None,
        )
        changed = result.files_modified + result.files_created + result.files_deleted + [item["to"] for item in result.files_renamed]
        return ToolResult(
            tool_name=self.spec.name,
            status="success" if result.applied else "failed",
            observation="Applied the approved change set." if result.applied else "Failed to apply the approved change set.",
            files_changed=changed,
            payload=result.model_dump(),
            next_recommended_action="post_apply_validation" if result.applied else None,
        )


class _DirectFileWriteTool(BaseTool):
    operation_name = "write"

    def check_permission(self, payload: dict[str, Any], context: ToolPermissionContext) -> PermissionDecision:
        return PermissionDecision.ask("Direct file writes require explicit approval and should normally use apply_change_set.")

    def call(self, payload: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        return ToolResult(
            tool_name=self.spec.name,
            status="failed",
            observation="Direct file mutation tools are registered for audit visibility; normal writes must go through apply_change_set.",
            payload={"errors": ["use apply_change_set for proposed file writes"]},
        )


class CreateFileTool(_DirectFileWriteTool):
    spec = ToolSpec(
        name="create_file",
        description="Create a file after explicit approval; normal proposed writes use apply_change_set.",
        operation="write",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"], "additionalProperties": True},
        read_only=False,
        concurrency_safe=False,
        requires_approval_by_default=True,
        side_effect_level="write",
        permission_required=True,
        parallel_safe=False,
        can_be_retried=False,
    )


class ModifyFileTool(_DirectFileWriteTool):
    spec = ToolSpec(
        name="modify_file",
        description="Modify a file after explicit approval; normal proposed writes use apply_change_set.",
        operation="write",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"], "additionalProperties": True},
        read_only=False,
        concurrency_safe=False,
        requires_approval_by_default=True,
        side_effect_level="write",
        permission_required=True,
        parallel_safe=False,
        can_be_retried=False,
    )


class DeleteFileTool(_DirectFileWriteTool):
    spec = ToolSpec(
        name="delete_file",
        description="Delete a file after explicit approval; normal proposed writes use apply_change_set.",
        operation="write",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}, "justification": {"type": "string"}}, "required": ["path", "justification"], "additionalProperties": True},
        read_only=False,
        concurrency_safe=False,
        requires_approval_by_default=True,
        side_effect_level="write",
        permission_required=True,
        parallel_safe=False,
        can_be_retried=False,
    )


class RenameFileTool(_DirectFileWriteTool):
    spec = ToolSpec(
        name="rename_file",
        description="Rename a file after explicit approval; normal proposed writes use apply_change_set.",
        operation="write",
        input_schema={"type": "object", "properties": {"from": {"type": "string"}, "to": {"type": "string"}}, "required": ["from", "to"], "additionalProperties": True},
        read_only=False,
        concurrency_safe=False,
        requires_approval_by_default=True,
        side_effect_level="write",
        permission_required=True,
        parallel_safe=False,
        can_be_retried=False,
    )


class GenerateEditTool(BaseTool):
    spec = ToolSpec(
        name="generate_edit",
        description="Prepare a validated proposal-only patch for already identified text files without writing files.",
        operation="edit",
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
        proposal_errors: list[str] = []
        for relative_path in target_files[:4]:
            target = validate_repo_file(context.request.project_path, relative_path)
            if not is_supported_text_file(target):
                continue
            content = target.read_text(encoding="utf-8", errors="replace")
            raw_proposal = model_generate_edit_proposal(relative_path, content, context.request.task, payload)
            if raw_proposal is None:
                proposal_errors.append(f"{relative_path}: no model proposal was returned")
                continue
            proposal, reason = validate_edit_proposal_detailed(relative_path, content, raw_proposal, context.request.task)
            if proposal is None:
                repaired = repair_edit_proposal(relative_path, content, raw_proposal, context.request.task, invalid_reason=reason)
                if repaired is None:
                    proposal_errors.append(f"{relative_path}: {reason or 'proposal validation failed'}")
                    continue
                proposal, repair_reason = validate_edit_proposal_detailed(relative_path, content, repaired, context.request.task)
                if proposal is None:
                    proposal_errors.append(f"{relative_path}: {repair_reason or reason or 'proposal validation failed'}")
                    continue
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
        error_text = "; ".join(proposal_errors[:4])
        return ToolResult(
            tool_name=self.spec.name,
            status=status,
            observation="Prepared a proposed edit. No file was written." if proposals else f"No safe edit proposal could be generated. {error_text}".strip(),
            files_read=target_files,
            payload={"edit_proposals": proposals, "applied": False, **({"proposal_error": error_text} if error_text and not proposals else {})},
            next_recommended_action="write_file" if proposals else None,
        )


class AskClarificationTool(BaseTool):
    spec = ToolSpec(
        name="ask_clarification",
        description="Stop the loop and ask the user for a precise missing file, scope, approval, or workflow detail.",
        operation="clarification",
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
        operation="final_answer",
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


CHANGE_SET_PROPOSAL_PROMPT = """\
You are RepoOperator's change-set generator. Return JSON only.
Prepare a proposal-only multi-file ChangeSetProposal. Do not claim changes were applied.
Store complete proposed file content in the JSON, not markdown snippets.
Schema:
{
  "plan": {
    "summary": "short plan summary",
    "target_files": [],
    "operations": ["modify"],
    "evidence_files": [],
    "constraints": [],
    "validation_requirements": []
  },
  "changes": [
    {
      "path": "repo-relative path",
      "operation": "modify | create | delete | rename",
      "rename_to": null,
      "summary": "short user-visible summary",
      "proposed_content": "complete replacement content for create/modify",
      "delete_justification": null,
      "risk_notes": []
    }
  ]
}
Rules:
- Use modify for existing files, create for new files, delete only when explicitly justified, rename only when the target path is clear.
- proposed_content must be complete source/text with no markdown fences.
- Never write files; this is only a proposal.
"""


def model_generate_change_set_proposal(
    *,
    task: str,
    repo: str,
    evidence_contents: dict[str, str],
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    try:
        raw = OpenAICompatibleModelClient().generate_text(
            ModelGenerationRequest(
                system_prompt=CHANGE_SET_PROPOSAL_PROMPT,
                user_prompt=json.dumps(
                    {
                        "task": task,
                        "repo": repo,
                        "change_plan": json_safe(payload.get("change_plan") or {}),
                        "target_files": payload.get("target_files") or [],
                        "new_file_paths": payload.get("new_file_paths") or [],
                        "delete_plan": payload.get("delete_plan") or [],
                        "rename_plan": payload.get("rename_plan") or [],
                        "constraints": payload.get("constraints") or [],
                        "coding_style_notes": payload.get("coding_style_notes") or [],
                        "validation_requirements": payload.get("validation_requirements") or [],
                        "evidence_files": {path: content[:80_000] for path, content in evidence_contents.items()},
                    },
                    ensure_ascii=False,
                ),
            )
        )
        parsed = parse_json_object(raw)
        return parsed if parsed else None
    except Exception:
        return None


def change_set_from_model_payload(
    raw_payload: dict[str, Any],
    *,
    task: str,
    evidence_contents: dict[str, str],
    payload: dict[str, Any],
):
    from repooperator_worker.agent_core.change_set import ChangePlan, ChangeSetProposal, ProposedFileChange, stable_proposal_id

    plan_payload = raw_payload.get("plan") if isinstance(raw_payload.get("plan"), dict) else {}
    changes_payload = raw_payload.get("changes") if isinstance(raw_payload.get("changes"), list) else []
    target_files = [str(item).strip().lstrip("/") for item in plan_payload.get("target_files") or payload.get("target_files") or [] if str(item).strip()]
    plan = ChangePlan(
        summary=str(plan_payload.get("summary") or payload.get("reason_summary") or task),
        target_files=target_files,
        operations=[item for item in plan_payload.get("operations") or [] if item in {"modify", "create", "delete", "rename"}],
        evidence_files=[str(item) for item in plan_payload.get("evidence_files") or evidence_contents.keys()],
        constraints=[str(item) for item in plan_payload.get("constraints") or payload.get("constraints") or []],
        validation_requirements=[str(item) for item in plan_payload.get("validation_requirements") or payload.get("validation_requirements") or []],
    )
    changes = []
    for item in changes_payload:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or item.get("file") or "").strip().lstrip("/")
        if not path:
            continue
        operation = item.get("operation") if item.get("operation") in {"modify", "create", "delete", "rename"} else ("modify" if path in evidence_contents else "create")
        original = evidence_contents.get(path)
        change = ProposedFileChange(
            path=path,
            operation=operation,
            summary=str(item.get("summary") or "Prepare proposed file change."),
            original_content=original,
            proposed_content=None if operation == "delete" else str(item.get("proposed_content") or ""),
            rename_to=item.get("rename_to") if item.get("rename_to") is not None else None,
            delete_justification=item.get("delete_justification") if item.get("delete_justification") is not None else None,
            risk_notes=[str(note) for note in item.get("risk_notes") or []],
        )
        changes.append(_change_with_diff_counts(change))
    if not plan.operations:
        plan.operations = sorted({change.operation for change in changes})  # type: ignore[assignment]
    if not plan.target_files:
        plan.target_files = [change.path for change in changes]
    return ChangeSetProposal(
        proposal_id=str(raw_payload.get("proposal_id") or stable_proposal_id(plan.summary, plan.target_files)),
        plan=plan,
        changes=changes,
    )


def repair_change_set_proposal(
    raw_payload: dict[str, Any],
    *,
    task: str,
    evidence_contents: dict[str, str],
    validation_errors: list[str],
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    stripped = json_safe(raw_payload)
    repaired_any = False
    for change in stripped.get("changes") or []:
        if not isinstance(change, dict) or not isinstance(change.get("proposed_content"), str):
            continue
        without_fences = strip_markdown_fences(change["proposed_content"])
        if without_fences != change["proposed_content"]:
            change["proposed_content"] = without_fences
            repaired_any = True
    if repaired_any:
        return stripped
    try:
        raw = OpenAICompatibleModelClient().generate_text(
            ModelGenerationRequest(
                system_prompt=(
                    "Repair this RepoOperator ChangeSetProposal. Return JSON only. "
                    "Fix the validation errors. All create/modify proposed_content values must be complete files with no markdown fences."
                ),
                user_prompt=json.dumps(
                    {
                        "task": task,
                        "invalid_change_set": json_safe(raw_payload),
                        "validation_errors": validation_errors,
                        "evidence_files": {path: content[:80_000] for path, content in evidence_contents.items()},
                        "context": json_safe(payload),
                    },
                    ensure_ascii=False,
                ),
            )
        )
        repaired = parse_json_object(raw)
        return repaired if repaired else None
    except Exception:
        return None


def _change_with_diff_counts(change):
    before = "" if change.operation == "create" else str(change.original_content or "")
    after = "" if change.operation == "delete" else str(change.proposed_content or "")
    diff = summarize_diff(before, after, limit=1_000_000)
    change.additions = sum(1 for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++"))
    change.deletions = sum(1 for line in diff.splitlines() if line.startswith("-") and not line.startswith("---"))
    return change


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
    return payload if isinstance(payload, dict) else None


def validate_edit_proposal(relative_path: str, original: str, payload: dict[str, Any], task: str) -> dict[str, Any] | None:
    proposal, _reason = validate_edit_proposal_detailed(relative_path, original, payload, task)
    return proposal


def validate_edit_proposal_detailed(relative_path: str, original: str, payload: dict[str, Any], task: str) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(payload, dict):
        return None, "proposal payload is not an object"
    if str(payload.get("file") or relative_path) != relative_path:
        return None, "proposal file does not match target file"
    proposed = str(payload.get("proposed_content") or "")
    if not proposed.strip() or proposed == original or len(proposed) > max(200_000, len(original) * 5):
        return None, "proposed content is empty, unchanged, or implausibly large"
    common_reason = common_proposal_validation_error(relative_path, original, proposed, task)
    if common_reason:
        return None, common_reason
    original_structure = extract_source_structure(relative_path, original)
    proposed_structure = extract_source_structure(relative_path, proposed)
    risk_notes = [str(item) for item in payload.get("risk_notes") or []]
    risk_text = " ".join(risk_notes).lower()
    missing_classes = sorted(set(original_structure["classes"]) - set(proposed_structure["classes"]))
    if missing_classes and not mentions_removal_justification(task, risk_text, missing_classes):
        return None, f"proposal removes existing class declarations without justification: {', '.join(missing_classes)}"
    missing_public = sorted(set(original_structure["public_members"]) - set(proposed_structure["public_members"]))
    if missing_public and not mentions_removal_justification(task, risk_text, missing_public):
        return None, f"proposal removes public declarations without justification: {', '.join(missing_public)}"
    missing_fields = sorted(set(original_structure["serialized_or_public_fields"]) - set(proposed_structure["serialized_or_public_fields"]))
    if missing_fields and not mentions_removal_justification(task, risk_text, missing_fields):
        return None, f"proposal removes public or serialized fields without justification: {', '.join(missing_fields)}"
    missing_lifecycle = sorted(set(original_structure["unity_lifecycle_methods"]) - set(proposed_structure["unity_lifecycle_methods"]))
    for method in missing_lifecycle:
        if method == "Update" and method_body_is_empty(original, method):
            continue
        if not mentions_removal_justification(task, risk_text, [method]):
            return None, f"proposal removes lifecycle method without justification: {method}"
    if relative_path.lower().endswith(".cs") and not csharp_roughly_valid(proposed):
        return None, "proposal has unbalanced C# braces"
    if unsafe_bitwise_change(original, proposed):
        return None, "proposal appears to change bitwise logic into boolean logic"
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
    }, None


def common_proposal_validation_error(relative_path: str, original: str, proposed: str, task: str) -> str | None:
    suffix = Path(relative_path).suffix.lower()
    if re.search(r"^\s*```", proposed, flags=re.MULTILINE):
        return "source content contains markdown fences"
    lowered = proposed.lower()
    commentary_markers = (
        "here is the",
        "i changed",
        "i updated",
        "as requested",
        "the rest of the file",
        "omitted for brevity",
        "truncated",
        "[truncated]",
    )
    if any(marker in lowered for marker in commentary_markers):
        return "source content contains model commentary or truncation markers"
    if suffix in {".py", ".js", ".ts", ".tsx", ".jsx", ".cs", ".java", ".kt", ".go", ".rs", ".c", ".cpp", ".h", ".hpp"}:
        balanced = balanced_delimiters(proposed)
        if balanced:
            return balanced
    if suffix == ".py":
        try:
            ast.parse(proposed)
        except SyntaxError as exc:
            return f"Python syntax error: {exc.msg}"
    if suffix == ".json":
        try:
            json.loads(proposed)
        except json.JSONDecodeError as exc:
            return f"JSON parse error: {exc.msg}"
    if suffix == ".toml":
        try:
            tomllib.loads(proposed)
        except tomllib.TOMLDecodeError as exc:
            return f"TOML parse error: {exc}"
    if suffix in {".yaml", ".yml"}:
        yaml_error = yaml_parse_error(proposed)
        if yaml_error:
            return yaml_error
    original_declarations = primary_declarations(relative_path, original)
    proposed_declarations = primary_declarations(relative_path, proposed)
    missing = sorted(set(original_declarations) - set(proposed_declarations))
    if missing and not mentions_removal_justification(task, "", missing):
        return "proposal removes primary declarations without justification: " + ", ".join(missing[:6])
    original_lines = max(1, len(original.splitlines()))
    proposed_lines = len(proposed.splitlines())
    if original_lines >= 20 and proposed_lines < original_lines * 0.45 and not mentions_removal_justification(task, "", list(original_declarations)):
        return "proposal deletes a large portion of the file without justification"
    return None


def repair_edit_proposal(
    relative_path: str,
    original: str,
    payload: dict[str, Any],
    task: str,
    *,
    invalid_reason: str | None,
) -> dict[str, Any] | None:
    proposed = str(payload.get("proposed_content") or "")
    stripped = strip_markdown_fences(proposed)
    if stripped != proposed:
        repaired = {**payload, "proposed_content": stripped, "summary": str(payload.get("summary") or "Repair proposal formatting.")}
        if validate_edit_proposal(relative_path, original, repaired, task):
            return repaired
    try:
        raw = OpenAICompatibleModelClient().generate_text(
            ModelGenerationRequest(
                system_prompt=(
                    "Repair this RepoOperator edit proposal. Return JSON only. "
                    "The proposed_content must be complete source for the target file, contain no markdown fences, "
                    "preserve unrelated declarations, and fix the validation error."
                ),
                user_prompt=json.dumps(
                    {
                        "task": task,
                        "file": relative_path,
                        "original_content": original[:80_000],
                        "invalid_proposal": json_safe(payload),
                        "validation_error": invalid_reason,
                    },
                    ensure_ascii=False,
                ),
            )
        )
    except Exception:
        return None
    repaired_payload = parse_json_object(raw)
    return repaired_payload if repaired_payload else None


def strip_markdown_fences(content: str) -> str:
    stripped = (content or "").strip()
    if not stripped.startswith("```"):
        return content
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines) + ("\n" if content.endswith("\n") else "")


def balanced_delimiters(content: str) -> str | None:
    pairs = {")": "(", "]": "[", "}": "{"}
    opens = set(pairs.values())
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
            continue
        if char in opens:
            stack.append(char)
        elif char in pairs:
            if not stack or stack.pop() != pairs[char]:
                return f"unbalanced delimiter: {char}"
    if stack:
        return "unbalanced delimiters"
    return None


def yaml_parse_error(content: str) -> str | None:
    try:
        import yaml  # type: ignore
    except Exception:
        return None
    try:
        yaml.safe_load(content)
    except Exception as exc:  # noqa: BLE001
        return f"YAML parse error: {exc}"
    return None


def primary_declarations(relative_path: str, content: str) -> set[str]:
    suffix = Path(relative_path).suffix.lower()
    declarations: set[str] = set()
    if suffix == ".py":
        declarations.update(re.findall(r"^\s*(?:class|def)\s+([A-Za-z_][A-Za-z0-9_]*)\b", content, flags=re.MULTILINE))
    elif suffix in {".js", ".ts", ".tsx", ".jsx"}:
        declarations.update(re.findall(r"\b(?:export\s+)?(?:function|class)\s+([A-Za-z_][A-Za-z0-9_]*)\b", content))
        declarations.update(re.findall(r"\b(?:export\s+)?const\s+([A-Za-z_][A-Za-z0-9_]*)\s*=", content))
    elif suffix in {".java", ".kt", ".cs", ".go", ".rs", ".c", ".cpp", ".h", ".hpp"}:
        declarations.update(re.findall(r"\b(?:class|interface|struct|enum|func|fn)\s+([A-Za-z_][A-Za-z0-9_]*)\b", content))
    return declarations


def summarize_code_change(content: str) -> str:
    declarations = sorted(primary_declarations("source.py", content))
    if declarations:
        return f"{len(content.splitlines())} line(s), declarations: {', '.join(declarations[:6])}"
    imports = re.findall(r"^\s*(?:import|from|using)\s+([A-Za-z0-9_.]+)", content, flags=re.MULTILINE)
    if imports:
        return f"{len(content.splitlines())} line(s), imports: {', '.join(imports[:6])}"
    return f"{len(content.splitlines())} line(s)"


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
        r"\b(?:public|private|protected|internal)?\s*(?:static\s+)?(?:void|bool|int|string|float|double|[A-Za-z_][A-Za-z0-9_<>,\[\]]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
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
    public_methods = re.findall(
        r"\bpublic\s+(?:static\s+)?(?:void|bool|int|string|float|double|[A-Za-z_][A-Za-z0-9_<>,\[\]]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
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
        "unity_lifecycle_methods": [],
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


def _source_notes(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    notes: list[dict[str, str]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        url = str(record.get("url") or "")
        if not url:
            continue
        notes.append(
            {
                "title": str(record.get("title") or url)[:200],
                "url": url,
                "source": str(record.get("source") or "")[:160],
                "fetched_at": str(record.get("fetched_at") or ""),
            }
        )
    return notes[:12]


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
