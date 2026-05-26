from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

from repooperator_worker.agent_core.events import append_activity_event, utc_now
from repooperator_worker.agent_core.final_response import build_agent_response
from repooperator_worker.schemas import AgentRunRequest, AgentRunResponse
from repooperator_worker.services.common import resolve_project_path
from repooperator_worker.services.model_client import ModelGenerationRequest, OpenAICompatibleModelClient
from repooperator_worker.services.response_quality_service import (
    clean_user_visible_response,
    language_guidance_for_task,
    user_prefers_korean,
)
from repooperator_worker.services.retrieval_service import SKIP_DIRS

REPOSITORY_REVIEW_SUFFIXES = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".kt", ".swift", ".go", ".rs",
    ".rb", ".php", ".cs", ".c", ".cpp", ".h", ".hpp", ".sh", ".sql",
    ".yaml", ".yml", ".toml", ".json", ".ini", ".cfg", ".properties", ".gradle",
    ".md", ".rst", ".txt", ".html", ".css",
}
REPOSITORY_REVIEW_FILENAMES = {
    "dockerfile", "makefile", "requirements.txt", "requirements.in",
    "pyproject.toml", "package.json", "readme", "readme.md",
}
REPOSITORY_REVIEW_BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip", ".tar",
    ".gz", ".tgz", ".7z", ".rar", ".mp3", ".mp4", ".mov", ".wav", ".onnx",
    ".pt", ".pth", ".bin", ".sqlite", ".db", ".pyc",
}
REPOSITORY_REVIEW_EXTRA_SKIP_DIRS = {
    ".git", ".claude", "node_modules", "runtime", ".next", "dist", "build",
    "out", "coverage", ".venv", "venv", "__pycache__",
}
MAX_REPOSITORY_REVIEW_FILES = 12
MAX_REPOSITORY_REVIEW_BYTES = 120_000
MAX_REPOSITORY_REVIEW_PROMPT_CHARS = 22_000
NONPUBLIC_MODEL_DELTA_TYPE = "reasoning" + "_delta"


def should_use_repository_wide_review(classifier: Any) -> bool:
    """Return True when no specific files are mentioned and the task is broad-scope.

    Routing is evidence-based: if the caller has already identified target files,
    skip the expensive repository-wide scan.  The decision does NOT depend on
    workflow-bucket fields from the previous classifier. Those fields are
    intentionally absent from ClassifierResult.
    """
    if getattr(classifier, "target_files", None) or getattr(classifier, "mentioned_files", None):
        return False
    return True


def run_repository_review(
    *,
    request: AgentRunRequest,
    run_id: str,
    classifier: Any | None = None,
    skills_used: list[str] | None = None,
) -> AgentRunResponse:
    started = time.perf_counter()
    activity_events: list[dict[str, Any]] = []
    repo_path = resolve_project_path(request.project_path)

    _emit(
        activity_events,
        run_id=run_id,
        request=request,
        activity_id="repository-review-plan",
        event_type="activity_completed",
        phase="Thinking",
        label="Planned repository review",
        status="completed",
        current_action="Planning a bounded repository review.",
        observation="No files have been reviewed yet.",
        next_action="Inventory readable files and skip generated, binary, dependency, or oversized paths.",
        detail="RepoOperator will review selected files individually and summarize completed evidence only.",
    )

    inventory = inventory_repository_review_files(repo_path)
    selected = inventory["selected"]
    skipped = inventory["skipped"]
    labels = review_progress_labels(selected)
    _emit(
        activity_events,
        run_id=run_id,
        request=request,
        activity_id="repository-inventory",
        event_type="activity_completed",
        phase="Searching",
        label="Listed repository files",
        status="completed",
        current_action="Inventorying repository files.",
        observation=f"Selected {len(selected)} readable file(s); skipped {len(skipped)} file(s).",
        next_action="Review selected files one at a time.",
        detail=f"Selected {len(selected)} readable file(s) and skipped {len(skipped)} unsupported, generated, dependency, or large file(s).",
        aggregate={"files_selected": len(selected), "files_skipped": len(skipped), "searches_count": 1},
    )

    reviewed: list[dict[str, Any]] = []
    timed_out: list[dict[str, Any]] = []
    read_failures: list[dict[str, str]] = []
    try:
        client: OpenAICompatibleModelClient | None = OpenAICompatibleModelClient()
    except Exception:
        client = None

    for relative_path in selected:
        if _run_cancelled(run_id):
            break
        file_started = time.perf_counter()
        started_at = utc_now()
        activity_id = _activity_id("review-file", relative_path)
        label = labels.get(relative_path) or relative_path
        read_summary = progress_summary_for_file(relative_path=relative_path, phase="reading")
        _emit(
            activity_events,
            run_id=run_id,
            request=request,
            activity_id=activity_id,
            event_type="activity_started",
            phase="Reading files",
            label=label,
            status="running",
            related_files=[relative_path],
            started_at=started_at,
            **read_summary,
        )
        read_result = read_review_file(repo_path, relative_path)
        if read_result.get("error"):
            read_failures.append({"file": relative_path, "reason": str(read_result["error"])})
            failed_summary = progress_summary_for_file(
                relative_path=relative_path,
                phase="failed",
                observation=str(read_result["error"]),
            )
            _emit(
                activity_events,
                run_id=run_id,
                request=request,
                activity_id=activity_id,
                event_type="activity_failed",
                phase="Reading files",
                label=label,
                status="failed",
                related_files=[relative_path],
                started_at=started_at,
                duration_ms=int((time.perf_counter() - file_started) * 1000),
                **failed_summary,
            )
            continue
        content = str(read_result["content"])
        observation = file_read_observation(relative_path, content, bool(read_result.get("truncated")))
        review_summary = progress_summary_for_file(
            relative_path=relative_path,
            phase="reviewing",
            content_preview=content[:1400],
            observation=observation,
        )
        _emit(
            activity_events,
            run_id=run_id,
            request=request,
            activity_id=activity_id,
            event_type="activity_updated",
            phase="Reviewing",
            label=label,
            status="running",
            related_files=[relative_path],
            started_at=started_at,
            **review_summary,
        )

        def emit_review_delta(delta: str) -> None:
            _emit(
                activity_events,
                run_id=run_id,
                request=request,
                activity_id=activity_id,
                event_type="activity_delta",
                phase="Reviewing",
                label=label,
                status="running",
                related_files=[relative_path],
                started_at=started_at,
                detail_delta=delta,
                observation_delta=delta,
            )

        review_result = review_single_file(
            request=request,
            relative_path=relative_path,
            content=content,
            truncated=bool(read_result.get("truncated")),
            client=client,
            on_delta=emit_review_delta,
            should_cancel=lambda: _run_cancelled(run_id),
        )
        if review_result.get("cancelled"):
            break
        if review_result.get("timed_out"):
            timed_out.append(review_result)
            _emit(
                activity_events,
                run_id=run_id,
                request=request,
                activity_id=activity_id,
                event_type="activity_failed",
                phase="Reviewing",
                label=label,
                status="failed",
                current_action="Skipping this file after timeout.",
                observation=f"Model review did not return within {review_result.get('elapsed_seconds')}s.",
                next_action="Continue with the remaining selected files.",
                detail=f"Timed out after {review_result.get('elapsed_seconds')}s; continuing with the next file.",
                related_files=[relative_path],
                started_at=started_at,
                duration_ms=int((time.perf_counter() - file_started) * 1000),
            )
            continue
        reviewed.append(review_result)
        completed_summary = progress_summary_for_file(
            relative_path=relative_path,
            phase="completed",
            content_preview=content[:1400],
            observation=_truncate(str(review_result.get("summary") or "Completed file-level review.")),
        )
        _emit(
            activity_events,
            run_id=run_id,
            request=request,
            activity_id=activity_id,
            event_type="activity_completed",
            phase="Reviewing",
            label=label,
            status="completed",
            related_files=[relative_path],
            started_at=started_at,
            duration_ms=int((time.perf_counter() - file_started) * 1000),
            **completed_summary,
        )

    counters = {
        "files_read_count": len(reviewed) + len(timed_out),
        "files_reviewed_count": len(reviewed),
        "files_skipped_count": len(skipped) + len(read_failures),
        "searches_count": 1,
        "timed_out_count": len(timed_out),
        "commands_count": 0,
        "edits_count": 0,
    }
    _emit(
        activity_events,
        run_id=run_id,
        request=request,
        activity_id="repository-review-aggregate",
        event_type="activity_completed",
        phase="Searching",
        label=f"Explored {counters['files_read_count']} files, searched 1 time",
        status="completed",
        current_action="Aggregating completed file reviews.",
        observation=(
            f"Reviewed {counters['files_reviewed_count']} file(s), timed out on "
            f"{counters['timed_out_count']} file(s), and skipped {counters['files_skipped_count']} file(s)."
        ),
        next_action="Prepare the final answer from completed evidence only.",
        detail="Completed repository review aggregation.",
        aggregate=counters,
        duration_ms=int((time.perf_counter() - started) * 1000),
    )
    response = format_repository_review_response(
        request=request,
        selected_files=selected,
        reviewed=reviewed,
        timed_out=timed_out,
        skipped=skipped,
        read_failures=read_failures,
    )
    _emit(
        activity_events,
        run_id=run_id,
        request=request,
        activity_id="repository-review-final-summary",
        event_type="activity_completed",
        phase="Finished",
        label="Prepared evidence-based review summary",
        status="completed",
        current_action="Preparing final summary.",
        observation="Timed-out and skipped files are separated from confirmed findings.",
        next_action="Return the answer to the chat.",
        detail="The final answer only includes confirmed findings from files that completed review.",
    )
    return build_agent_response(
        request,
        response=response,
        files_read=[item["file"] for item in reviewed],
        graph_path="agent_core:repository_review",
        intent_classification=classifier_intent(classifier),
        activity_events=activity_events,
        run_id=run_id,
        skills_used=skills_used or [],
        context_source="repository_wide_review",
    )


def inventory_repository_review_files(repo_path: Path) -> dict[str, list[Any]]:
    candidates: list[Path] = []
    skipped: list[dict[str, str]] = []
    for path in sorted(repo_path.rglob("*"), key=lambda p: (len(p.relative_to(repo_path).parts), str(p).lower())):
        if not path.is_file():
            continue
        rel = path.relative_to(repo_path)
        reason = review_skip_reason(path, rel)
        if reason:
            skipped.append({"file": str(rel), "reason": reason})
        else:
            candidates.append(path)
    priority_names = {"readme.md", "readme", "pyproject.toml", "package.json", "requirements.txt"}
    candidates.sort(
        key=lambda p: (
            0 if p.name.lower() in priority_names else 1,
            0 if p.suffix.lower() in {".py", ".kt", ".java", ".ts", ".tsx", ".js"} else 1,
            len(p.relative_to(repo_path).parts),
            str(p).lower(),
        )
    )
    selected = candidates[:MAX_REPOSITORY_REVIEW_FILES]
    for path in candidates[MAX_REPOSITORY_REVIEW_FILES:]:
        skipped.append({"file": str(path.relative_to(repo_path)), "reason": "review file limit reached"})
    return {"selected": [str(path.relative_to(repo_path)) for path in selected], "skipped": skipped}


def review_skip_reason(path: Path, relative_path: Path) -> str | None:
    if is_stale_duplicate_copy(relative_path):
        return "stale duplicate copy"
    parts = {part.lower() for part in relative_path.parts}
    if parts & {item.lower() for item in SKIP_DIRS | frozenset(REPOSITORY_REVIEW_EXTRA_SKIP_DIRS)}:
        return "generated, dependency, cache, or hidden workspace path"
    suffix = path.suffix.lower()
    name = path.name.lower()
    if suffix in REPOSITORY_REVIEW_BINARY_SUFFIXES:
        return "binary or unsupported file type"
    if suffix not in REPOSITORY_REVIEW_SUFFIXES and name not in REPOSITORY_REVIEW_FILENAMES:
        return "unsupported file type"
    try:
        size = path.stat().st_size
    except OSError:
        return "could not stat file"
    if size > MAX_REPOSITORY_REVIEW_BYTES:
        return f"larger than {MAX_REPOSITORY_REVIEW_BYTES} bytes"
    return None


def is_stale_duplicate_copy(relative_path: Path) -> bool:
    suffix = relative_path.suffix.lower()
    name = relative_path.name.lower()
    if re.search(r"\.(?:bak|orig)$", name, flags=re.IGNORECASE):
        return True
    return suffix in REPOSITORY_REVIEW_SUFFIXES and bool(re.search(r"(?:\s+\d+|\s+copy)(?=\.[^.]+$)", name, flags=re.IGNORECASE))


def review_progress_labels(relative_paths: list[str]) -> dict[str, str]:
    basenames: dict[str, int] = {}
    for relative_path in relative_paths:
        name = Path(relative_path).name
        basenames[name] = basenames.get(name, 0) + 1
    labels: dict[str, str] = {}
    for relative_path in relative_paths:
        path = Path(relative_path)
        if basenames.get(path.name, 0) <= 1:
            labels[relative_path] = path.name
            continue
        parent = "root" if str(path.parent) == "." else str(path.parent)
        labels[relative_path] = f"{path.name} · {parent}"
    return labels


def classifier_intent(classifier: Any | None) -> str:
    if classifier is None:
        return "repo_analysis"
    if isinstance(classifier, dict):
        return str(classifier.get("intent") or "repo_analysis")
    return str(getattr(classifier, "intent", "repo_analysis"))


def read_review_file(repo_path: Path, relative_path: str) -> dict[str, Any]:
    path = (repo_path / relative_path).resolve()
    try:
        path.relative_to(repo_path.resolve())
    except ValueError:
        return {"error": "path is outside the active repository"}
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return {"error": f"could not read file: {exc}"}
    if b"\0" in raw[:4096]:
        return {"error": "binary content detected"}
    truncated = len(raw) > MAX_REPOSITORY_REVIEW_PROMPT_CHARS
    return {"content": raw[:MAX_REPOSITORY_REVIEW_PROMPT_CHARS].decode("utf-8", errors="replace"), "truncated": truncated}


def review_single_file(
    *,
    request: AgentRunRequest,
    relative_path: str,
    content: str,
    truncated: bool,
    client: OpenAICompatibleModelClient | None,
    on_delta: Any | None = None,
    should_cancel: Any | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    if client is None:
        return {
            "file": relative_path,
            "summary": fallback_file_review_summary(relative_path, content, truncated),
            "confirmed": True,
            "fallback": True,
            "elapsed_seconds": 0,
        }
    try:
        prompt = ModelGenerationRequest(
            system_prompt=(
                "You are RepoOperator performing a file-level code review. Use only the provided file "
                "content. Return concise visible review notes: purpose, confirmed issues, improvement "
                "opportunities, and evidence. If no issue is confirmed, say so. Do not include non-public deliberation.\n"
                + language_guidance_for_task(request.task)
            ),
            user_prompt=(
                f"Repository: {Path(request.project_path).name}\nFile: {relative_path}\n"
                f"Content truncated: {'yes' if truncated else 'no'}\n\n"
                f"User review request:\n{request.task}\n\nFile content:\n{content}"
            ),
        )
        pieces: list[str] = []
        stream_text = getattr(client, "stream_text", None)
        if callable(stream_text):
            for delta in stream_text(prompt):
                if should_cancel and should_cancel():
                    return {
                        "file": relative_path,
                        "cancelled": True,
                        "elapsed_seconds": int(time.perf_counter() - started),
                        "summary": "Cancelled before file-level review completed.",
                    }
                delta_type = delta.get("type")
                text = str(delta.get("delta") or "")
                if not text:
                    continue
                if delta_type == "assistant_delta":
                    pieces.append(text)
                    if on_delta:
                        on_delta(text)
                elif delta_type == NONPUBLIC_MODEL_DELTA_TYPE:
                    continue
        raw = "".join(pieces) if pieces else client.generate_text(prompt)
        if should_cancel and should_cancel():
            return {
                "file": relative_path,
                "cancelled": True,
                "elapsed_seconds": int(time.perf_counter() - started),
                "summary": "Cancelled before file-level review completed.",
            }
        clean, _reasoning = clean_user_visible_response(raw, user_task=request.task)
        return {"file": relative_path, "summary": clean.strip(), "confirmed": True, "elapsed_seconds": int(time.perf_counter() - started)}
    except (ValueError, RuntimeError, TimeoutError) as exc:
        if is_timeout_exception(exc):
            return {
                "file": relative_path,
                "timed_out": True,
                "error": "model_timeout",
                "elapsed_seconds": max(1, int(time.perf_counter() - started)),
                "summary": "Timed out before file-level review completed.",
            }
        return {
            "file": relative_path,
            "summary": f"Could not complete model review for this file: {safe_error_summary(exc)}",
            "confirmed": False,
            "error": "model_error",
            "elapsed_seconds": int(time.perf_counter() - started),
        }


def format_repository_review_response(
    *,
    request: AgentRunRequest,
    selected_files: list[str],
    reviewed: list[dict[str, Any]],
    timed_out: list[dict[str, Any]],
    skipped: list[dict[str, str]],
    read_failures: list[dict[str, str]],
) -> str:
    if user_prefers_korean(request.task):
        lines = ["파일별 코드 리뷰가 완료되지 않았습니다.", "", "타임아웃 또는 읽기 실패 파일은 확인된 문제로 다루지 않았습니다."] if not reviewed else [
            f"분석 가능한 파일 {len(selected_files)}개 중 {len(reviewed)}개를 파일별로 검토했습니다.",
            "",
            "## 확인된 파일별 결과",
        ]
        for item in reviewed:
            lines.extend([f"- `{item['file']}`", f"  {one_line(str(item.get('summary') or '검토 완료'))}"])
        if timed_out:
            lines.extend(["", "## 타임아웃으로 검토하지 못한 파일"])
            lines.extend(f"- `{item['file']}`: {item.get('elapsed_seconds', 0)}초 후 타임아웃" for item in timed_out)
        if skipped or read_failures:
            lines.extend(["", "## 제외되거나 읽지 못한 파일"])
            for item in [*skipped[:12], *read_failures[:12]]:
                lines.append(f"- `{item['file']}`: {item['reason']}")
        lines.extend(["", "위 결론은 실제로 읽고 검토가 끝난 파일에만 근거합니다."])
        return "\n".join(lines)

    lines = ["File-by-file code review did not complete.", "", "Timed-out or unreadable files are not used as confirmed findings."] if not reviewed else [
        f"Reviewed {len(reviewed)} of {len(selected_files)} selected readable file(s).",
        "",
        "## Confirmed File-Level Results",
    ]
    for item in reviewed:
        lines.extend([f"- `{item['file']}`", f"  {one_line(str(item.get('summary') or 'Review completed.'))}"])
    if timed_out:
        lines.extend(["", "## Not Reviewed Due To Timeout"])
        lines.extend(f"- `{item['file']}`: timed out after {item.get('elapsed_seconds', 0)}s" for item in timed_out)
    if skipped or read_failures:
        lines.extend(["", "## Skipped Or Unreadable Files"])
        for item in [*skipped[:12], *read_failures[:12]]:
            lines.append(f"- `{item['file']}`: {item['reason']}")
    lines.extend(["", "I only treated completed per-file reviews as confirmed evidence. Skipped and timed-out files are listed separately."])
    return "\n".join(lines)


def file_read_observation(relative_path: str, content: str, truncated: bool) -> str:
    suffix = " Content was truncated for review." if truncated else ""
    return f"Read {len(content.splitlines())} line(s) from `{relative_path}`.{suffix}"


def progress_summary_for_file(*, relative_path: str, phase: str, content_preview: str = "", observation: str = "") -> dict[str, str]:
    path = Path(relative_path)
    descriptor = file_descriptor(path)
    if phase == "reading":
        return {
            "detail": f"Reading `{relative_path}`.",
            "current_action": f"Reading {descriptor}.",
            "observation": "No content has been inspected yet.",
            "next_action": "Review the file purpose and look for confirmed issues from its contents.",
            "safe_reasoning_summary_delta": f"`{path.name}` is being inspected as {descriptor}.",
        }
    if phase == "reviewing":
        metadata = content_metadata_summary(relative_path, content_preview)
        return {
            "detail": f"Read file content. Reviewing {descriptor}.",
            "current_action": f"Checking {descriptor} for concrete, file-backed findings.",
            "observation": metadata or observation or f"Read `{relative_path}`.",
            "next_action": "Use the file-level review result as evidence only if it completes.",
            "safe_reasoning_summary_delta": phase_summary_from_file(relative_path, metadata),
        }
    if phase == "completed":
        return {
            "detail": "Completed file-level review.",
            "current_action": "File review complete.",
            "observation": observation or "No confirmed issue was reported from this file.",
            "next_action": "Move to the next selected file or aggregate completed results.",
            "safe_reasoning_summary_delta": f"`{path.name}` has a completed review result and can be used as evidence.",
        }
    return {
        "detail": f"Could not review `{relative_path}`.",
        "current_action": "Skipping this file.",
        "observation": observation or "The file could not be read safely.",
        "next_action": "Continue with the remaining selected files.",
        "safe_reasoning_summary_delta": f"`{path.name}` will not be used as confirmed evidence.",
    }


def file_descriptor(path: Path) -> str:
    suffix = path.suffix.lower()
    name = path.name.lower()
    if name in {"readme", "readme.md"} or suffix in {".md", ".rst", ".txt"}:
        return "documentation"
    if name in {"package.json", "pyproject.toml", "requirements.txt", "requirements.in"}:
        return "dependency or packaging metadata"
    if suffix in {".yml", ".yaml", ".toml", ".json", ".ini", ".cfg", ".properties", ".gradle"}:
        return "configuration"
    if suffix in {".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".kt", ".swift", ".go", ".rs"}:
        return f"{suffix.lstrip('.')} source code"
    if name == "dockerfile":
        return "container configuration"
    return "readable repository file"


def content_metadata_summary(relative_path: str, content_preview: str) -> str:
    path = Path(relative_path)
    name = path.name.lower()
    suffix = path.suffix.lower()
    lines = content_preview.splitlines()
    if name == "package.json":
        try:
            payload = json.loads(content_preview)
            return f"Found package metadata with {len(payload.get('dependencies') or {})} dependencies and {len(payload.get('devDependencies') or {})} dev dependencies."
        except json.JSONDecodeError:
            return "Found package metadata, but the preview was not enough to parse it fully."
    if suffix == ".py":
        defs = sum(1 for line in lines if line.lstrip().startswith(("def ", "async def ", "class ")))
        return f"Found Python source with {defs} visible function/class definition(s) in the preview."
    if suffix in {".md", ".rst", ".txt"} or name in {"readme", "readme.md"}:
        headings = sum(1 for line in lines if line.lstrip().startswith("#"))
        return f"Found documentation with {headings} visible heading(s) in the preview."
    return f"Read {len(lines)} preview line(s)."


def phase_summary_from_file(relative_path: str, metadata: str) -> str:
    path = Path(relative_path)
    if path.name.lower() in {"readme", "readme.md"}:
        return f"`{path.name}` documents setup or usage, so the review checks whether guidance matches repository evidence."
    if path.suffix.lower() in {".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".kt", ".swift", ".go", ".rs"}:
        return f"`{path.name}` is source code; the review focuses on concrete behavior visible in that file."
    return metadata or f"`{path.name}` is being reviewed from its file content."


def fallback_file_review_summary(relative_path: str, content: str, truncated: bool) -> str:
    suffix = " Content was truncated before review." if truncated else ""
    return f"`{relative_path}` was read successfully. It contains {len(content.splitlines())} line(s). No model review was available, so this is a structural observation only.{suffix}"


def _emit(events: list[dict[str, Any]], **kwargs: Any) -> None:
    event = append_activity_event(**kwargs)
    events.append(event)


def _activity_id(prefix: str, value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.:-]+", "-", value.strip()).strip("-")
    return f"{prefix}:{safe[:160] or uuid.uuid4().hex[:8]}"


def _run_cancelled(run_id: str) -> bool:
    try:
        from repooperator_worker.services.event_service import get_run

        run = get_run(run_id) or {}
    except Exception:
        return False
    return run.get("status") in {"cancelled", "cancelling"}


def is_timeout_exception(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "timeout" in text or "timed out" in text or isinstance(exc, TimeoutError)


def safe_error_summary(exc: BaseException) -> str:
    return " ".join(str(exc).split())[:220] or exc.__class__.__name__


def one_line(text: str) -> str:
    return " ".join(text.split())[:900]


def _truncate(text: str, limit: int = 240) -> str:
    cleaned = " ".join(text.split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip() + "..."
