from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from repooperator_worker.agent_core.actions import AgentAction, ActionResult
from repooperator_worker.agent_core.context_budget import ContextBudget, estimate_chars
from repooperator_worker.agent_core.events import append_activity_event
from repooperator_worker.agent_core.final_synthesis import collect_file_contents
from repooperator_worker.agent_core.request_parsing import extract_file_tokens
from repooperator_worker.agent_core.state import AgentCoreState
from repooperator_worker.agent_core.tools.builtin import is_supported_text_file
from repooperator_worker.agent_core.tools.registry import get_default_tool_registry
from repooperator_worker.schemas import AgentRunRequest, AgentRunResponse
from repooperator_worker.services.common import ensure_relative_to_repo, resolve_project_path
from repooperator_worker.services.ide_bridge_service import get_ide_context
from repooperator_worker.services.json_safe import json_safe
from repooperator_worker.services.model_client import ModelGenerationRequest, OpenAICompatibleModelClient


SOURCE_SUFFIXES = {".cs", ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".kt", ".swift", ".go", ".rs", ".rb", ".php", ".c", ".cpp", ".h", ".hpp"}
TEXT_SUFFIXES = SOURCE_SUFFIXES | {".md", ".txt", ".rst", ".json", ".toml", ".yaml", ".yml", ".ini", ".cfg", ".gradle"}
SEARCH_SKIP_DIRS = {".git", ".claude", "node_modules", "runtime", ".next", "dist", "build", "out", "coverage", ".venv", "venv", "__pycache__"}
PLANNER_ACTION_TYPES = set(get_default_tool_registry().allowed_action_types())


NEXT_ACTION_PROMPT = """\
You are RepoOperator's bounded next-action planner. Return JSON only.
Choose one safe primitive action from the available tool specs. Do not include non-public deliberation.
Schema:
{
  "action_type": "one of available_actions",
  "reason_summary": "short user-visible reason",
  "target_files": [],
  "target_symbols": [],
  "search_queries": [],
  "text_queries": [],
  "symbol_queries": [],
  "file_globs": [],
  "query": "optional text search query",
  "path_globs": [],
  "command": [],
  "expected_output": "short description",
  "requires_approval": false,
  "confidence": 0.0,
  "enough_evidence": false,
  "visible_work_note": {
    "goal": "short user-visible goal for this step",
    "why_this_action": "1-2 sentence user-visible reason for choosing this action",
    "evidence_needed": [],
    "uncertainty": [],
    "safety_note": null
  }
}
Prefer gathering missing evidence before answering. Commands are policy-previewed later; never request direct shell execution.
Use search_text for content/regex grep-like searches instead of shell commands.
visible_work_note is for the user-facing work trace only. Do not include non-public deliberation. Summarize the decision for the user in 1-2 sentences.
"""


@dataclass
class TaskFrame:
    user_goal: str
    mentioned_files: list[str] = field(default_factory=list)
    mentioned_symbols: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    requested_outputs: list[str] = field(default_factory=list)
    likely_needed_tools: list[str] = field(default_factory=list)
    likely_capabilities: list[str] = field(default_factory=list)
    answer_style: str | None = None
    safety_notes: list[str] = field(default_factory=list)
    uncertainty: list[str] = field(default_factory=list)
    should_ask_clarification: bool = False
    clarification_question: str | None = None


def propose_next_action_with_model(
    request: AgentRunRequest,
    state: AgentCoreState,
    task_frame: TaskFrame,
    *,
    model_client_factory: Callable[[], Any] = OpenAICompatibleModelClient,
) -> AgentAction | None:
    registry = get_default_tool_registry()
    available_tool_specs = registry.specs_for_model(
        capabilities=_capability_hints_for_model(registry, task_frame),
        tool_names=task_frame.likely_needed_tools,
    )
    available_actions = [str(item["name"]) for item in available_tool_specs if item.get("name")]
    try:
        raw = model_client_factory().generate_text(
            ModelGenerationRequest(
                system_prompt=NEXT_ACTION_PROMPT,
                user_prompt=json.dumps(
                    {
                        "task": request.task,
                        "task_frame": json_safe(task_frame),
                        "context_packet": json_safe(state.context_packet or {}),
                        "available_actions": available_actions,
                        "available_tools": available_tool_specs,
                        "state": {
                            "observations": state.observations[-8:],
                            "files_read": state.files_read,
                            "files_changed": state.files_changed,
                            "commands_run": state.commands_run,
                            "actions_taken": [action.model_dump() for action in state.actions_taken[-8:]],
                            **planner_result_context(state),
                            "plan": state.plan[-6:],
                            "loop_iteration": state.loop_iteration,
                            "budgets": {
                                "max_file_reads": state.max_file_reads,
                                "max_commands": state.max_commands,
                            },
                        },
                        "safety_constraints": [
                            "All target files must stay inside the repository.",
                            "Commands must be previewed through command policy before running.",
                            "Mutating commands require approval and must not run automatically.",
                            "Final answers need gathered evidence unless the user only needs a simple clarification.",
                        ],
                    },
                    ensure_ascii=False,
                ),
            )
        )
        payload = _parse_json(raw)
    except Exception:
        return None
    return validate_model_next_action(payload, request, state, task_frame)


def _capability_hints_for_model(registry, task_frame: TaskFrame) -> list[str]:
    hints = [str(item).strip() for item in task_frame.likely_capabilities if str(item).strip()]
    for tool_hint in task_frame.likely_needed_tools:
        hints.extend(registry.capabilities_for_tool(str(tool_hint), available_only=True))
    return _dedupe(hints)


def validate_model_next_action(payload: dict[str, Any], request: AgentRunRequest, state: AgentCoreState, task_frame: TaskFrame) -> AgentAction | None:
    action_type = str(payload.get("action_type") or "")
    allowed_action_types = set(get_default_tool_registry().allowed_action_types())
    if action_type not in allowed_action_types:
        return None
    try:
        confidence = float(payload.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < 0.55:
        return None
    reason = _safe_reason_summary(payload.get("reason_summary") or f"Use {action_type} for the next safe step.")
    target_files = [str(item).strip().lstrip("/") for item in payload.get("target_files") or [] if str(item).strip()]
    target_symbols = [str(item).strip() for item in payload.get("target_symbols") or payload.get("symbol_queries") or [] if str(item).strip()]
    search_queries = [str(item).strip() for item in payload.get("search_queries") or [] if str(item).strip()]
    text_queries = [str(item).strip() for item in payload.get("text_queries") or [] if str(item).strip()]
    file_globs = [str(item).strip() for item in payload.get("file_globs") or [] if str(item).strip()]
    command = [str(item) for item in payload.get("command") or [] if str(item)]
    expected = str(payload.get("expected_output") or "")
    visible_work_note = validate_visible_work_note(payload.get("visible_work_note"))

    if action_type in {"read_file", "generate_edit", "generate_change_set"}:
        resolved = resolve_target_files(request, target_files, preferred=known_context_files(request, state))
        if not resolved:
            queries = _dedupe([*target_files, *target_symbols, *search_queries, *file_globs])
            if queries or text_queries:
                return AgentAction(
                    type="search_files",
                    reason_summary="Resolve model-proposed targets before reading or proposing edits.",
                    target_symbols=target_symbols,
                    expected_output="Ranked repo-contained candidate files.",
                    payload=_action_payload_with_note({"queries": queries, "text_queries": text_queries, "file_globs": file_globs, "source": "model_planner"}, visible_work_note),
                )
            return None
        unread = [path for path in resolved if path not in state.files_read]
        if action_type == "read_file":
            if not unread:
                return None
            return AgentAction(type="read_file", reason_summary=reason, target_files=unread, expected_output=expected, payload=_action_payload_with_note({}, visible_work_note))
        valid_edit_targets = set(current_edit_target_files(state, task_frame, request, model_targets=resolved))
        if not valid_edit_targets:
            queries = _dedupe([*target_files, *target_symbols, *search_queries, *file_globs])
            if queries or text_queries:
                return AgentAction(
                    type="search_files",
                    reason_summary="Find validated edit targets before preparing a patch.",
                    target_symbols=target_symbols,
                    expected_output="Ranked repo-contained candidate files.",
                    payload=_action_payload_with_note({"queries": queries, "text_queries": text_queries, "file_globs": file_globs, "source": "model_planner"}, visible_work_note),
                )
            return None
        if not state.files_read and unread:
            return AgentAction(type="read_file", reason_summary="Read target files before preparing an edit proposal.", target_files=unread, expected_output="File contents for edit proposal.", payload=_action_payload_with_note({}, visible_work_note))
        unread_valid = [path for path in valid_edit_targets if path not in state.files_read]
        if unread_valid:
            return AgentAction(type="read_file", reason_summary="Read target files before preparing an edit proposal.", target_files=unread_valid, expected_output="File contents for edit proposal.", payload=_action_payload_with_note({}, visible_work_note))
        return AgentAction(type="generate_change_set", reason_summary=reason, target_files=list(valid_edit_targets), expected_output=expected, payload=_action_payload_with_note({"source": "model_planner", "current_edit_targets": list(valid_edit_targets)}, visible_work_note))

    if action_type == "search_files":
        queries = _dedupe([*search_queries, *target_files, *file_globs, *target_symbols])
        if not queries and not text_queries:
            return None
        if _has_search_for(state, queries or text_queries):
            return None
        return AgentAction(
            type="search_files",
            reason_summary=reason,
            target_symbols=target_symbols,
            expected_output=expected or "Ranked repo-contained candidate files.",
            payload=_action_payload_with_note({"queries": queries, "text_queries": text_queries, "file_globs": file_globs, "source": "model_planner"}, visible_work_note),
        )

    if action_type == "search_text":
        query = str(payload.get("query") or (text_queries[0] if text_queries else "")).strip()
        if not query or _has_search_text_for(state, query):
            return None
        return AgentAction(
            type="search_text",
            reason_summary=reason,
            expected_output=expected or "Repo-contained text matches.",
            payload=_action_payload_with_note({
                "query": query,
                "path_globs": [str(item).strip() for item in payload.get("path_globs") or file_globs if str(item).strip()],
                "max_results": int(payload.get("max_results") or 50),
                "case_sensitive": bool(payload.get("case_sensitive")),
                "regex": bool(payload.get("regex")),
                "context_lines": int(payload.get("context_lines") or 0),
                "source": "model_planner",
            }, visible_work_note),
        )

    if action_type in {"preview_command", "inspect_git_state", "run_approved_command"}:
        if not command:
            return None
        if action_type == "run_approved_command" and not (_latest_command_preview(state, command) and _preview_read_only(_latest_command_preview(state, command).command_result)):
            return AgentAction(
                type="inspect_git_state" if command[:1] == ["git"] else "preview_command",
                reason_summary=reason,
                command=command,
                expected_output="Command safety classification.",
                payload=_action_payload_with_note({}, visible_work_note),
            )
        preview_action = "inspect_git_state" if command[:1] == ["git"] else "preview_command"
        if not _has_command_preview(state, command):
            return AgentAction(type=preview_action, reason_summary=reason, command=command, expected_output="Command safety classification.", payload=_action_payload_with_note({}, visible_work_note))
        preview = _latest_command_preview(state, command)
        if preview and preview.status == "success" and _preview_read_only(preview.command_result) and not _has_command_run(state, command):
            return AgentAction(type="run_approved_command", reason_summary=reason, command=command, expected_output=expected or "Command output.", payload=_action_payload_with_note({}, visible_work_note))
        return None

    if action_type == "ask_clarification":
        attempted_search = any(action.type in {"search_files", "search_text", "inspect_repo_tree"} for action in state.actions_taken)
        if not attempted_search and not payload.get("requires_approval"):
            return None
        return AgentAction(type="ask_clarification", reason_summary=reason, payload=_action_payload_with_note({"question": str(payload.get("question") or reason)}, visible_work_note))

    if action_type == "final_answer":
        enough = bool(payload.get("enough_evidence")) and has_substantive_evidence(state)
        if not enough:
            return None
        return AgentAction(type="final_answer", reason_summary=reason, payload=_action_payload_with_note({}, visible_work_note))

    if action_type == "analyze_repository":
        if _has_action(state, "analyze_repository"):
            return None
        return AgentAction(type="analyze_repository", reason_summary=reason, expected_output=expected, payload=_action_payload_with_note({"classifier": state.classifier_result}, visible_work_note))

    if action_type == "inspect_repo_tree" and not _has_action(state, "inspect_repo_tree"):
        return AgentAction(type="inspect_repo_tree", reason_summary=reason, expected_output=expected, payload=_action_payload_with_note({}, visible_work_note))
    return None


def build_task_frame(request: AgentRunRequest, state: AgentCoreState) -> TaskFrame:
    # Prefer RequestUnderstanding facts; keep ClassifierResult only as a target-file
    # compatibility source for older steering/test helpers.
    ru = getattr(state, "request_understanding", None)
    classifier = state.classifier_result

    # Merge deterministic file tokens with model-extracted mentions.
    ru_files = list(getattr(ru, "mentioned_files", None) or []) if ru else []
    cl_files = list(getattr(classifier, "target_files", None) or [])
    mentioned_files = _dedupe([*ru_files, *cl_files, *_file_tokens(request.task)])

    ru_symbols = list(getattr(ru, "mentioned_symbols", None) or []) if ru else []
    cl_symbols = list(getattr(classifier, "target_symbols", None) or [])
    symbols = _dedupe([*ru_symbols, *cl_symbols, *symbol_tokens(request.task)])

    # Capability hints come from RequestUnderstanding.likely_needed_tools only.
    # Do NOT route by old workflow-bucket fields.
    capabilities: list[str] = []
    if ru:
        for tool_hint in (getattr(ru, "likely_needed_tools", None) or []):
            capabilities.append(f"weak_tool:{tool_hint}")
    if getattr(classifier, "needs_tool", None):
        capabilities.append(f"weak_tool:{classifier.needs_tool}")
    if mentioned_files or symbols:
        capabilities.append("file_read")
    if not capabilities:
        capabilities.append("open_planning")

    constraints = list(getattr(ru, "constraints", None) or []) if ru else []
    ide_context = ide_context_for_request(request, state)
    if ide_context and ide_context.get("diagnostics"):
        constraints.append("Use active editor diagnostics as bugfix evidence when they match the request.")
    requested_outputs = list(getattr(ru, "requested_outputs", None) or []) if ru else []
    likely_needed_tools = list(getattr(ru, "likely_needed_tools", None) or []) if ru else []
    if mentioned_files and not constraints:
        constraints.append("Use explicitly mentioned files before broader context.")

    needs_clarification = bool(
        (getattr(ru, "needs_clarification", None) or getattr(classifier, "needs_clarification", None))
        and not mentioned_files
    )
    clarification_question = (
        (getattr(ru, "clarification_question", None) or getattr(classifier, "clarification_question", None))
    )

    return TaskFrame(
        user_goal=request.task,
        mentioned_files=mentioned_files,
        mentioned_symbols=symbols,
        constraints=constraints,
        requested_outputs=_dedupe(requested_outputs),
        likely_needed_tools=_dedupe(likely_needed_tools),
        likely_capabilities=_dedupe(capabilities),
        answer_style="concise_synthesis",
        safety_notes=list(getattr(ru, "safety_notes", None) or []) if ru else [],
        uncertainty=list(getattr(ru, "uncertainties", None) or []) if ru else [],
        should_ask_clarification=needs_clarification,
        clarification_question=clarification_question,
    )


def planner_result_context(state: AgentCoreState) -> dict[str, Any]:
    summaries = [_summarize_action_result(result) for result in state.action_results[-8:]]
    total = estimate_chars(summaries)
    budget = ContextBudget(max_chars=60_000, reserved_final_answer_chars=4_000)
    compacted = False
    if total > budget.max_tool_result_chars:
        summaries = summaries[-4:]
        compacted = True
    return {
        "action_results": summaries,
        "planner_context_compaction": {
            "compacted": compacted,
            "total_chars_before": total,
            "included_results": len(summaries),
            "max_tool_result_chars": budget.max_tool_result_chars,
        },
    }


def files_from_recent_context(request: AgentRunRequest) -> list[str]:
    files: list[str] = []
    for item in request.conversation_history[-8:]:
        metadata = item.metadata or {}
        for key in ("files_read", "resolved_files"):
            for path in metadata.get(key) or []:
                if isinstance(path, str):
                    files.append(path)
        files.extend(_file_tokens(item.content or ""))
    return files


def symbol_tokens(text: str) -> list[str]:
    symbols: list[str] = []
    generic_request_words = {"Add", "Fix", "Update", "Change", "Refactor", "Implement", "Explain", "Analyze", "Review", "Summarize"}
    for match in re.finditer(r"\b([A-Z][A-Za-z0-9_]{2,})\b", text or ""):
        token = match.group(1)
        if token in generic_request_words:
            continue
        if "." not in token and token not in symbols:
            symbols.append(token)
    return symbols[:8]


def resolve_target_files(request: AgentRunRequest, target_files: list[str], *, preferred: list[str] | None = None) -> list[str]:
    repo = resolve_project_path(request.project_path).resolve()
    preferred = preferred or []
    all_files = searchable_repo_files(repo)
    resolved: list[str] = []
    for item in target_files:
        cleaned = str(item).strip().strip("`'\"")
        if not cleaned:
            continue
        try:
            candidate = ensure_relative_to_repo(repo, cleaned)
            if candidate.is_file():
                rel = str(candidate.relative_to(repo))
                if rel not in resolved:
                    resolved.append(rel)
                    continue
        except ValueError:
            pass
        lowered = cleaned.lower()
        preferred_matches = [path for path in preferred if path.lower() == lowered or Path(path).name.lower() == Path(cleaned).name.lower()]
        matches = preferred_matches or [path for path in all_files if path.lower() == lowered]
        if not matches:
            basename = Path(cleaned).name.lower()
            matches = [path for path in all_files if Path(path).name.lower() == basename]
        for rel in sorted(matches, key=file_match_priority):
            if rel not in resolved:
                resolved.append(rel)
                break
    return resolved[:8]


def searchable_repo_files(repo: Path) -> list[str]:
    files: list[str] = []
    for path in repo.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(repo)
        if any(part.lower() in SEARCH_SKIP_DIRS for part in rel.parts):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name.lower() not in {"readme", "makefile", "dockerfile"}:
            continue
        files.append(str(rel))
    return files


def file_match_priority(path: str) -> tuple[int, int, str]:
    parts = [part.lower() for part in Path(path).parts]
    generated = int(any(part in SEARCH_SKIP_DIRS for part in parts))
    source_bonus = 0 if Path(path).suffix.lower() in SOURCE_SUFFIXES else 1
    script_bonus = 0 if any(part in {"assets", "scripts", "src", "app", "apps"} for part in parts) else 1
    return (generated, source_bonus + script_bonus, path.lower())


def known_context_files(request: AgentRunRequest, state: AgentCoreState) -> list[str]:
    prior: list[str] = []
    if isinstance(state.context_packet, dict):
        prior = [str(item) for item in state.context_packet.get("prior_files_read") or []]
    ide_files = ide_context_files(request, state)
    return _dedupe([*ide_files, *state.files_read, *prior, *files_from_recent_context(request)])


def emit_target_resolution(state: AgentCoreState, request: AgentRunRequest, requested: list[str], resolved: list[str]) -> None:
    if _has_resolution_event(state):
        return
    state.observations.append(f"Resolved target files: {', '.join(resolved)}.")
    append_activity_event(
        run_id=state.run_id,
        request=request,
        activity_id="resolve-target-files",
        event_type="activity_completed",
        phase="Searching",
        label="Resolved target files",
        status="completed",
        observation=f"Resolved {len(resolved)} target file(s).",
        current_action="Resolving mentioned file names to repo-relative paths.",
        next_action="Read resolved files before answering.",
        related_files=resolved,
        aggregate={"requested_files": requested, "resolved_files": resolved},
    )


def command_needed_for_task(frame: TaskFrame, state: AgentCoreState) -> list[str] | None:
    text_command = command_needed_for_text(frame.user_goal)
    if text_command and not _has_command_run(state, text_command):
        return text_command
    if pending_commit_context(frame) and _has_command_run(state, ["git", "log", "--oneline", "-n", "5"]) and not _has_command_run(state, ["git", "status", "--short"]):
        return ["git", "status", "--short"]
    return None


def command_needed_for_text(text: str) -> list[str] | None:
    lowered = (text or "").lower()
    if ("git log" in lowered) or ("commit" in lowered and "recent" in lowered) or ("커밋" in text and "최근" in text):
        return ["git", "log", "--oneline", "-n", "5"]
    return None


def pending_commit_context(frame: TaskFrame) -> bool:
    lowered = frame.user_goal.lower()
    return "commit it" in lowered or "commit changes" in lowered or "커밋해" in frame.user_goal or "커밋 해" in frame.user_goal


def edit_requested(frame: TaskFrame) -> bool:
    if explanation_only_requested(frame.user_goal, frame.requested_outputs):
        return False
    tool_hints = {str(item).strip() for item in frame.likely_needed_tools}
    caps = {str(item).strip() for item in frame.likely_capabilities}
    requested_outputs = {_normalise_marker(item) for item in frame.requested_outputs}
    structured_edit_outputs = {
        "edit_proposal",
        "code_change_proposal",
        "patch_proposal",
        "implementation_" + "im" + "provement",
    }
    structured_edit = (
        "generate_change_set" in tool_hints
        or "generate_edit" in tool_hints
        or "weak_tool:generate_change_set" in caps
        or "weak_tool:generate_edit" in caps
        or "weak_edit" in caps
        or bool(requested_outputs & structured_edit_outputs)
    )
    return structured_edit or edit_requested_text(frame.user_goal)


def edit_requested_text(text: str) -> bool:
    # Fallback only for older callers that have not provided structured
    # RequestUnderstanding facts. Do not expand this into language-specific
    # request routing; prefer likely_needed_tools/requested_outputs instead.
    lowered = (text or "").lower()
    return bool(re.search(r"\b(edit|patch|add|fix|implement|refactor|change|update|support)\b", lowered)) or any(
        term in text for term in ("추가", "고쳐", "구현", "수정")
    )


def explanation_only_requested(text: str, requested_outputs: list[str] | None = None) -> bool:
    lowered = (text or "").lower().strip()
    outputs = " ".join(str(item).lower() for item in requested_outputs or [])
    asks_how = bool(re.search(r"\bhow\s+(would|do|can|should)\b", lowered)) or any(term in text for term in ("어떻게", "어떤 식으로"))
    asks_apply = bool(re.search(r"\b(apply|generate patch|prepare patch)\b", lowered)) or any(term in text for term in ("해줘", "적용", "패치"))
    return asks_how and not asks_apply and "patch" not in outputs and "diff" not in outputs


def validate_visible_work_note(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    note = {
        "goal": _safe_note_text(value.get("goal"), limit=160),
        "why_this_action": _safe_note_text(value.get("why_this_action"), limit=260),
        "evidence_needed": _safe_note_list(value.get("evidence_needed"), item_limit=120, max_items=6),
        "uncertainty": _safe_note_list(value.get("uncertainty"), item_limit=140, max_items=6),
        "safety_note": _safe_note_text(value.get("safety_note"), limit=220),
    }
    if not note["goal"] and not note["why_this_action"] and not note["evidence_needed"] and not note["uncertainty"] and not note["safety_note"]:
        return None
    return json_safe({key: item for key, item in note.items() if item not in (None, "", [], {})})


def _action_payload_with_note(payload: dict[str, Any], visible_work_note: dict[str, Any] | None) -> dict[str, Any]:
    if not visible_work_note:
        return payload
    return {**payload, "visible_work_note": visible_work_note}


def _safe_note_text(value: Any, *, limit: int) -> str | None:
    text = " ".join(str(value or "").split())
    if not text:
        return None
    if _contains_nonpublic_reasoning_marker(text):
        return None
    return text if len(text) <= limit else text[: max(0, limit - 3)].rstrip() + "..."


def _safe_note_list(value: Any, *, item_limit: int, max_items: int) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = _safe_note_text(item, limit=item_limit)
        if text:
            out.append(text)
        if len(out) >= max_items:
            break
    return out


def _normalise_marker(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _contains_nonpublic_reasoning_marker(text: str) -> bool:
    lowered = text.lower()
    markers = (
        "<think>",
        "chain-" + "of-thought",
        "chain " + "of thought",
        "private " + "reasoning",
        "hidden " + "reasoning",
    )
    return any(marker in lowered for marker in markers)


def likely_edit_file_queries(frame: TaskFrame) -> list[str]:
    queries = list(frame.mentioned_files)
    return _dedupe(queries) or ["*.py", "*.js", "*.ts", "*.tsx", "*.jsx", "*.cs", "*.go", "*.rs", "*.java", "*.kt", "*.swift", "*.rb", "*.php"]


def likely_feature_context_files(request: AgentRunRequest) -> list[str]:
    repo = resolve_project_path(request.project_path).resolve()
    priority = [
        "README.md",
        "readme.md",
        "main.py",
        "app.py",
        "server.py",
        "src/main.py",
        "src/app.py",
        "index.js",
        "index.ts",
        "src/index.js",
        "src/index.ts",
        "src/index.tsx",
        "package.json",
        "pyproject.toml",
        "requirements.txt",
    ]
    files: list[str] = []
    seen: set[str] = set()
    for path in priority:
        target = repo / path
        if not target.is_file() or not is_supported_text_file(target):
            continue
        marker = str(target.resolve()).lower()
        if marker in seen:
            continue
        seen.add(marker)
        files.append(path)
    return files[:4]


def candidate_files_from_results(state: AgentCoreState, *, edit_related: bool = False) -> list[str]:
    min_score = 18.0 if edit_related else 1.0
    detail_candidates = current_search_candidate_files(state, min_score=min_score)
    if detail_candidates:
        return detail_candidates[:8]
    candidates: list[str] = []
    for result in state.action_results:
        for path in result.payload.get("candidates") or []:
            if isinstance(path, str) and path not in candidates:
                candidates.append(path)
    return candidates[:8]


def current_search_candidate_files(state: AgentCoreState, *, min_score: float = 1.0) -> list[str]:
    candidates: list[str] = []
    for result in reversed(state.action_results):
        details = result.payload.get("candidate_details") or []
        if not details:
            continue
        for detail in sorted(details, key=lambda item: -float(item.get("score") or 0.0)):
            path = str(detail.get("path") or "")
            score = float(detail.get("score") or 0.0)
            if path and score >= min_score and path not in candidates:
                candidates.append(path)
        if candidates:
            return candidates
    return []


def current_edit_target_files(
    state: AgentCoreState,
    frame: TaskFrame,
    request: AgentRunRequest,
    *,
    model_targets: list[str] | None = None,
) -> list[str]:
    explicit = resolve_target_files(request, frame.mentioned_files, preferred=known_context_files(request, state))
    model_set = list(model_targets or [])
    ide_targets = ide_edit_target_files(request, state, frame)
    high_confidence = current_search_candidate_files(state, min_score=24.0)[:2]
    candidates = _dedupe([*explicit, *ide_targets, *model_set, *high_confidence])
    valid: list[str] = []
    for path in candidates:
        if path not in state.files_read:
            continue
        if path in valid:
            continue
        if path in explicit or path in ide_targets or path in high_confidence:
            valid.append(path)
        elif path in model_set and (path in explicit or path in ide_targets or path in high_confidence):
            valid.append(path)
    return valid[:3]


def ide_context_for_request(request: AgentRunRequest, state: AgentCoreState) -> dict[str, Any] | None:
    packet = state.context_packet if isinstance(state.context_packet, dict) else {}
    packet_context = packet.get("ide_context") if isinstance(packet.get("ide_context"), dict) else None
    if packet_context:
        return json_safe(packet_context)
    return get_ide_context(project_path=request.project_path, branch=request.branch)


def ide_context_files(request: AgentRunRequest, state: AgentCoreState) -> list[str]:
    context = ide_context_for_request(request, state)
    if not context:
        return []
    return _dedupe(
        [
            *([str(context.get("active_file"))] if context.get("active_file") else []),
            *[str(item) for item in context.get("open_files") or [] if str(item)],
        ]
    )


def ide_edit_target_files(request: AgentRunRequest, state: AgentCoreState, frame: TaskFrame) -> list[str]:
    context = ide_context_for_request(request, state)
    if not context or not context.get("active_file"):
        return []
    active_file = str(context.get("active_file") or "")
    if not active_file:
        return []
    explicit = resolve_target_files(request, frame.mentioned_files, preferred=[active_file, *ide_context_files(request, state)])
    if explicit and active_file not in explicit:
        return []
    if not _ide_context_relevant_for_edit(context, frame, explicit):
        return []
    return [active_file]


def _ide_context_relevant_for_edit(context: dict[str, Any], frame: TaskFrame, explicit: list[str]) -> bool:
    if explicit:
        return True
    text = frame.user_goal.lower()
    if any(term in text for term in ("current file", "active file", "this file", "selection", "selected text", "cursor")):
        return True
    if str(context.get("selected_text") or "").strip():
        return True
    diagnostics = context.get("diagnostics") if isinstance(context.get("diagnostics"), list) else []
    if diagnostics and any(term in text for term in ("fix", "bug", "error", "diagnostic", "failing", "broken", "고쳐", "수정")):
        return True
    return edit_requested(frame) and not frame.mentioned_files


def project_summary_files(request: AgentRunRequest) -> list[str]:
    repo = resolve_project_path(request.project_path).resolve()
    priority = ["README.md", "readme.md", "package.json", "pyproject.toml", "apps/web/package.json", "apps/local-worker/pyproject.toml"]
    files: list[str] = []
    seen_resolved: set[str] = set()
    for path in priority:
        target = repo / path
        if not target.is_file():
            continue
        if not is_supported_text_file(target):
            continue
        marker = str(target.resolve()).lower()
        if marker in seen_resolved:
            continue
        seen_resolved.add(marker)
        files.append(path)
    return files[:4]


def has_substantive_evidence(state: AgentCoreState) -> bool:
    if collect_file_contents(state):
        return True
    if state.commands_run:
        return True
    if _latest_edit_proposal(state) or _repository_review_response(state):
        return True
    if state.pending_approval:
        return True
    return False


def _existing_target_files(request: AgentRunRequest, target_files: list[str]) -> list[str]:
    return resolve_target_files(request, target_files)


def _safe_reason_summary(value: Any) -> str:
    text = " ".join(str(value or "").split())
    if _contains_nonpublic_reasoning_marker(text):
        return "Choose the next safe primitive action."
    return text[:180] or "Choose the next safe primitive action."


def _summarize_action_result(result: ActionResult) -> dict[str, Any]:
    return json_safe(
        {
            "status": result.status,
            "observation": result.observation[:240],
            "files_read": result.files_read,
            "files_changed": result.files_changed,
            "command": result.command_result.get("display_command") if result.command_result else None,
            "payload_keys": sorted(result.payload.keys()),
            "candidates": result.payload.get("candidates") or [],
            "matches": (result.payload.get("matches") or [])[:8],
        }
    )


def _has_resolution_event(state: AgentCoreState) -> bool:
    return any(item.startswith("Resolved target files:") for item in state.observations)


def _has_search_for(state: AgentCoreState, queries: list[str]) -> bool:
    wanted = {normalize_search_query(item) for item in queries if normalize_search_query(item)}
    for action in state.actions_taken:
        if action.type != "search_files":
            continue
        previous = {normalize_search_query(item) for item in action.payload.get("queries") or [] if normalize_search_query(item)}
        if wanted & previous or not wanted:
            return True
    return False


def _has_search_text_for(state: AgentCoreState, query: str) -> bool:
    wanted = normalize_search_query(query)
    return any(action.type == "search_text" and normalize_search_query(action.payload.get("query") or "") == wanted for action in state.actions_taken)


def normalize_search_query(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _has_action(state: AgentCoreState, action_type: str) -> bool:
    return any(action.type == action_type for action in state.actions_taken)


def _has_command_preview(state: AgentCoreState, command: list[str]) -> bool:
    return any(action.command == command and action.type in {"preview_command", "inspect_git_state"} for action in state.actions_taken)


def _latest_command_preview(state: AgentCoreState, command: list[str]) -> ActionResult | None:
    preview_action_ids = {
        action.action_id
        for action in state.actions_taken
        if action.command == command and action.type in {"preview_command", "inspect_git_state"}
    }
    for result in reversed(state.action_results):
        if result.action_id in preview_action_ids:
            return result
    return None


def _latest_unrun_read_only_preview(state: AgentCoreState) -> ActionResult | None:
    for result in reversed(state.action_results):
        command = list((result.command_result or {}).get("command") or [])
        if command and _preview_read_only(result.command_result) and not _has_command_run(state, command):
            return result
    return None


def _has_command_run(state: AgentCoreState, command: list[str]) -> bool:
    return any(action.command == command and action.type == "run_approved_command" for action in state.actions_taken)


def _preview_read_only(command_result: dict[str, Any] | None) -> bool:
    return bool(command_result and command_result.get("read_only") and not command_result.get("needs_approval") and not command_result.get("blocked"))


def _repository_review_response(state: AgentCoreState) -> AgentRunResponse | None:
    for result in reversed(state.action_results):
        response = result.payload.get("response")
        if isinstance(response, AgentRunResponse):
            return response
        if isinstance(response, dict):
            try:
                return AgentRunResponse.model_validate(response)
            except Exception:
                continue
    return None


def _latest_command_result(state: AgentCoreState) -> dict[str, Any] | None:
    for result in reversed(state.action_results):
        if result.command_result and result.command_result.get("exit_code") is not None:
            return result.command_result
    return None


def _latest_edit_proposal(state: AgentCoreState) -> dict[str, Any] | None:
    for result in reversed(state.action_results):
        proposals = result.payload.get("edit_proposals") or []
        if proposals:
            return {"applied": bool(result.payload.get("applied")), "proposals": proposals}
    return None


def _format_edit_proposal(payload: dict[str, Any], *, ide_context: dict[str, Any] | None = None) -> str:
    proposals = [item for item in payload.get("proposals") or [] if isinstance(item, dict)]
    if not proposals:
        return "I prepared no file changes because there was not enough safe evidence to build a minimal patch."
    sections = ["I prepared a proposed patch only. No files were modified in this run."]
    active_file = str((ide_context or {}).get("active_file") or "")
    if active_file and any(str(item.get("file") or "") == active_file for item in proposals):
        sections.append(f"Used active editor context for `{active_file}`.")
    for item in proposals[:3]:
        file_path = str(item.get("file") or "unknown file")
        before = str(item.get("before_summary") or "before state recorded")
        after = str(item.get("after_summary") or "after state recorded")
        diff = str(item.get("diff_summary") or "").strip()
        notes = [str(note) for note in item.get("risk_notes") or [] if str(note)]
        notes_text = ("\nRisk notes: " + "; ".join(notes)) if notes else ""
        sections.append(
            f"\n`{file_path}`\nBefore: {before}\nAfter: {after}{notes_text}\n\n```diff\n{diff[:3000]}\n```"
        )
    return "\n".join(sections)


def _format_command_result(result: dict[str, Any], *, pending_commit: bool = False) -> str:
    command = str(result.get("display_command") or shlex.join(list(result.get("command") or [])))
    stdout = str(result.get("stdout") or "").strip()
    stderr = str(result.get("stderr") or "").strip()
    status = result.get("status") or "ok"
    body = stdout or stderr or "No output."
    suffix = ""
    if pending_commit:
        suffix = "\n\nI did not create a commit. Committing requires an explicit approval path and a commit message."
    return f"Ran `{command}` and finished with status `{status}`.\n\n```text\n{body[:4000]}\n```{suffix}"


def _file_tokens(task: str) -> list[str]:
    return extract_file_tokens(task)


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return result


def _parse_json(text: str) -> dict[str, Any]:
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
