from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

from repooperator_worker.agent_core.actions import AgentAction, ActionResult
from repooperator_worker.agent_core.state import AgentCoreState, AgentSubtask
from repooperator_worker.agent_core.tools.builtin import is_supported_text_file
from repooperator_worker.schemas import AgentRunRequest
from repooperator_worker.services.common import resolve_project_path
from repooperator_worker.services.json_safe import json_safe


SOURCE_SUFFIXES = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".cs", ".java", ".kt", ".swift", ".go", ".rs",
    ".rb", ".php", ".c", ".cpp", ".h", ".hpp",
}
CONFIG_BASENAMES = {
    "package.json", "pyproject.toml", "requirements.txt", "build.gradle", "settings.gradle",
    "cargo.toml", "go.mod", "pom.xml", "tsconfig.json", "next.config.ts", "vite.config.ts",
    "makefile", "dockerfile",
}
DOC_SUFFIXES = {".md", ".rst", ".txt"}
ENTRYPOINT_STEMS = {"main", "app", "bot", "index", "server", "cli", "worker"}
SKIP_DIRS = {".git", ".claude", "node_modules", "runtime", ".next", "dist", "build", "out", "coverage", ".venv", "venv", "__pycache__"}


def should_ask_clarification_now(state: AgentCoreState, request: AgentRunRequest, frame: Any) -> bool:
    """Return true only after safe evidence gathering is unavailable or exhausted."""
    if not _repository_accessible(request):
        return True
    if _external_dependency_required(request.task):
        return True
    missing = minimum_evidence_missing_for_task(state, request, frame)
    if not missing:
        return False
    if _early_irreversible_request(request.task, frame) and not _has_any_evidence_action(state):
        return True
    if not _has_any_evidence_action(state):
        return False
    if next_evidence_gathering_action(state, request, frame) is not None:
        return False
    return _evidence_attempts_exhausted(state, request, frame)


def minimum_evidence_missing_for_task(state: AgentCoreState, request: AgentRunRequest, frame: Any) -> list[str]:
    missing: list[str] = []
    shape = request_shape(request, frame)
    if not _repository_accessible(request):
        return ["repository is unavailable or inaccessible"]
    if shape in {"summary", "broad"} and not _has_action(state, "inspect_repo_tree") and not _has_inventory(state):
        missing.append("repository structure or inventory")
    if shape == "summary" and not _has_read_high_signal(state):
        missing.append("high-signal documentation, config, or entrypoint evidence")
    if shape == "broad":
        if not _has_inventory(state):
            missing.append("bounded source inventory")
        if not state.files_read:
            missing.append("first analysis batch")
    if shape == "follow_up" and not state.files_read and not prior_context_files(request):
        missing.append("relevant prior or source evidence")
    if shape == "edit":
        if not _has_action(state, "inspect_repo_tree") and not _has_inventory(state):
            missing.append("repository structure")
        if not _has_likely_implementation_evidence(state):
            missing.append("likely implementation area")
        if (_has_action(state, "generate_change_set") or _has_action(state, "generate_edit")) and not _latest_valid_edit_proposal(state):
            missing.append("validated proposal-only change")
    return _dedupe(missing)


def next_evidence_gathering_action(state: AgentCoreState, request: AgentRunRequest, frame: Any) -> AgentAction | None:
    shape = request_shape(request, frame)
    if not _repository_accessible(request):
        return None

    if shape == "broad":
        return _next_broad_evidence_action(state, request)
    if shape == "edit":
        return _next_edit_evidence_action(state, request, frame)
    if shape == "follow_up":
        return _next_followup_evidence_action(state, request, frame)
    return _next_summary_evidence_action(state, request)


def next_recovery_action(
    state: AgentCoreState,
    request: AgentRunRequest,
    frame: Any,
    failed_action: AgentAction,
    result: ActionResult,
) -> AgentAction | None:
    signature = action_signature(failed_action)
    if signature and state.failed_action_signatures.count(signature) > 1:
        state.strategy_shifts.append(f"Changed strategy after repeated ineffective action: {signature}")

    if failed_action.type == "search_text":
        query = normalize_search_query(failed_action.payload.get("query") or "")
        if query and query not in state.zero_result_queries:
            state.zero_result_queries.append(query)
        basename_queries = query_terms_for_request(request.task)[:4]
        if basename_queries and not _has_search_for(state, basename_queries):
            return _with_note(
                AgentAction(
                    type="search_files",
                    reason_summary="Switch from content search to filename and symbol search.",
                    expected_output="Candidate files ranked by filename, symbol, or text evidence.",
                    payload={"queries": basename_queries, "source": "recovery_filename_search"},
                ),
                "Try a different evidence source.",
                "Content search did not find a match, so I am switching to file and symbol discovery.",
                ["Ranked file candidates"],
            )
        broadened = broaden_search_query(query)
        if broadened and broadened not in state.zero_result_queries and not _has_search_text_for(state, broadened):
            return _with_note(
                AgentAction(
                    type="search_text",
                    reason_summary="Broaden the content search after no useful matches.",
                    expected_output="Broader text matches or confirmation that none exist.",
                    payload={"query": broadened, "max_results": 80, "source": "recovery_broaden_search"},
                ),
                "Recover from an empty content search.",
                "The previous search produced no useful evidence, so I am broadening the query before asking for clarification.",
                ["Broader text matches"],
            )

    if failed_action.type == "search_files":
        queries = [normalize_search_query(item) for item in failed_action.payload.get("queries") or []]
        for query in queries:
            if query and query not in state.zero_result_queries:
                state.zero_result_queries.append(query)
        explicit_missing = {normalize_search_query(item) for item in getattr(frame, "mentioned_files", []) or []}
        if explicit_missing and explicit_missing & set(queries):
            return None
        unread_entrypoints = [path for path in likely_entrypoint_and_config_files(request) if path not in state.files_read]
        if unread_entrypoints:
            return _with_note(
                AgentAction(
                    type="read_file",
                    reason_summary="Read entrypoint and config files after search produced no candidates.",
                    target_files=unread_entrypoints[:4],
                    expected_output="Entrypoint/config evidence for a new search strategy.",
                ),
                "Recover from an empty file search.",
                "The file search did not identify a target, so I am reading generic entrypoints and config files to derive better symbols.",
                ["Entrypoint imports, routes, handlers, or module names"],
            )
        if not _has_action(state, "inspect_repo_tree"):
            return _with_note(
                AgentAction(type="inspect_repo_tree", reason_summary="Inspect repository structure after an empty search."),
                "Inspect repository structure.",
                "Search did not produce candidates, so repository shape is the next cheap source of evidence.",
                ["Top-level repository entries"],
            )

    if failed_action.type == "read_file":
        missing = [Path(path).name for path in failed_action.target_files if path]
        if missing and not _has_search_for(state, missing):
            return _with_note(
                AgentAction(
                    type="search_files",
                    reason_summary="Search by basename after a file read failed or was skipped.",
                    expected_output="Repo-contained candidate files by basename.",
                    payload={"queries": missing, "source": "recovery_read_basename"},
                ),
                "Recover from a failed file read.",
                "The requested path could not be read, so I am searching by basename before asking the user.",
                ["Repo-relative candidate paths"],
            )

    if failed_action.type in {"generate_change_set", "generate_edit"}:
        if result.payload.get("proposal_error"):
            return AgentAction(type="final_answer", reason_summary="Report why no validated proposal could be prepared.")
        return next_evidence_gathering_action(state, request, frame)

    if failed_action.type in {"preview_command", "inspect_git_state", "run_approved_command"}:
        return AgentAction(type="final_answer", reason_summary="Summarize command safety or validation failure and the safer next step.")

    return next_evidence_gathering_action(state, request, frame)


def ensure_subtasks(state: AgentCoreState, request: AgentRunRequest, frame: Any) -> None:
    if state.subtasks:
        return
    shape = request_shape(request, frame)
    if shape == "edit":
        specs = [
            ("locate", "Locate relevant implementation area", "Find the files or modules that likely own the requested change.", ["list_files", "search", "read_file"]),
            ("understand", "Understand current behavior/data/API flow", "Read enough source to understand the current implementation path.", ["read_file", "search"]),
            ("propose", "Prepare proposal-only change", "Create a proposal-only edit without writing files.", ["edit"]),
            ("validate", "Validate proposal", "Reject, repair, or report malformed proposals.", ["edit"]),
            ("summarize", "Summarize result and limitations", "Report the evidence, proposal status, and any remaining uncertainty.", ["final_answer"]),
        ]
    elif shape == "broad":
        specs = [
            ("inventory", "Inventory source tree", "Collect a readable source inventory.", ["list_files", "search"]),
            ("batch", "Analyze first bounded batch", "Read docs, configs, entrypoints, and representative source files.", ["read_file"]),
            ("remaining", "Summarize analyzed files and remaining groups", "Describe what was covered and what remains.", ["final_answer"]),
            ("continue", "Continue next batch if budget allows", "Continue safely when budget remains.", ["read_file"]),
        ]
    elif shape == "follow_up":
        specs = [
            ("locate", "Locate relevant domain files", "Use prior context and source search to identify relevant files.", ["search", "read_file"]),
            ("inspect", "Inspect representative source/data", "Read enough current source to answer with evidence.", ["read_file"]),
            ("answer", "Answer with evidence and uncertainty", "Answer from checked files and name any uncertainty.", ["final_answer"]),
        ]
    else:
        specs = [
            ("shape", "Identify repository shape", "Inspect structure or inventory.", ["list_files"]),
            ("evidence", "Read high-signal evidence", "Read docs, config, and entrypoint files.", ["read_file"]),
            ("synthesize", "Synthesize grounded summary", "Answer from gathered evidence.", ["final_answer"]),
        ]
    state.subtasks = [
        AgentSubtask(
            id=f"{shape}:{suffix}",
            title=title,
            goal=goal,
            planned_operations=ops,
            status="running" if index == 0 else "pending",
        )
        for index, (suffix, title, goal, ops) in enumerate(specs)
    ]
    state.current_subtask_id = state.subtasks[0].id if state.subtasks else None


def update_subtasks_after_action(state: AgentCoreState, action: AgentAction, result: ActionResult, operation: str) -> None:
    subtask = current_subtask(state)
    if subtask is None:
        return
    subtask.attempts += 1
    if operation not in subtask.completed_operations and result.status == "success":
        subtask.completed_operations.append(operation)
    for path in result.files_read:
        if path not in subtask.evidence_files:
            subtask.evidence_files.append(path)
    if result.status in {"failed", "timed_out", "cancelled"}:
        subtask.status = "failed" if result.status != "failed" else "blocked"
        subtask.blocker = result.observation or "; ".join(result.errors) or "The action did not produce useful evidence."
    elif result.status in {"success", "skipped"} and _subtask_has_completed_enough(subtask, action, result, operation):
        subtask.status = "completed"
        _advance_subtask(state)


def block_current_subtask(state: AgentCoreState, blocker: str) -> None:
    subtask = current_subtask(state)
    if subtask is None:
        return
    subtask.status = "blocked"
    subtask.blocker = blocker


def current_subtask(state: AgentCoreState) -> AgentSubtask | None:
    if not state.current_subtask_id:
        return state.subtasks[0] if state.subtasks else None
    return next((item for item in state.subtasks if item.id == state.current_subtask_id), None)


def action_operation(action_type: str) -> str:
    if action_type == "inspect_repo_tree":
        return "list_files"
    if action_type in {"search_files", "search_text"}:
        return "search"
    if action_type in {"read_file", "read_many_files"}:
        return "read_file"
    if action_type == "analyze_repository":
        return "analyze_repository"
    if action_type in {"preview_command", "inspect_git_state", "run_approved_command"}:
        return "command"
    if action_type in {"generate_change_set", "generate_edit"}:
        return "edit"
    if action_type == "validate_change_set":
        return "validation"
    if action_type in {"apply_change_set", "create_file", "modify_file", "delete_file", "rename_file"}:
        return "write"
    if action_type == "ask_clarification":
        return "clarification"
    if action_type == "final_answer":
        return "final_answer"
    return action_type


def record_ineffective_action(state: AgentCoreState, action: AgentAction, result: ActionResult) -> None:
    if not _is_ineffective_result(action, result):
        return
    signature = action_signature(action)
    if signature:
        state.failed_action_signatures.append(signature)
    if action.type == "search_text":
        query = normalize_search_query(action.payload.get("query") or "")
        if query and query not in state.zero_result_queries:
            state.zero_result_queries.append(query)
    elif action.type == "search_files":
        for item in [*(action.payload.get("queries") or []), *(action.payload.get("text_queries") or [])]:
            query = normalize_search_query(item)
            if query and query not in state.zero_result_queries:
                state.zero_result_queries.append(query)


def request_shape(request: AgentRunRequest, frame: Any) -> str:
    if _broad_requested(frame):
        return "broad"
    if _edit_requested(frame):
        return "edit"
    if _followup_requested(frame) or prior_context_files(request):
        return "follow_up"
    return "summary"


def repository_file_inventory(request: AgentRunRequest) -> list[str]:
    try:
        repo = resolve_project_path(request.project_path).resolve()
    except ValueError:
        return []
    files: list[str] = []
    for path in repo.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(repo)
        if any(part.lower() in SKIP_DIRS for part in rel.parts):
            continue
        if not is_supported_text_file(path):
            continue
        files.append(str(rel))
    return sorted(files, key=file_rank)


def group_inventory(files: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {
        "entrypoints": [],
        "configs": [],
        "app/service modules": [],
        "UI/components": [],
        "tests": [],
        "docs": [],
        "other": [],
    }
    for path in files:
        group = file_group(path)
        groups.setdefault(group, []).append(path)
    return {key: value for key, value in groups.items() if value}


def first_batch_files(request: AgentRunRequest, *, max_files: int = 8) -> list[str]:
    inventory = repository_file_inventory(request)
    groups = group_inventory(inventory)
    ordered: list[str] = []
    for group in ("docs", "configs", "entrypoints", "app/service modules", "UI/components", "tests", "other"):
        candidates = groups.get(group) or []
        take = 2 if group in {"docs", "configs", "entrypoints"} else 1
        for path in candidates[:take]:
            if path not in ordered:
                ordered.append(path)
        if len(ordered) >= max_files:
            break
    return ordered[:max_files]


def likely_entrypoint_and_config_files(request: AgentRunRequest) -> list[str]:
    files = repository_file_inventory(request)
    selected = [path for path in files if file_group(path) in {"docs", "configs", "entrypoints"}]
    return selected[:8]


def query_terms_for_request(task: str) -> list[str]:
    terms: list[str] = []
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", task or ""):
        lowered = token.lower()
        if lowered in {"the", "and", "for", "with", "this", "that", "into", "from", "please"}:
            continue
        terms.append(token)
    for token in re.findall(r"`([^`]{2,80})`", task or ""):
        terms.append(token)
    return _dedupe(terms)[:8]


def normalize_search_query(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def action_signature(action: AgentAction) -> str:
    if action.type == "search_text":
        return f"search_text:{normalize_search_query(action.payload.get('query') or '')}"
    if action.type == "search_files":
        queries = [normalize_search_query(item) for item in action.payload.get("queries") or []]
        text_queries = [normalize_search_query(item) for item in action.payload.get("text_queries") or []]
        return "search_files:" + "|".join([item for item in [*queries, *text_queries] if item])
    if action.type == "read_file":
        return "read_file:" + "|".join(sorted(action.target_files))
    if action.command:
        return f"{action.type}:{shlex.join(action.command)}"
    return action.type


def _next_summary_evidence_action(state: AgentCoreState, request: AgentRunRequest) -> AgentAction | None:
    if not _has_action(state, "inspect_repo_tree"):
        return _with_note(
            AgentAction(type="inspect_repo_tree", reason_summary="Inspect repository structure before answering."),
            "Identify repository shape.",
            "A structure pass is the cheapest safe evidence for a project-level answer.",
            ["Top-level repository entries"],
        )
    batch = [path for path in likely_entrypoint_and_config_files(request) if path not in state.files_read]
    if batch and len(state.files_read) < 6:
        return _with_note(
            AgentAction(
                type="read_file",
                reason_summary="Read high-signal project files before answering.",
                target_files=batch[:4],
                expected_output="Documentation, config, or entrypoint evidence.",
            ),
            "Read high-signal project evidence.",
            "Project and architecture summaries need repository evidence before a user-facing answer.",
            ["Docs, config, entrypoints"],
        )
    return None


def _next_broad_evidence_action(state: AgentCoreState, request: AgentRunRequest) -> AgentAction | None:
    if not _has_action(state, "inspect_repo_tree"):
        return _with_note(
            AgentAction(type="inspect_repo_tree", reason_summary="Inspect repository structure before broad analysis."),
            "Inventory repository shape.",
            "Broad analysis starts with a cheap repository shape check.",
            ["Top-level repository entries"],
        )
    source_globs = [f"*{suffix}" for suffix in sorted(SOURCE_SUFFIXES | DOC_SUFFIXES | {".json", ".toml", ".yaml", ".yml", ".gradle", ".xml"})]
    if not _has_inventory(state):
        return _with_note(
            AgentAction(
                type="search_files",
                reason_summary="Inventory readable source and high-signal files for broad analysis.",
                expected_output="Grouped readable source inventory.",
                payload={"queries": source_globs, "max_results": 50, "source": "broad_inventory"},
            ),
            "Inventory readable source files.",
            "The request is broad, so I am collecting a bounded source inventory before reading a batch.",
            ["Readable docs, configs, entrypoints, modules, tests"],
        )
    batch = [path for path in first_batch_files(request, max_files=8) if path not in state.files_read]
    if batch:
        return _with_note(
            AgentAction(
                type="read_file",
                reason_summary="Read the first bounded batch for broad analysis.",
                target_files=batch[:8],
                expected_output="File-role evidence for the first broad-analysis batch.",
                payload={"analysis_batch": "first_bounded_batch"},
            ),
            "Analyze the first bounded batch.",
            "I am reading docs, configs, entrypoints, and representative modules before summarizing what remains.",
            ["File roles for read files", "Remaining groups"],
        )
    return None


def _next_edit_evidence_action(state: AgentCoreState, request: AgentRunRequest, frame: Any) -> AgentAction | None:
    if not _has_action(state, "inspect_repo_tree"):
        return _with_note(
            AgentAction(type="inspect_repo_tree", reason_summary="Inspect repository before locating implementation targets."),
            "Locate implementation area.",
            "The request asks for a code change, so I am first checking repository shape.",
            ["Repository structure"],
        )
    candidates = _candidate_files_from_search_results(state)
    unread_candidates = [path for path in candidates if path not in state.files_read]
    if unread_candidates:
        return _with_note(
            AgentAction(
                type="read_file",
                reason_summary="Read likely implementation candidates found by repository search.",
                target_files=unread_candidates[:3],
                expected_output="Candidate implementation contents.",
            ),
            "Read likely implementation candidates.",
            "A search found candidate files, and reading them is needed before any proposal-only edit.",
            ["Current implementation details"],
        )
    queries = _edit_discovery_queries(request, frame)
    if queries and not _has_search_for(state, queries):
        return _with_note(
            AgentAction(
                type="search_files",
                reason_summary="Search for likely implementation areas before asking which file to edit.",
                expected_output="Ranked candidate implementation files.",
                payload={"queries": queries, "text_queries": queries[:6], "source": "edit_discovery"},
            ),
            "Search for implementation ownership.",
            "No explicit target file was confirmed, so I am searching generic symbols and task terms before asking for clarification.",
            ["Candidate source files"],
        )
    context_files = [path for path in likely_entrypoint_and_config_files(request) if path not in state.files_read]
    if context_files:
        return _with_note(
            AgentAction(
                type="read_file",
                reason_summary="Read generic entrypoint/config context before clarifying the edit target.",
                target_files=context_files[:4],
                expected_output="Entrypoint/config evidence for implementation discovery.",
            ),
            "Read entrypoint and config context.",
            "The initial search did not identify a safe target, so I am using entrypoints and config to derive the next target.",
            ["Imports, routes, handlers, or module boundaries"],
        )
    return None


def _next_followup_evidence_action(state: AgentCoreState, request: AgentRunRequest, frame: Any) -> AgentAction | None:
    prior = [path for path in prior_context_files(request) if path not in state.files_read]
    if prior:
        return _with_note(
            AgentAction(
                type="read_file",
                reason_summary="Reuse prior context files before answering the follow-up.",
                target_files=prior[:4],
                expected_output="Previously referenced source evidence.",
            ),
            "Reuse prior context.",
            "This looks like a follow-up, so I am reading the previously referenced source evidence first.",
            ["Prior files from the conversation"],
        )
    queries = _dedupe([*getattr(frame, "mentioned_symbols", []), *query_terms_for_request(request.task)])
    if queries and not _has_search_for(state, queries):
        return _with_note(
            AgentAction(
                type="search_files",
                reason_summary="Search relevant source candidates for the follow-up.",
                expected_output="Candidate source files for the follow-up topic.",
                payload={"queries": queries, "text_queries": queries[:6], "source": "followup_discovery"},
            ),
            "Locate follow-up source evidence.",
            "Prior evidence is not enough, so I am doing a small source scan before answering.",
            ["Relevant source candidates"],
        )
    candidates = [path for path in _candidate_files_from_search_results(state) if path not in state.files_read]
    if candidates:
        return _with_note(
            AgentAction(type="read_file", reason_summary="Read the best follow-up source candidates.", target_files=candidates[:3]),
            "Inspect follow-up source evidence.",
            "Reading the best candidates lets me answer with evidence and uncertainty instead of guessing.",
            ["Representative source contents"],
        )
    return None


def _with_note(action: AgentAction, goal: str, why: str, evidence_needed: list[str]) -> AgentAction:
    payload = dict(action.payload or {})
    payload.setdefault(
        "visible_work_note",
        {
            "goal": goal,
            "why_this_action": why,
            "evidence_needed": evidence_needed,
        },
    )
    action.payload = payload
    return action


def _repository_accessible(request: AgentRunRequest) -> bool:
    try:
        return resolve_project_path(request.project_path).is_dir()
    except Exception:
        return False


def _external_dependency_required(task: str) -> bool:
    lowered = (task or "").lower()
    return any(term in lowered for term in ("api key", "credential", "secret", "password", "login", "oauth token"))


def _early_irreversible_request(task: str, frame: Any) -> bool:
    lowered = (task or "").lower()
    mutating = any(term in lowered for term in ("delete", "remove files", "drop database", "push", "deploy", "publish", "commit"))
    return mutating and not getattr(frame, "mentioned_files", [])


def _has_any_evidence_action(state: AgentCoreState) -> bool:
    return any(action.type in {"inspect_repo_tree", "search_files", "search_text", "read_file", "analyze_repository"} for action in state.actions_taken)


def _has_action(state: AgentCoreState, action_type: str) -> bool:
    return any(action.type == action_type for action in state.actions_taken)


def _has_inventory(state: AgentCoreState) -> bool:
    return any(action.type == "search_files" and action.payload.get("source") == "broad_inventory" for action in state.actions_taken)


def _has_read_high_signal(state: AgentCoreState) -> bool:
    return any(file_group(path) in {"docs", "configs", "entrypoints"} for path in state.files_read)


def _has_likely_implementation_evidence(state: AgentCoreState) -> bool:
    if any(file_group(path) in {"app/service modules", "UI/components", "entrypoints"} for path in state.files_read):
        return True
    return bool(_candidate_files_from_search_results(state))


def _has_search_for(state: AgentCoreState, queries: list[str]) -> bool:
    wanted = {normalize_search_query(item) for item in queries if normalize_search_query(item)}
    for action in state.actions_taken:
        if action.type != "search_files":
            continue
        previous = {normalize_search_query(item) for item in [*(action.payload.get("queries") or []), *(action.payload.get("text_queries") or [])]}
        if wanted and wanted & previous:
            return True
    return False


def _has_search_text_for(state: AgentCoreState, query: str) -> bool:
    wanted = normalize_search_query(query)
    return any(action.type == "search_text" and normalize_search_query(action.payload.get("query") or "") == wanted for action in state.actions_taken)


def _candidate_files_from_search_results(state: AgentCoreState) -> list[str]:
    candidates: list[str] = []
    for result in reversed(state.action_results):
        details = result.payload.get("candidate_details") or []
        if details:
            for detail in sorted(details, key=lambda item: -float(item.get("score") or 0.0)):
                path = str(detail.get("path") or "")
                if path and path not in candidates:
                    candidates.append(path)
            if candidates:
                return candidates[:8]
        for path in result.payload.get("candidates") or []:
            if isinstance(path, str) and path not in candidates:
                candidates.append(path)
    return candidates[:8]


def _edit_discovery_queries(request: AgentRunRequest, frame: Any) -> list[str]:
    explicit = [str(item) for item in getattr(frame, "mentioned_files", []) or []]
    symbols = [str(item) for item in getattr(frame, "mentioned_symbols", []) or []]
    terms = query_terms_for_request(request.task)
    if explicit or symbols or terms:
        return _dedupe([*explicit, *symbols, *terms])
    return [f"*{suffix}" for suffix in sorted(SOURCE_SUFFIXES)]


def _broad_requested(frame: Any) -> bool:
    text = " ".join([getattr(frame, "user_goal", "") or "", *[str(item) for item in getattr(frame, "requested_outputs", []) or []]]).lower()
    broad_scope = any(term in text for term in ("all", "every", "whole", "entire", "directory structure", "source tree", "codebase", "repository-wide", "전체", "모든"))
    broad_object = any(term in text for term in ("source", "file", "module", "project", "repo", "repository", "directory", "folder", "구조", "파일", "모듈"))
    return broad_scope and broad_object


def _edit_requested(frame: Any) -> bool:
    tools = {str(item) for item in getattr(frame, "likely_needed_tools", []) or []}
    outputs = {str(item).lower() for item in getattr(frame, "requested_outputs", []) or []}
    goal = str(getattr(frame, "user_goal", "") or "").lower()
    if "generate_change_set" in tools or "generate_edit" in tools or any("edit" in item or "patch" in item or "change" in item for item in outputs):
        return True
    return bool(re.search(r"\b(add|implement|fix|refactor|change|update|support)\b", goal)) or any(term in goal for term in ("추가", "고쳐", "구현", "수정"))


def _followup_requested(frame: Any) -> bool:
    goal = str(getattr(frame, "user_goal", "") or "").lower()
    return any(term in goal for term in ("previous", "above", "earlier", "that part", "방금", "위에서", "그 파일", "그 부분"))


def _evidence_attempts_exhausted(state: AgentCoreState, request: AgentRunRequest, frame: Any) -> bool:
    shape = request_shape(request, frame)
    if shape == "edit":
        return _has_action(state, "inspect_repo_tree") and (len(state.zero_result_queries) >= 1 or any(action.type == "search_files" for action in state.actions_taken)) and bool(state.files_read or state.strategy_shifts)
    if shape == "broad":
        return _has_inventory(state) and bool(state.files_read)
    return _has_action(state, "inspect_repo_tree") or bool(state.files_read)


def _latest_valid_edit_proposal(state: AgentCoreState) -> bool:
    return any(result.payload.get("change_set_proposal") or result.payload.get("edit_proposals") for result in state.action_results)


def _is_ineffective_result(action: AgentAction, result: ActionResult) -> bool:
    if result.status in {"failed", "skipped", "timed_out"}:
        return True
    if action.type == "search_files":
        return not bool(result.payload.get("candidates"))
    if action.type == "search_text":
        return not bool(result.payload.get("matches"))
    return False


def _subtask_has_completed_enough(subtask: AgentSubtask, action: AgentAction, result: ActionResult, operation: str) -> bool:
    if result.status not in {"success", "skipped"}:
        return False
    if action.type in {"generate_change_set", "generate_edit"}:
        return bool(result.payload.get("edit_proposals") or result.payload.get("proposal_error"))
    return operation in subtask.planned_operations


def _advance_subtask(state: AgentCoreState) -> None:
    if not state.subtasks:
        return
    current_index = next((index for index, item in enumerate(state.subtasks) if item.id == state.current_subtask_id), 0)
    for next_item in state.subtasks[current_index + 1:]:
        if next_item.status == "pending":
            next_item.status = "running"
            state.current_subtask_id = next_item.id
            return
    state.current_subtask_id = state.subtasks[current_index].id


def prior_context_files(request: AgentRunRequest) -> list[str]:
    files: list[str] = []
    for message in request.conversation_history[-8:]:
        metadata = message.metadata or {}
        for key in ("files_read", "resolved_files"):
            for path in metadata.get(key) or []:
                if isinstance(path, str):
                    files.append(path)
    return _dedupe(files)


def file_group(path: str) -> str:
    rel = Path(path)
    parts = {part.lower() for part in rel.parts}
    name = rel.name.lower()
    stem = rel.stem.lower()
    suffix = rel.suffix.lower()
    if "test" in parts or "tests" in parts or name.startswith("test_") or name.endswith((".test.ts", ".spec.ts", ".test.js", ".spec.js")):
        return "tests"
    if suffix in DOC_SUFFIXES or name.startswith("readme"):
        return "docs"
    if name in CONFIG_BASENAMES:
        return "configs"
    if suffix in SOURCE_SUFFIXES and (stem in ENTRYPOINT_STEMS or str(rel).lower().startswith(("src/main.", "src/app.", "app/main."))):
        return "entrypoints"
    if suffix in {".tsx", ".jsx", ".css", ".html"} or parts & {"components", "pages", "ui", "views"}:
        return "UI/components"
    if suffix in SOURCE_SUFFIXES:
        return "app/service modules"
    return "other"


def file_rank(path: str) -> tuple[int, int, str]:
    group_order = {
        "docs": 0,
        "configs": 1,
        "entrypoints": 2,
        "app/service modules": 3,
        "UI/components": 4,
        "tests": 5,
        "other": 6,
    }
    return (group_order.get(file_group(path), 9), len(Path(path).parts), path.lower())


def broaden_search_query(query: str) -> str | None:
    if not query:
        return None
    parts = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", query)
    if len(parts) > 1:
        return parts[0]
    if len(query) > 4:
        return query[: max(3, len(query) // 2)]
    return None


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return out
