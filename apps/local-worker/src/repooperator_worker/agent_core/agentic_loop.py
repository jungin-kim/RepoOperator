"""Model-driven agentic decision loop (native tool calling).

This is the Phase 1 replacement for the deterministic ``choose_graph_next_action``
priority chain as the *primary* planner. Instead of Python heuristics deciding
the next tool, the model sees the real tool schemas plus the running
think -> act -> observe transcript and emits a native tool call. Each graph
iteration calls :func:`propose_next_action_with_tool_calling`, which maps the
model's chosen tool call into an :class:`AgentAction`; execution, permissions,
secret redaction, and budgets remain enforced by the existing tool orchestrator
and graph budget checks.

When the model is not tool-calling-capable, not configured, or declines to
produce a usable call, this returns ``None`` and the deterministic choosers in
``graph_routes.py`` take over as a safety fallback.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from repooperator_worker.agent_core.actions import AgentAction
from repooperator_worker.agent_core.model_profile import detect_model_profile
from repooperator_worker.agent_core.state import AgentCoreState
from repooperator_worker.agent_core.tools.registry import get_default_tool_registry
from repooperator_worker.config import Settings, get_settings
from repooperator_worker.schemas import AgentRunRequest
from repooperator_worker.services.json_safe import json_safe
from repooperator_worker.services.model_client import (
    ModelGenerationRequest,
    build_model_client,
    resolve_model_provider,
)

MAX_TRANSCRIPT_ACTIONS = 12
# The transcript window slides in blocks rather than one action at a time.
# Dropping the oldest action on every step changes the message prefix on every
# model call, which defeats provider prompt caching (OpenAI/vLLM/Ollama prefix
# caches, Anthropic cache_control). Sliding by a block keeps the prefix
# byte-identical for TRANSCRIPT_WINDOW_BLOCK consecutive steps.
TRANSCRIPT_WINDOW_BLOCK = 8
# Read-only evidence tools the model may request several of in one response.
# Only these are batched; anything that mutates, runs a command, touches the
# network, or needs approval still goes one call per model turn.
BATCHABLE_ACTION_TYPES = frozenset(
    {
        "inspect_repo_tree",
        "read_file",
        "read_many_files",
        "search_files",
        "search_text",
        "inspect_symbol",
        "analyze_file",
        "inspect_git_state",
    }
)
MAX_BATCHED_ACTIONS = 6
_RUN_CACHE_LIMIT = 256
# Budget for prior conversation turns carried into the loop (multi-turn memory).
# Older turns beyond this budget are dropped so the transcript can never
# overflow the model window — a lightweight, always-on compaction of history.
MAX_HISTORY_CHARS = 6000
MAX_HISTORY_TURNS = 8
MAX_OBSERVATION_CHARS = 1500
# Auto-compact: when older turns fall outside the keep-window above, they are
# summarized by one model call (instead of being silently dropped) so the agent
# keeps the gist of the earlier conversation.
HISTORY_SUMMARY_INPUT_CHARS = 8000   # cap on dropped-turn text fed to the summarizer
HISTORY_SUMMARY_MAX_TOKENS = 400     # summary output budget
HISTORY_SUMMARY_MAX_CHARS = 1500     # hard cap on the stored summary
MAX_ANSWER_CHARS = 8000

# The system prompt is prose, edited often, so it lives in a file and is
# re-read when the file changes — no worker restart needed. A user copy at
# ~/.repooperator/prompts/agent.md overrides the bundled one.
_BUNDLED_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "agent.md"
_PROMPT_CACHE: dict[str, Any] = {"path": None, "mtime": None, "text": ""}


def _prompt_path() -> Path:
    override = Path.home() / ".repooperator" / "prompts" / "agent.md"
    return override if override.is_file() else _BUNDLED_PROMPT_PATH


def agent_system_prompt() -> str:
    """The agent's system prompt, reloaded when its file changes."""
    path = _prompt_path()
    try:
        mtime = path.stat().st_mtime
    except OSError as exc:
        raise FileNotFoundError(f"Agent system prompt not found: {path}") from exc
    if _PROMPT_CACHE["path"] != path or _PROMPT_CACHE["mtime"] != mtime:
        _PROMPT_CACHE.update(path=path, mtime=mtime, text=path.read_text(encoding="utf-8"))
    return _PROMPT_CACHE["text"]


def __getattr__(name: str) -> Any:
    # Back-compat for callers that imported the old module constant.
    if name == "AGENTIC_SYSTEM_PROMPT":
        return agent_system_prompt()
    raise AttributeError(name)


def endpoint_configured(settings: Settings) -> bool:
    """Whether a reachable model endpoint + model name are configured."""

    if not settings.openai_model:
        return False
    if resolve_model_provider(settings) == "anthropic":
        return bool(settings.openai_api_key or settings.openai_base_url)
    return bool(settings.openai_base_url)


def tool_calling_available(settings: Settings | None = None) -> bool:
    """Whether the model-driven loop should act as the primary planner."""

    settings = settings or get_settings()
    if not settings.agentic_tool_calling:
        return False
    if not endpoint_configured(settings):
        return False
    return bool(detect_model_profile(settings=settings).supports_tool_calls)


def propose_next_action_with_tool_calling(
    request: AgentRunRequest,
    state: AgentCoreState,
    task_frame: Any,
    *,
    client_factory: Callable[[Settings], Any] = build_model_client,
    settings: Settings | None = None,
) -> AgentAction | None:
    """Ask the model for the next tool call and map it to an AgentAction."""

    settings = settings or get_settings()
    if not tool_calling_available(settings):
        return None

    # Read-only calls the model already asked for in its previous response run
    # first, without another model round trip.
    queued = _pop_queued_action(state)
    if queued is not None:
        return queued

    registry = get_default_tool_registry()
    allowed = set(registry.allowed_action_types())
    tool_specs = _run_tool_specs(registry, state, task_frame)
    if not tool_specs:
        return None

    messages = _build_transcript(request, state, task_frame)
    client = client_factory(settings)
    try:
        profile = detect_model_profile(settings=settings)
        max_tokens = min(4096, profile.max_output_tokens)
    except Exception:
        max_tokens = 4096

    def _ask(transcript: list[dict[str, Any]]) -> Any | None:
        try:
            return client.generate_with_tools(
                system_prompt=agent_system_prompt(),
                messages=transcript,
                tools=tool_specs,
                tool_choice="auto",
                max_output_tokens=max_tokens,
            )
        except Exception:
            return None

    response = _ask(messages)
    if response is None:
        return None
    actions = _actions_from_response(response, allowed)
    if not actions:
        return None

    # Policy checks no longer silently veto the model. A violation is returned
    # to the model as the result of its call (like a failed tool), so it can
    # correct itself with full context. Only if it repeats the violation does
    # the deterministic planner take over for this step.
    for attempt in range(MAX_GATE_FEEDBACK_RETRIES + 1):
        violation = _gate_violation(actions[0], state, task_frame)
        if violation is None:
            break
        if attempt == MAX_GATE_FEEDBACK_RETRIES:
            return None
        _bump_usage("gate_feedback_retries")
        messages = messages + _feedback_messages(response, actions[0], violation)
        response = _ask(messages)
        if response is None:
            return None
        actions = _actions_from_response(response, allowed)
        if not actions:
            return None

    action = actions[0]
    _queue_batched_actions(state, action, actions[1:])
    return action


MAX_GATE_FEEDBACK_RETRIES = 1


def _gate_violation(action: AgentAction, state: AgentCoreState, task_frame: Any) -> dict[str, str] | None:
    """Why this action breaks the run's policy, or None when it is fine.

    Each rule names the problem and the way forward, in the shape of a failed
    tool result. Rules only express WHAT must hold; they never pick the tool.
    """
    kind = action.type
    if kind in {"final_answer", "ask_clarification"} and not _has_min_evidence(state):
        return {
            "code": "no_evidence_yet",
            "problem": "You tried to answer before reading anything from the repository this run.",
            "hint": "Inspect the tree or read the relevant files first (inspect_repo_tree, read_file, search_text).",
        }
    if (
        kind not in _EDIT_PRODUCING_ACTION_TYPES
        and _is_change_request(task_frame)
        and _has_min_evidence(state)
        and not _change_applied(state)
    ):
        return {
            "code": "change_not_applied",
            "problem": "The user asked for a change and nothing has been applied yet. Describing a change does not modify files.",
            "hint": "Call generate_change_set (or generate_edit) with the target files you already found.",
        }
    if kind in {"final_answer", "ask_clarification"} and _edit_generation_came_up_empty(state) and not _change_applied(state):
        return {
            "code": "edit_came_up_empty",
            "problem": "The last edit attempt produced no change, so the request is not done.",
            "hint": "Retry the edit with the other edit tool (generate_edit <-> generate_change_set) and a more specific instruction.",
        }
    if kind in _EDIT_PRODUCING_ACTION_TYPES and not _is_change_request(task_frame) and _looks_read_only(task_frame):
        return {
            "code": "read_only_request",
            "problem": "The user asked a question; they did not ask to modify files.",
            "hint": "Answer from the evidence you gathered with final_answer. Do not generate patches.",
        }
    if kind == "ask_clarification" and _looks_read_only(task_frame) and _has_min_evidence(state):
        return {
            "code": "answer_available",
            "problem": "You already read files relevant to this question.",
            "hint": "Answer it with final_answer, citing the files. Ask only if the evidence truly cannot answer it.",
        }
    if kind not in {"preview_command", "run_approved_command", "run_validation_command", "request_command_approval", "inspect_git_state"}:
        try:
            from repooperator_worker.agent_core.planner import command_needed_for_text, edit_requested_text

            goal = str(getattr(task_frame, "user_goal", "") or "")
            needed = command_needed_for_text(goal)
            if needed and not edit_requested_text(goal) and not _has_command_evidence(state, needed):
                return {
                    "code": "command_not_run",
                    "problem": f"The user asked to run `{' '.join(needed)}` and it has not been run or requested.",
                    "hint": "Call preview_command with that command; it goes through the approval flow.",
                }
        except Exception:
            pass
    return None


def _feedback_messages(response: Any, action: AgentAction, violation: dict[str, str]) -> list[dict[str, Any]]:
    """The rejected call plus a tool-shaped rejection, appended to the transcript."""
    feedback = json.dumps({"ok": False, **violation, "note": "Not executed."}, ensure_ascii=False)
    calls = list(getattr(response, "tool_calls", ()) or ())
    if calls:
        call = calls[0]
        call_id = str(call.id or action.action_id)
        return [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": json.dumps(dict(call.arguments or {}), ensure_ascii=False, sort_keys=True)},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": call_id, "content": feedback},
        ]
    # Text-only reply (an implicit final_answer): answer it as the runtime.
    text = str(getattr(response, "text", "") or "")[:2000]
    return [
        {"role": "assistant", "content": text or "(answer)"},
        {"role": "user", "content": "[runtime] " + feedback},
    ]


def _bump_usage(name: str) -> None:
    from repooperator_worker.services import usage_tracker

    usage_tracker.bump(name)


# ---------------------------------------------------------------------------
# Per-run caches. Process-local: a worker restart only costs one cache miss.


def _run_key(state: Any) -> str | None:
    run_id = getattr(state, "run_id", None)
    return str(run_id) if run_id else None


def _bounded_put(cache: dict, key: Any, value: Any) -> None:
    if key not in cache and len(cache) >= _RUN_CACHE_LIMIT:
        cache.pop(next(iter(cache)))
    cache[key] = value


_RUN_TOOL_NAMES: dict[str, list[str]] = {}
_RUN_STABLE_PAYLOAD: dict[str, str] = {}
_RUN_ACTION_QUEUE: dict[str, dict[str, Any]] = {}


def _run_tool_specs(registry, state: Any, task_frame: Any) -> list[dict[str, Any]]:
    """Tool specs for this model call, kept stable for the whole run.

    Tool definitions sit at the very front of the cached prefix, so a tool list
    that changes between steps (task-frame hints drift as evidence arrives)
    invalidates the cache for the entire request. The first call fixes the set;
    later calls only append newly hinted tools at the end.
    """
    hinted_tools = [str(item) for item in getattr(task_frame, "likely_needed_tools", []) or []]
    specs = registry.specs_for_model(capabilities=_capability_hints(registry, task_frame), tool_names=hinted_tools)
    key = _run_key(state)
    if key is None:
        return specs
    previous = _RUN_TOOL_NAMES.get(key)
    if previous is None:
        _bounded_put(_RUN_TOOL_NAMES, key, [str(spec.get("name")) for spec in specs])
        return specs
    current_names = [str(spec.get("name")) for spec in specs]
    names = list(previous) + [name for name in current_names if name not in previous]
    if names != previous:
        _bounded_put(_RUN_TOOL_NAMES, key, names)
    by_name = {str(spec.get("name")): spec for spec in specs}
    missing = [name for name in names if name not in by_name]
    if missing:
        for spec in registry.specs_for_model(tool_names=missing, include_default=False):
            by_name.setdefault(str(spec.get("name")), spec)
    return [by_name[name] for name in names if name in by_name]


def _queue_batched_actions(state: Any, first: AgentAction, rest: list[AgentAction]) -> None:
    """Queue the read-only calls that followed ``first`` in the same response."""
    key = _run_key(state)
    if key is None:
        return
    _RUN_ACTION_QUEUE.pop(key, None)
    if first.type not in BATCHABLE_ACTION_TYPES:
        return
    batch: list[AgentAction] = []
    for extra in rest:
        if extra.type not in BATCHABLE_ACTION_TYPES or len(batch) >= MAX_BATCHED_ACTIONS:
            break
        batch.append(extra)
    if batch:
        _bounded_put(_RUN_ACTION_QUEUE, key, {"after": first.action_id, "actions": batch})


def _pop_queued_action(state: Any) -> AgentAction | None:
    """Next queued read-only call, if the batch is still in sequence.

    The queue is dropped as soon as anything other than the batch ran (a
    deterministic chooser, an approval, a recovery step), so a stale read never
    fires after the repository changed.
    """
    key = _run_key(state)
    if key is None:
        return None
    entry = _RUN_ACTION_QUEUE.get(key)
    if not entry:
        return None
    taken = list(getattr(state, "actions_taken", []) or [])
    if not taken or getattr(taken[-1], "action_id", None) != entry["after"]:
        _RUN_ACTION_QUEUE.pop(key, None)
        return None
    already_read = set(getattr(state, "files_read", []) or [])
    queue: list[AgentAction] = entry["actions"]
    while queue:
        candidate = queue.pop(0)
        # Skip reads that earlier steps already covered (graph_routes has the
        # full repeat check, but importing it here would be circular).
        redundant = candidate.type == "read_file" and candidate.target_files and set(candidate.target_files) <= already_read
        if not redundant:
            entry["after"] = candidate.action_id
            if not queue:
                _RUN_ACTION_QUEUE.pop(key, None)
            return candidate
    _RUN_ACTION_QUEUE.pop(key, None)
    return None


def run_usage(run_id: str) -> dict[str, Any]:
    """Token usage for a run (all model calls, incl. prompt-cache hits)."""
    from repooperator_worker.services import usage_tracker

    return usage_tracker.snapshot(run_id)


def _has_command_evidence(state: Any, command: list[str]) -> bool:
    joined = " ".join(command)
    for item in getattr(state, "commands_run", []) or []:
        if joined in str(item):
            return True
    return bool(getattr(state, "pending_approval", None))


_EDIT_PRODUCING_ACTION_TYPES = frozenset(
    {
        "generate_change_set",
        "generate_edit",
        "modify_file",
        "create_file",
        "delete_file",
        "rename_file",
        "apply_change_set",
    }
)


_READ_ONLY_PHRASES = (
    "읽고", "읽어", "설명해", "알려줘", "분석해", "요약해", "보여줘", "확인해",
    "무슨", "어떻게 동작", "어떤 역할", "파악해", "찾아줘", "찾아봐", "뭐가 있",
    "반환해", "뭐야", "뭐였지", "몇 개", "무엇", "뭘 ", "기억나",
    "explain", "describe", "summarize", "analyze", "analyse", "what does", "how does",
    "walk me through", "tell me", "show me", "read the", "review", "find the", "look for",
)


def _is_planning_only(task_frame: Any) -> bool:
    """A request for a plan/breakdown should produce a plan, not a patch."""
    try:
        from repooperator_worker.agent_core.intent import is_planning_request

        return is_planning_request(str(getattr(task_frame, "user_goal", "") or ""))
    except Exception:
        return False


def _looks_read_only(task_frame: Any) -> bool:
    """Whether the user's own words clearly ask to read/explain, not to change.

    ``edit_requested`` also turns True from weak model tool-hints, which
    misfires on questions like "read the main file and explain how it works" —
    that used to push a read request down the edit path.
    """
    try:
        from repooperator_worker.agent_core.planner import edit_requested_text

        text = str(getattr(task_frame, "user_goal", "") or "")
        if not text:
            return False
        if edit_requested_text(text):
            return False
        lowered = text.lower()
        return any(p in text or p in lowered for p in _READ_ONLY_PHRASES)
    except Exception:
        return False


def _is_change_request(task_frame: Any) -> bool:
    """Whether the task asks to modify the repository (vs. only explain/read).

    Gate on the USER'S OWN WORDS only (edit_requested_text). The broader
    edit_requested(frame) also fires from weak model tool-hints, which kept
    misclassifying plain questions ("관리자 명령어 뭐 있어?") as change requests
    — the gate then blocked final_answer until the loop budget ran out and the
    user got an "insufficient evidence" template.
    """
    try:
        from repooperator_worker.agent_core.planner import edit_requested_text

        if _looks_read_only(task_frame) or _is_planning_only(task_frame):
            return False
        text = str(getattr(task_frame, "user_goal", "") or "")
        return bool(edit_requested_text(text))
    except Exception:
        return False


def _change_applied(state: AgentCoreState) -> bool:
    """Whether a file change has actually been applied to the working tree."""
    return bool(getattr(state, "files_changed", None))


def _edit_generation_came_up_empty(state: AgentCoreState) -> bool:
    """The latest generate_edit/generate_change_set attempt produced nothing.

    Kept local (not imported from graph_routes) to avoid a circular import.
    """
    actions = list(getattr(state, "actions_taken", []) or [])
    results = list(getattr(state, "action_results", []) or [])
    if not results:
        return False
    for action, result in zip(reversed(actions[-len(results):]), reversed(results)):
        if str(getattr(action, "type", "") or "") not in {"generate_edit", "generate_change_set"}:
            continue
        payload = getattr(result, "payload", None) or {}
        if payload.get("edit_proposals") or (payload.get("change_set_proposal") or {}).get("changes"):
            return False
        return str(getattr(result, "status", "") or "") in {"skipped", "failed"}
    return False


_EVIDENCE_ACTION_TYPES = frozenset(
    {
        "inspect_repo_tree",
        "read_file",
        "read_many_files",
        "search_files",
        "search_text",
        "analyze_file",
        "analyze_repository",
        "inspect_symbol",
        "run_approved_command",
        "inspect_git_state",
    }
)


def _has_min_evidence(state: AgentCoreState) -> bool:
    """Whether the run has gathered any repository evidence yet."""

    if getattr(state, "files_read", None):
        return True
    if getattr(state, "commands_run", None):
        return True
    for action in getattr(state, "actions_taken", []) or []:
        if getattr(action, "type", None) in _EVIDENCE_ACTION_TYPES:
            return True
    return False


def _capability_hints(registry, task_frame: Any) -> list[str]:
    hints = [str(item).strip() for item in getattr(task_frame, "likely_capabilities", []) or [] if str(item).strip()]
    for tool_hint in getattr(task_frame, "likely_needed_tools", []) or []:
        hints.extend(registry.capabilities_for_tool(str(tool_hint), available_only=True))
    seen: list[str] = []
    for hint in hints:
        if hint and hint not in seen:
            seen.append(hint)
    return seen


# Per-run cache of history summaries, keyed by (thread_id, dropped-turn count,
# content hash) so repeated loop iterations reuse one model call, and an unchanged
# older history reuses it across runs too.
_HISTORY_SUMMARY_CACHE: dict[tuple[str, int, str], str] = {}
_HISTORY_SUMMARY_SYSTEM = (
    "You compress earlier turns of a software-engineering conversation into a terse, "
    "factual summary for an AI coding agent. Preserve concrete decisions, file/path and "
    "symbol names, constraints, and any unresolved questions. Omit pleasantries. "
    "Output at most 8 short bullet lines; no preamble."
)


def _summarize_dropped_history(dropped: list[dict[str, str]], thread_id: str | None) -> str | None:
    """Summarize older, out-of-window turns with one model call. None on any failure.

    Cached so it costs at most one call per distinct older-history state.
    """
    if not dropped:
        return None
    joined = "\n\n".join(f"{turn['role']}: {turn['content']}" for turn in dropped)
    if len(joined) > HISTORY_SUMMARY_INPUT_CHARS:
        # Keep the most recent of the dropped turns when over budget.
        joined = joined[-HISTORY_SUMMARY_INPUT_CHARS:]
    key = (str(thread_id or ""), len(dropped), hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16])
    cached = _HISTORY_SUMMARY_CACHE.get(key)
    if cached is not None:
        return cached or None

    settings = get_settings()
    if not endpoint_configured(settings):
        return None
    try:
        client = build_model_client(settings)
        text = client.generate_text(
            ModelGenerationRequest(
                system_prompt=_HISTORY_SUMMARY_SYSTEM,
                user_prompt=f"Summarize these earlier conversation turns:\n\n{joined}",
                max_output_tokens=HISTORY_SUMMARY_MAX_TOKENS,
            )
        )
    except Exception:
        return None
    summary = (text or "").strip()[:HISTORY_SUMMARY_MAX_CHARS]
    if len(_HISTORY_SUMMARY_CACHE) > 256:
        _HISTORY_SUMMARY_CACHE.clear()
    _HISTORY_SUMMARY_CACHE[key] = summary
    return summary or None


def _recent_history_messages(request: AgentRunRequest) -> list[dict[str, Any]]:
    """Prior conversation turns for multi-turn memory, with auto-compaction.

    The newest turns are kept verbatim (up to MAX_HISTORY_TURNS / MAX_HISTORY_CHARS).
    Older turns that fall outside that window are not silently dropped: they are
    compacted into a single summary turn (one cached model call) so the agent
    retains the gist of the earlier conversation. The trailing user turn that
    duplicates the current task is skipped so the task is not sent twice.
    """
    history = list(getattr(request, "conversation_history", []) or [])
    current_task = (getattr(request, "task", "") or "").strip()

    # Filtered, chronological usable turns.
    usable: list[dict[str, str]] = []
    for item in history:
        role = getattr(item, "role", None) if not isinstance(item, dict) else item.get("role")
        content = getattr(item, "content", None) if not isinstance(item, dict) else item.get("content")
        if role not in {"user", "assistant"}:
            continue
        text = str(content or "").strip()
        if not text or text == current_task:
            continue
        usable.append({"role": role, "content": text})

    # Keep the newest turns within budget (unchanged window semantics).
    kept_rev: list[dict[str, Any]] = []
    used = 0
    for turn in reversed(usable):
        if len(kept_rev) >= MAX_HISTORY_TURNS:
            break
        if used + len(turn["content"]) > MAX_HISTORY_CHARS and kept_rev:
            break
        kept_rev.append({"role": turn["role"], "content": turn["content"][:MAX_HISTORY_CHARS]})
        used += len(turn["content"])
    kept = list(reversed(kept_rev))

    dropped = usable[: len(usable) - len(kept)]
    messages: list[dict[str, Any]] = []
    if dropped:
        summary = _summarize_dropped_history(dropped, getattr(request, "thread_id", None))
        if summary:
            messages.append(
                {
                    "role": "user",
                    "content": f"[Earlier conversation summary — {len(dropped)} prior turn(s) compacted]\n{summary}",
                }
            )
    messages.extend(kept)
    return messages


def _stable_task_payload(request: AgentRunRequest, state: AgentCoreState, task_frame: Any) -> str:
    """The run's opening user turn, serialized once and reused byte-for-byte.

    task_frame and context_packet are snapshotted at the first model call. What
    the agent learns afterwards reaches the model through tool results, so the
    opening turn never needs to change within a run.
    """
    key = _run_key(state)
    if key is not None and key in _RUN_STABLE_PAYLOAD:
        return _RUN_STABLE_PAYLOAD[key]
    payload = json.dumps(
        {
            "task": request.task,
            "task_frame": json_safe(task_frame),
            "context_packet": json_safe(getattr(state, "context_packet", {}) or {}),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    if key is not None:
        _bounded_put(_RUN_STABLE_PAYLOAD, key, payload)
    return payload


def _transcript_window_start(total: int) -> int:
    """Index of the first action kept, advancing in TRANSCRIPT_WINDOW_BLOCK steps."""
    if total <= MAX_TRANSCRIPT_ACTIONS:
        return 0
    overflow = total - MAX_TRANSCRIPT_ACTIONS
    blocks = -(-overflow // TRANSCRIPT_WINDOW_BLOCK)  # ceil
    return blocks * TRANSCRIPT_WINDOW_BLOCK


def _run_status_message(state: AgentCoreState) -> dict[str, Any]:
    """Per-step progress, placed LAST so it never invalidates the cached prefix."""
    status = {
        "files_read": list(getattr(state, "files_read", []) or []),
        "files_changed": list(getattr(state, "files_changed", []) or []),
        "budgets": {
            "max_file_reads": getattr(state, "max_file_reads", None),
            "max_commands": getattr(state, "max_commands", None),
            "loop_iteration": getattr(state, "loop_iteration", None),
        },
    }
    return {"role": "user", "content": "[run status]\n" + json.dumps(status, ensure_ascii=False, sort_keys=True)}


def _build_transcript(request: AgentRunRequest, state: AgentCoreState, task_frame: Any) -> list[dict[str, Any]]:
    """Build the model transcript so that each step only APPENDS to the last one.

    Layout: history -> stable task turn -> (omitted-steps note) -> tool calls and
    results -> run status. Everything before the run status is identical to the
    previous step's request, so provider prompt caches keep hitting.
    """
    messages: list[dict[str, Any]] = []
    messages.extend(_recent_history_messages(request))
    messages.append({"role": "user", "content": _stable_task_payload(request, state, task_frame)})

    actions = list(getattr(state, "actions_taken", []) or [])
    results = list(getattr(state, "action_results", []) or [])
    pairs = list(zip(actions, results))
    start = _transcript_window_start(len(pairs))
    if start:
        messages.append(
            {
                "role": "user",
                "content": f"[{start} earlier tool step(s) omitted; files already read are listed in the run status]",
            }
        )
    for action, result in pairs[start:]:
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": action.action_id,
                        "type": "function",
                        "function": {
                            "name": action.type,
                            "arguments": json.dumps(_action_arguments(action), ensure_ascii=False, sort_keys=True),
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": action.action_id,
                "content": _observation_text(result),
            }
        )
    messages.append(_run_status_message(state))
    return messages


def _action_arguments(action: AgentAction) -> dict[str, Any]:
    args: dict[str, Any] = dict(action.payload or {})
    if action.target_files:
        args.setdefault("target_files", list(action.target_files))
    if action.target_symbols:
        args.setdefault("target_symbols", list(action.target_symbols))
    if action.command:
        args.setdefault("command", list(action.command))
    return args


def _observation_text(result: Any) -> str:
    if result is None:
        return "(no result)"
    parts = [f"status={getattr(result, 'status', 'unknown')}"]
    observation = str(getattr(result, "observation", "") or "")
    if observation:
        parts.append(observation)
    files_read = list(getattr(result, "files_read", []) or [])
    if files_read:
        parts.append("files_read: " + ", ".join(files_read[:10]))
    payload = getattr(result, "payload", {}) or {}
    if isinstance(payload, dict):
        candidates = payload.get("candidates")
        if isinstance(candidates, list) and candidates:
            names = [str(c.get("path") if isinstance(c, dict) else c) for c in candidates[:10]]
            parts.append("candidates: " + ", ".join(names))
        matches = payload.get("matches")
        if isinstance(matches, list) and matches:
            parts.append(f"matches: {len(matches)}")
    text = "\n".join(parts)
    if len(text) > MAX_OBSERVATION_CHARS:
        text = text[:MAX_OBSERVATION_CHARS] + "\n…(truncated)"
    return text


def _action_from_call(call: Any, response: Any, allowed: set[str]) -> AgentAction | None:
    name = str(call.name or "")
    if name not in allowed:
        return None
    args = dict(call.arguments or {})
    reason = str(args.get("reason_summary") or (response.text or "") or f"Use {name} for the next step.").strip()
    action = AgentAction(
        type=name,  # type: ignore[arg-type]
        reason_summary=(reason or f"Use {name}.")[:300],
        payload=json_safe(args),
    )
    _map_common_fields(action, args)
    return action


def _actions_from_response(response: Any, allowed: set[str]) -> list[AgentAction]:
    """Every usable tool call in the response, in order.

    Models that support parallel tool calls often ask for several reads at
    once. Previously only ``tool_calls[0]`` was used and the rest were silently
    dropped, costing one extra model round trip per file.
    """
    if getattr(response, "has_tool_calls", False):
        actions: list[AgentAction] = []
        for index, call in enumerate(response.tool_calls):
            action = _action_from_call(call, response, allowed)
            if action is None:
                if index == 0:
                    return []
                break
            actions.append(action)
        return actions

    text = (getattr(response, "text", "") or "").strip()
    if text:
        return [
            AgentAction(
                type="final_answer",
                reason_summary="Answer from gathered evidence.",
                payload={"model_answer": text[:MAX_ANSWER_CHARS]},
            )
        ]
    return []


def _action_from_response(response: Any, allowed: set[str]) -> AgentAction | None:
    """First action of the response (kept for callers that need exactly one)."""
    actions = _actions_from_response(response, allowed)
    return actions[0] if actions else None


def _map_common_fields(action: AgentAction, args: dict[str, Any]) -> None:
    files = _string_list(args, ("target_files", "paths", "files", "path", "file"))
    if files:
        action.target_files = files
    symbols = _string_list(args, ("target_symbols", "symbols", "symbol", "symbol_queries"))
    if symbols:
        action.target_symbols = symbols
    command = args.get("command")
    if isinstance(command, list) and command:
        action.command = [str(item) for item in command]
    expected = args.get("expected_output")
    if expected:
        action.expected_output = str(expected)


def _string_list(args: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for key in keys:
        value = args.get(key)
        if isinstance(value, str) and value.strip().startswith("[") and value.strip().endswith("]"):
            # The model intermittently double-serializes list arguments —
            # target_files arrives as the STRING '["calc.py"]', which then
            # resolves to no file at all and kills the edit.
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    value = parsed
            except (ValueError, TypeError):
                pass
        if isinstance(value, str) and value.strip():
            out.append(value.strip().lstrip("/"))
        elif isinstance(value, list):
            out.extend(str(item).strip().lstrip("/") for item in value if str(item).strip())
    deduped: list[str] = []
    for item in out:
        if item and item not in deduped:
            deduped.append(item)
    return deduped
