"""Model-backed worker subagents for the supervisor path.

Phase 3 replaces the supervisor's canned worker reports with real, bounded
model-driven subagents. Each worker runs a small think -> act -> observe loop
scoped to its role and file group, using a role-specific NON-MUTATING tool subset
(base read tools plus, per role, web-read / git-read / command-classify tools)
through the same ToolOrchestrator (permissions, hooks, secret redaction still
apply), then synthesizes a structured report. Mutation, command execution, git
writes, and change-set apply remain with the parent graph and its approval gates.

When native tool calling is unavailable (not configured, or the flag is off),
:func:`run_worker_subagent` returns ``None`` so the supervisor falls back to its
deterministic worker behavior. This mirrors the primary-loop fallback design.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from repooperator_worker.agent_core.actions import AgentAction
from repooperator_worker.agent_core.agentic_loop import tool_calling_available
from repooperator_worker.agent_core.hooks import HookManager
from repooperator_worker.agent_core.tool_orchestrator import ToolOrchestrator
from repooperator_worker.agent_core.tools.registry import get_default_tool_registry
from repooperator_worker.config import Settings, get_settings
from repooperator_worker.schemas import AgentRunRequest
from repooperator_worker.services.json_safe import json_safe
from repooperator_worker.services.model_client import build_model_client

# Base evidence tools every worker may use (repo-contained, read-only).
SUBAGENT_READONLY_TOOLS = frozenset(
    {"inspect_repo_tree", "search_files", "search_text", "read_file", "read_many_files", "analyze_file"}
)

# Role-specific NON-MUTATING tools so each worker can actually do its job instead
# of only reading files. These are read/network-read/git-read/classify tools —
# mutation, command execution, git writes, and change-set apply stay with the
# parent graph and its approval gates. Tools still pass through the same
# ToolOrchestrator (permissions, hooks, secret redaction) when invoked.
ROLE_EXTRA_TOOLS: dict[str, frozenset[str]] = {
    "WebResearchAgent": frozenset({"search_web", "fetch_url", "summarize_web_evidence"}),
    "GitAgent": frozenset({"git_status", "git_diff", "git_log", "inspect_git_state"}),
    "ValidationAgent": frozenset({"preview_command"}),
    "TestAgent": frozenset({"preview_command"}),
}


def allowed_tools_for_role(role: str, registry) -> set[str]:
    """The non-mutating tool set a given worker role may use, intersected with
    what the registry actually exposes."""
    wanted = set(SUBAGENT_READONLY_TOOLS) | set(ROLE_EXTRA_TOOLS.get(role, frozenset()))
    return wanted & set(registry.allowed_action_types())


MAX_SUBAGENT_STEPS = 6
MAX_CALLS_PER_STEP = 6
MAX_OBSERVATION_CHARS = 1200

SUBAGENT_SYSTEM_PROMPT = """\
You are a {role} subagent inside RepoOperator, working on a bounded slice of a
larger repository task. You may only use the non-mutating tools available to your
role (reading, searching, and — for some roles — web lookups, git inspection, or
command classification). You cannot modify files, run commands, or push.

- Use tools to inspect exactly the files in your scope; do not wander.
- After you have enough evidence, stop calling tools and reply with a short
  plain-text report of what you found for your scope (2-5 sentences). Do not
  include hidden reasoning.
"""


def run_worker_subagent(
    task: dict[str, Any],
    *,
    request: AgentRunRequest,
    run_id: str,
    settings: Settings | None = None,
    client_factory: Callable[[Settings], Any] = build_model_client,
    orchestrator: ToolOrchestrator | None = None,
) -> dict[str, Any] | None:
    """Run a bounded model-driven subagent for one work unit.

    Returns a worker report dict, or ``None`` when tool calling is unavailable
    so the caller can use its deterministic fallback.
    """

    settings = settings or get_settings()
    if not tool_calling_available(settings):
        return None

    role = str(task.get("role") or task.get("assigned_worker_role") or "AnalysisAgent")
    scope = task.get("scope") or task.get("group")
    files = [str(item) for item in task.get("input_files") or task.get("files") or []]

    registry = get_default_tool_registry()
    allowed = allowed_tools_for_role(role, registry)
    # tool_names surfaces exactly these specs even though role tools (git/web/
    # preview) are deferred by default; include_default=False keeps the set scoped.
    tool_specs = [
        spec
        for spec in registry.specs_for_model(tool_names=allowed, include_default=False)
        if spec.get("name") in allowed
    ]
    if not tool_specs:
        return None

    orchestrator = orchestrator or ToolOrchestrator(
        run_id=run_id,
        request=request,
        registry=registry,
        hook_manager=HookManager(),
    )

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": request.task,
                    "your_role": role,
                    "your_scope": scope,
                    "files_in_scope": files,
                    "goal": task.get("goal") or f"Analyze the {scope or 'assigned'} files for the current task.",
                },
                ensure_ascii=False,
            ),
        }
    ]
    system_prompt = SUBAGENT_SYSTEM_PROMPT.format(role=role)

    client = client_factory(settings)
    files_analyzed: list[str] = []
    summary = ""
    for _ in range(MAX_SUBAGENT_STEPS):
        try:
            response = client.generate_with_tools(
                system_prompt=system_prompt,
                messages=messages,
                tools=tool_specs,
                tool_choice="auto",
                max_output_tokens=1024,
            )
        except Exception:
            break

        if not getattr(response, "has_tool_calls", False):
            summary = (getattr(response, "text", "") or "").strip()
            break

        # Run every allowed call the model asked for in this response. Models
        # with parallel tool calling batch their reads; taking only the first
        # call dropped the rest and cost an extra round trip per file.
        calls = []
        for call in response.tool_calls[:MAX_CALLS_PER_STEP]:
            if call.name not in allowed:
                break
            calls.append(call)
        if not calls:
            summary = (getattr(response, "text", "") or "").strip()
            break

        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": json.dumps(dict(call.arguments), ensure_ascii=False)}}
                    for call in calls
                ],
            }
        )
        for call in calls:
            action = _action_from_call(call)
            result = orchestrator.execute_action(action)
            for path in result.files_read or []:
                if path not in files_analyzed:
                    files_analyzed.append(path)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": _observation_text(result)})

    if not summary:
        summary = f"{role} gathered evidence for {scope or 'the assigned scope'} across {len(files_analyzed)} file(s)."

    return _report(role, task, files, files_analyzed, summary)


def _action_from_call(call: Any) -> AgentAction:
    args = dict(call.arguments or {})
    action = AgentAction(
        type=call.name,  # type: ignore[arg-type]
        reason_summary=str(args.get("reason_summary") or f"Subagent {call.name}")[:200],
        payload=json_safe(args),
    )
    files = args.get("target_files") or args.get("paths") or args.get("path")
    if isinstance(files, str):
        action.target_files = [files.strip().lstrip("/")]
    elif isinstance(files, list):
        action.target_files = [str(item).strip().lstrip("/") for item in files if str(item).strip()]
    return action


def _observation_text(result: Any) -> str:
    parts = [f"status={getattr(result, 'status', 'unknown')}"]
    observation = str(getattr(result, "observation", "") or "")
    if observation:
        parts.append(observation)
    files_read = list(getattr(result, "files_read", []) or [])
    if files_read:
        parts.append("files_read: " + ", ".join(files_read[:10]))
    text = "\n".join(parts)
    return text[:MAX_OBSERVATION_CHARS]


def _report(role: str, task: dict[str, Any], files: list[str], files_analyzed: list[str], summary: str) -> dict[str, Any]:
    report = {
        "worker": role,
        "work_unit_id": task.get("id") or task.get("task_id"),
        "task_id": task.get("task_id"),
        "role": role,
        "scope": task.get("scope") or task.get("group"),
        "group": task.get("group"),
        "files": files,
        "files_analyzed": files_analyzed or files,
        "findings": [summary] if summary else [],
        "summary": summary,
        "recommended_next_actions": ["Reduce this subagent report into the parent evidence summary."],
        "status": "completed",
        "subagent": True,
    }
    if role == "EditPlanningAgent":
        report["risk_notes"] = ["Change plan still requires proposal validation before it can be shown as valid."]
    return report
