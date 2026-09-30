"""Context-window usage snapshot for the UI (Claude Code / Codex style).

Returns the served model's context window plus a token breakdown of the fixed
overhead (system prompt, loaded tools, MCP tools, skills) and the deferred tools
that are only loaded on demand. The per-chat message tokens are added by the
client from the active chat's transcript, so the window usage is per chat room.
"""

from __future__ import annotations

import json
from typing import Any

from repooperator_worker.agent_core.model_profile import detect_model_profile
from repooperator_worker.config import get_settings


def _tokens(value: Any) -> int:
    """Rough token estimate (~4 chars/token), matching the packer's heuristic."""

    if value is None:
        return 0
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return max(0, round(len(text) / 4))


def context_window_snapshot(thread_id: str | None = None) -> dict[str, Any]:
    settings = get_settings()
    profile = detect_model_profile(settings=settings)

    system_prompt_tokens = _system_prompt_tokens()
    system_tools_tokens, system_tools_deferred = _tool_tokens()
    mcp_tools_tokens, mcp_tools_deferred = _mcp_tokens()
    skills_tokens = _skills_tokens()

    # Show the ACTUAL repo-context tokens from this chat's most recent run.
    # Before any run has packed context there is no real usage yet, so we show
    # 0 (Claude Code / Codex style — measured usage only, not a reserved budget).
    repo_context_tokens = 0
    repo_context_actual = False
    usage = _thread_context_usage(thread_id)
    if usage and usage.get("input_tokens"):
        repo_context_tokens = int(usage["input_tokens"])
        repo_context_actual = True

    return {
        "model_name": profile.model_name,
        "provider": profile.provider,
        "context_window": profile.context_window,
        "max_output_tokens": profile.max_output_tokens,
        "repo_context_actual": repo_context_actual,
        "components": {
            "system_prompt": system_prompt_tokens,
            "system_tools": system_tools_tokens,
            "mcp_tools": mcp_tools_tokens,
            "skills": skills_tokens,
            "repo_context": repo_context_tokens,
        },
        "deferred": {
            "system_tools": system_tools_deferred,
            "mcp_tools": mcp_tools_deferred,
        },
    }


def _thread_context_usage(thread_id: str | None) -> dict[str, Any] | None:
    try:
        from repooperator_worker.services.context_usage_service import get_context_usage

        return get_context_usage(thread_id)
    except Exception:
        return None


def _system_prompt_tokens() -> int:
    try:
        from repooperator_worker.agent_core.agentic_loop import agent_system_prompt

        return _tokens(agent_system_prompt())
    except Exception:
        return 0


def _tool_tokens() -> tuple[int, int]:
    try:
        from repooperator_worker.agent_core.tools.registry import get_default_tool_registry

        registry = get_default_tool_registry()
        loaded = registry.specs_for_model(include_default=True)
        everything = registry.specs_for_model(include_deferred=True)
        loaded_names = {str(s.get("name")) for s in loaded}
        deferred = [s for s in everything if str(s.get("name")) not in loaded_names]
        return _tokens(loaded), _tokens(deferred)
    except Exception:
        return 0, 0


def _mcp_tokens() -> tuple[int, int]:
    try:
        from repooperator_worker.agent_core.mcp import get_default_mcp_registry

        registry = get_default_mcp_registry()
        enabled = registry.specs_for_model(enabled_only=True)
        allservers = registry.specs_for_model(enabled_only=False)
        enabled_ids = {str(s.get("id")) for s in enabled}
        deferred = [s for s in allservers if str(s.get("id")) not in enabled_ids]
        return _tokens(enabled), _tokens(deferred)
    except Exception:
        return 0, 0


def _skills_tokens() -> int:
    try:
        from repooperator_worker.services.skills_service import discover_skills

        return _tokens(discover_skills())
    except Exception:
        return 0
