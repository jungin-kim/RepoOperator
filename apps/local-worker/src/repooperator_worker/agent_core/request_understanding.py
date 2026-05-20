"""
Request understanding — extracts facts and constraints from the user's task.

This replaces the old intent-classifier approach. Instead of assigning workflow
buckets, we extract:
  - what the user is actually asking for (user_goal)
  - which files / symbols they mentioned (mentioned_files, mentioned_symbols)
  - constraints the user stated explicitly (constraints)
  - what kind of output they expect (requested_outputs)
  - which tools would likely help, as a *weak hint only* (likely_needed_tools)
  - safety notes from the task text (safety_notes)
  - things that are unclear (uncertainties)
  - a clarification question if the task is genuinely ambiguous (clarification_question)

IMPORTANT: This module must NOT assign authoritative workflow buckets.
  - Do not add legacy intent constants.
  - Do not populate old workflow-routing fields.
  - Do not branch planner behavior on any field here.
  - likely_needed_tools is a *weak hint* to the planner, never an authoritative route.
  - The planner must choose safe primitive actions based on evidence needs and tool specs.
"""
from __future__ import annotations

import dataclasses
import json
from typing import Any

from repooperator_worker.agent_core.request_parsing import extract_file_tokens
from repooperator_worker.schemas import AgentRunRequest
from repooperator_worker.services.model_client import ModelGenerationRequest, OpenAICompatibleModelClient

_UNDERSTANDING_PROMPT = """\
You are RepoOperator's task-understanding layer. Read the user's request and \
return a JSON object that captures *facts and constraints* — NOT workflow buckets \
or routing decisions.

Return ONLY a JSON object matching this schema:
{
  "user_goal": "<concise restatement of what the user wants>",
  "mentioned_files": ["<repo-relative paths explicitly named by the user>"],
  "mentioned_symbols": ["<class/function/variable names explicitly named>"],
  "constraints": ["<explicit constraints the user stated, e.g. 'only look at X'>"],
  "requested_outputs": ["<output types requested, e.g. 'explanation', 'diff', 'list'>"],
  "likely_needed_tools": ["<weak tool hints: read_file | search_files | run_command | generate_change_set | generate_edit | ask_clarification>"],
  "safety_notes": ["<any safety or scope constraints implicit in the task>"],
  "uncertainties": ["<things that are unclear or ambiguous>"],
  "needs_clarification": false,
  "clarification_question": null
}

Rules:
- Do not invent files or symbols not mentioned.
- likely_needed_tools must only contain values from the allowed set.
- Never assign a workflow category or routing bucket.
- If the task mentions no specific files, mentioned_files must be [].
- Keep strings short and factual.
"""

_ALLOWED_TOOL_HINTS = frozenset({
    "read_file", "search_files", "search_text", "run_command",
    "generate_change_set", "generate_edit", "ask_clarification", "inspect_repo_tree",
})


@dataclasses.dataclass
class RequestUnderstanding:
    user_goal: str = ""
    mentioned_files: list[str] = dataclasses.field(default_factory=list)
    mentioned_symbols: list[str] = dataclasses.field(default_factory=list)
    constraints: list[str] = dataclasses.field(default_factory=list)
    requested_outputs: list[str] = dataclasses.field(default_factory=list)
    likely_needed_tools: list[str] = dataclasses.field(default_factory=list)
    safety_notes: list[str] = dataclasses.field(default_factory=list)
    uncertainties: list[str] = dataclasses.field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: str | None = None

    def model_dump(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def understand_request(request: AgentRunRequest) -> RequestUnderstanding:
    """Extract facts and constraints from the user request.

    Falls back to purely deterministic extraction if the model call fails.
    """
    try:
        raw = OpenAICompatibleModelClient().generate_text(
            ModelGenerationRequest(
                system_prompt=_UNDERSTANDING_PROMPT,
                user_prompt=json.dumps(
                    {
                        "task": request.task,
                        "recent_messages": [
                            {"role": m.role, "content": m.content[:400]}
                            for m in request.conversation_history[-6:]
                        ],
                    },
                    ensure_ascii=False,
                ),
            )
        )
        payload = _parse_json(raw)
    except Exception:
        payload = {}
    return _build_understanding(payload, request)


def request_understanding_to_classifier_result(ru: RequestUnderstanding, request: AgentRunRequest):
    """Adapter: converts RequestUnderstanding to the legacy ClassifierResult.

    This exists only for backward compatibility with code that still reads
    ClassifierResult fields.  The adapter DOES NOT populate routing fields
    from the previous workflow-bucket classifier.
    Those fields are absent from ClassifierResult entirely.
    """
    from repooperator_worker.agent_core.state import ClassifierResult  # local import avoids cycle
    return ClassifierResult(
        intent="ambiguous",
        confidence=0.0,
        target_files=list(ru.mentioned_files),
        target_symbols=list(ru.mentioned_symbols),
        requested_action=ru.user_goal[:120] if ru.user_goal else "",
        needs_tool=ru.likely_needed_tools[0] if ru.likely_needed_tools else None,
        needs_clarification=ru.needs_clarification,
        clarification_question=ru.clarification_question,
        raw={"request_understanding": ru.model_dump()},
    )


# ── Internals ─────────────────────────────────────────────────────────────────

def _build_understanding(payload: dict[str, Any], request: AgentRunRequest) -> RequestUnderstanding:
    # Always extract file tokens deterministically so a malformed model response
    # does not lose explicit file mentions.
    deterministic_files = extract_file_tokens(request.task)
    model_files = [_safe_public_text(f, limit=240).lstrip("/") for f in payload.get("mentioned_files") or [] if _safe_public_text(f, limit=240)]
    mentioned_files = _dedupe([*model_files, *deterministic_files])

    model_symbols = _safe_public_list(payload.get("mentioned_symbols"), limit=120)
    tool_hints = [
        str(t).strip() for t in payload.get("likely_needed_tools") or []
        if str(t).strip() in _ALLOWED_TOOL_HINTS
    ]
    return RequestUnderstanding(
        user_goal=_safe_public_text(payload.get("user_goal") or request.task, limit=200) or request.task[:200],
        mentioned_files=mentioned_files,
        mentioned_symbols=model_symbols,
        constraints=_safe_public_list(payload.get("constraints"), limit=180),
        requested_outputs=_safe_public_list(payload.get("requested_outputs"), limit=80),
        likely_needed_tools=tool_hints,
        safety_notes=_safe_public_list(payload.get("safety_notes"), limit=220),
        uncertainties=_safe_public_list(payload.get("uncertainties"), limit=220),
        needs_clarification=bool(payload.get("needs_clarification")),
        clarification_question=_safe_public_text(payload.get("clarification_question"), limit=240),
    )


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


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


def _safe_public_list(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = _safe_public_text(item, limit=limit)
        if text:
            result.append(text)
    return result


def _safe_public_text(value: Any, *, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if not text or _contains_nonpublic_reasoning_marker(text):
        return ""
    return text if len(text) <= limit else text[: max(0, limit - 3)].rstrip() + "..."


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
