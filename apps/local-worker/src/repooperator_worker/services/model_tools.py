"""Native tool-calling primitives shared across model providers.

This module gives RepoOperator a provider-agnostic representation of tool
definitions and tool calls so the agent core can drive a real
think -> act -> observe loop with native function calling instead of
prompt-embedded JSON. The converters translate the internal ``ToolSpec``
JSON schema into the OpenAI ``tools`` shape and the Anthropic ``tools``
shape, and the parsers normalize the two providers' responses back into a
single :class:`ToolCall` / :class:`ToolCallResponse` pair.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": dict(self.arguments)}


@dataclass(frozen=True)
class ToolCallResponse:
    """A normalized model response that may contain text and/or tool calls."""

    text: str = ""
    reasoning: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str | None = None
    raw: dict[str, Any] | None = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def usage(self) -> dict[str, int]:
        """Normalized token usage, including prompt-cache hits when reported.

        Keys: ``input_tokens`` (all prompt tokens, cached included),
        ``cached_input_tokens`` (served from the provider's prompt cache),
        ``cache_write_tokens`` (Anthropic cache creation), ``output_tokens``.
        Empty when the provider reported nothing.
        """
        return normalize_usage(self.raw)

    def model_dump(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "reasoning": self.reasoning,
            "tool_calls": [call.model_dump() for call in self.tool_calls],
            "finish_reason": self.finish_reason,
        }


def normalize_usage(raw: Any) -> dict[str, int]:
    """Read token usage from an OpenAI-compatible or Anthropic response body."""

    if not isinstance(raw, dict):
        return {}
    usage = raw.get("usage")
    out: dict[str, int] = {}
    if isinstance(usage, dict):
        def _int(value: Any) -> int:
            try:
                return int(value or 0)
            except (TypeError, ValueError):
                return 0

        if "input_tokens" in usage:  # Anthropic Messages API
            cached = _int(usage.get("cache_read_input_tokens"))
            written = _int(usage.get("cache_creation_input_tokens"))
            out["input_tokens"] = _int(usage.get("input_tokens")) + cached + written
            out["cached_input_tokens"] = cached
            out["cache_write_tokens"] = written
            out["output_tokens"] = _int(usage.get("output_tokens"))
        else:  # OpenAI-compatible (OpenAI, vLLM, Ollama /v1, DashScope, Gemini compat)
            details = usage.get("prompt_tokens_details") or {}
            out["input_tokens"] = _int(usage.get("prompt_tokens"))
            out["cached_input_tokens"] = _int(details.get("cached_tokens") if isinstance(details, dict) else 0)
            out["output_tokens"] = _int(usage.get("completion_tokens"))
    elif "prompt_eval_count" in raw:  # native Ollama body
        out["input_tokens"] = int(raw.get("prompt_eval_count") or 0)
        out["output_tokens"] = int(raw.get("eval_count") or 0)
    return {key: value for key, value in out.items() if value}


def _normalized_schema(schema: Any) -> dict[str, Any]:
    """Return a JSON-schema object that both providers accept.

    Providers require an ``object`` schema with a ``properties`` map. Tools
    that declare no schema still need a valid empty object schema.
    """

    if not isinstance(schema, dict) or not schema:
        return {"type": "object", "properties": {}}
    normalized = dict(schema)
    normalized.setdefault("type", "object")
    if normalized["type"] == "object":
        normalized.setdefault("properties", {})
    return normalized


def to_openai_tools(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert internal model tool specs to the OpenAI ``tools`` array."""

    tools: list[dict[str, Any]] = []
    for spec in specs or []:
        name = str(spec.get("name") or "").strip()
        if not name:
            continue
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(spec.get("description") or spec.get("prompt_summary") or name),
                    "parameters": _normalized_schema(spec.get("input_schema")),
                },
            }
        )
    return tools


def to_anthropic_tools(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert internal model tool specs to the Anthropic ``tools`` array."""

    tools: list[dict[str, Any]] = []
    for spec in specs or []:
        name = str(spec.get("name") or "").strip()
        if not name:
            continue
        tools.append(
            {
                "name": name,
                "description": str(spec.get("description") or spec.get("prompt_summary") or name),
                "input_schema": _normalized_schema(spec.get("input_schema")),
            }
        )
    return tools


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def parse_openai_response(response_payload: dict[str, Any]) -> ToolCallResponse:
    """Normalize an OpenAI ``/chat/completions`` response."""

    try:
        choice = response_payload["choices"][0]
        message = choice.get("message") or {}
    except (KeyError, IndexError, TypeError):
        return ToolCallResponse(raw=response_payload)

    tool_calls: list[ToolCall] = []
    for index, entry in enumerate(message.get("tool_calls") or []):
        if not isinstance(entry, dict):
            continue
        function = entry.get("function") or {}
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        tool_calls.append(
            ToolCall(
                id=str(entry.get("id") or f"call_{index}"),
                name=name,
                arguments=_parse_arguments(function.get("arguments")),
            )
        )

    text = str(message.get("content") or "")
    reasoning = str(message.get("reasoning_content") or message.get("thinking") or "")
    return ToolCallResponse(
        text=text,
        reasoning=reasoning,
        tool_calls=tuple(tool_calls),
        finish_reason=str(choice.get("finish_reason")) if choice.get("finish_reason") else None,
        raw=response_payload,
    )


def parse_anthropic_response(response_payload: dict[str, Any]) -> ToolCallResponse:
    """Normalize an Anthropic ``/v1/messages`` response."""

    content = response_payload.get("content")
    if not isinstance(content, list):
        return ToolCallResponse(raw=response_payload)

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for index, block in enumerate(content):
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text_parts.append(str(block.get("text") or ""))
        elif block_type == "thinking":
            reasoning_parts.append(str(block.get("thinking") or ""))
        elif block_type == "tool_use":
            name = str(block.get("name") or "").strip()
            if not name:
                continue
            arguments = block.get("input")
            tool_calls.append(
                ToolCall(
                    id=str(block.get("id") or f"toolu_{index}"),
                    name=name,
                    arguments=arguments if isinstance(arguments, dict) else {},
                )
            )

    return ToolCallResponse(
        text="".join(text_parts),
        reasoning="".join(reasoning_parts),
        tool_calls=tuple(tool_calls),
        finish_reason=str(response_payload.get("stop_reason")) if response_payload.get("stop_reason") else None,
        raw=response_payload,
    )
