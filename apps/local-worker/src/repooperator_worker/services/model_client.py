import json
import socket
import time
from dataclasses import dataclass
from typing import Any, Iterator
from urllib import error, request

from repooperator_worker.config import Settings, get_settings
from repooperator_worker.services import usage_tracker
from repooperator_worker.services.model_tools import (
    normalize_usage,
    ToolCallResponse,
    parse_anthropic_response,
    parse_openai_response,
    to_anthropic_tools,
    to_openai_tools,
)


NONPUBLIC_MODEL_DELTA_TYPE = "reasoning" + "_delta"

_ANTHROPIC_PROVIDERS = {"anthropic", "claude"}
_ANTHROPIC_VERSION = "2023-06-01"
_ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com/v1"


@dataclass(frozen=True)
class ModelGenerationRequest:
    system_prompt: str
    user_prompt: str
    max_output_tokens: int | None = None


def _post_json(
    *,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: int,
    settings: Settings,
) -> dict[str, Any]:
    """POST JSON and return the decoded response with RepoOperator diagnostics."""

    http_request = request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with request.urlopen(http_request, timeout=timeout) as response:
            decoded = json.loads(response.read().decode("utf-8"))
        usage_tracker.record(normalize_usage(decoded))
        return decoded
    except error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        if exc.code == 404 and "model" in error_body.lower():
            raise RuntimeError(
                "The configured model was not found by the model endpoint. "
                f"Effective worker model: {settings.openai_model or 'not configured'}. "
                f"Configured provider: {settings.configured_model_provider or 'unknown'}. "
                f"Base URL: {settings.openai_base_url or 'not configured'}. "
                "Re-run onboarding or reload/restart the worker after changing model settings."
            ) from exc
        raise RuntimeError(f"Model API request failed with status {exc.code}: {error_body}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise RuntimeError(
            "Model API request timed out. The local worker reached the model backend, but the model did not respond in time."
        ) from exc
    except error.URLError as exc:
        if "timed out" in str(exc.reason).lower():
            raise RuntimeError(
                "Model API request timed out. The local worker reached the model backend, but the model did not respond in time."
            ) from exc
        raise RuntimeError(f"Model API connection failed: {exc.reason}") from exc


class OpenAICompatibleModelClient:
    def __init__(self) -> None:
        self._settings = get_settings()

    @property
    def model_name(self) -> str:
        if not self._settings.openai_model:
            raise ValueError("OPENAI_MODEL is not configured.")
        return self._settings.openai_model

    def generate_text(self, prompt: ModelGenerationRequest) -> str:
        http_request = request.Request(
            url=self._chat_completions_url,
            data=json.dumps(self._build_payload(prompt)).encode("utf-8"),
            headers=self._build_headers(),
            method="POST",
        )

        try:
            with request.urlopen(
                http_request,
                timeout=self._settings.model_request_timeout_seconds,
            ) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 404 and "model" in error_body.lower():
                raise RuntimeError(
                    "The configured model was not found by the model endpoint. "
                    f"Effective worker model: {self._settings.openai_model or 'not configured'}. "
                    f"Configured provider: {self._settings.configured_model_provider or 'unknown'}. "
                    f"Base URL: {self._settings.openai_base_url or 'not configured'}. "
                    "Re-run onboarding or reload/restart the worker after changing model settings."
                ) from exc
            raise RuntimeError(
                f"Model API request failed with status {exc.code}: {error_body}"
            ) from exc
        except TimeoutError as exc:
            raise RuntimeError(
                "Model API request timed out. The local worker reached the model backend, but the model did not respond in time."
            ) from exc
        except socket.timeout as exc:
            raise RuntimeError(
                "Model API request timed out. The local worker reached the model backend, but the model did not respond in time."
            ) from exc
        except error.URLError as exc:
            if "timed out" in str(exc.reason).lower():
                raise RuntimeError(
                    "Model API request timed out. The local worker reached the model backend, but the model did not respond in time."
                ) from exc
            raise RuntimeError(f"Model API connection failed: {exc.reason}") from exc

        response_text = _extract_response_text(response_payload).strip()
        if not response_text:
            raise RuntimeError("Model API response was empty.")
        return response_text

    def stream_text(self, prompt: ModelGenerationRequest) -> Iterator[dict[str, str]]:
        payload = {**self._build_payload(prompt), "stream": True}
        http_request = request.Request(
            url=self._chat_completions_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._build_headers(),
            method="POST",
        )
        stream_started = time.monotonic()
        try:
            with request.urlopen(
                http_request,
                timeout=self._settings.model_request_timeout_seconds,
            ) as response:
                for raw_line in response:
                    if time.monotonic() - stream_started > self._settings.model_request_timeout_seconds:
                        # urlopen's timeout only bounds gaps BETWEEN bytes — a
                        # slow-but-steady stream could run forever.
                        break
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = _extract_stream_delta(chunk)
                    if delta:
                        yield delta
        except (error.HTTPError, error.URLError, TimeoutError, socket.timeout):
            text = self.generate_text(prompt)
            reasoning, content = split_visible_reasoning(text)
            if reasoning:
                yield {"type": NONPUBLIC_MODEL_DELTA_TYPE, "delta": reasoning}
            if content:
                yield {"type": "assistant_delta", "delta": content}

    def generate_with_tools(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        max_output_tokens: int | None = None,
    ) -> ToolCallResponse:
        """Run one native tool-calling turn against an OpenAI-compatible API.

        ``messages`` use the canonical OpenAI chat shape (including ``tool``
        role results and assistant ``tool_calls``). ``tools`` are internal
        model tool specs; they are converted to the OpenAI ``tools`` array.
        """

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "system", "content": system_prompt}, *messages],
        }
        if tools:
            payload["tools"] = to_openai_tools(tools)
            payload["tool_choice"] = tool_choice or "auto"
        if max_output_tokens:
            payload["max_tokens"] = int(max_output_tokens)
        payload.update(self._ollama_options())
        response_payload = _post_json(
            url=self._chat_completions_url,
            payload=payload,
            headers=self._build_headers(),
            timeout=self._settings.model_request_timeout_seconds,
            settings=self._settings,
        )
        return parse_openai_response(response_payload)

    @property
    def _chat_completions_url(self) -> str:
        base_url = self._settings.openai_base_url
        if not base_url:
            raise ValueError("OPENAI_BASE_URL is not configured.")
        return f"{base_url}/chat/completions"

    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        api_key = self._settings.openai_api_key
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        elif self._settings.configured_model_connection_mode != "local-runtime":
            raise ValueError(
                "OPENAI_API_KEY is not configured. "
                "Set it in ~/.repooperator/config.json or via the OPENAI_API_KEY environment variable."
            )
        # Local runtimes (Ollama, vLLM) do not require an API key.
        return headers

    def _build_payload(self, prompt: ModelGenerationRequest) -> dict:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt},
            ],
            # Uncapped output let a slow local model generate for 30+ minutes
            # on a single call. Callers that legitimately need long output
            # (full-file edit proposals) override max_output_tokens.
            "max_tokens": int(prompt.max_output_tokens or 4096),
        }
        payload.update(self._ollama_options())
        return payload

    def _ollama_options(self) -> dict:
        """For Ollama, pass the configured context window as num_ctx so the model
        actually allocates that much KV cache. Only for Ollama — vLLM/remote
        OpenAI endpoints reject an unknown ``options`` field."""
        if (self._settings.configured_model_provider or "").lower() != "ollama":
            return {}
        window = getattr(self._settings, "configured_context_window", None)
        if not window:
            return {}
        return {"options": {"num_ctx": int(window)}}


def _extract_response_text(response_payload: dict) -> str:
    try:
        message = response_payload["choices"][0]["message"]
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("thinking") or ""
        if reasoning and content:
            return f"<think>{reasoning}</think>\n{content}"
        return content or reasoning
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Model API response did not contain a chat completion message.") from exc


def _extract_stream_delta(chunk: dict) -> dict[str, str] | None:
    try:
        delta = chunk["choices"][0].get("delta") or {}
    except (KeyError, IndexError, TypeError):
        return None
    reasoning = delta.get("reasoning_content") or delta.get("thinking")
    if reasoning:
        return {"type": NONPUBLIC_MODEL_DELTA_TYPE, "delta": str(reasoning)}
    content = delta.get("content")
    if content:
        return {"type": "assistant_delta", "delta": str(content)}
    return None


def split_visible_reasoning(text: str) -> tuple[str, str]:
    stripped = text or ""
    if "<think>" not in stripped.lower():
        return "", stripped
    lowered = stripped.lower()
    start = lowered.find("<think>")
    end = lowered.find("</think>", start)
    if start == -1 or end == -1:
        return "", stripped.replace("<think>", "").replace("</think>", "")
    reasoning = stripped[start + len("<think>") : end].strip()
    content = (stripped[:start] + stripped[end + len("</think>") :]).strip()
    return reasoning, content


def _openai_messages_to_anthropic(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate canonical OpenAI chat messages into Anthropic message blocks.

    OpenAI ``tool`` results become Anthropic ``tool_result`` content blocks and
    assistant ``tool_calls`` become ``tool_use`` blocks. Adjacent messages of
    the same role are merged so the transcript stays valid for the Messages API.
    """

    translated: list[dict[str, Any]] = []

    def _append(role: str, block: dict[str, Any]) -> None:
        if translated and translated[-1]["role"] == role:
            translated[-1]["content"].append(block)
        else:
            translated.append({"role": role, "content": [block]})

    for message in messages:
        role = message.get("role")
        if role == "system":
            # Callers pass the system prompt separately; skip stray system turns.
            continue
        if role == "tool":
            _append(
                "user",
                {
                    "type": "tool_result",
                    "tool_use_id": str(message.get("tool_call_id") or ""),
                    "content": str(message.get("content") or ""),
                },
            )
            continue
        if role == "assistant":
            text = str(message.get("content") or "")
            if text:
                _append("assistant", {"type": "text", "text": text})
            for entry in message.get("tool_calls") or []:
                function = entry.get("function") or {}
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        arguments = {}
                _append(
                    "assistant",
                    {
                        "type": "tool_use",
                        "id": str(entry.get("id") or ""),
                        "name": str(function.get("name") or ""),
                        "input": arguments if isinstance(arguments, dict) else {},
                    },
                )
            continue
        # Default to a user text block.
        _append("user", {"type": "text", "text": str(message.get("content") or "")})

    return translated


def _with_message_cache_breakpoint(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark the end of the stable conversation prefix for Anthropic prompt caching.

    Anthropic caches only up to explicit ``cache_control`` breakpoints. The
    agent transcript is append-only except for its final block (the per-step
    run status), so the breakpoint goes on the block just before that one:
    the next step's request extends exactly this prefix and reads it from cache.
    Together with the system and tools breakpoints this stays within the
    four-breakpoint limit.
    """

    if not messages:
        return messages
    last = messages[-1]
    blocks = last.get("content") or []
    if len(blocks) >= 2:
        target_message, index = last, len(blocks) - 2
    elif len(messages) >= 2 and messages[-2].get("content"):
        target_message, index = messages[-2], len(messages[-2]["content"]) - 1
    else:
        return messages
    block = target_message["content"][index]
    if block.get("type") in {"text", "tool_result", "tool_use"}:
        target_message["content"][index] = {**block, "cache_control": {"type": "ephemeral"}}
    return messages


class AnthropicModelClient:
    """Native Anthropic Messages API client with tool-use support."""

    def __init__(self) -> None:
        self._settings = get_settings()

    @property
    def model_name(self) -> str:
        if not self._settings.openai_model:
            raise ValueError("Model name is not configured.")
        return self._settings.openai_model

    @property
    def _messages_url(self) -> str:
        base_url = self._settings.openai_base_url or _ANTHROPIC_DEFAULT_BASE_URL
        return f"{base_url.rstrip('/')}/messages"

    def _build_headers(self) -> dict[str, str]:
        api_key = self._settings.openai_api_key
        if not api_key:
            raise ValueError(
                "An Anthropic API key is not configured. "
                "Set it in ~/.repooperator/config.json or via the OPENAI_API_KEY environment variable."
            )
        return {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
        }

    def generate_with_tools(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        max_output_tokens: int | None = None,
    ) -> ToolCallResponse:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": int(max_output_tokens or 4096),
            "messages": _with_message_cache_breakpoint(_openai_messages_to_anthropic(messages)),
        }
        if system_prompt:
            payload["system"] = [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]
        if tools:
            payload["tools"] = to_anthropic_tools(tools)
            if payload["tools"]:
                payload["tools"][-1] = {**payload["tools"][-1], "cache_control": {"type": "ephemeral"}}
            if tool_choice == "required":
                payload["tool_choice"] = {"type": "any"}
            elif tool_choice and tool_choice not in {"auto", "none"}:
                payload["tool_choice"] = {"type": "tool", "name": tool_choice}
            else:
                payload["tool_choice"] = {"type": tool_choice or "auto"}
        response_payload = _post_json(
            url=self._messages_url,
            payload=payload,
            headers=self._build_headers(),
            timeout=self._settings.model_request_timeout_seconds,
            settings=self._settings,
        )
        return parse_anthropic_response(response_payload)

    def generate_text(self, prompt: ModelGenerationRequest) -> str:
        response = self.generate_with_tools(
            system_prompt=prompt.system_prompt,
            messages=[{"role": "user", "content": prompt.user_prompt}],
        )
        text = response.text.strip()
        if not text:
            raise RuntimeError("Model API response was empty.")
        return text


def resolve_model_provider(settings: Settings | None = None) -> str:
    """Return the effective model provider family: ``anthropic`` or ``openai``."""

    settings = settings or get_settings()
    provider = (settings.configured_model_provider or "").strip().lower()
    if provider in _ANTHROPIC_PROVIDERS:
        return "anthropic"
    model = (settings.openai_model or settings.configured_model_name or "").lower()
    if "claude" in model or provider in _ANTHROPIC_PROVIDERS:
        return "anthropic"
    return "openai"


def build_model_client(settings: Settings | None = None) -> Any:
    """Construct the tool-calling model client for the configured provider.

    All OpenAI-compatible providers (OpenAI remote, Ollama, vLLM) share the
    ``/chat/completions`` client; Anthropic uses its native Messages client.
    """

    if resolve_model_provider(settings) == "anthropic":
        return AnthropicModelClient()
    return OpenAICompatibleModelClient()
