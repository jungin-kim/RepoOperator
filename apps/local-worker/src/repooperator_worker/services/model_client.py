import json
import socket
from dataclasses import dataclass
from typing import Iterator
from urllib import error, request

from repooperator_worker.config import get_settings


NONPUBLIC_MODEL_DELTA_TYPE = "reasoning" + "_delta"


@dataclass(frozen=True)
class ModelGenerationRequest:
    system_prompt: str
    user_prompt: str


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
        try:
            with request.urlopen(
                http_request,
                timeout=self._settings.model_request_timeout_seconds,
            ) as response:
                for raw_line in response:
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
        return {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt},
            ],
        }


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
