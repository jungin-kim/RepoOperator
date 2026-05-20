from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from repooperator_worker.config import Settings, get_settings
from repooperator_worker.services.json_safe import json_safe


@dataclass(frozen=True)
class ModelProfile:
    provider: str
    model_name: str
    context_window: int
    max_output_tokens: int
    supports_streaming: bool
    supports_tool_calls: bool
    supports_json_schema: bool
    supports_reasoning_signal: bool
    tokenizer_hint: str
    compression_strategy: str

    def model_dump(self) -> dict[str, Any]:
        payload = json_safe(self)
        payload["supports_" + "reasoning" + "_delta"] = payload.pop("supports_reasoning_signal")
        return payload


_KNOWN_MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "gpt-4o": {"context_window": 128_000, "max_output_tokens": 16_384, "tokenizer_hint": "o200k_base", "compression_strategy": "balanced"},
    "gpt-4.1": {"context_window": 1_000_000, "max_output_tokens": 32_768, "tokenizer_hint": "o200k_base", "compression_strategy": "generous"},
    "gpt-5": {"context_window": 400_000, "max_output_tokens": 128_000, "tokenizer_hint": "o200k_base", "compression_strategy": "balanced"},
    "claude-3.5": {"context_window": 200_000, "max_output_tokens": 8_192, "tokenizer_hint": "anthropic", "compression_strategy": "balanced"},
    "llama": {"context_window": 32_000, "max_output_tokens": 4_096, "tokenizer_hint": "sentencepiece", "compression_strategy": "aggressive"},
    "mistral": {"context_window": 32_000, "max_output_tokens": 4_096, "tokenizer_hint": "sentencepiece", "compression_strategy": "aggressive"},
}


def detect_model_profile(
    *,
    settings: Settings | None = None,
    provider: str | None = None,
    model_name: str | None = None,
    provider_metadata: dict[str, Any] | None = None,
) -> ModelProfile:
    settings = settings or get_settings()
    selected_provider = provider or settings.configured_model_provider or "openai-compatible"
    selected_model = model_name or settings.configured_model_name or settings.openai_model or "unknown-model"
    metadata = provider_metadata or {}
    known = _known_profile_for(selected_model)
    context_window = int(metadata.get("context_window") or known.get("context_window") or _fallback_context_window(selected_model))
    max_output_tokens = int(metadata.get("max_output_tokens") or known.get("max_output_tokens") or min(8_192, max(1_024, context_window // 8)))
    tokenizer_hint = str(metadata.get("tokenizer_hint") or known.get("tokenizer_hint") or _tokenizer_hint(selected_provider, selected_model))
    strategy = str(metadata.get("compression_strategy") or known.get("compression_strategy") or _compression_strategy(context_window))
    return ModelProfile(
        provider=str(selected_provider),
        model_name=str(selected_model),
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        supports_streaming=bool(metadata.get("supports_streaming", True)),
        supports_tool_calls=bool(metadata.get("supports_tool_calls", False)),
        supports_json_schema=bool(metadata.get("supports_json_schema", False)),
        supports_reasoning_signal=bool(metadata.get("supports_" + "reasoning" + "_delta", False)),
        tokenizer_hint=tokenizer_hint,
        compression_strategy=strategy,
    )


def _known_profile_for(model_name: str) -> dict[str, Any]:
    lowered = (model_name or "").lower()
    for key, value in _KNOWN_MODEL_PROFILES.items():
        if key in lowered:
            return value
    return {}


def _fallback_context_window(model_name: str) -> int:
    lowered = (model_name or "").lower()
    if any(token in lowered for token in ("mini", "small", "7b", "8b")):
        return 16_000
    if any(token in lowered for token in ("32k", "70b", "large")):
        return 32_000
    return 64_000


def _tokenizer_hint(provider: str, model_name: str) -> str:
    lowered = f"{provider} {model_name}".lower()
    if "openai" in lowered or "gpt" in lowered:
        return "o200k_base"
    if "anthropic" in lowered or "claude" in lowered:
        return "anthropic"
    return "unknown"


def _compression_strategy(context_window: int) -> str:
    if context_window <= 32_000:
        return "aggressive"
    if context_window >= 200_000:
        return "generous"
    return "balanced"
