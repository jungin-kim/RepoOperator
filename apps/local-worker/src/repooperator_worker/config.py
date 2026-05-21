import os
import json
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProviderSettings:
    provider: str
    base_url: str | None
    token: str | None


WRITE_MODE_READ_ONLY = "read-only"
WRITE_MODE_WRITE_WITH_APPROVAL = "write-with-approval"
WRITE_MODE_AUTO_APPLY = "auto-apply"
_VALID_WRITE_MODES = {WRITE_MODE_READ_ONLY, WRITE_MODE_WRITE_WITH_APPROVAL, WRITE_MODE_AUTO_APPLY}
AVAILABLE_WRITE_MODES = [
    WRITE_MODE_READ_ONLY,
    WRITE_MODE_WRITE_WITH_APPROVAL,
    WRITE_MODE_AUTO_APPLY,
]
PERMISSION_MODE_BASIC = "basic"
PERMISSION_MODE_AUTO_REVIEW = "auto_review"
PERMISSION_MODE_DEFAULT = "default"
PERMISSION_MODE_PLAN_ONLY = "plan_only"
PERMISSION_MODE_PROPOSAL_ONLY = "proposal_only"
PERMISSION_MODE_ACCEPT_EDITS = "accept_edits"
PERMISSION_MODE_AUTO_READONLY = "auto_readonly"
PERMISSION_MODE_FULL_ACCESS = "full_access"
PERMISSION_MODE_ROUTINE_SAFE = "routine_safe"
PERMISSION_MODE_HEADLESS_SAFE = "headless_safe"
AVAILABLE_PERMISSION_MODES = [
    PERMISSION_MODE_DEFAULT,
    PERMISSION_MODE_PLAN_ONLY,
    PERMISSION_MODE_PROPOSAL_ONLY,
    PERMISSION_MODE_ACCEPT_EDITS,
    PERMISSION_MODE_AUTO_READONLY,
    PERMISSION_MODE_BASIC,
    PERMISSION_MODE_AUTO_REVIEW,
    PERMISSION_MODE_FULL_ACCESS,
    PERMISSION_MODE_ROUTINE_SAFE,
    PERMISSION_MODE_HEADLESS_SAFE,
]


@dataclass(frozen=True)
class Settings:
    repo_base_dir: Path
    default_command_timeout_seconds: int
    git_clone_timeout_seconds: int
    git_fetch_timeout_seconds: int
    git_push_timeout_seconds: int
    gitlab_base_url: str | None
    gitlab_token: str | None
    github_base_url: str | None
    github_token: str | None
    openai_base_url: str | None
    openai_api_key: str | None
    openai_model: str | None
    model_request_timeout_seconds: int
    repooperator_config_path: Path
    repooperator_home_dir: Path
    configured_git_provider: str | None
    configured_repository_sources: list[dict]
    configured_model_connection_mode: str | None
    configured_model_provider: str | None
    configured_model_name: str | None
    config_loaded_at: str
    config_hash: str
    permission_mode: str
    write_mode: str
    composio_api_key: str | None

    def get_provider_settings(self, provider: str) -> ProviderSettings:
        normalized = provider.strip().lower()
        if normalized == "gitlab":
            return ProviderSettings(
                provider="gitlab",
                base_url=self.gitlab_base_url,
                token=self.gitlab_token,
            )
        if normalized == "github":
            return ProviderSettings(
                provider="github",
                base_url=self.github_base_url,
                token=self.github_token,
            )
        raise ValueError(f"Unsupported git provider: {provider}")


def get_settings() -> Settings:
    repooperator_config_path = _get_repooperator_config_path()
    runtime_config = _load_runtime_config(repooperator_config_path)
    repo_base_dir = Path(
        os.getenv("LOCAL_REPO_BASE_DIR", Path.home() / ".repooperator" / "repos")
    ).expanduser().resolve()

    return Settings(
        repo_base_dir=repo_base_dir,
        default_command_timeout_seconds=int(
            os.getenv("REPOOPERATOR_COMMAND_TIMEOUT_SECONDS", "30")
        ),
        git_clone_timeout_seconds=int(os.getenv("REPOOPERATOR_GIT_CLONE_TIMEOUT_SECONDS", "300")),
        git_fetch_timeout_seconds=int(os.getenv("REPOOPERATOR_GIT_FETCH_TIMEOUT_SECONDS", "120")),
        git_push_timeout_seconds=int(os.getenv("REPOOPERATOR_GIT_PUSH_TIMEOUT_SECONDS", "180")),
        gitlab_base_url=_resolve_provider_value(
            env_value=os.getenv("GITLAB_BASE_URL"),
            runtime_config=runtime_config,
            provider="gitlab",
            key="baseUrl",
            normalizer=_normalize_optional_url,
        ),
        gitlab_token=_resolve_provider_value(
            env_value=os.getenv("GITLAB_TOKEN"),
            runtime_config=runtime_config,
            provider="gitlab",
            key="token",
            normalizer=_normalize_optional_value,
        ),
        github_base_url=_resolve_provider_value(
            env_value=os.getenv("GITHUB_BASE_URL"),
            runtime_config=runtime_config,
            provider="github",
            key="baseUrl",
            normalizer=_normalize_optional_url,
        ),
        github_token=_resolve_provider_value(
            env_value=os.getenv("GITHUB_TOKEN"),
            runtime_config=runtime_config,
            provider="github",
            key="token",
            normalizer=_normalize_optional_value,
        ),
        openai_base_url=_resolve_model_base_url(runtime_config),
        openai_api_key=_resolve_model_api_key(runtime_config),
        openai_model=_resolve_model_name(runtime_config),
        model_request_timeout_seconds=int(
            os.getenv("REPOOPERATOR_MODEL_REQUEST_TIMEOUT_SECONDS", "120")
        ),
        repooperator_config_path=repooperator_config_path,
        repooperator_home_dir=repooperator_config_path.parent,
        configured_git_provider=_resolve_configured_git_provider(runtime_config),
        configured_repository_sources=_resolve_configured_repository_sources(runtime_config),
        configured_model_connection_mode=_resolve_configured_model_connection_mode(runtime_config),
        configured_model_provider=_resolve_configured_model_provider(runtime_config),
        configured_model_name=_resolve_configured_model_name(runtime_config),
        config_loaded_at=_config_loaded_at(repooperator_config_path),
        config_hash=_safe_config_hash(runtime_config),
        permission_mode=_resolve_permission_mode(runtime_config),
        write_mode=_resolve_write_mode(runtime_config),
        composio_api_key=_normalize_optional_value(os.getenv("REPOOPERATOR_COMPOSIO_API_KEY")),
    )


def _get_repooperator_config_path() -> Path:
    configured = os.getenv("REPOOPERATOR_CONFIG_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    repooperator_path = (Path.home() / ".repooperator" / "config.json").resolve()
    return repooperator_path


def _load_runtime_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _resolve_provider_value(
    env_value: str | None,
    runtime_config: dict,
    provider: str,
    key: str,
    normalizer,
) -> str | None:
    env_normalized = normalizer(env_value)
    if env_normalized is not None:
        return env_normalized

    for provider_config in _iter_repository_source_configs(runtime_config):
        if provider_config.get("provider") == provider:
            return normalizer(provider_config.get(key))

    provider_config = runtime_config.get("gitProvider")
    if not isinstance(provider_config, dict):
        return None
    if provider_config.get("provider") != provider:
        return None
    return normalizer(provider_config.get(key))


def _iter_repository_source_configs(runtime_config: dict) -> list[dict]:
    sources = runtime_config.get("repositorySources")
    if not isinstance(sources, list):
        return []
    return [source for source in sources if isinstance(source, dict)]


def _normalize_optional_url(value: str | None) -> str | None:
    normalized = _normalize_optional_value(value)
    if normalized is None:
        return None
    return normalized.rstrip("/")


def _normalize_optional_value(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _resolve_configured_git_provider(runtime_config: dict) -> str | None:
    provider_config = runtime_config.get("gitProvider")
    if not isinstance(provider_config, dict):
        return None
    provider = _normalize_optional_value(provider_config.get("provider"))
    if provider in {"gitlab", "github", "local"}:
        return provider
    return None


def _resolve_configured_repository_sources(runtime_config: dict) -> list[dict]:
    sources: list[dict] = []
    for source in _iter_repository_source_configs(runtime_config):
        provider = _normalize_optional_value(source.get("provider"))
        if provider not in {"gitlab", "github", "local"}:
            continue
        sources.append(_safe_repository_source_summary(source))
    provider_config = runtime_config.get("gitProvider")
    if isinstance(provider_config, dict):
        provider = _normalize_optional_value(provider_config.get("provider"))
        if provider in {"gitlab", "github", "local"}:
            sources.insert(0, _safe_repository_source_summary(provider_config))
    return _dedupe_repository_source_summaries(sources)


def _safe_repository_source_summary(source: dict) -> dict:
    provider = _normalize_optional_value(source.get("provider"))
    summary = {
        "provider": provider,
        "baseUrl": _normalize_optional_url(source.get("baseUrl")),
        "tokenConfigured": bool(_normalize_optional_value(source.get("token"))),
    }
    owner = _normalize_optional_value(source.get("owner"))
    if owner:
        summary["owner"] = owner
    return summary


def _dedupe_repository_source_summaries(sources: list[dict]) -> list[dict]:
    seen: set[tuple[str | None, str | None, str | None]] = set()
    deduped: list[dict] = []
    for source in sources:
        key = (source.get("provider"), source.get("baseUrl"), source.get("owner"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(source)
    return deduped


_LOCAL_RUNTIME_PROVIDERS = {"ollama", "vllm"}


def _resolve_configured_model_connection_mode(runtime_config: dict) -> str | None:
    model_config = runtime_config.get("model")
    if not isinstance(model_config, dict):
        return None
    connection_mode = _normalize_optional_value(model_config.get("connectionMode"))
    if connection_mode in {"local-runtime", "remote-api"}:
        return connection_mode
    provider = _normalize_optional_value(model_config.get("provider"))
    if provider in _LOCAL_RUNTIME_PROVIDERS:
        return "local-runtime"
    if provider:
        return "remote-api"
    return None


def _resolve_configured_model_provider(runtime_config: dict) -> str | None:
    model_config = runtime_config.get("model")
    if not isinstance(model_config, dict):
        return None
    return _normalize_optional_value(model_config.get("provider"))


def _resolve_configured_model_name(runtime_config: dict) -> str | None:
    model_config = runtime_config.get("model")
    if not isinstance(model_config, dict):
        return None
    return _normalize_optional_value(model_config.get("model"))


def _resolve_model_base_url(runtime_config: dict) -> str | None:
    env_value = _normalize_optional_url(os.getenv("OPENAI_BASE_URL"))
    if env_value is not None:
        return env_value
    model_config = runtime_config.get("model")
    if isinstance(model_config, dict):
        return _normalize_optional_url(model_config.get("baseUrl"))
    return None


def _resolve_model_api_key(runtime_config: dict) -> str | None:
    env_value = _normalize_optional_value(os.getenv("OPENAI_API_KEY"))
    if env_value is not None:
        return env_value
    model_config = runtime_config.get("model")
    if isinstance(model_config, dict):
        return _normalize_optional_value(model_config.get("apiKey"))
    return None


def _resolve_model_name(runtime_config: dict) -> str | None:
    env_value = _normalize_optional_value(os.getenv("OPENAI_MODEL"))
    if env_value is not None:
        return env_value
    return _resolve_configured_model_name(runtime_config)


def _config_loaded_at(config_path: Path) -> str:
    try:
        timestamp = config_path.stat().st_mtime
    except OSError:
        timestamp = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))


def _safe_config_hash(runtime_config: dict) -> str:
    redacted = json.loads(json.dumps(runtime_config))
    model = redacted.get("model")
    if isinstance(model, dict) and model.get("apiKey"):
        model["apiKey"] = f"present:{hashlib.sha256(str(model['apiKey']).encode('utf-8')).hexdigest()[:12]}"
    for source in redacted.get("repositorySources") or []:
        if isinstance(source, dict) and source.get("token"):
            source["token"] = f"present:{hashlib.sha256(str(source['token']).encode('utf-8')).hexdigest()[:12]}"
    provider = redacted.get("gitProvider")
    if isinstance(provider, dict) and provider.get("token"):
        provider["token"] = f"present:{hashlib.sha256(str(provider['token']).encode('utf-8')).hexdigest()[:12]}"
    raw = json.dumps(redacted, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _resolve_write_mode(runtime_config: dict) -> str:
    """Return the legacy write-mode equivalent for older code paths."""
    env_value = os.getenv("REPOOPERATOR_WRITE_MODE")
    if env_value and env_value.strip().lower() in _VALID_WRITE_MODES:
        return env_value.strip().lower()

    permission_mode = _resolve_permission_mode(runtime_config)
    if permission_mode == PERMISSION_MODE_FULL_ACCESS:
        return WRITE_MODE_AUTO_APPLY
    if permission_mode in {
        PERMISSION_MODE_BASIC,
        PERMISSION_MODE_DEFAULT,
        PERMISSION_MODE_AUTO_REVIEW,
        PERMISSION_MODE_ACCEPT_EDITS,
        PERMISSION_MODE_PLAN_ONLY,
        PERMISSION_MODE_PROPOSAL_ONLY,
        PERMISSION_MODE_AUTO_READONLY,
        PERMISSION_MODE_ROUTINE_SAFE,
        PERMISSION_MODE_HEADLESS_SAFE,
    }:
        return WRITE_MODE_AUTO_APPLY

    permissions = runtime_config.get("permissions")
    if isinstance(permissions, dict):
        mode = _normalize_optional_value(permissions.get("writeMode"))
        if mode and mode.lower() in _VALID_WRITE_MODES:
            return mode.lower()

    return WRITE_MODE_AUTO_APPLY


def _resolve_permission_mode(runtime_config: dict) -> str:
    env_value = os.getenv("REPOOPERATOR_PERMISSION_MODE")
    if env_value:
        normalized = _normalize_permission_mode(env_value)
        if normalized:
            return normalized

    permissions = runtime_config.get("permissions")
    if isinstance(permissions, dict):
        mode = _normalize_permission_mode(permissions.get("mode"))
        if mode:
            return mode
        legacy = _normalize_optional_value(permissions.get("writeMode"))
        if legacy == WRITE_MODE_WRITE_WITH_APPROVAL:
            return PERMISSION_MODE_AUTO_REVIEW
        if legacy == WRITE_MODE_AUTO_APPLY:
            return PERMISSION_MODE_FULL_ACCESS
        if legacy == WRITE_MODE_READ_ONLY:
            return PERMISSION_MODE_BASIC

    return PERMISSION_MODE_BASIC


def _normalize_permission_mode(value: str | None) -> str | None:
    normalized = _normalize_optional_value(value)
    if normalized is None:
        return None
    lowered = normalized.strip().lower().replace("-", "_")
    aliases = {
        "basic": PERMISSION_MODE_BASIC,
        "default": PERMISSION_MODE_DEFAULT,
        "plan": PERMISSION_MODE_PLAN_ONLY,
        "plan_only": PERMISSION_MODE_PLAN_ONLY,
        "proposal": PERMISSION_MODE_PROPOSAL_ONLY,
        "proposal_only": PERMISSION_MODE_PROPOSAL_ONLY,
        "accept_edits": PERMISSION_MODE_ACCEPT_EDITS,
        "auto_readonly": PERMISSION_MODE_AUTO_READONLY,
        "auto_read_only": PERMISSION_MODE_AUTO_READONLY,
        "readonly": PERMISSION_MODE_AUTO_READONLY,
        "routine_safe": PERMISSION_MODE_ROUTINE_SAFE,
        "headless_safe": PERMISSION_MODE_HEADLESS_SAFE,
        "auto_review": PERMISSION_MODE_AUTO_REVIEW,
        "autoreview": PERMISSION_MODE_AUTO_REVIEW,
        "full_access": PERMISSION_MODE_FULL_ACCESS,
        "fullaccess": PERMISSION_MODE_FULL_ACCESS,
        WRITE_MODE_READ_ONLY: PERMISSION_MODE_BASIC,
        WRITE_MODE_WRITE_WITH_APPROVAL: PERMISSION_MODE_AUTO_REVIEW,
        WRITE_MODE_AUTO_APPLY: PERMISSION_MODE_FULL_ACCESS,
    }
    return aliases.get(lowered)
