from pathlib import Path

from pydantic import BaseModel, Field, field_validator


SUPPORTED_REPOSITORY_PROVIDERS = {"gitlab", "github", "local"}


def _normalize_project_path(value: str) -> str:
    path = Path(value)
    if not value.strip():
        raise ValueError("project_path must not be empty")
    if path.is_absolute():
        return str(path)
    if ".." in path.parts:
        raise ValueError("project_path must not escape its configured base")
    return value.strip("/")


class GitProviderMetadata(BaseModel):
    provider: str | None = Field(default=None, description="Git provider identifier.")
    clone_url: str | None = Field(
        default=None,
        description="Clone URL used when the repository is missing locally.",
    )
    default_branch: str | None = Field(
        default=None,
        description="Provider-reported default branch if known.",
    )


class ThreadRepositorySnapshot(BaseModel):
    project_path: str
    git_provider: str
    local_repo_path: str
    branch: str | None = None
    head_sha: str | None = None
    cloned: bool = False
    is_git_repository: bool = True
    message: str = ""

    @field_validator("git_provider")
    @classmethod
    def validate_repository_provider(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in SUPPORTED_REPOSITORY_PROVIDERS:
            raise ValueError("git_provider must be one of: gitlab, github, local")
        return normalized


class ThreadMessagePayload(BaseModel):
    id: str
    role: str
    content: str
    timestamp: str
    metadata: dict | None = None

    @field_validator("id", "content", "timestamp")
    @classmethod
    def validate_required_thread_strings(cls, value: str, info) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name} must not be empty")
        return value

    @field_validator("role")
    @classmethod
    def validate_thread_role(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"user", "assistant", "system"}:
            raise ValueError("role must be one of: user, assistant, system")
        return normalized


class ThreadUpsertRequest(BaseModel):
    id: str
    title: str
    repo: ThreadRepositorySnapshot
    messages: list[ThreadMessagePayload]
    created_at: str
    updated_at: str

    @field_validator("id", "title", "created_at", "updated_at")
    @classmethod
    def validate_thread_strings(cls, value: str, info) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name} must not be empty")
        return value


class RepoOpenRequest(BaseModel):
    project_path: str = Field(
        ...,
        description=(
            "Repository identifier. Use a provider path like 'group/project' for "
            "gitlab or 'owner/repo' for github, or an absolute filesystem path for local projects."
        ),
    )
    branch: str | None = Field(
        default=None,
        description="Branch to fetch and check out for provider-backed repositories or local git repositories.",
    )
    git_provider: str | None = Field(
        default=None,
        description="Repository source identifier such as 'gitlab', 'github', or 'local'.",
    )
    git: GitProviderMetadata | None = Field(
        default=None,
        description="Optional git provider metadata used for clone and future provider integration.",
    )
    client_request_id: str | None = Field(
        default=None,
        description="Client-generated repository-open operation id used to ignore stale concurrent opens.",
    )

    @field_validator("project_path")
    @classmethod
    def validate_project_path(cls, value: str) -> str:
        return _normalize_project_path(value)

    @field_validator("branch")
    @classmethod
    def validate_branch(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("git_provider")
    @classmethod
    def validate_git_provider(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if not normalized:
            return None
        if normalized not in SUPPORTED_REPOSITORY_PROVIDERS:
            raise ValueError("git_provider must be one of: gitlab, github, local")
        return normalized

    @field_validator("client_request_id")
    @classmethod
    def validate_client_request_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class FileReadRequest(BaseModel):
    project_path: str
    relative_path: str
    encoding: str = "utf-8"
    max_bytes: int = Field(default=100_000, ge=1, le=2_000_000)

    @field_validator("project_path", "relative_path")
    @classmethod
    def validate_relative_values(cls, value: str, info) -> str:
        if info.field_name == "project_path":
            return _normalize_project_path(value)
        path = Path(value)
        if not value.strip():
            raise ValueError("relative_path must not be empty")
        if path.is_absolute():
            raise ValueError("relative_path must be relative")
        if ".." in path.parts:
            raise ValueError("relative_path must not escape its base directory")
        return value.strip("/")


class FileWriteRequest(BaseModel):
    project_path: str
    relative_path: str
    content: str
    encoding: str = "utf-8"

    @field_validator("project_path", "relative_path")
    @classmethod
    def validate_relative_values(cls, value: str, info) -> str:
        if info.field_name == "project_path":
            return _normalize_project_path(value)
        path = Path(value)
        if not value.strip():
            raise ValueError("relative_path must not be empty")
        if path.is_absolute():
            raise ValueError("relative_path must be relative")
        if ".." in path.parts:
            raise ValueError("relative_path must not escape its base directory")
        return value.strip("/")


class CommandRunRequest(BaseModel):
    project_path: str
    command: str = Field(..., description="Command parsed into argv and executed through RepoOperator command policy.")
    timeout_seconds: int | None = Field(default=None, ge=1, le=600)
    approval_id: str | None = None
    remember_for_session: bool = False

    @field_validator("project_path")
    @classmethod
    def validate_project_path(cls, value: str) -> str:
        return _normalize_project_path(value)

    @field_validator("command")
    @classmethod
    def validate_command(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("command must not be empty")
        return value


class GitDiffRequest(BaseModel):
    project_path: str
    staged: bool = False
    relative_paths: list[str] | None = None

    @field_validator("project_path")
    @classmethod
    def validate_project_path(cls, value: str) -> str:
        return _normalize_project_path(value)

    @field_validator("relative_paths")
    @classmethod
    def validate_relative_paths(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        cleaned: list[str] = []
        for item in value:
            path = Path(item)
            if not item.strip():
                raise ValueError("relative_paths items must not be empty")
            if path.is_absolute():
                raise ValueError("relative_paths items must be relative")
            if ".." in path.parts:
                raise ValueError("relative_paths items must not escape the repo base directory")
            cleaned.append(item.strip("/"))
        return cleaned


class GitBranchCreateRequest(BaseModel):
    project_path: str
    branch: str
    from_ref: str = "HEAD"
    checkout: bool = True
    approval_id: str | None = None
    remember_for_session: bool = False

    @field_validator("project_path", "branch", "from_ref")
    @classmethod
    def validate_values(cls, value: str, info) -> str:
        if info.field_name == "project_path":
            return _normalize_project_path(value)
        if not value.strip():
            raise ValueError(f"{info.field_name} must not be empty")
        return value.strip()


class GitCommitRequest(BaseModel):
    project_path: str
    message: str
    stage_all: bool = True
    approval_id: str | None = None
    stage_approval_id: str | None = None
    remember_for_session: bool = False

    @field_validator("project_path")
    @classmethod
    def validate_project_path_for_commit(cls, value: str) -> str:
        return _normalize_project_path(value)

    @field_validator("message")
    @classmethod
    def validate_commit_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be empty")
        return value.strip()


class GitPushRequest(BaseModel):
    project_path: str
    branch: str
    remote: str = "origin"
    set_upstream: bool = True
    git_provider: str | None = None
    approval_id: str | None = None
    remember_for_session: bool = False

    @field_validator("project_path", "branch", "remote")
    @classmethod
    def validate_push_values(cls, value: str, info) -> str:
        if info.field_name == "project_path":
            return _normalize_project_path(value)
        if not value.strip():
            raise ValueError(f"{info.field_name} must not be empty")
        return value.strip()

    @field_validator("git_provider")
    @classmethod
    def validate_push_provider(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if not normalized:
            return None
        if normalized not in {"gitlab", "github"}:
            raise ValueError("git_provider must be one of: gitlab, github")
        return normalized


class GitMergeRequestCreateRequest(BaseModel):
    project_path: str
    git_provider: str
    source_branch: str
    target_branch: str
    title: str
    description: str | None = None
    approval_id: str | None = None
    remember_for_session: bool = False

    @field_validator("project_path")
    @classmethod
    def validate_project_path_for_mr(cls, value: str) -> str:
        return _normalize_project_path(value)

    @field_validator("git_provider")
    @classmethod
    def validate_mr_provider(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"gitlab"}:
            raise ValueError("git_provider must be one of: gitlab")
        return normalized

    @field_validator("source_branch", "target_branch", "title")
    @classmethod
    def validate_required_strings(cls, value: str, info) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name} must not be empty")
        return value.strip()

    @field_validator("description")
    @classmethod
    def normalize_description(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class ConversationMessage(BaseModel):
    role: str
    content: str
    metadata: dict | None = None

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"user", "assistant", "system"}:
            raise ValueError("role must be one of: user, assistant, system")
        return normalized

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must not be empty")
        return value


class AgentRunRequest(BaseModel):
    project_path: str = Field(
        ...,
        description="Repository identifier for the opened project. Local projects use an absolute filesystem path.",
    )
    task: str = Field(..., description="User task sent to the centralized model backend.")
    git_provider: str | None = Field(
        default=None,
        description="Repository source identifier for the opened project.",
    )
    branch: str | None = Field(
        default=None,
        description="Branch that was active when the repository was opened.",
    )
    thread_id: str | None = Field(
        default=None,
        description="Client thread id used to scope active run state and progress.",
    )
    conversation_history: list[ConversationMessage] = Field(
        default_factory=list,
        description="Recent conversation turns (user + assistant) for write confirmation context.",
    )

    @field_validator("project_path")
    @classmethod
    def validate_project_path(cls, value: str) -> str:
        return _normalize_project_path(value)

    @field_validator("task")
    @classmethod
    def validate_task(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("task must not be empty")
        return value.strip()

    @field_validator("git_provider")
    @classmethod
    def validate_git_provider(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if not normalized:
            return None
        if normalized not in SUPPORTED_REPOSITORY_PROVIDERS:
            raise ValueError("git_provider must be one of: gitlab, github, local")
        return normalized

    @field_validator("branch")
    @classmethod
    def validate_branch(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("thread_id")
    @classmethod
    def validate_thread_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class IDEContextUpdateRequest(BaseModel):
    active_file: str | None = None
    selected_text: str | None = None
    open_files: list[str] = Field(default_factory=list)
    diagnostics: list[dict] = Field(default_factory=list)
    cursor_position: dict | None = None
    workspace_root: str
    branch: str | None = None
    editor: str | None = None
    timestamp: float | str | None = None

    @field_validator("active_file", "selected_text", "workspace_root", "branch", "editor")
    @classmethod
    def normalize_optional_strings(cls, value: str | None, info) -> str | None:
        if value is None:
            if info.field_name == "workspace_root":
                raise ValueError("workspace_root must not be empty")
            return None
        stripped = value.strip()
        if not stripped and info.field_name == "workspace_root":
            raise ValueError("workspace_root must not be empty")
        return stripped or None


class GitBranchListRequest(BaseModel):
    project_path: str

    @field_validator("project_path")
    @classmethod
    def validate_project_path(cls, value: str) -> str:
        return _normalize_project_path(value)


class PermissionModeRequest(BaseModel):
    mode: str | None = None
    write_mode: str | None = None

    @field_validator("mode", "write_mode")
    @classmethod
    def validate_write_mode(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        allowed = {
            "default",
            "plan",
            "plan_only",
            "plan-only",
            "proposal",
            "proposal_only",
            "proposal-only",
            "accept_edits",
            "accept-edits",
            "auto_readonly",
            "auto-readonly",
            "routine_safe",
            "routine-safe",
            "headless_safe",
            "headless-safe",
            "basic",
            "auto_review",
            "auto-review",
            "full_access",
            "full-access",
            "read-only",
            "write-with-approval",
            "auto-apply",
        }
        if normalized not in allowed:
            raise ValueError(
                "permission mode must be one of: default, plan_only, proposal_only, accept_edits, auto_readonly, full_access, routine_safe, headless_safe"
            )
        return normalized


class GitCheckoutRequest(BaseModel):
    project_path: str
    branch: str
    approval_id: str | None = None
    remember_for_session: bool = False

    @field_validator("project_path", "branch")
    @classmethod
    def validate_checkout_values(cls, value: str, info) -> str:
        if info.field_name == "project_path":
            return _normalize_project_path(value)
        if not value.strip():
            raise ValueError("branch must not be empty")
        return value.strip()


class AgentProposeFileRequest(BaseModel):
    project_path: str = Field(
        ...,
        description="Repository identifier for the opened project. Local projects use an absolute filesystem path.",
    )
    relative_path: str = Field(
        ...,
        description="Target file path relative to the repository root.",
    )
    instruction: str = Field(
        ...,
        description="Requested change instruction for the target file.",
    )

    @field_validator("project_path", "relative_path")
    @classmethod
    def validate_relative_values(cls, value: str, info) -> str:
        if info.field_name == "project_path":
            return _normalize_project_path(value)
        path = Path(value)
        if not value.strip():
            raise ValueError("relative_path must not be empty")
        if path.is_absolute():
            raise ValueError("relative_path must be relative")
        if ".." in path.parts:
            raise ValueError("relative_path must not escape its base directory")
        return value.strip("/")

    @field_validator("instruction")
    @classmethod
    def validate_instruction(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("instruction must not be empty")
        return value.strip()
