from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    repo_base_dir: str
    configured_git_provider: str | None = None
    configured_repository_source: str | None = None
    configured_repository_sources: list[dict] = []
    configured_model_connection_mode: str | None = None
    configured_model_provider: str | None = None
    configured_model_name: str | None = None
    configured_model_base_url: str | None = None
    config_loaded_at: str | None = None
    config_source_path: str | None = None
    config_hash: str | None = None
    write_mode: str = "read-only"
    permission_mode: str = "basic"
    sandbox_scope: str = "repository"
    approval_policy: dict = {}
    tool_permissions: dict = {}
    recent_projects: list[str] = []


class PermissionModeResponse(BaseModel):
    mode: str = "basic"
    write_mode: str
    available_modes: list[str]
    unsupported_modes: list[str] = []
    sandbox: dict = {}
    approval: dict = {}
    tools: dict = {}
    profile: dict = {}


class ProviderProjectSummary(BaseModel):
    git_provider: str
    project_path: str
    display_name: str
    default_branch: str | None = None
    source: str
    is_git_repository: bool = True


class ProviderProjectsResponse(BaseModel):
    git_provider: str
    configured_git_provider: str | None = None
    projects: list[ProviderProjectSummary]
    recent_projects: list[ProviderProjectSummary]


class RecentProjectsResponse(BaseModel):
    projects: list[ProviderProjectSummary]


class ProviderBranchSummary(BaseModel):
    name: str
    is_default: bool = False


class ProviderBranchesResponse(BaseModel):
    git_provider: str
    project_path: str
    default_branch: str | None = None
    branches: list[ProviderBranchSummary]


class ThreadRepositorySummary(BaseModel):
    project_path: str
    git_provider: str
    local_repo_path: str
    branch: str | None = None
    head_sha: str | None = None
    cloned: bool = False
    is_git_repository: bool = True
    message: str = ""


class ThreadMessageSummary(BaseModel):
    id: str
    role: str
    content: str
    timestamp: str
    metadata: dict | None = None


class ThreadSummary(BaseModel):
    id: str
    title: str
    repo: ThreadRepositorySummary
    messages: list[ThreadMessageSummary]
    created_at: str
    updated_at: str


class ThreadListResponse(BaseModel):
    threads: list[ThreadSummary]


class RepoOpenResponse(BaseModel):
    project_path: str
    git_provider: str
    local_repo_path: str
    branch: str | None = None
    head_sha: str | None = None
    cloned: bool
    is_git_repository: bool
    message: str


class RepoOpenPlanResponse(BaseModel):
    project_path: str
    git_provider: str
    local_repo_path: str
    local_checkout_exists: bool
    open_mode: str
    message: str


class FileReadResponse(BaseModel):
    project_path: str
    relative_path: str
    content: str
    truncated: bool
    bytes_read: int


class FileWriteResponse(BaseModel):
    project_path: str
    relative_path: str
    bytes_written: int
    message: str


class CommandRunResponse(BaseModel):
    project_path: str
    command: str
    timeout_seconds: int
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool


class GitDiffResponse(BaseModel):
    project_path: str
    diff: str


class GitBranchCreateResponse(BaseModel):
    project_path: str
    branch: str
    from_ref: str
    head_sha: str
    message: str


class GitCommitResponse(BaseModel):
    project_path: str
    branch: str
    commit_sha: str
    message: str


class GitPushResponse(BaseModel):
    project_path: str
    remote: str
    branch: str
    message: str


class GitMergeRequestCreateResponse(BaseModel):
    project_path: str
    git_provider: str
    title: str
    web_url: str
    iid: str
    state: str


class AgentRunResponse(BaseModel):
    project_path: str
    git_provider: str | None = None
    active_repository_source: str | None = None
    active_repository_path: str | None = None
    active_branch: str | None = None
    task: str
    model: str
    branch: str | None = None
    repo_root_name: str
    context_summary: str
    top_level_entries: list[str]
    readme_included: bool
    diff_included: bool
    is_git_repository: bool
    files_read: list[str] = []
    response: str
    # Write-intent routing fields (populated when response_type != "assistant_answer")
    response_type: str = "assistant_answer"
    proposal_relative_path: str | None = None
    proposal_original_content: str | None = None
    proposal_proposed_content: str | None = None
    proposal_context_summary: str | None = None
    clarification_candidates: list[str] = []
    selected_target_file: str | None = None
    intent_classification: str | None = None
    graph_path: str | None = None
    agent_flow: str = "langgraph"
    proposal_error_details: str | None = None
    command_approval: dict | None = None
    command_result: dict | None = None
    commands_planned: list[str] = []
    commands_run: list[str] = []
    recommendation_context: dict | None = None
    recommendation_context_loaded: bool = False
    selected_recommendation_ids: list[str] = []
    plan_id: str | None = None
    plan_steps: list[str] = []
    proposal_validation_status: str | None = None
    retry_count: int = 0
    effective_worker_model: str | None = None
    configured_model: str | None = None
    run_id: str | None = None
    skills_used: list[str] = []
    thread_context_files: list[str] = []
    thread_context_symbols: list[str] = []
    context_source: str | None = None
    context_reference_resolver: str | None = None
    resolved_reference_type: str | None = None
    resolved_files: list[str] = []
    resolved_symbols: list[str] = []
    reference_confidence: float | None = None
    reference_clarification_needed: bool | None = None
    validation_status: str | None = None
    plan_steps_summary: list[dict] = []
    activity_events: list[dict] = []
    edit_archive: list[dict] = []
    change_set_proposal: dict | None = None
    loop_iteration: int = 0
    stop_reason: str | None = None


class LocalBranchSummary(BaseModel):
    name: str
    is_current: bool = False


class GitBranchListResponse(BaseModel):
    project_path: str
    current_branch: str | None
    branches: list[LocalBranchSummary]


class GitCheckoutResponse(BaseModel):
    project_path: str
    branch: str
    head_sha: str | None
    message: str


class AgentProposeFileResponse(BaseModel):
    project_path: str
    relative_path: str
    model: str
    context_summary: str
    original_content: str
    proposed_content: str
