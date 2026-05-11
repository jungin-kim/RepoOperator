"use client";

export type PermissionMode = "basic" | "auto_review" | "full_access";
export type LegacyWriteMode = "read-only" | "write-with-approval" | "auto-apply";

export type WorkerHealthPayload = {
  status: string;
  service: string;
  repo_base_dir: string;
  configured_git_provider?: string | null;
  configured_repository_source?: string | null;
  configured_repository_sources?: Array<{
    provider?: string | null;
    baseUrl?: string | null;
    tokenConfigured?: boolean;
    owner?: string;
  }>;
  configured_model_connection_mode?: string | null;
  configured_model_provider?: string | null;
  configured_model_name?: string | null;
  configured_model_base_url?: string | null;
  config_loaded_at?: string | null;
  config_source_path?: string | null;
  config_hash?: string | null;
  write_mode?: LegacyWriteMode;
  permission_mode?: PermissionMode;
  sandbox_scope?: string;
  approval_policy?: Record<string, boolean>;
  tool_permissions?: Record<string, string>;
  recent_projects?: string[];
};

export type PermissionModePayload = {
  mode: PermissionMode;
  write_mode: LegacyWriteMode;
  available_modes: PermissionMode[];
  unsupported_modes?: PermissionMode[];
  sandbox?: Record<string, boolean | string>;
  approval?: Record<string, boolean>;
  tools?: Record<string, string>;
  profile?: Record<string, unknown>;
};

export type ProviderProjectSummary = {
  git_provider: string;
  project_path: string;
  display_name: string;
  default_branch?: string | null;
  source: string;
  is_git_repository?: boolean;
};

export type ProviderProjectsPayload = {
  git_provider: string;
  configured_git_provider?: string | null;
  projects: ProviderProjectSummary[];
  recent_projects: ProviderProjectSummary[];
};

export type RecentProjectsPayload = {
  projects: ProviderProjectSummary[];
};

export type ProviderBranchSummary = {
  name: string;
  is_default: boolean;
};

export type ProviderBranchesPayload = {
  git_provider: string;
  project_path: string;
  default_branch?: string | null;
  branches: ProviderBranchSummary[];
};

export type RepoOpenPayload = {
  project_path: string;
  git_provider: string;
  local_repo_path: string;
  branch?: string | null;
  head_sha?: string | null;
  cloned: boolean;
  is_git_repository: boolean;
  message: string;
};

export type RepoOpenPlanPayload = {
  project_path: string;
  git_provider: string;
  local_repo_path: string;
  local_checkout_exists: boolean;
  open_mode: "clone" | "refresh" | "local" | "unknown";
  message: string;
};

export type FileReadPayload = {
  project_path: string;
  relative_path: string;
  content: string;
  truncated: boolean;
  bytes_read: number;
};

export type LocalBranchSummary = {
  name: string;
  is_current: boolean;
};

export type GitBranchListPayload = {
  project_path: string;
  current_branch: string | null;
  branches: LocalBranchSummary[];
};

export type GitBranchCreatePayload = {
  project_path: string;
  branch: string;
  from_ref: string;
  head_sha: string;
  message: string;
};

export type GitCheckoutPayload = {
  project_path: string;
  branch: string;
  head_sha: string | null;
  message: string;
};

export type AgentProposeFilePayload = {
  project_path: string;
  relative_path: string;
  model: string;
  context_summary: string;
  original_content: string;
  proposed_content: string;
};

export type FileWritePayload = {
  project_path: string;
  relative_path: string;
  bytes_written: number;
  message: string;
};

export type AgentRunPayload = {
  project_path: string;
  git_provider?: string | null;
  active_repository_source?: string | null;
  active_repository_path?: string | null;
  active_branch?: string | null;
  task: string;
  model: string;
  branch?: string | null;
  repo_root_name: string;
  context_summary: string;
  top_level_entries: string[];
  readme_included: boolean;
  diff_included: boolean;
  is_git_repository: boolean;
  files_read: string[];
  response: string;
  // Response metadata
  response_type?: "assistant_answer" | "change_proposal" | "edit_applied" | "permission_required" | "clarification" | "proposal_error" | "command_approval" | "command_result" | "command_denied" | "command_error" | "agent_error";
  proposal_relative_path?: string | null;
  proposal_original_content?: string | null;
  proposal_proposed_content?: string | null;
  proposal_context_summary?: string | null;
  clarification_candidates?: string[];
  selected_target_file?: string | null;
  intent_classification?: string | null;
  graph_path?: string | null;
  agent_flow?: string | null;
  proposal_error_details?: string | null;
  command_approval?: CommandApprovalPayload | null;
  command_result?: CommandResultPayload | null;
  run_id?: string | null;
  skills_used?: string[];
  thread_context_files?: string[];
  thread_context_symbols?: string[];
  context_source?: string | null;
  context_reference_resolver?: string | null;
  resolved_reference_type?: string | null;
  resolved_files?: string[];
  resolved_symbols?: string[];
  reference_confidence?: number | null;
  reference_clarification_needed?: boolean | null;
  validation_status?: string | null;
  commands_planned?: string[];
  commands_run?: string[];
  recommendation_context?: Record<string, unknown> | null;
  recommendation_context_loaded?: boolean;
  selected_recommendation_ids?: string[];
  plan_id?: string | null;
  plan_steps?: string[];
  proposal_validation_status?: string | null;
  retry_count?: number;
  effective_worker_model?: string | null;
  configured_model?: string | null;
  plan_steps_summary?: Array<{
    step_index: number;
    description: string;
    intent: string;
    file?: string | null;
    elapsed_ms?: number | null;
    has_proposal?: boolean;
  }>;
  activity_events?: AgentActivityEvent[];
  edit_archive?: EditArchiveRecord[];
  loop_iteration?: number;
  stop_reason?: string | null;
};

export type AgentActivityEvent = {
  type?: "progress_delta" | string;
  id?: string;
  activity_id?: string | null;
  run_id?: string;
  thread_id?: string | null;
  repo?: string | null;
  branch?: string | null;
  sequence?: number | null;
  timestamp?: string | null;
  persisted?: boolean;
  event_type?: string | null;
  visibility?: "user" | "debug" | "internal" | string | null;
  display?: "primary" | "secondary" | "hidden" | string | null;
  phase?: string;
  label?: string;
  detail?: string;
  detail_delta?: string | null;
  message?: string;
  safe_reasoning_summary?: string | null;
  safe_reasoning_summary_delta?: string | null;
  summary_delta?: string | null;
  evidence_needed?: string[];
  uncertainty?: string[];
  safety_note?: string | null;
  operation?: string | null;
  action_type?: string | null;
  tool_name?: string | null;
  current_action?: string | null;
  observation?: string | null;
  observation_delta?: string | null;
  next_action?: string | null;
  next_action_delta?: string | null;
  related_search_query?: string | null;
  aggregate?: Record<string, unknown> | null;
  status?: "pending" | "running" | "completed" | "failed" | "waiting" | string;
  started_at?: string | null;
  ended_at?: string | null;
  duration_ms?: number | null;
  elapsed_ms?: number | null;
  files?: string[];
  command?: string | string[] | null;
  related_command?: string | string[] | null;
  proposal_id?: string | null;
};

export type EditArchiveRecord = {
  file_path: string;
  status: "proposed" | "applied" | "rejected" | "failed" | string;
  additions: number;
  deletions: number;
  diff?: string | null;
  summary?: string | null;
  timestamp?: string | null;
  proposal_id?: string | null;
  plan_id?: string | null;
  apply_result?: string | null;
  tests?: string[];
};

export type AgentRunRecord = {
  id: string;
  thread_id?: string | null;
  repo?: string | null;
  branch?: string | null;
  task_summary?: string;
  status: "pending" | "running" | "waiting_approval" | "cancelling" | "completed" | "failed" | "cancelled" | "timed_out" | string;
  started_at?: string;
  completed_at?: string | null;
  final_result?: AgentRunPayload | null;
  error?: string | null;
};

export type CommandApprovalPayload = {
  type: "command_approval";
  approval_id: string;
  command: string[];
  display_command: string;
  cwd?: string | null;
  risk: "low" | "medium" | "high" | string;
  read_only: boolean;
  needs_network: boolean;
  touches_outside_repo: boolean;
  needs_approval: boolean;
  blocked?: boolean;
  reason: string;
  pattern?: string;
  options?: string[];
  next_command_approval?: CommandApprovalPayload;
};

export type CommandResultPayload = CommandApprovalPayload & {
  status: string;
  exit_code: number;
  stdout: string;
  stderr: string;
  duration_ms?: number;
};

export type ThreadMessagePayload = {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: string;
  metadata?: AgentRunPayload | Record<string, unknown> | null;
};

export type ThreadRecordPayload = {
  id: string;
  title: string;
  repo: RepoOpenPayload;
  messages: ThreadMessagePayload[];
  created_at: string;
  updated_at: string;
};

export type ThreadListPayload = {
  threads: ThreadRecordPayload[];
};

export class LocalWorkerClientError extends Error {
  status: number;

  constructor(message: string, status = 500) {
    super(message);
    this.name = "LocalWorkerClientError";
    this.status = status;
  }
}

async function parseWorkerResponse<T>(response: Response): Promise<T> {
  const payload = (await response.json()) as T & { detail?: string };

  if (!response.ok) {
    throw new LocalWorkerClientError(
      payload.detail || "The local worker request did not complete successfully.",
      response.status,
    );
  }

  return payload;
}

export async function getWorkerHealth(): Promise<WorkerHealthPayload> {
  const response = await fetch("/api/worker/health", { cache: "no-store" });
  return parseWorkerResponse<WorkerHealthPayload>(response);
}

export async function getPermissionMode(): Promise<PermissionModePayload> {
  const response = await fetch("/api/worker/permissions", { cache: "no-store" });
  return parseWorkerResponse<PermissionModePayload>(response);
}

export async function updatePermissionMode(mode: PermissionMode): Promise<PermissionModePayload> {
  const response = await fetch("/api/worker/permissions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode }),
  });
  return parseWorkerResponse<PermissionModePayload>(response);
}

export async function runApprovedCommand(input: {
  command: string[];
  approval_id?: string;
  remember_for_session?: boolean;
  decision?: string;
}): Promise<CommandResultPayload> {
  const response = await fetch("/api/worker/commands/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      argv: input.command,
      approval_id: input.approval_id,
      remember_for_session: input.remember_for_session,
      decision: input.decision,
    }),
  });
  return parseWorkerResponse<CommandResultPayload>(response);
}

export async function getProviderProjects(input: {
  git_provider: string;
  search?: string;
}): Promise<ProviderProjectsPayload> {
  const query = new URLSearchParams({
    git_provider: input.git_provider,
  });
  if (input.search?.trim()) {
    query.set("search", input.search.trim());
  }

  const response = await fetch(`/api/worker/provider/projects?${query.toString()}`, {
    cache: "no-store",
  });
  return parseWorkerResponse<ProviderProjectsPayload>(response);
}

export async function getRecentProjects(input: { limit?: number } = {}): Promise<RecentProjectsPayload> {
  const query = new URLSearchParams();
  if (input.limit) {
    query.set("limit", String(input.limit));
  }

  const suffix = query.toString() ? `?${query.toString()}` : "";
  const response = await fetch(`/api/worker/provider/recent-projects${suffix}`, {
    cache: "no-store",
  });
  return parseWorkerResponse<RecentProjectsPayload>(response);
}

export async function getProviderBranches(input: {
  git_provider: string;
  project_path: string;
}): Promise<ProviderBranchesPayload> {
  const query = new URLSearchParams({
    git_provider: input.git_provider,
    project_path: input.project_path,
  });

  const response = await fetch(`/api/worker/provider/branches?${query.toString()}`, {
    cache: "no-store",
  });
  return parseWorkerResponse<ProviderBranchesPayload>(response);
}

export async function openRepository(input: {
  project_path: string;
  branch?: string;
  git_provider?: string;
  client_request_id?: string;
}): Promise<RepoOpenPayload> {
  const response = await fetch("/api/worker/repo-open", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });

  return parseWorkerResponse<RepoOpenPayload>(response);
}

export async function getRepositoryOpenPlan(input: {
  project_path: string;
  branch?: string;
  git_provider?: string;
  client_request_id?: string;
}): Promise<RepoOpenPlanPayload> {
  const response = await fetch("/api/worker/repo-open-plan", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });

  return parseWorkerResponse<RepoOpenPlanPayload>(response);
}

export type ConversationMessage = {
  role: "user" | "assistant" | "system";
  content: string;
  metadata?: Record<string, unknown> | AgentRunPayload | null;
};

export async function runAgentTask(input: {
  project_path: string;
  git_provider?: string;
  branch?: string;
  task: string;
  conversation_history?: ConversationMessage[];
}): Promise<AgentRunPayload> {
  const response = await fetch("/api/worker/agent-run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });

  return parseWorkerResponse<AgentRunPayload>(response);
}

type NonPublicModelDeltaType = `${"reasoning"}_${"delta"}`;

export type AgentProgressEvent =
  | { type: "progress"; node: string; message: string }
  | (AgentActivityEvent & { type: "progress_delta" })
  | { type: "assistant_delta"; delta: string }
  | { type: NonPublicModelDeltaType; delta: string; source?: string }
  | { type: "command_delta"; delta?: string; message?: string }
  | { type: "done"; result: AgentRunPayload }
  | { type: "final_message"; result: AgentRunPayload }
  | { type: "error"; message: string };

export async function* streamAgentTask(input: {
  project_path: string;
  git_provider?: string;
  branch?: string;
  thread_id?: string;
  task: string;
  conversation_history?: ConversationMessage[];
}): AsyncGenerator<AgentProgressEvent> {
  const response = await fetch("/api/worker/agent/run/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new LocalWorkerClientError(text || "Stream request failed.", response.status);
  }

  if (!response.body) {
    throw new LocalWorkerClientError("No response body for stream.", 500);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const dataStr = line.slice(6).trim();
        if (dataStr === "[DONE]") return;
        try {
          yield JSON.parse(dataStr) as AgentProgressEvent;
        } catch {
          // skip malformed events
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

export async function getAgentRun(runId: string): Promise<AgentRunRecord> {
  const response = await fetch(`/api/worker/agent/runs/${encodeURIComponent(runId)}`, { cache: "no-store" });
  return parseWorkerResponse<AgentRunRecord>(response);
}

export async function getActiveAgentRuns(threadId?: string): Promise<{ runs: AgentRunRecord[] }> {
  const suffix = threadId ? `?thread_id=${encodeURIComponent(threadId)}` : "";
  const response = await fetch(`/api/worker/agent/runs/active${suffix}`, { cache: "no-store" });
  return parseWorkerResponse<{ runs: AgentRunRecord[] }>(response);
}

export async function getAgentRunEvents(runId: string, afterSequence = 0): Promise<{ events: AgentActivityEvent[] }> {
  const response = await fetch(`/api/worker/agent/runs/${encodeURIComponent(runId)}/events?after_sequence=${afterSequence}`, { cache: "no-store" });
  return parseWorkerResponse<{ events: AgentActivityEvent[] }>(response);
}

export async function steerAgentRun(runId: string, content: string): Promise<{ status: string }> {
  const response = await fetch(`/api/worker/agent/runs/${encodeURIComponent(runId)}/steer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content }),
  });
  return parseWorkerResponse<{ status: string }>(response);
}

export async function cancelAgentRun(runId: string): Promise<{ status: string }> {
  const response = await fetch(`/api/worker/agent/runs/${encodeURIComponent(runId)}/cancel`, {
    method: "POST",
  });
  return parseWorkerResponse<{ status: string }>(response);
}

export async function readRepositoryFile(input: {
  project_path: string;
  relative_path: string;
}): Promise<FileReadPayload> {
  const response = await fetch("/api/worker/fs-read", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });

  return parseWorkerResponse<FileReadPayload>(response);
}

export async function listLocalBranches(input: {
  project_path: string;
}): Promise<GitBranchListPayload> {
  const query = new URLSearchParams({ project_path: input.project_path });
  const response = await fetch(`/api/worker/git-branches?${query.toString()}`, {
    cache: "no-store",
  });
  return parseWorkerResponse<GitBranchListPayload>(response);
}

export async function createLocalBranch(input: {
  project_path: string;
  branch: string;
  from_ref?: string;
  checkout?: boolean;
}): Promise<GitBranchCreatePayload> {
  const response = await fetch("/api/worker/git-branch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      project_path: input.project_path,
      branch: input.branch,
      from_ref: input.from_ref ?? "HEAD",
      checkout: input.checkout ?? true,
    }),
  });
  return parseWorkerResponse<GitBranchCreatePayload>(response);
}

export async function checkoutLocalBranch(input: {
  project_path: string;
  branch: string;
}): Promise<GitCheckoutPayload> {
  const response = await fetch("/api/worker/git-checkout", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  return parseWorkerResponse<GitCheckoutPayload>(response);
}

export async function proposeFileEdit(input: {
  project_path: string;
  relative_path: string;
  instruction: string;
}): Promise<AgentProposeFilePayload> {
  const response = await fetch("/api/worker/propose-file", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  return parseWorkerResponse<AgentProposeFilePayload>(response);
}

export async function writeRepositoryFile(input: {
  project_path: string;
  relative_path: string;
  content: string;
}): Promise<FileWritePayload> {
  const response = await fetch("/api/worker/fs-write", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  return parseWorkerResponse<FileWritePayload>(response);
}

export async function generateApplySummary(input: {
  project_path: string;
  branch?: string | null;
  relative_path: string;
  user_request?: string;
  proposal_summary?: string;
  diff_summary?: string;
}): Promise<{ response: string; response_type: string; relative_path: string }> {
  const response = await fetch("/api/worker/agent/apply-summary", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  return parseWorkerResponse<{ response: string; response_type: string; relative_path: string }>(response);
}

export async function listThreads(): Promise<ThreadListPayload> {
  const response = await fetch("/api/worker/threads", { cache: "no-store" });
  return parseWorkerResponse<ThreadListPayload>(response);
}

export async function saveThread(input: ThreadRecordPayload): Promise<ThreadRecordPayload> {
  const response = await fetch("/api/worker/threads", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });

  return parseWorkerResponse<ThreadRecordPayload>(response);
}
