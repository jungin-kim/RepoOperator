import type { Page, Route } from "@playwright/test";

// ── Shared fixture types ──────────────────────────────────────────────────────

export type MockRepo = {
  git_provider: string;
  project_path: string;
  branch: string;
};

export type MockRunOpts = {
  runId: string;
  threadId: string;
  repo: MockRepo;
  status?: "running" | "completed" | "failed" | "cancelled" | "waiting_approval" | "cancelling";
  progressEvents?: MockProgressEvent[];
  assistantDeltaChunks?: string[];
  finalResponse?: string;
};

export type MockProgressEvent = {
  id?: string;
  activity_id?: string;
  event_type?: string;
  visibility?: "user" | "debug" | "internal" | string;
  display?: "primary" | "secondary" | "hidden" | string;
  phase: string;
  label: string;
  status: "running" | "completed";
  sequence: number;
  safe_reasoning_summary?: string;
  current_action?: string;
  observation?: string;
  next_action?: string;
  evidence_needed?: string[];
  uncertainty?: string[];
  safety_note?: string;
  operation?: string;
  action_type?: string;
  tool_name?: string;
  related_search_query?: string;
  files?: string[];
  command?: string | string[];
  aggregate?: Record<string, unknown>;
};

// ── Default fixtures ──────────────────────────────────────────────────────────

export const DEFAULT_REPO: MockRepo = {
  git_provider: "local",
  project_path: "/mock/repo",
  branch: "main",
};

export function buildMockRunRecord(opts: MockRunOpts) {
  return {
    id: opts.runId,
    thread_id: opts.threadId,
    repo: opts.repo.project_path,
    branch: opts.repo.branch,
    task_summary: "mock task",
    status: opts.status ?? "running",
    started_at: new Date().toISOString(),
    completed_at: opts.status && ["completed", "failed", "cancelled"].includes(opts.status) ? new Date().toISOString() : null,
    final_result: opts.finalResponse
      ? buildFinalResult(opts.runId, opts.threadId, opts.finalResponse, opts.progressEvents)
      : null,
    error: null,
  };
}

export function buildFinalResult(runId: string, threadId: string, response: string, progressEvents?: MockProgressEvent[]) {
  return {
    run_id: runId,
    thread_id: threadId,
    response,
    response_type: "assistant_answer",
    stop_reason: "completed",
    loop_iteration: 1,
    files_read: [],
    activity_events: (progressEvents ?? []).map((ev) => ({
      ...ev,
      id: ev.id ?? `${runId}-ev-${ev.sequence}`,
      activity_id: ev.activity_id ?? `act-${ev.sequence}`,
      run_id: runId,
      thread_id: threadId,
      event_type: ev.event_type ?? "progress_delta",
      type: "progress_delta",
      status: "completed",
      timestamp: new Date().toISOString(),
    })),
  };
}

export function buildProgressEvents(runId: string, threadId: string, events: MockProgressEvent[]) {
  return events.map((ev) => ({
    id: ev.id ?? `${runId}-ev-${ev.sequence}`,
    activity_id: ev.activity_id ?? `act-${ev.sequence}`,
    run_id: runId,
    thread_id: threadId,
    event_type: ev.event_type ?? "progress_delta",
    type: "progress_delta",
    visibility: ev.visibility,
    display: ev.display,
    phase: ev.phase,
    label: ev.label,
    status: ev.status,
    sequence: ev.sequence,
    safe_reasoning_summary: ev.safe_reasoning_summary,
    current_action: ev.current_action,
    observation: ev.observation,
    next_action: ev.next_action,
    evidence_needed: ev.evidence_needed,
    uncertainty: ev.uncertainty,
    safety_note: ev.safety_note,
    operation: ev.operation,
    action_type: ev.action_type,
    tool_name: ev.tool_name,
    related_search_query: ev.related_search_query,
    files: ev.files,
    command: ev.command,
    aggregate: ev.aggregate,
    timestamp: new Date().toISOString(),
  }));
}

// ── Route helpers ─────────────────────────────────────────────────────────────

export async function mockHealthConnected(page: Page) {
  await page.route("/api/worker/health", async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        status: "ok",
        configured_repository_source: "local",
        configured_git_provider: "local",
        configured_model_connection_mode: "direct",
        configured_model_provider: "mock",
        configured_model_name: "mock-model",
        permission_mode: "basic",
        recent_projects: [DEFAULT_REPO.project_path],
      }),
    });
  });
}

export async function mockListThreads(page: Page, threads: { id: string; title: string; repo: MockRepo; messages: unknown[] }[]) {
  await page.route("/api/worker/threads", async (route: Route) => {
    if (route.request().method() !== "GET") {
      await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ ok: true }) });
      return;
    }
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        threads: threads.map((t) => ({
          id: t.id,
          title: t.title,
          repo: {
            git_provider: t.repo.git_provider,
            project_path: t.repo.project_path,
            branch: t.repo.branch,
            local_repo_path: t.repo.project_path,
            is_git_repository: true,
          },
          messages: t.messages,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        })),
      }),
    });
  });
}

export async function mockSaveThread(page: Page) {
  await page.route("/api/worker/threads", async (route: Route) => {
    if (route.request().method() === "POST") {
      await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ ok: true }) });
    } else {
      await route.continue();
    }
  });
}

export async function mockOpenRepository(page: Page, repo: MockRepo) {
  await page.route("/api/worker/repo-open", async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        git_provider: repo.git_provider,
        project_path: repo.project_path,
        branch: repo.branch,
        local_repo_path: repo.project_path,
        is_git_repository: true,
      }),
    });
  });
  await page.route("/api/worker/repo-open-plan", async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ open_mode: "local" }),
    });
  });
}

export async function mockGetAgentRun(page: Page, runRecord: ReturnType<typeof buildMockRunRecord>) {
  await page.route(`/api/worker/agent/runs/${runRecord.id}`, async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(runRecord),
    });
  });
}

export async function mockGetAgentRunDynamic(
  page: Page,
  runId: string,
  getRunRecord: () => ReturnType<typeof buildMockRunRecord>,
) {
  await page.route(`/api/worker/agent/runs/${runId}`, async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(getRunRecord()),
    });
  });
}

export async function mockGetAgentRunEvents(page: Page, runId: string, events: unknown[]) {
  await page.route(`/api/worker/agent/runs/${runId}/events*`, async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ events }),
    });
  });
}

export async function mockGetAgentRunEventsDynamic(page: Page, runId: string, getEvents: () => unknown[]) {
  await page.route(`/api/worker/agent/runs/${runId}/events*`, async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ events: getEvents() }),
    });
  });
}

export async function mockGetActiveRuns(page: Page, runs: ReturnType<typeof buildMockRunRecord>[]) {
  await page.route("/api/worker/agent/runs/active*", async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ runs }),
    });
  });
}

export async function mockGetActiveRunsDynamic(page: Page, getRuns: () => ReturnType<typeof buildMockRunRecord>[]) {
  await page.route("/api/worker/agent/runs/active*", async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ runs: getRuns() }),
    });
  });
}

export async function mockDebugRoutes(page: Page, runs: () => ReturnType<typeof buildMockRunRecord>[] = () => []) {
  await page.route("/api/worker/debug/runtime", async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        worker: { status: "ok", service: "mock" },
        model: { provider: "mock", connection_mode: "direct", name: "mock-model" },
        permissions: { mode: "basic", sandbox: {}, approval: {}, tools: {} },
        repository: { source: "local", project_path: DEFAULT_REPO.project_path, branch: DEFAULT_REPO.branch },
        agent: { orchestration_mode: "mock" },
        active_runs: runs(),
        recent_runs: [],
      }),
    });
  });
  await page.route("/api/worker/debug/memory", async (route: Route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ items: [], graph: { nodes: [], edges: [] } }) });
  });
  await page.route("/api/worker/debug/context", async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        model_profile: { provider: "mock", model_name: "mock-model", context_window: 128000, max_output_tokens: 4096, compression_strategy: "balanced" },
        latest_pack: null,
        recent_packs: [],
      }),
    });
  });
  await page.route("/api/worker/debug/skills", async (route: Route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ skills: [] }) });
  });
  await page.route("/api/worker/debug/integrations", async (route: Route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ integrations: [] }) });
  });
  await page.route("/api/worker/tools", async (route: Route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ tools: [], permissions: {} }) });
  });
}

// Emit a mocked SSE stream for streamAgentTask.
export async function mockStreamAgentTask(page: Page, chunks: string[]) {
  await page.route("/api/worker/stream", async (route: Route) => {
    const body = chunks.join("");
    await route.fulfill({
      status: 200,
      headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" },
      body,
    });
  });
}

export function sseEvent(data: object): string {
  return `data: ${JSON.stringify(data)}\n\n`;
}
