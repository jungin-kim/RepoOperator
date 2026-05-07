/**
 * E2E tests for chat thread / run rehydration.
 *
 * These tests intercept all local-worker API routes so no real worker process is needed.
 * Run with:  npm --prefix apps/web run test:e2e
 *
 * If Playwright browsers are not installed:
 *   npx playwright install --with-deps chromium
 * then re-run the test command.
 *
 * CI note: set E2E_BASE_URL to an already-running server to skip the built-in webServer start.
 */
import { test, expect, type Page } from "@playwright/test";
import {
  DEFAULT_REPO,
  buildMockRunRecord,
  buildProgressEvents,
  buildFinalResult,
  mockHealthConnected,
  mockListThreads,
  mockSaveThread,
  mockOpenRepository,
  mockGetAgentRun,
  mockGetAgentRunDynamic,
  mockGetAgentRunEvents,
  mockGetAgentRunEventsDynamic,
  mockGetActiveRuns,
  mockGetActiveRunsDynamic,
  mockDebugRoutes,
} from "./fixtures/mock-worker";

// ── Helpers ───────────────────────────────────────────────────────────────────

const RUN_ID = "run_test_001";
const THREAD_ID = "thread_test_001";
const USER_MSG = "이 레포가 뭐 하는 프로젝트인지 알아내줘.";
const FINAL_RESPONSE = "This repository is a local-first coding agent proxy.";
const WORK_NOTE = "I’m inspecting the repository structure to find the best entry evidence.";

const PROGRESS_EVENTS = [
  { phase: "Thinking", label: "Loaded context", status: "completed" as const, sequence: 1, visibility: "debug", display: "secondary" },
  { phase: "Planning", label: "Framed request", status: "completed" as const, sequence: 2, visibility: "debug", display: "secondary" },
  {
    phase: "Planning",
    label: "Mapped restore plan",
    status: "completed" as const,
    sequence: 3,
    activity_id: "summary-note-1",
    event_type: "work_trace",
    visibility: "user",
    display: "primary",
    safe_reasoning_summary: WORK_NOTE,
  },
  {
    phase: "Searching",
    label: "Searching file contents",
    status: "completed" as const,
    sequence: 4,
    activity_id: "activity-search-1",
    event_type: "work_trace",
    visibility: "user",
    display: "primary",
    related_search_query: "repooperator-active-thread",
    aggregate: { action_type: "search_text", query: "repooperator-active-thread" },
  },
  {
    phase: "Reading files",
    label: "Reading repository files",
    status: "running" as const,
    sequence: 5,
    activity_id: "activity-read-1",
    event_type: "work_trace",
    visibility: "user",
    display: "primary",
    files: ["apps/web/src/components/chat/ChatApp.tsx"],
    aggregate: { action_type: "read_file" },
  },
];

function repoIdentityKey(repo = DEFAULT_REPO) {
  const provider = encodeURIComponent(repo.git_provider || "local");
  const path = encodeURIComponent((repo.project_path || "unknown").replace(/\\/g, "/").replace(/\/+$/, ""));
  const branch = encodeURIComponent(repo.branch || "default");
  return `${provider}:${path}:${branch}`;
}

function activeThreadKey(repo = DEFAULT_REPO) {
  return `repooperator-active-thread:${repoIdentityKey(repo)}`;
}

function buildThread(overrides: { messages?: unknown[] } = {}) {
  return {
    id: THREAD_ID,
    title: "mock/repo",
    repo: DEFAULT_REPO,
    messages: overrides.messages ?? [
      {
        id: "msg-user-1",
        role: "user",
        content: USER_MSG,
        timestamp: new Date().toISOString(),
      },
    ],
  };
}

async function setupBaseRoutes(page: Page) {
  await mockHealthConnected(page);
  await mockSaveThread(page);

  // Provider / branch endpoints — return empty lists for local provider
  await page.route("/api/worker/provider/projects*", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ projects: [], recent_projects: [] }) }),
  );
  await page.route("/api/worker/provider/branches*", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ branches: [] }) }),
  );
  await page.route("/api/worker/provider/recent-projects*", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ projects: [] }) }),
  );
  // Local branches
  await page.route("/api/worker/git-branches*", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ branches: [{ name: "main", is_current: true }], current_branch: "main" }),
    }),
  );
}

async function setStorageForThread(page: Page, threadId: string, runId?: string, repoKey?: string) {
  await page.addInitScript(
    ({ threadId, runId, repoKey }) => {
      const key = repoKey || "repooperator-active-thread:local:%2Fmock%2Frepo:main";
      const identity = key.replace("repooperator-active-thread:", "");
      localStorage.setItem(key, threadId);
      localStorage.setItem("repooperator-active-repo-identity", identity);
      if (runId) {
        localStorage.setItem(`repooperator-active-run-id:${threadId}`, runId);
      }
    },
    { threadId, runId, repoKey: repoKey || activeThreadKey() },
  );
}

async function revealCompletedWorkLog(page: Page) {
  const showWorkLog = page.getByRole("button", { name: /Show work log/i });
  if (await showWorkLog.isVisible().catch(() => false)) {
    await showWorkLog.click();
  }
}

// ── Scenario A: Active thread survives navigation during active run ────────────

test("A: active thread survives navigation away and back during active run", async ({ page }) => {
  const runRecord = buildMockRunRecord({
    runId: RUN_ID,
    threadId: THREAD_ID,
    repo: DEFAULT_REPO,
    status: "running",
    progressEvents: PROGRESS_EVENTS,
  });

  const events = buildProgressEvents(RUN_ID, THREAD_ID, PROGRESS_EVENTS);

  await setupBaseRoutes(page);
  await mockListThreads(page, [buildThread()]);
  await mockOpenRepository(page, DEFAULT_REPO);
  await mockGetActiveRuns(page, [runRecord]);
  await mockGetAgentRun(page, runRecord);
  await mockGetAgentRunEvents(page, RUN_ID, events);
  await mockDebugRoutes(page, () => [runRecord]);

  await setStorageForThread(page, THREAD_ID, RUN_ID);
  await page.goto("/app");

  await expect(page.getByText(USER_MSG)).toBeVisible({ timeout: 5000 });
  await expect(page.getByText(WORK_NOTE)).toBeVisible({ timeout: 5000 });
  await expect(page.locator(".agent-transcript-section")).toHaveCount(1);
  await expect(page.getByTestId("stop-run-button")).toBeVisible();

  // Navigate away to the real debug route, which unmounts ChatApp, then return.
  await page.goto("/debug");
  await expect(page.locator(".debug-card:has-text('Worker') .debug-row:has-text('Active runs') strong")).toHaveText("1");
  await page.getByRole("link", { name: "Back to app" }).click();

  // The thread must still be selected (user message visible)
  await expect(page.getByText(USER_MSG)).toBeVisible({ timeout: 5000 });
  await expect(page.locator(".sidebar-thread.sidebar-item-active")).toContainText("mock/repo");

  await expect(page.getByText(WORK_NOTE)).toBeVisible({ timeout: 5000 });
  await expect(page.locator(".agent-transcript-section")).toHaveCount(1);
  await expect(page.getByTestId("stop-run-button")).toBeVisible();
  await expect(page.getByText("Loaded context")).toHaveCount(0);
  await expect(page.getByText("Framed request")).toHaveCount(0);

  // No duplicate assistant messages for this run
  const assistantMessages = page.locator('[data-testid="assistant-message"]');
  const count = await assistantMessages.count();
  expect(count).toBeLessThanOrEqual(1);
});

test("B: active run completes while user is on debug page and rehydrates on return", async ({ page }) => {
  const runningRun = buildMockRunRecord({
    runId: RUN_ID,
    threadId: THREAD_ID,
    repo: DEFAULT_REPO,
    status: "running",
    progressEvents: PROGRESS_EVENTS,
  });
  const finalResult = buildFinalResult(RUN_ID, THREAD_ID, FINAL_RESPONSE, PROGRESS_EVENTS);
  const completedRun = buildMockRunRecord({
    runId: RUN_ID,
    threadId: THREAD_ID,
    repo: DEFAULT_REPO,
    status: "completed",
    finalResponse: FINAL_RESPONSE,
    progressEvents: PROGRESS_EVENTS,
  });
  completedRun.final_result = finalResult;

  let currentRun = runningRun;
  let currentEvents: unknown[] = buildProgressEvents(RUN_ID, THREAD_ID, PROGRESS_EVENTS);
  let currentActiveRuns = [runningRun];

  await setupBaseRoutes(page);
  await mockListThreads(page, [buildThread()]);
  await mockOpenRepository(page, DEFAULT_REPO);
  await mockGetActiveRunsDynamic(page, () => currentActiveRuns);
  await mockGetAgentRunDynamic(page, RUN_ID, () => currentRun);
  await mockGetAgentRunEventsDynamic(page, RUN_ID, () => currentEvents);
  await mockDebugRoutes(page, () => currentActiveRuns);

  await setStorageForThread(page, THREAD_ID, RUN_ID);
  await page.goto("/app");
  await expect(page.getByText(USER_MSG)).toBeVisible({ timeout: 5000 });
  await expect(page.getByText(WORK_NOTE)).toBeVisible({ timeout: 5000 });

  await page.goto("/debug");
  await expect(page.getByText("RepoOperator Debug")).toBeVisible();

  currentRun = completedRun;
  currentActiveRuns = [];
  currentEvents = [
    ...buildProgressEvents(RUN_ID, THREAD_ID, PROGRESS_EVENTS),
    {
      id: `${RUN_ID}-final`,
      run_id: RUN_ID,
      thread_id: THREAD_ID,
      type: "final_message",
      event_type: "final_message",
      result: finalResult,
      sequence: 20,
      timestamp: new Date().toISOString(),
    },
  ];

  await page.getByRole("link", { name: "Back to app" }).click();
  await expect(page.getByText(USER_MSG)).toBeVisible({ timeout: 5000 });
  await expect(page.getByText(FINAL_RESPONSE, { exact: false })).toBeVisible({ timeout: 8000 });
  await revealCompletedWorkLog(page);
  await expect(page.getByText(WORK_NOTE)).toBeVisible({ timeout: 5000 });
  await expect(page.locator(".agent-transcript-section")).toHaveCount(1);
  await expect(page.getByTestId("stop-run-button")).toHaveCount(0);

  const assistantMsgs = page.locator('[data-testid="assistant-message"]');
  expect(await assistantMsgs.count()).toBeLessThanOrEqual(1);
});

// ── Scenario B: Completed run rehydrates from persisted backend events ─────────

test("B1: completed run rehydrates final answer and progress from backend events", async ({ page }) => {
  const finalResult = buildFinalResult(RUN_ID, THREAD_ID, FINAL_RESPONSE, PROGRESS_EVENTS);
  const completedRun = buildMockRunRecord({
    runId: RUN_ID,
    threadId: THREAD_ID,
    repo: DEFAULT_REPO,
    status: "completed",
    finalResponse: FINAL_RESPONSE,
    progressEvents: PROGRESS_EVENTS,
  });
  completedRun.final_result = finalResult;

  const events = [
    ...buildProgressEvents(RUN_ID, THREAD_ID, PROGRESS_EVENTS),
    {
      id: `${RUN_ID}-final`,
      run_id: RUN_ID,
      thread_id: THREAD_ID,
      type: "final_message",
      event_type: "final_message",
      result: finalResult,
      sequence: 10,
      timestamp: new Date().toISOString(),
    },
  ];

  const threadWithRun = buildThread({
    messages: [
      { id: "msg-user-1", role: "user", content: USER_MSG, timestamp: new Date().toISOString() },
    ],
  });

  await setupBaseRoutes(page);
  await mockListThreads(page, [threadWithRun]);
  await mockOpenRepository(page, DEFAULT_REPO);
  await mockGetActiveRuns(page, []);
  await mockGetAgentRun(page, completedRun);
  await mockGetAgentRunEvents(page, RUN_ID, events);

  await setStorageForThread(page, THREAD_ID, RUN_ID);
  await page.goto("/app");
  await page.waitForTimeout(800);

  // After rehydrating a completed run, the final answer should appear
  await expect(page.getByText(FINAL_RESPONSE, { exact: false })).toBeVisible({ timeout: 8000 });

  // activeRunId should be cleared (no pending indicator)
  // The composer should not be in "pending" state after terminal run
  const stopButton = page.locator('[data-testid="stop-run-button"]');
  const isStopVisible = await stopButton.isVisible().catch(() => false);
  expect(isStopVisible, "Stop button must not be visible for terminal run").toBe(false);

  // No duplicate assistant messages
  const assistantMsgs = page.locator('[data-testid="assistant-message"]');
  const count = await assistantMsgs.count();
  expect(count).toBeLessThanOrEqual(1);
});

test("C: transcript replacement hides low-value labels and expands structured detail rows", async ({ page }) => {
  const runRecord = buildMockRunRecord({
    runId: RUN_ID,
    threadId: THREAD_ID,
    repo: DEFAULT_REPO,
    status: "running",
    progressEvents: PROGRESS_EVENTS,
  });

  await setupBaseRoutes(page);
  await mockListThreads(page, [buildThread()]);
  await mockOpenRepository(page, DEFAULT_REPO);
  await mockGetActiveRuns(page, [runRecord]);
  await mockGetAgentRun(page, runRecord);
  await mockGetAgentRunEvents(page, RUN_ID, buildProgressEvents(RUN_ID, THREAD_ID, PROGRESS_EVENTS));

  await setStorageForThread(page, THREAD_ID, RUN_ID);
  await page.goto("/app");

  await expect(page.getByText(WORK_NOTE)).toBeVisible({ timeout: 5000 });
  for (const hiddenLabel of [
    "Loaded context",
    "Framed request",
    "Recorded observation",
    "Chose next action",
    "Inspect repository tree",
  ]) {
    await expect(page.getByText(hiddenLabel)).toHaveCount(0);
  }

  const group = page.locator(".agent-transcript-section").first();
  await expect(group).toBeVisible();
  await group.locator(".agent-section-summary").click();
  await expect(group.locator(".agent-detail-item")).toHaveCount(2);
  await expect(group.getByText("repooperator-active-thread", { exact: false })).toBeVisible();
  await expect(group.getByText("apps/web/src/components/chat/ChatApp.tsx", { exact: false })).toBeVisible();
});

// ── Scenario C: Delayed assistant_delta does not make UI look stuck ────────────

test("C1: progress_delta events keep run alive when assistant_delta is delayed", async ({ page }) => {
  const runRecord = buildMockRunRecord({
    runId: RUN_ID,
    threadId: THREAD_ID,
    repo: DEFAULT_REPO,
    status: "running",
    progressEvents: PROGRESS_EVENTS,
  });
  const events = buildProgressEvents(RUN_ID, THREAD_ID, PROGRESS_EVENTS);

  await setupBaseRoutes(page);
  await mockListThreads(page, [buildThread()]);
  await mockOpenRepository(page, DEFAULT_REPO);
  await mockGetActiveRuns(page, [runRecord]);
  await mockGetAgentRun(page, runRecord);
  await mockGetAgentRunEvents(page, RUN_ID, events);

  await setStorageForThread(page, THREAD_ID, RUN_ID);
  await page.goto("/app");
  await page.waitForTimeout(600);

  await expect(page.getByText(WORK_NOTE)).toBeVisible({ timeout: 5000 });

  // No empty/blank assistant message should have been created
  const emptyAssistant = page.locator('[data-testid="assistant-message"]:has-text("")');
  // We only check that a spurious blank card was not inserted
  const emptyCount = await emptyAssistant.count();
  expect(emptyCount).toBe(0);
});

test("D: running transcript detail updates to completed without duplication", async ({ page }) => {
  const runningProgress = [
    {
      phase: "Searching",
      label: "Searching file contents",
      status: "running" as const,
      sequence: 1,
      activity_id: "same-search",
      event_type: "work_trace",
      visibility: "user",
      display: "primary",
      related_search_query: "route navigation",
      aggregate: { action_type: "search_text", query: "route navigation" },
    },
  ];
  const completedProgress = [
    {
      ...runningProgress[0],
      status: "completed" as const,
      sequence: 2,
    },
  ];
  const runRecord = buildMockRunRecord({
    runId: RUN_ID,
    threadId: THREAD_ID,
    repo: DEFAULT_REPO,
    status: "running",
    progressEvents: runningProgress,
  });
  let currentEvents = buildProgressEvents(RUN_ID, THREAD_ID, runningProgress);

  await setupBaseRoutes(page);
  await mockListThreads(page, [buildThread()]);
  await mockOpenRepository(page, DEFAULT_REPO);
  await mockGetActiveRuns(page, [runRecord]);
  await mockGetAgentRun(page, runRecord);
  await mockGetAgentRunEventsDynamic(page, RUN_ID, () => currentEvents);

  await setStorageForThread(page, THREAD_ID, RUN_ID);
  await page.goto("/app");

  const group = page.locator(".agent-transcript-section").first();
  await expect(group).toBeVisible({ timeout: 5000 });
  await expect(group.locator(".agent-detail-item")).toHaveCount(1);
  await expect(group.locator(".agent-detail-status")).toHaveText("running");

  currentEvents = buildProgressEvents(RUN_ID, THREAD_ID, completedProgress);
  const completedSummary = group.locator("button.agent-section-summary");
  await expect(completedSummary).toBeVisible({ timeout: 5000 });
  await completedSummary.click();
  await expect(group.locator(".agent-detail-status")).toHaveText("completed", { timeout: 5000 });
  await expect(page.locator(".agent-transcript-section")).toHaveCount(1);
  await expect(group.locator(".agent-detail-item")).toHaveCount(1);
});

// ── Scenario D: final_message without assistant_delta creates final message ────

test("D1: final_message without assistant_delta still creates assistant message", async ({ page }) => {
  const finalResult = buildFinalResult(RUN_ID, THREAD_ID, FINAL_RESPONSE, PROGRESS_EVENTS);
  const completedRun = buildMockRunRecord({
    runId: RUN_ID,
    threadId: THREAD_ID,
    repo: DEFAULT_REPO,
    status: "completed",
    finalResponse: FINAL_RESPONSE,
    progressEvents: PROGRESS_EVENTS,
  });

  // Events have only progress_delta + final_message, no assistant_delta
  const events = [
    ...buildProgressEvents(RUN_ID, THREAD_ID, PROGRESS_EVENTS),
    {
      id: `${RUN_ID}-final`,
      run_id: RUN_ID,
      thread_id: THREAD_ID,
      type: "final_message",
      event_type: "final_message",
      result: finalResult,
      sequence: 10,
      timestamp: new Date().toISOString(),
    },
  ];

  await setupBaseRoutes(page);
  await mockListThreads(page, [buildThread()]);
  await mockOpenRepository(page, DEFAULT_REPO);
  await mockGetActiveRuns(page, []);
  await mockGetAgentRun(page, completedRun);
  await mockGetAgentRunEvents(page, RUN_ID, events);

  await setStorageForThread(page, THREAD_ID, RUN_ID);
  await page.goto("/app");
  await page.waitForTimeout(800);

  await expect(page.getByText(FINAL_RESPONSE, { exact: false })).toBeVisible({ timeout: 8000 });
});

// ── Scenario E: assistant_delta + final_message deduplication ─────────────────

test("E: assistant_delta plus final_message does not duplicate the answer", async ({ page }) => {
  const finalResult = buildFinalResult(RUN_ID, THREAD_ID, FINAL_RESPONSE, PROGRESS_EVENTS);
  const completedRun = buildMockRunRecord({
    runId: RUN_ID,
    threadId: THREAD_ID,
    repo: DEFAULT_REPO,
    status: "completed",
    finalResponse: FINAL_RESPONSE,
    progressEvents: PROGRESS_EVENTS,
  });

  // Both assistant_delta chunks and final_message present
  const events = [
    ...buildProgressEvents(RUN_ID, THREAD_ID, PROGRESS_EVENTS),
    {
      id: `${RUN_ID}-ad-1`,
      run_id: RUN_ID,
      thread_id: THREAD_ID,
      type: "assistant_delta",
      event_type: "assistant_delta",
      delta: FINAL_RESPONSE,
      sequence: 9,
      timestamp: new Date().toISOString(),
    },
    {
      id: `${RUN_ID}-final`,
      run_id: RUN_ID,
      thread_id: THREAD_ID,
      type: "final_message",
      event_type: "final_message",
      result: finalResult,
      sequence: 10,
      timestamp: new Date().toISOString(),
    },
  ];

  await setupBaseRoutes(page);
  await mockListThreads(page, [buildThread()]);
  await mockOpenRepository(page, DEFAULT_REPO);
  await mockGetActiveRuns(page, []);
  await mockGetAgentRun(page, completedRun);
  await mockGetAgentRunEvents(page, RUN_ID, events);

  await setStorageForThread(page, THREAD_ID, RUN_ID);
  await page.goto("/app");
  await page.waitForTimeout(800);

  await expect(page.getByText(FINAL_RESPONSE, { exact: false })).toBeVisible({ timeout: 8000 });

  // Detect obvious doubling: two separate assistant message containers each showing it.
  const assistantMsgs = page.locator('[data-testid="assistant-message"]');
  const count = await assistantMsgs.count();
  expect(count).toBeLessThanOrEqual(1);
});

// ── Scenario F: non-terminal statuses keep run active ─────────────────────────

test("F: waiting_approval and cancelling statuses keep active run visible", async ({ page }) => {
  for (const status of ["waiting_approval", "cancelling"] as const) {
    const runRecord = buildMockRunRecord({
      runId: RUN_ID,
      threadId: THREAD_ID,
      repo: DEFAULT_REPO,
      status,
      progressEvents: PROGRESS_EVENTS,
    });
    const events = buildProgressEvents(RUN_ID, THREAD_ID, PROGRESS_EVENTS);

    await setupBaseRoutes(page);
    await mockListThreads(page, [buildThread()]);
    await mockOpenRepository(page, DEFAULT_REPO);
    await mockGetActiveRuns(page, [runRecord]);
    await mockGetAgentRun(page, runRecord);
    await mockGetAgentRunEvents(page, RUN_ID, events);

    await setStorageForThread(page, THREAD_ID, RUN_ID);
    await page.goto("/app");
    await page.waitForTimeout(600);

    // For non-terminal statuses the run should remain active — no final answer yet
    const finalText = await page.getByText(FINAL_RESPONSE, { exact: false }).isVisible().catch(() => false);
    expect(finalText, `status=${status}: final answer must not appear for non-terminal run`).toBe(false);
  }
});

// ── Scenario G: work trace fields rehydrate and merge ────────────────────────

test("G: work trace rehydrates safe summaries and merges by activity_id", async ({ page }) => {
  const workTraceEvents = [
    {
      phase: "Decision",
      label: "Chose next action",
      status: "running" as const,
      sequence: 1,
      activity_id: "decision-1",
      event_type: "work_trace",
      visibility: "user",
      display: "primary",
      safe_reasoning_summary: "The user named README.md, so I will read that file before answering.",
      current_action: "Read README.md.",
      evidence_needed: ["README.md contents"],
    },
    {
      phase: "Decision",
      label: "Chose next action",
      status: "completed" as const,
      sequence: 2,
      activity_id: "decision-1",
      event_type: "work_trace",
      visibility: "user",
      display: "primary",
      observation: "Read 12 lines.",
      next_action: "Prepare answer from README.md.",
      safety_note: "Read-only file access.",
    },
  ];
  const finalResult = buildFinalResult(RUN_ID, THREAD_ID, FINAL_RESPONSE, workTraceEvents);
  const completedRun = buildMockRunRecord({
    runId: RUN_ID,
    threadId: THREAD_ID,
    repo: DEFAULT_REPO,
    status: "completed",
    finalResponse: FINAL_RESPONSE,
    progressEvents: workTraceEvents,
  });
  completedRun.final_result = finalResult;

  const events: unknown[] = buildProgressEvents(RUN_ID, THREAD_ID, workTraceEvents).map((event) => ({
    ...event,
    event_type: "work_trace",
    hidden_reasoning: "do-not-render-hidden",
    private_reasoning: "do-not-render-private",
  }));
  events.push({
    id: `${RUN_ID}-final`,
    run_id: RUN_ID,
    thread_id: THREAD_ID,
    type: "final_message",
    event_type: "final_message",
    result: finalResult,
    sequence: 10,
    timestamp: new Date().toISOString(),
  });

  await setupBaseRoutes(page);
  await mockListThreads(page, [buildThread()]);
  await mockOpenRepository(page, DEFAULT_REPO);
  await mockGetActiveRuns(page, []);
  await mockGetAgentRun(page, completedRun);
  await mockGetAgentRunEvents(page, RUN_ID, events);

  await setStorageForThread(page, THREAD_ID, RUN_ID);
  await page.goto("/app");
  await page.waitForTimeout(800);

  await expect(page.getByText("The user named README.md", { exact: false })).toBeVisible({ timeout: 8000 });
  await expect(page.getByText("Chose next action")).toHaveCount(0);
  await expect(page.getByText("do-not-render-hidden")).toHaveCount(0);
  await expect(page.getByText("do-not-render-private")).toHaveCount(0);
});

test("H: active thread storage is scoped by repo branch", async ({ page }) => {
  const devRepo = { ...DEFAULT_REPO, branch: "dev" };
  const mainThread = {
    id: "thread-main",
    title: "mock/repo main",
    repo: DEFAULT_REPO,
    messages: [{ id: "main-msg", role: "user", content: "main branch thread", timestamp: new Date().toISOString() }],
  };
  const devThread = {
    id: "thread-dev",
    title: "mock/repo dev",
    repo: devRepo,
    messages: [{ id: "dev-msg", role: "user", content: "dev branch thread", timestamp: new Date().toISOString() }],
  };

  await setupBaseRoutes(page);
  await mockListThreads(page, [mainThread, devThread]);
  await mockOpenRepository(page, devRepo);
  await mockGetActiveRuns(page, []);
  await mockDebugRoutes(page);

  await setStorageForThread(page, "thread-dev", undefined, activeThreadKey(devRepo));
  await page.goto("/app");

  await expect(page.getByText("dev branch thread")).toBeVisible({ timeout: 5000 });
  await expect(page.getByText("main branch thread")).toHaveCount(0);

  await page.goto("/debug");
  await page.getByRole("link", { name: "Back to app" }).click();
  await expect(page.getByText("dev branch thread")).toBeVisible({ timeout: 5000 });
  await expect(page.getByText("main branch thread")).toHaveCount(0);
});
