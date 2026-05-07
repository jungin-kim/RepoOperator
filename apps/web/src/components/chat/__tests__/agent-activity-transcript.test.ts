import { describe, expect, it } from "vitest";
import { buildAgentActivityTranscript } from "../agent-activity-transcript";
import type { ProgressStep } from "../progress-types";

function step(overrides: Partial<ProgressStep> & { activityId: string }): ProgressStep {
  return {
    status: "completed",
    ...overrides,
  };
}

function statusStep(text: string, overrides: Partial<ProgressStep> & { activityId: string }): ProgressStep {
  return step({
    label: "Planning",
    safeReasoningSummary: text,
    eventType: "work_trace",
    visibility: "user",
    display: "primary",
    ...overrides,
  });
}

function toolStep(
  actionType: string,
  overrides: Partial<ProgressStep> & { activityId: string },
): ProgressStep {
  return step({
    eventType: "work_trace",
    visibility: "user",
    display: "primary",
    ...overrides,
    aggregate: { ...(overrides.aggregate || {}), tool: actionType, action_type: actionType },
  });
}

describe("buildAgentActivityTranscript sections", () => {
  it("returns empty array for empty steps", () => {
    expect(buildAgentActivityTranscript([])).toEqual([]);
  });

  it("status note starts a section", () => {
    const sections = buildAgentActivityTranscript([
      statusStep("I will map the restore path first.", { activityId: "note:1" }),
    ]);

    expect(sections).toHaveLength(1);
    expect(sections[0].statusText).toBe("I will map the restore path first.");
  });

  it("actions after status note attach to that section", () => {
    const sections = buildAgentActivityTranscript([
      statusStep("I will inspect thread persistence.", { activityId: "note:1" }),
      toolStep("search_text", { activityId: "search:1", relatedSearchQuery: "active thread" }),
      toolStep("read_file", { activityId: "read:1", files: ["ChatApp.tsx"] }),
    ]);

    expect(sections).toHaveLength(1);
    expect(sections[0].details.map((detail) => detail.kind)).toEqual(["search", "read_file"]);
    expect(sections[0].summary).toMatchObject({ searches: 1, filesRead: 1 });
  });

  it("next status note starts a new section", () => {
    const sections = buildAgentActivityTranscript([
      statusStep("First phase.", { activityId: "note:1" }),
      toolStep("search_text", { activityId: "search:1", relatedSearchQuery: "thread" }),
      statusStep("Second phase.", { activityId: "note:2" }),
      toolStep("read_file", { activityId: "read:1", files: ["run-event-state.ts"] }),
    ]);

    expect(sections).toHaveLength(2);
    expect(sections[0].details).toHaveLength(1);
    expect(sections[1].details).toHaveLength(1);
    expect(sections[1].statusText).toBe("Second phase.");
  });

  it("completed sections are collapsible by default", () => {
    const sections = buildAgentActivityTranscript([
      statusStep("Completed phase.", { activityId: "note:1" }),
      toolStep("read_file", { activityId: "read:1", files: ["README.md"], status: "completed" }),
    ]);

    expect(sections[0]).toMatchObject({
      status: "completed",
      collapsible: true,
      collapsedByDefault: true,
      isCurrent: false,
    });
  });

  it("current running section is expanded and not collapsible", () => {
    const sections = buildAgentActivityTranscript([
      statusStep("Running phase.", { activityId: "note:1", status: "running" }),
      toolStep("run_approved_command", {
        activityId: "cmd:1",
        command: "npm test",
        status: "running",
      }),
    ]);

    expect(sections[0]).toMatchObject({
      status: "running",
      collapsible: false,
      collapsedByDefault: false,
      isCurrent: true,
    });
  });

  it("running command updates to completed in the same section", () => {
    const sections = buildAgentActivityTranscript([
      statusStep("Run checks.", { activityId: "note:1" }),
      toolStep("run_approved_command", { activityId: "cmd:1", command: "npm test", status: "running" }),
      toolStep("run_approved_command", { activityId: "cmd:1", command: "npm test", status: "completed" }),
    ]);

    expect(sections[0].details).toHaveLength(1);
    expect(sections[0].details[0]).toMatchObject({ kind: "command", status: "completed" });
    expect(sections[0].status).toBe("completed");
  });

  it("duplicate activity_id does not create duplicate action rows", () => {
    const sections = buildAgentActivityTranscript([
      statusStep("Search once.", { activityId: "note:1" }),
      toolStep("search_text", { activityId: "same", relatedSearchQuery: "repoIdentityKey", status: "running" }),
      toolStep("search_text", { activityId: "same", relatedSearchQuery: "repoIdentityKey", status: "completed" }),
    ]);

    expect(sections[0].details).toHaveLength(1);
  });

  it("low-value labels do not create sections", () => {
    const sections = buildAgentActivityTranscript([
      step({ activityId: "debug:1", label: "Loaded context", display: "secondary", visibility: "debug" }),
      step({ activityId: "debug:2", label: "Framed request", display: "secondary", visibility: "debug" }),
      step({ activityId: "debug:3", label: "Recorded observation", display: "secondary", visibility: "debug" }),
      step({ activityId: "debug:4", label: "Chose next action", display: "secondary", visibility: "debug" }),
    ]);

    expect(sections).toEqual([]);
  });

  it("debug secondary actions do not appear in primary sections", () => {
    const sections = buildAgentActivityTranscript([
      statusStep("Visible phase.", { activityId: "note:1" }),
      toolStep("search_text", {
        activityId: "debug-search",
        relatedSearchQuery: "hidden",
        display: "secondary",
        visibility: "debug",
      }),
    ]);

    expect(sections).toHaveLength(1);
    expect(sections[0].details).toEqual([]);
  });

  it("action before status note creates quiet implicit section only for primary work", () => {
    const sections = buildAgentActivityTranscript([
      toolStep("search_text", { activityId: "search:1", relatedSearchQuery: "implicit work" }),
    ]);

    expect(sections).toHaveLength(1);
    expect(sections[0].statusText).toBe("Working");
    expect(sections[0].details).toHaveLength(1);
  });

  it("completed-only search/read/list/command rows produce completed section status and summary", () => {
    const sections = buildAgentActivityTranscript([
      statusStep("Map files.", { activityId: "note:1" }),
      toolStep("search_text", { activityId: "search:1", relatedSearchQuery: "threads" }),
      toolStep("inspect_repo_tree", { activityId: "list:1", aggregate: { path: "apps/web/src" } }),
      toolStep("read_file", { activityId: "read:1", files: ["ChatApp.tsx", "run-event-state.ts"] }),
      toolStep("run_approved_command", { activityId: "cmd:1", command: "git status", aggregate: { exit_code: 0 } }),
    ]);

    expect(sections[0].status).toBe("completed");
    expect(sections[0].summary).toMatchObject({ searches: 1, filesListed: 1, filesRead: 2, commandsRun: 1 });
    expect(sections[0].summaryText).toContain("검색 1회");
    expect(sections[0].summaryText).toContain("ran 1 command");
  });

  it("failed detail makes section failed", () => {
    const sections = buildAgentActivityTranscript([
      statusStep("Run command.", { activityId: "note:1" }),
      toolStep("run_approved_command", { activityId: "cmd:1", command: "npm test", status: "failed" }),
    ]);

    expect(sections[0].status).toBe("failed");
  });

  it("waiting detail makes section waiting", () => {
    const sections = buildAgentActivityTranscript([
      statusStep("Need approval.", { activityId: "note:1" }),
      toolStep("run_approved_command", { activityId: "cmd:1", command: "git push", status: "waiting_approval" }),
    ]);

    expect(sections[0].status).toBe("waiting");
  });

  it("overall finalizeRunning converts running rows and section to completed", () => {
    const sections = buildAgentActivityTranscript([
      statusStep("Finishing.", { activityId: "note:1", status: "running" }),
      toolStep("read_file", { activityId: "read:1", files: ["README.md"], status: "running" }),
    ], { finalizeRunning: true });

    expect(sections[0].status).toBe("completed");
    expect(sections[0].details[0].status).toBe("completed");
  });

  it("edit rows include file path and additions/deletions when metadata exists", () => {
    const sections = buildAgentActivityTranscript([
      statusStep("Prepare edit.", { activityId: "note:1" }),
      toolStep("generate_edit", {
        activityId: "edit:1",
        safetyNote: "Proposal only. No files were written.",
        aggregate: {
          edit_archive: [
            {
              file_path: "ChatApp.tsx",
              additions: 12,
              deletions: 4,
              status: "proposed",
              summary: "Restore active thread",
              diff: "@@",
            },
          ],
        },
      }),
    ]);

    expect(sections[0].edits[0]).toMatchObject({
      path: "ChatApp.tsx",
      additions: 12,
      deletions: 4,
      status: "proposed",
      safetyNote: "Proposal only. No files were written.",
    });
    expect(sections[0].summary.filesEdited).toBe(1);
  });

  it("edit rows do not invent additions/deletions when missing", () => {
    const sections = buildAgentActivityTranscript([
      statusStep("Prepare edit.", { activityId: "note:1" }),
      toolStep("generate_edit", { activityId: "edit:1", files: ["ChatApp.tsx"] }),
    ]);

    expect(sections[0].edits[0]).toMatchObject({ path: "ChatApp.tsx" });
    expect(sections[0].edits[0].additions).toBeUndefined();
    expect(sections[0].edits[0].deletions).toBeUndefined();
  });
});
