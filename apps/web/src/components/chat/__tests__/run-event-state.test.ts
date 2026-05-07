import { describe, expect, it } from "vitest";
import type { AgentRunPayload } from "../../../lib/local-worker-client";
import { buildAgentActivityTranscript } from "../agent-activity-transcript";
import {
  mergeRunEventsIntoProgressSteps,
  progressStepsForCompletedRun,
  type AgentRunEvent,
} from "../run-event-state";

function progressEvent(overrides: Partial<AgentRunEvent>): AgentRunEvent {
  return {
    id: `event-${overrides.sequence || 1}`,
    type: "progress_delta",
    event_type: "work_trace",
    run_id: "run-1",
    thread_id: "thread-1",
    activity_id: "activity-1",
    sequence: 1,
    phase: "Searching",
    label: "Searching file contents",
    status: "completed",
    aggregate: { action_type: "search_text", query: "repoIdentityKey" },
    ...overrides,
  };
}

function finalResult(overrides: Partial<AgentRunPayload> = {}): AgentRunPayload {
  return {
    project_path: "/repo",
    git_provider: "local",
    task: "test",
    model: "mock",
    branch: "main",
    repo_root_name: "repo",
    context_summary: "",
    top_level_entries: [],
    readme_included: false,
    diff_included: false,
    is_git_repository: true,
    files_read: [],
    response: "Done.",
    activity_events: [],
    edit_archive: [],
    ...overrides,
  };
}

describe("run-event-state transcript reconstruction", () => {
  it("completed run with persisted progress events reconstructs transcript", () => {
    const steps = progressStepsForCompletedRun([
      progressEvent({ activity_id: "search-1", aggregate: { action_type: "search_text", query: "active run" } }),
    ]);
    const transcript = buildAgentActivityTranscript(steps, { finalizeRunning: true });

    expect(transcript[0].details).toHaveLength(1);
    expect(transcript[0].details[0].kind).toBe("search");
  });

  it("completed run falls back to final_result.activity_events when stored events are missing", () => {
    const result = finalResult({
      activity_events: [
        progressEvent({
          activity_id: "read-1",
          phase: "Reading files",
          label: "README.md",
          files: ["README.md"],
          aggregate: { action_type: "read_file" },
        }),
      ],
    });
    const steps = progressStepsForCompletedRun([], result);
    const transcript = buildAgentActivityTranscript(steps, { finalizeRunning: true });

    expect(transcript[0].details).toHaveLength(1);
    expect(transcript[0].details[0].kind).toBe("read_file");
    expect(steps[0].files).toEqual(["README.md"]);
  });

  it("does not duplicate transcript items after rehydrate merges duplicate activity_id", () => {
    const steps = mergeRunEventsIntoProgressSteps([
      progressEvent({ activity_id: "same", status: "running", sequence: 1 }),
      progressEvent({ activity_id: "same", status: "completed", sequence: 2 }),
    ]);
    const transcript = buildAgentActivityTranscript(steps);

    expect(steps).toHaveLength(1);
    expect(transcript[0].details).toHaveLength(1);
  });

  it("attaches edit_archive metadata to completed transcript state", () => {
    const result = finalResult({
      response_type: "change_proposal",
      proposal_relative_path: "ChatApp.tsx",
      edit_archive: [
        {
          file_path: "ChatApp.tsx",
          additions: 3,
          deletions: 1,
          status: "proposed",
          diff: "@@",
        },
      ],
    });
    const steps = progressStepsForCompletedRun([], result);
    const transcript = buildAgentActivityTranscript(steps, { finalizeRunning: true });

    expect(transcript[0].edits[0]).toMatchObject({ path: "ChatApp.tsx", additions: 3, deletions: 1 });
    expect(transcript[0].edits[0].safetyNote).toContain("No files were written");
  });
});
