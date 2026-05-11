import { describe, expect, it } from "vitest";
import type { AgentRunPayload } from "../../../lib/local-worker-client";
import { buildAgentActivityTranscript } from "../agent-activity-transcript";
import {
  mergeRunEventsIntoProgressSteps,
  progressStepFromEvent,
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

  it("progressStepFromEvent preserves structured action metadata", () => {
    const step = progressStepFromEvent(progressEvent({
      activity_id: "read-structured",
      operation: "read_file",
      action_type: "read_file",
      tool_name: "read_file",
      files: ["README.md"],
      aggregate: { action_type: "read_file", file_path: "README.md", line_count: 3 },
    }));

    expect(step.operation).toBe("read_file");
    expect(step.actionType).toBe("read_file");
    expect(step.toolName).toBe("read_file");
    expect(step.aggregate?.action_type).toBe("read_file");
    expect(step.files).toEqual(["README.md"]);
  });

  it("preserves command and edit proposal metadata", () => {
    const command = progressStepFromEvent(progressEvent({
      activity_id: "cmd-structured",
      operation: "command",
      action_type: "run_approved_command",
      tool_name: "run_approved_command",
      command: ["git", "status", "--short"],
      aggregate: { action_type: "run_approved_command", display_command: "git status --short", exit_code: 0 },
    }));
    const edit = progressStepFromEvent(progressEvent({
      activity_id: "edit-structured",
      phase: "Editing",
      label: "Prepared patch",
      operation: "edit",
      action_type: "generate_edit",
      tool_name: "generate_edit",
      files: ["main.py"],
      proposal_id: "proposal:main.py",
      aggregate: { action_type: "generate_edit", edit_archive: [{ file_path: "main.py", additions: 2, deletions: 1 }] },
    }));

    expect(command.command).toEqual(["git", "status", "--short"]);
    expect(command.aggregate?.display_command).toBe("git status --short");
    expect(edit.proposalId).toBe("proposal:main.py");
    expect(edit.aggregate?.edit_archive).toBeTruthy();
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
