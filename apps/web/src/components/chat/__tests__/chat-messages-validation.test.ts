import { describe, expect, it } from "vitest";
import type { AgentRunPayload } from "../../../lib/local-worker-client";
import { renderableValidationResult } from "../validation-result";

function metadata(overrides: Partial<AgentRunPayload>): AgentRunPayload {
  return {
    project_path: "/repo",
    git_provider: "local",
    task: "test",
    model: "mock",
    repo_root_name: "repo",
    context_summary: "",
    top_level_entries: [],
    readme_included: false,
    diff_included: false,
    is_git_repository: true,
    files_read: [],
    response: "Done.",
    ...overrides,
  };
}

describe("ValidationResultCard real validation gate", () => {
  it("does not expose a renderable result for bare success status", () => {
    const payload = metadata({ validation_result: { status: "success" } });

    expect(renderableValidationResult(payload)).toBeNull();
  });

  it("allows post-apply validation with kind and status", () => {
    const payload = metadata({
      validation_result: {
        kind: "post_apply",
        source: "post_apply",
        status: "passed",
        display_command: "npm test",
      },
    });
    expect(renderableValidationResult(payload)).toMatchObject({ kind: "post_apply", status: "passed" });
    expect(renderableValidationResult(payload)).toMatchObject({ display_command: "npm test" });
  });

  it("maps legacy post-apply status only with explicit validation context", () => {
    const bare = metadata({ post_apply_validation_status: "passed" });
    const withSelection = metadata({
      post_apply_validation_status: "passed",
      validation_command_selection: { candidates: [], reason: "selected" },
    });

    expect(renderableValidationResult(bare)).toBeNull();
    expect(renderableValidationResult(withSelection)).toMatchObject({ kind: "post_apply", status: "passed" });
  });
});
