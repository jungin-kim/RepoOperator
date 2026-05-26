import { describe, expect, it } from "vitest";
import { formatBudgetUsage, formatCarryoverSummary, formatTargetCandidates } from "../../../app/debug/context-format";

describe("debug context formatting", () => {
  it("renders context budget/window information", () => {
    expect(formatBudgetUsage({
      estimated_input_tokens: 1200,
      estimated_output_reserve: 800,
      estimated_total_tokens: 2000,
      context_window: 8000,
      usage_ratio: 0.25,
    })).toBe("2000 / 8000 tokens (25%)");
  });

  it("renders memory carryover and target candidates without normal-chat internals", () => {
    expect(formatTargetCandidates([{ path: "src/app.py", score: 91, role: "entrypoints", prior_reused: true }])).toContain("src/app.py");
    expect(formatCarryoverSummary({
      carryover_summaries: [{ kind: "prior_edit_target_evidence", selected_target_files: ["src/app.py"], candidate_count: 1 }],
    })).toContain("prior_edit_target_evidence: src/app.py");
  });
});
