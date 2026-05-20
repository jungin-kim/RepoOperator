import { describe, expect, it } from "vitest";
import {
  operationCountsText,
  operationDescription,
  proposalApplyWarning,
} from "../proposal-card-copy";
import type { ChangeProposal } from "../proposal-card-copy";

function baseProposal(overrides: Partial<ChangeProposal> = {}): ChangeProposal {
  return {
    id: "proposal-1",
    runId: "run-1",
    proposalId: "change-1",
    projectPath: "/repo",
    branch: "main",
    relativePath: "README.md",
    originalContent: "old",
    proposedContent: "new",
    model: "test",
    status: "proposed",
    ...overrides,
  };
}

describe("ProposalCard wording helpers", () => {
  it("does not use single-file modify-only wording for multi-file change sets", () => {
    const proposal = baseProposal({
      changeSetProposal: {
        proposal_id: "change-1",
        changes: [
          { path: "src/a.ts", operation: "modify" },
          { path: "src/b.ts", operation: "create" },
          { path: "src/c.ts", operation: "delete" },
        ],
      },
    });

    const text = proposalApplyWarning(proposal, proposal.changeSetProposal?.changes || [], true);
    expect(text).toContain("apply the listed change-set operations to 3 files");
    expect(text).toContain("Operations: 1 modify, 1 create, 1 delete.");
    expect(text).not.toMatch(/modify only/);
  });

  it("uses operation-specific language for create and delete rows", () => {
    expect(operationDescription("create")).toBe("created");
    expect(operationDescription("delete")).toBe("deleted");
    expect(operationCountsText([
      { path: "new.ts", operation: "create" },
      { path: "old.ts", operation: "delete" },
    ])).toBe("1 create, 1 delete");
  });

  it("keeps old single-file compatibility wording", () => {
    const proposal = baseProposal();
    const text = proposalApplyWarning(proposal, [{ path: "README.md", operation: "modify" }], false);
    expect(text).toContain("RepoOperator will apply the single-file modification to README.md");
  });
});
