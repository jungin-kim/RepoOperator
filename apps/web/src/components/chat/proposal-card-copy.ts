import type { ChangeSetProposalPayload } from "@/lib/local-worker-client";

export type ProposalStatus = "proposed" | "applied" | "rejected" | "failed";

export type ChangeProposal = {
  id: string;
  runId?: string | null;
  proposalId?: string | null;
  projectPath: string;
  branch: string | null | undefined;
  relativePath: string;
  originalContent: string;
  proposedContent: string;
  model: string;
  status: ProposalStatus;
  changeSetProposal?: ChangeSetProposalPayload | null;
  appliedAt?: string | null;
};

export type ProposalChange = NonNullable<ChangeSetProposalPayload["changes"]>[number];

export function operationLabel(operation: string | undefined): string {
  const normalized = String(operation || "modify").toLowerCase();
  if (normalized === "create") return "create";
  if (normalized === "delete") return "delete";
  if (normalized === "rename") return "rename";
  return "modify";
}

export function operationDescription(operation: string | undefined): string {
  const normalized = operationLabel(operation);
  if (normalized === "create") return "created";
  if (normalized === "delete") return "deleted";
  if (normalized === "rename") return "renamed";
  return "modified";
}

export function operationCounts(changes: ProposalChange[]): Record<string, number> {
  return changes.reduce<Record<string, number>>((counts, change) => {
    const op = operationLabel(change.operation);
    counts[op] = (counts[op] || 0) + 1;
    return counts;
  }, {});
}

export function operationCountsText(changes: ProposalChange[]): string {
  const counts = operationCounts(changes);
  const order = ["modify", "create", "delete", "rename"];
  return order
    .filter((op) => counts[op])
    .map((op) => `${counts[op]} ${op}`)
    .join(", ");
}

export function proposalApplyWarning(
  proposal: ChangeProposal,
  changes: ProposalChange[],
  hasChangeSet: boolean,
): string {
  const fileCount = changes.length || 1;
  if (!hasChangeSet) {
    return `Review the diff before applying. RepoOperator will apply the single-file modification to ${proposal.relativePath} on the current branch after approval.`;
  }
  const base = `Review the diff before applying. RepoOperator will apply the listed change-set operations to ${fileCount} file${fileCount === 1 ? "" : "s"} on the current branch after approval.`;
  const counts = operationCountsText(changes);
  return counts ? `${base} Operations: ${counts}.` : base;
}
