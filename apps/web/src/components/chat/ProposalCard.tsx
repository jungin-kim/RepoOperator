"use client";

import { useState } from "react";
import {
  applyChangeSet,
  rejectChangeSet,
  LocalWorkerClientError,
  type AgentRunPayload,
  type AgentProposeFilePayload,
  type ChangeSetProposalPayload,
} from "@/lib/local-worker-client";

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

interface ProposalCardProps {
  proposal: ChangeProposal;
  writeMode: "basic" | "auto_review" | "full_access";
  onStatusChange: (id: string, status: ProposalStatus, message?: string, result?: AgentRunPayload) => void;
}

/** Build a simple unified-style line diff for display (no external library needed). */
function buildLineDiff(original: string, proposed: string): Array<{ kind: "ctx" | "add" | "del"; line: string }> {
  const oldLines = original.split("\n");
  const newLines = proposed.split("\n");

  // Simple longest-common-subsequence diff (good enough for small files)
  const m = oldLines.length;
  const n = newLines.length;
  const dp: number[][] = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));

  for (let i = m - 1; i >= 0; i--) {
    for (let j = n - 1; j >= 0; j--) {
      if (oldLines[i] === newLines[j]) {
        dp[i][j] = 1 + dp[i + 1][j + 1];
      } else {
        dp[i][j] = Math.max(dp[i + 1][j], dp[i][j + 1]);
      }
    }
  }

  const result: Array<{ kind: "ctx" | "add" | "del"; line: string }> = [];
  let i = 0;
  let j = 0;
  while (i < m || j < n) {
    if (i < m && j < n && oldLines[i] === newLines[j]) {
      result.push({ kind: "ctx", line: oldLines[i] });
      i++;
      j++;
    } else if (j < n && (i >= m || dp[i + 1][j] <= dp[i][j + 1])) {
      result.push({ kind: "add", line: newLines[j] });
      j++;
    } else {
      result.push({ kind: "del", line: oldLines[i] });
      i++;
    }
  }

  return result;
}

function DiffView({ original, proposed }: { original: string; proposed: string }) {
  const diff = buildLineDiff(original, proposed);
  const hasChanges = diff.some((d) => d.kind !== "ctx");

  if (!hasChanges) {
    return (
      <div className="proposal-diff-empty">No changes detected between original and proposed content.</div>
    );
  }

  return (
    <div className="proposal-diff">
      {diff.map((entry, idx) => (
        <div
          key={idx}
          className={`proposal-diff-line proposal-diff-line-${entry.kind}`}
        >
          <span className="proposal-diff-gutter" aria-hidden="true">
            {entry.kind === "add" ? "+" : entry.kind === "del" ? "−" : " "}
          </span>
          <span className="proposal-diff-text">{entry.line}</span>
        </div>
      ))}
    </div>
  );
}

export function ProposalCard({ proposal, writeMode, onStatusChange }: ProposalCardProps) {
  const [applying, setApplying] = useState(false);
  const [showFull, setShowFull] = useState(false);
  const [selectedPath, setSelectedPath] = useState<string | null>(null);

  const isSettled = proposal.status !== "proposed";
  const changes = proposal.changeSetProposal?.changes?.length
    ? proposal.changeSetProposal.changes
    : [{
        path: proposal.relativePath,
        operation: "modify",
        original_content: proposal.originalContent,
        proposed_content: proposal.proposedContent,
        summary: proposal.changeSetProposal?.plan?.summary,
      }];
  const selectedChange = changes.find((change) => change.path === selectedPath) ?? changes[0];
  const fileCount = changes.length;

  async function handleApply() {
    if (applying || isSettled) return;
    setApplying(true);
    try {
      if (proposal.runId && proposal.proposalId) {
        const result = await applyChangeSet({
          run_id: proposal.runId,
          proposal_id: proposal.proposalId,
        });
        onStatusChange(proposal.id, "applied", `Applied proposal ${proposal.proposalId}`, result);
      } else {
        throw new Error("This proposal is missing a run id and cannot be applied safely.");
      }
    } catch (err) {
      const msg =
        err instanceof LocalWorkerClientError || err instanceof Error
          ? err.message
          : "Unable to apply the changes.";
      onStatusChange(proposal.id, "failed", msg);
    } finally {
      setApplying(false);
    }
  }

  function handleReject() {
    if (proposal.runId && proposal.proposalId) {
      void rejectChangeSet({ run_id: proposal.runId, proposal_id: proposal.proposalId })
        .then((result) => onStatusChange(proposal.id, "rejected", "Change proposal rejected.", result))
        .catch(() => onStatusChange(proposal.id, "rejected", "Change proposal rejected."));
      return;
    }
    onStatusChange(proposal.id, "rejected", "Change proposal rejected.");
  }

  const statusLabel: Record<ProposalStatus, string> = {
    proposed: "Pending approval",
    applied: "Applied",
    rejected: "Rejected",
    failed: "Failed to apply",
  };

  const statusClass: Record<ProposalStatus, string> = {
    proposed: "proposal-status-proposed",
    applied: "proposal-status-applied",
    rejected: "proposal-status-rejected",
    failed: "proposal-status-failed",
  };

  return (
    <div className={`proposal-card${isSettled ? ` proposal-card-${proposal.status}` : ""}`}>
      {/* Header */}
      <div className="proposal-card-header">
        <div className="proposal-card-header-left">
          <span className="proposal-card-icon" aria-hidden="true" />
          <div>
            <div className="proposal-card-title">
              RepoOperator prepared proposed changes for {fileCount} file{fileCount === 1 ? "" : "s"}
            </div>
            <div className="proposal-card-path">Proposed changes only. No files modified yet.</div>
          </div>
        </div>
        <span className={`proposal-status-badge ${statusClass[proposal.status]}`}>
          {statusLabel[proposal.status]}
        </span>
      </div>

      {proposal.branch && (
        <div className="proposal-card-meta">
          Branch: <strong>{proposal.branch}</strong>
          {" · "}Model: <strong>{proposal.model}</strong>
          {proposal.proposalId ? (
            <>
              {" · "}Proposal: <strong>{proposal.proposalId}</strong>
            </>
          ) : null}
        </div>
      )}

      <div className="proposal-file-list">
        {changes.map((change) => (
          <button
            key={`${change.operation}-${change.path}`}
            className={`proposal-file-row${selectedChange?.path === change.path ? " proposal-file-row-selected" : ""}`}
            type="button"
            onClick={() => setSelectedPath(change.path)}
          >
            <span className="proposal-file-op">{change.operation}</span>
            <span className="proposal-file-path">{change.path}</span>
            <span className="proposal-file-stats">
              +{change.additions ?? countChangedLines(change.original_content ?? "", change.proposed_content ?? "").added}
              {" / "}
              -{change.deletions ?? countChangedLines(change.original_content ?? "", change.proposed_content ?? "").removed}
            </span>
            <span className="proposal-file-validation">{change.validation_status || proposal.changeSetProposal?.validation?.status || "pending"}</span>
          </button>
        ))}
      </div>

      {/* Diff preview */}
      <div className="proposal-diff-wrapper">
        <div className="proposal-diff-titlebar">
          <span>{selectedChange?.path || proposal.relativePath}</span>
          <button
            className="proposal-diff-toggle"
            type="button"
            onClick={() => setShowFull((v) => !v)}
          >
            {showFull ? "Collapse" : "Expand full diff"}
          </button>
          <button
            className="proposal-diff-toggle"
            type="button"
            onClick={() => void navigator.clipboard?.writeText(formatSelectedDiff(selectedChange, proposal))}
          >
            Copy diff
          </button>
        </div>
        {selectedChange?.summary ? <p className="proposal-risk-note">{selectedChange.summary}</p> : null}
        {selectedChange?.risk_notes?.length ? (
          <div className="proposal-risk-note">Risk notes: {selectedChange.risk_notes.join("; ")}</div>
        ) : null}
        {proposal.changeSetProposal?.validation?.errors?.length ? (
          <div className="proposal-validation-errors">
            {proposal.changeSetProposal.validation.errors.join("; ")}
          </div>
        ) : null}
        <div
          className="proposal-diff-scroll"
          style={{ maxHeight: showFull ? "none" : "260px" }}
        >
          <DiffView
            original={selectedChange?.operation === "create" ? "" : selectedChange?.original_content ?? proposal.originalContent}
            proposed={selectedChange?.operation === "delete" ? "" : selectedChange?.proposed_content ?? proposal.proposedContent}
          />
        </div>
      </div>

      {/* Actions */}
      {!isSettled && (
        <div className="proposal-card-actions">
          <p className="proposal-warning">
            Review the diff before applying. RepoOperator will modify only{" "}
            <strong>{proposal.relativePath}</strong> on the current branch.
          </p>
          <div className="proposal-card-buttons">
            <button
              className="proposal-btn-apply"
              type="button"
              onClick={() => void handleApply()}
              disabled={applying}
            >
              {applying ? "Applying..." : "Apply changes"}
            </button>
            <button
              className="proposal-btn-reject"
              type="button"
              onClick={handleReject}
              disabled={applying}
            >
              Reject
            </button>
          </div>
        </div>
      )}

      {isSettled && proposal.status !== "proposed" && (
        <div className={`proposal-settled-notice proposal-settled-${proposal.status}`}>
          {proposal.status === "applied" && `Changes applied${proposal.appliedAt ? ` at ${proposal.appliedAt}` : ""}.`}
          {proposal.status === "rejected" && "Proposal rejected."}
          {proposal.status === "failed" && "Failed to apply changes."}
        </div>
      )}
    </div>
  );
}

function countChangedLines(original: string, proposed: string): { added: number; removed: number } {
  const diff = buildLineDiff(original, proposed);
  return {
    added: diff.filter((line) => line.kind === "add").length,
    removed: diff.filter((line) => line.kind === "del").length,
  };
}

function formatSelectedDiff(
  change: NonNullable<ChangeSetProposalPayload["changes"]>[number] | undefined,
  proposal: ChangeProposal,
): string {
  if (!change) return `${proposal.relativePath}\n`;
  return [
    `${change.operation} ${change.path}`,
    change.summary || "",
    ...(change.risk_notes || []),
    "",
    buildLineDiff(
      change.operation === "create" ? "" : change.original_content ?? proposal.originalContent,
      change.operation === "delete" ? "" : change.proposed_content ?? proposal.proposedContent,
    ).map((line) => `${line.kind === "add" ? "+" : line.kind === "del" ? "-" : " "}${line.line}`).join("\n"),
  ].filter(Boolean).join("\n");
}

/** Helper to turn an AgentProposeFilePayload into a ChangeProposal. */
export function proposalFromPayload(
  payload: AgentProposeFilePayload,
  opts: { projectPath: string; branch?: string | null },
): ChangeProposal {
  return {
    id: `${Date.now()}-proposal-${payload.relative_path}`,
    projectPath: opts.projectPath,
    branch: opts.branch,
    relativePath: payload.relative_path,
    originalContent: payload.original_content,
    proposedContent: payload.proposed_content,
    model: payload.model,
    status: "proposed",
  };
}

/** Helper to turn a change_proposal AgentRunPayload into a ChangeProposal. */
export function proposalFromRunPayload(
  payload: {
    model: string;
    run_id?: string | null;
    change_set_proposal?: ChangeSetProposalPayload | null;
    proposal_id?: string | null;
    proposal_relative_path?: string | null;
    proposal_original_content?: string | null;
    proposal_proposed_content?: string | null;
    apply_status?: string | null;
  },
  opts: { projectPath: string; branch?: string | null },
): ChangeProposal {
  const firstChange = payload.change_set_proposal?.changes?.[0];
  const relativePath = firstChange?.path ?? payload.proposal_relative_path ?? "unknown";
  return {
    id: payload.change_set_proposal?.proposal_id || `${Date.now()}-proposal-${relativePath}`,
    runId: payload.run_id,
    proposalId: payload.change_set_proposal?.proposal_id || payload.proposal_id || null,
    projectPath: opts.projectPath,
    branch: opts.branch,
    relativePath,
    originalContent: firstChange?.original_content ?? payload.proposal_original_content ?? "",
    proposedContent: firstChange?.proposed_content ?? payload.proposal_proposed_content ?? "",
    model: payload.model,
    status: payload.apply_status === "applied" ? "applied" : "proposed",
    changeSetProposal: payload.change_set_proposal,
    appliedAt: payload.change_set_proposal?.applied_at ?? null,
  };
}
