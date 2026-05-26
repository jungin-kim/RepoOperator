"use client";

import { useEffect, useRef, useState } from "react";
import type {
  AgentRunPayload,
  CommandResultPayload,
  EditArchiveRecord,
  PermissionMode,
  RepoOpenPayload,
} from "@/lib/local-worker-client";
import { MarkdownContent } from "./MarkdownContent";
import {
  ProposalCard,
  type ChangeProposal,
  type ProposalStatus,
} from "./ProposalCard";
import type { ProgressStep } from "./progress-types";
import { AgentActivityTranscript } from "./AgentActivityTranscript";
import { renderableValidationResult } from "./validation-result";

export type ChatMessage = {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: Date;
  metadata?: AgentRunPayload;
  proposal?: ChangeProposal;
  progressSteps?: ProgressStep[];
};

function CommandApprovalCard({
  metadata,
  onDecision,
}: {
  metadata: AgentRunPayload;
  onDecision?: (metadata: AgentRunPayload, decision: "yes" | "yes_session" | "no_explain") => void;
}) {
  const approval = metadata.command_approval;
  if (!approval) return null;
  return (
      <div className={`command-card command-card-${approval.risk}`}>
        <div className="command-card-heading">Command approval required</div>
        {metadata.response ? <MarkdownContent content={metadata.response} /> : null}
        <p>{approval.reason}</p>
      <dl className="command-card-grid">
        <div>
          <dt>Command</dt>
          <dd><code>{approval.display_command}</code></dd>
        </div>
        <div>
          <dt>Working directory</dt>
          <dd>{approval.cwd || "No repository opened"}</dd>
        </div>
        <div>
          <dt>Risk</dt>
          <dd>{approval.risk}</dd>
        </div>
        <div>
          <dt>Read-only</dt>
          <dd>{approval.read_only ? "Yes" : "No"}</dd>
        </div>
        <div>
          <dt>Network</dt>
          <dd>{approval.needs_network ? "May use network" : "No network expected"}</dd>
        </div>
        <div>
          <dt>Outside repository</dt>
          <dd>{approval.touches_outside_repo ? "Yes" : "No"}</dd>
        </div>
      </dl>
      {approval.blocked ? (
        <div className="command-card-blocked">Blocked by RepoOperator safety policy.</div>
      ) : (
        <div className="command-card-actions">
          <button type="button" onClick={() => onDecision?.(metadata, "yes")}>Yes</button>
          <button type="button" onClick={() => onDecision?.(metadata, "yes_session")}>
            Yes, and don't ask again for this session
          </button>
          <button type="button" onClick={() => onDecision?.(metadata, "no_explain")}>
            No, explain another approach
          </button>
        </div>
      )}
    </div>
  );
}

function CommandResultCard({ result }: { result: CommandResultPayload }) {
  return (
    <div className="command-card">
      <div className="command-card-heading">Command result</div>
      <p><code>{result.display_command}</code> exited with {result.exit_code}.</p>
      <details open>
        <summary>Output</summary>
        <pre>{result.stdout || result.stderr || "No output"}</pre>
      </details>
    </div>
  );
}

export function ValidationResultCard({ metadata }: { metadata: AgentRunPayload }) {
  const result = renderableValidationResult(metadata);
  const status = result?.status;
  if (!status) return null;
  const commands = result?.candidate_commands || metadata.validation_commands || metadata.validation_command_selection?.candidates || [];
  return (
    <div className={`workflow-card workflow-card-${status === "failed" ? "failed" : status === "passed" ? "passed" : "neutral"}`}>
      <div className="workflow-card-heading">Validation result</div>
      <dl className="command-card-grid">
        <div>
          <dt>Status</dt>
          <dd>{status}</dd>
        </div>
        {result?.display_command ? (
          <div>
            <dt>Command</dt>
            <dd><code>{result.display_command}</code></dd>
          </div>
        ) : null}
        {metadata.validation_command_selection?.language ? (
          <div>
            <dt>Project</dt>
            <dd>{metadata.validation_command_selection.project_type || metadata.validation_command_selection.language}</dd>
          </div>
        ) : null}
      </dl>
      {commands.length ? (
        <div className="workflow-command-list">
          {commands.slice(0, 4).map((candidate) => (
            <div key={candidate.display_command || candidate.command.join(" ")} className="workflow-command-row">
              <code>{candidate.display_command || candidate.command.join(" ")}</code>
              <span>{candidate.requires_approval ? "approval required" : "safe"}</span>
            </div>
          ))}
        </div>
      ) : null}
      {result?.errors?.length ? <pre className="workflow-card-output">{result.errors.join("\n")}</pre> : null}
    </div>
  );
}

function GitApprovalCard({
  metadata,
  onDecision,
}: {
  metadata: AgentRunPayload;
  onDecision?: (metadata: AgentRunPayload, decision: "yes" | "yes_session" | "no_explain") => void;
}) {
  const approval = metadata.git_approval;
  const commandApproval = approval?.command_approval;
  if (!approval || !commandApproval) return null;
  const decisionMetadata = { ...metadata, command_approval: commandApproval };
  return (
    <div className="workflow-card workflow-card-git">
      <div className="workflow-card-heading">{approval.title || "Git approval required"}</div>
      <p>{approval.reason}</p>
      <dl className="command-card-grid">
        <div>
          <dt>Action</dt>
          <dd>{approval.kind}</dd>
        </div>
        {approval.message ? (
          <div>
            <dt>Message</dt>
            <dd>{approval.message}</dd>
          </div>
        ) : null}
        {approval.commit_summary?.validation_status ? (
          <div>
            <dt>Validation</dt>
            <dd>{approval.commit_summary.validation_status}</dd>
          </div>
        ) : null}
        {approval.branch ? (
          <div>
            <dt>Branch</dt>
            <dd>{approval.branch}</dd>
          </div>
        ) : null}
      </dl>
      {approval.files?.length ? (
        <div className="workflow-command-list">
          {approval.files.map((file) => <div key={file} className="workflow-command-row"><code>{file}</code></div>)}
        </div>
      ) : null}
      <div className="command-card-actions">
        <button type="button" onClick={() => onDecision?.(decisionMetadata, "yes")}>Approve</button>
        <button type="button" onClick={() => onDecision?.(decisionMetadata, "no_explain")}>Deny</button>
      </div>
    </div>
  );
}

function CommitSummaryCard({ metadata }: { metadata: AgentRunPayload }) {
  const summary = metadata.git_workflow?.commit_summary;
  if (!summary || metadata.git_approval) return null;
  return (
    <div className="workflow-card workflow-card-git">
      <div className="workflow-card-heading">Commit summary preview</div>
      <dl className="command-card-grid">
        <div>
          <dt>Message</dt>
          <dd>{summary.message}</dd>
        </div>
        <div>
          <dt>Validation</dt>
          <dd>{summary.validation_status || "unknown"}</dd>
        </div>
      </dl>
      {summary.files?.length ? (
        <div className="workflow-command-list">
          {summary.files.map((file) => <div key={file} className="workflow-command-row"><code>{file}</code></div>)}
        </div>
      ) : null}
    </div>
  );
}

function ChangedFilesArchive({ records }: { records?: EditArchiveRecord[] }) {
  const [selected, setSelected] = useState<EditArchiveRecord | null>(null);
  if (!records?.length) return null;
  return (
    <>
      <div className="changed-files-archive">
        <div className="changed-files-title">Changed files</div>
        <div className="changed-files-list">
          {records.map((record) => (
            <button
              key={`${record.proposal_id || record.file_path}-${record.status}`}
              className="changed-file-row"
              type="button"
              title={record.file_path}
              onClick={() => setSelected(record)}
            >
              <div className="changed-file-path">{compactFileName(record.file_path)}</div>
              <div className="changed-file-stats">
                {record.operation ? <span className="changed-file-operation">{record.operation}</span> : null}
                <span className="changed-file-add">+{record.additions}</span>
                <span className="changed-file-del">-{record.deletions}</span>
                <span className={`changed-file-status changed-file-status-${record.status}`}>
                  {record.status}
                </span>
              </div>
            </button>
          ))}
        </div>
      </div>
      {selected ? (
        <aside className="changed-file-drawer" aria-label="Changed file details">
          <div className="changed-file-drawer-card">
            <div className="changed-file-drawer-header">
              <div>
                <div className="changed-file-drawer-title">{compactFileName(selected.file_path)}</div>
                <div className="changed-file-drawer-path">{selected.file_path}</div>
              </div>
              <button type="button" onClick={() => setSelected(null)} aria-label="Close changed file details">×</button>
            </div>
            <div className="changed-file-drawer-stats">
              {selected.operation ? <span className="changed-file-operation">{selected.operation}</span> : null}
              <span className={`changed-file-status changed-file-status-${selected.status}`}>{selected.status}</span>
              <span className="changed-file-add">+{selected.additions}</span>
              <span className="changed-file-del">-{selected.deletions}</span>
            </div>
            {selected.summary ? <p className="changed-file-drawer-summary">{selected.summary}</p> : null}
            {selected.plan_id ? <p className="changed-file-drawer-meta">Related plan: {selected.plan_id}</p> : null}
            {selected.tests?.length ? (
              <div className="changed-file-drawer-tests">
                <span>Suggested tests</span>
                <ul>
                  {selected.tests.map((test) => <li key={test}>{test}</li>)}
                </ul>
              </div>
            ) : null}
            <div className="changed-file-drawer-actions">
              <button type="button" onClick={() => void navigator.clipboard?.writeText(selected.file_path)}>Copy path</button>
              <button type="button" onClick={() => void navigator.clipboard?.writeText(selected.diff || formatChangedFileRecord(selected))}>Copy diff</button>
            </div>
            <div className="proposal-diff-wrapper changed-file-diff-wrapper">
              <div className="proposal-diff-titlebar">
                <span>Proposed diff</span>
                <span className="changed-file-readonly-note">Read-only</span>
              </div>
              <div className="proposal-diff-scroll changed-file-diff-scroll">
                <UnifiedDiffView diff={selected.diff || formatChangedFileRecord(selected)} />
              </div>
            </div>
          </div>
        </aside>
      ) : null}
    </>
  );
}

function compactFileName(path: string): string {
  return path.split(/[\\/]/).filter(Boolean).at(-1) || path;
}

function UnifiedDiffView({ diff }: { diff: string }) {
  const lines = diff.split("\n");
  if (!diff.trim()) {
    return <div className="proposal-diff-empty">No diff is available for this change.</div>;
  }
  return (
    <div className="proposal-diff">
      {lines.map((line, index) => {
        const kind = diffLineKind(line);
        return (
          <div key={`${index}-${line}`} className={`proposal-diff-line proposal-diff-line-${kind}`}>
            <span className="proposal-diff-gutter" aria-hidden="true">
              {kind === "add" ? "+" : kind === "del" ? "−" : " "}
            </span>
            <span className="proposal-diff-text">{line}</span>
          </div>
        );
      })}
    </div>
  );
}

function diffLineKind(line: string): "ctx" | "add" | "del" {
  if (line.startsWith("+") && !line.startsWith("+++")) return "add";
  if (line.startsWith("-") && !line.startsWith("---")) return "del";
  return "ctx";
}

function formatChangedFileRecord(record: EditArchiveRecord): string {
  return [
    `${record.file_path} +${record.additions} -${record.deletions} ${record.status}`,
    record.summary || "",
    record.apply_result || "",
    record.diff || "",
  ].filter(Boolean).join("\n");
}

function ToolCard({ metadata }: { metadata: AgentRunPayload }) {
  const [open, setOpen] = useState(false);
  const filesRead = metadata.files_read ?? [];
  const fileCount = filesRead.length;

  const headerLabel = fileCount > 0
    ? `${fileCount} file${fileCount === 1 ? "" : "s"} read`
    : "Answer trust trace";

  return (
    <div className="tool-card">
      <button
        className="tool-card-header"
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        {headerLabel}
        <span className={`tool-card-caret${open ? " tool-card-caret-open" : ""}`}>▼</span>
      </button>
      {open && (
        <div className="tool-card-body">
          <div className="tool-meta-item">
            <span className="tool-meta-label">Source</span>
            <span className="tool-meta-value">
              {metadata.active_repository_source || metadata.git_provider || "unknown"}
            </span>
          </div>
          <div className="tool-meta-item">
            <span className="tool-meta-label">Project</span>
            <span className="tool-meta-value">
              {metadata.active_repository_path || metadata.project_path}
            </span>
          </div>
          <div className="tool-meta-item">
            <span className="tool-meta-label">Active branch</span>
            <span className="tool-meta-value">
              {metadata.active_branch || metadata.branch || "none"}
            </span>
          </div>
          {filesRead.length > 0 && (
            <div className="tool-meta-item" style={{ gridColumn: "1 / -1" }}>
              <span className="tool-meta-label">Files read</span>
              <ul style={{ margin: "4px 0 0", padding: 0, listStyle: "none" }}>
                {filesRead.map((f) => (
                  <li key={f} className="tool-meta-value" style={{ paddingBottom: "2px" }}>
                    {f}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {metadata.model && (
            <div className="tool-meta-item">
              <span className="tool-meta-label">Model</span>
              <span className="tool-meta-value">{metadata.model}</span>
            </div>
          )}
          {metadata.agent_flow && (
            <div className="tool-meta-item">
              <span className="tool-meta-label">Agent flow</span>
              <span className="tool-meta-value">{metadata.agent_flow}</span>
            </div>
          )}
          {metadata.validation_status && (
            <div className="tool-meta-item">
              <span className="tool-meta-label">Validation</span>
              <span className="tool-meta-value">{metadata.validation_status}</span>
            </div>
          )}
          {metadata.run_id && (
            <div className="tool-meta-item">
              <span className="tool-meta-label">Run ID</span>
              <span className="tool-meta-value">{metadata.run_id}</span>
            </div>
          )}
          {metadata.skills_used?.length ? (
            <div className="tool-meta-item">
              <span className="tool-meta-label">Skills used</span>
              <span className="tool-meta-value">{metadata.skills_used.join(", ")}</span>
            </div>
          ) : null}
          {metadata.selected_target_file && (
            <div className="tool-meta-item">
              <span className="tool-meta-label">Target file</span>
              <span className="tool-meta-value">{metadata.selected_target_file}</span>
            </div>
          )}
          {metadata.context_source && (
            <div className="tool-meta-item">
              <span className="tool-meta-label">Context source</span>
              <span className="tool-meta-value">{metadata.context_source}</span>
            </div>
          )}
          {metadata.context_reference_resolver && (
            <div className="tool-meta-item">
              <span className="tool-meta-label">Reference resolver</span>
              <span className="tool-meta-value">{metadata.context_reference_resolver}</span>
            </div>
          )}
          {metadata.resolved_reference_type && (
            <div className="tool-meta-item">
              <span className="tool-meta-label">Reference type</span>
              <span className="tool-meta-value">{metadata.resolved_reference_type}</span>
            </div>
          )}
          {metadata.reference_confidence !== undefined && metadata.reference_confidence !== null && (
            <div className="tool-meta-item">
              <span className="tool-meta-label">Reference confidence</span>
              <span className="tool-meta-value">{Math.round(metadata.reference_confidence * 100)}%</span>
            </div>
          )}
          {metadata.resolved_files?.length ? (
            <div className="tool-meta-item" style={{ gridColumn: "1 / -1" }}>
              <span className="tool-meta-label">Resolved files</span>
              <span className="tool-meta-value">{metadata.resolved_files.join(", ")}</span>
            </div>
          ) : null}
          {metadata.resolved_symbols?.length ? (
            <div className="tool-meta-item" style={{ gridColumn: "1 / -1" }}>
              <span className="tool-meta-label">Resolved symbols</span>
              <span className="tool-meta-value">{metadata.resolved_symbols.join(", ")}</span>
            </div>
          ) : null}
          {metadata.commands_run?.length ? (
            <div className="tool-meta-item" style={{ gridColumn: "1 / -1" }}>
              <span className="tool-meta-label">Commands run</span>
              <span className="tool-meta-value">{metadata.commands_run.join(", ")}</span>
            </div>
          ) : null}
          {metadata.commands_planned?.length ? (
            <div className="tool-meta-item" style={{ gridColumn: "1 / -1" }}>
              <span className="tool-meta-label">Commands planned</span>
              <span className="tool-meta-value">{metadata.commands_planned.join(", ")}</span>
            </div>
          ) : null}
          {metadata.recommendation_context_loaded ? (
            <div className="tool-meta-item">
              <span className="tool-meta-label">Recommendation context</span>
              <span className="tool-meta-value">loaded</span>
            </div>
          ) : null}
          {metadata.selected_recommendation_ids?.length ? (
            <div className="tool-meta-item" style={{ gridColumn: "1 / -1" }}>
              <span className="tool-meta-label">Selected recommendations</span>
              <span className="tool-meta-value">{metadata.selected_recommendation_ids.join(", ")}</span>
            </div>
          ) : null}
          {metadata.plan_steps?.length ? (
            <div className="tool-meta-item" style={{ gridColumn: "1 / -1" }}>
              <span className="tool-meta-label">Plan steps</span>
              <span className="tool-meta-value">{metadata.plan_steps.join(" → ")}</span>
            </div>
          ) : null}
          {metadata.thread_context_files?.length ? (
            <div className="tool-meta-item" style={{ gridColumn: "1 / -1" }}>
              <span className="tool-meta-label">Thread context files</span>
              <span className="tool-meta-value">{metadata.thread_context_files.join(", ")}</span>
            </div>
          ) : null}
          {metadata.thread_context_symbols?.length ? (
            <div className="tool-meta-item" style={{ gridColumn: "1 / -1" }}>
              <span className="tool-meta-label">Thread context symbols</span>
              <span className="tool-meta-value">{metadata.thread_context_symbols.join(", ")}</span>
            </div>
          ) : null}
          {metadata.branch && (
            <div className="tool-meta-item">
              <span className="tool-meta-label">Branch</span>
              <span className="tool-meta-value">{metadata.branch}</span>
            </div>
          )}
          {metadata.repo_root_name && (
            <div className="tool-meta-item">
              <span className="tool-meta-label">Repository</span>
              <span className="tool-meta-value">{metadata.repo_root_name}</span>
            </div>
          )}
          <div className="tool-meta-item">
            <span className="tool-meta-label">Git repo</span>
            <span className="tool-meta-value">
              {metadata.is_git_repository ? "yes" : "no"}
            </span>
          </div>
          {metadata.context_summary && (
            <div className="tool-meta-item" style={{ gridColumn: "1 / -1" }}>
              <span className="tool-meta-label">Context summary</span>
              <span
                className="tool-meta-value"
                style={{ fontFamily: "inherit", fontSize: "0.86rem", whiteSpace: "normal" }}
              >
                {metadata.context_summary}
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function formatTime(date: Date): string {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

interface ChatMessagesProps {
  messages: ChatMessage[];
  repoResult: RepoOpenPayload | null;
  questionPending: boolean;
  progressSteps?: ProgressStep[];
  streamedAnswer?: string;
  gitProvider: string;
  writeMode?: PermissionMode;
  onProposalStatusChange?: (id: string, status: ProposalStatus, message?: string, result?: AgentRunPayload) => void;
  onClarificationSelect?: (candidate: string) => void;
  onCommandDecision?: (metadata: AgentRunPayload, decision: "yes" | "yes_session" | "no_explain") => void;
}

export function ChatMessages({
  messages,
  repoResult,
  questionPending,
  progressSteps = [],
  streamedAnswer = "",
  gitProvider,
  writeMode = "default",
  onProposalStatusChange,
  onClarificationSelect,
  onCommandDecision,
}: ChatMessagesProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, questionPending, streamedAnswer]);

  const activeProvider = repoResult?.git_provider || gitProvider;
  const providerLabel =
    activeProvider === "local"
      ? "Local project"
      : activeProvider === "gitlab"
        ? "GitLab"
        : "GitHub";

  function speakerLabel(message: ChatMessage): string {
    return message.role === "user"
      ? "You"
      : message.role === "system"
        ? "Context"
        : "RepoOperator";
  }

  function formatMessageForCopy(message: ChatMessage): string {
    const lines = [
      `[${message.timestamp.toLocaleString()}] ${speakerLabel(message)}`,
      message.content,
    ];
    if (message.proposal) {
      lines.push(`Proposal: ${message.proposal.relativePath}`);
      lines.push(`Status: ${message.proposal.status}`);
    }
    if (message.metadata?.edit_archive?.length) {
      lines.push("Changed files:");
      for (const record of message.metadata.edit_archive) {
        lines.push(
          `- ${record.file_path} +${record.additions} -${record.deletions} (${record.status})${record.summary ? `: ${record.summary}` : ""}`,
        );
      }
    }
    if (message.metadata?.selected_target_file) {
      lines.push(`Target file: ${message.metadata.selected_target_file}`);
    }
    if (message.metadata?.clarification_candidates?.length) {
      lines.push(`Candidates: ${message.metadata.clarification_candidates.join(", ")}`);
    }
    if (message.metadata?.proposal_error_details) {
      lines.push(`Error: ${message.metadata.proposal_error_details}`);
    }
    return lines.filter(Boolean).join("\n");
  }

  function formatChatForCopy(): string {
    return messages.map(formatMessageForCopy).join("\n\n---\n\n");
  }

  async function copyText(text: string, id: string) {
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text);
      } else {
        const textarea = document.createElement("textarea");
        textarea.value = text;
        textarea.style.position = "fixed";
        textarea.style.opacity = "0";
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand("copy");
        document.body.removeChild(textarea);
      }
      setCopiedId(id);
      window.setTimeout(() => setCopiedId(null), 1800);
    } catch {
      setCopiedId(id);
      window.setTimeout(() => setCopiedId(null), 1800);
    }
  }

  return (
    <div className="chat-body">
      {repoResult && (
        <div className="repo-banner">
          <div className="repo-banner-icon" aria-hidden="true" />
          <div className="repo-banner-content">
            <div className="repo-banner-title" style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
              <span>Repository ready</span>
              {repoResult.branch ? (
                <span style={{ opacity: 0.75 }}>· {repoResult.branch}</span>
              ) : null}
            </div>
            <div className="repo-banner-detail">
              {repoResult.local_repo_path} · {providerLabel}
              {repoResult.head_sha ? ` · ${repoResult.head_sha.slice(0, 8)}` : ""}
            </div>
          </div>
        </div>
      )}

      {messages.length > 0 && (
        <div className="chat-copy-bar">
          <button
            className="message-copy-btn"
            type="button"
            onClick={() => void copyText(formatChatForCopy(), "chat")}
          >
            {copiedId === "chat" ? "Copied" : "Copy chat"}
          </button>
        </div>
      )}

      {messages.length === 0 && !questionPending ? (
        <div className="chat-empty">
          <div className="chat-empty-icon" aria-hidden="true" />
          <h2>RepoOperator</h2>
          <p>
            {repoResult
              ? "Repository is open. Ask a question about the codebase below."
              : "Select a repository above and click Open repository, then ask questions."}
          </p>
        </div>
      ) : (
        <>
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`message-group message-group-${msg.role}`}
              data-testid={msg.role === "assistant" ? "assistant-message" : undefined}
            >
              <div className="message-meta-row">
                <span className="message-role-label">{speakerLabel(msg)}</span>
                <button
                  className="message-copy-btn"
                  type="button"
                  onClick={() => void copyText(formatMessageForCopy(msg), msg.id)}
                >
                  {copiedId === msg.id ? "Copied" : "Copy message"}
                </button>
              </div>
              {msg.proposal ? (
                <>
                  {msg.progressSteps && msg.progressSteps.length > 0 ? (
                    <AgentActivityTranscript steps={msg.progressSteps} done={true} />
                  ) : null}
                  <ChangedFilesArchive records={msg.metadata?.edit_archive} />
                  <ProposalCard
                    proposal={msg.proposal}
                    writeMode={writeMode}
                    onStatusChange={onProposalStatusChange ?? (() => {})}
                  />
                  {msg.metadata ? (
                    <>
                      <ValidationResultCard metadata={msg.metadata} />
                      <CommitSummaryCard metadata={msg.metadata} />
                      <GitApprovalCard metadata={msg.metadata} onDecision={onCommandDecision} />
                    </>
                  ) : null}
                </>
              ) : msg.role === "assistant" && msg.metadata?.response_type === "permission_required" ? (
                <div className="message-bubble message-bubble-permission">
                  <div className="permission-callout">
                    <span className="permission-callout-icon" aria-hidden="true" />
                    <div>
                      <div className="permission-callout-title">Write permission required</div>
                      <div className="permission-callout-body">{msg.content}</div>
                    </div>
                  </div>
                </div>
              ) : msg.role === "assistant" && msg.metadata?.response_type === "clarification" ? (
                <div className="message-bubble message-bubble-md">
                  <MarkdownContent content={msg.content} />
                  {msg.metadata.clarification_candidates?.length ? (
                    <div className="clarification-options">
                      {msg.metadata.clarification_candidates.map((candidate) => (
                        <button
                          key={candidate}
                          className="clarification-option"
                          type="button"
                          onClick={() => onClarificationSelect?.(candidate)}
                        >
                          {candidate}
                        </button>
                      ))}
                    </div>
                  ) : null}
                </div>
              ) : msg.role === "assistant" && msg.metadata?.response_type === "command_approval" ? (
                <CommandApprovalCard metadata={msg.metadata} onDecision={onCommandDecision} />
              ) : msg.role === "assistant" && msg.metadata?.response_type === "git_approval" ? (
                <GitApprovalCard metadata={msg.metadata} onDecision={onCommandDecision} />
              ) : msg.role === "assistant" && msg.metadata?.response_type === "command_result" && msg.metadata.command_result ? (
                <CommandResultCard result={msg.metadata.command_result as CommandResultPayload} />
              ) : msg.role === "assistant" && msg.metadata?.response_type === "command_denied" ? (
                <CommandApprovalCard metadata={msg.metadata} onDecision={onCommandDecision} />
              ) : msg.role === "assistant" && msg.metadata?.response_type === "proposal_error" ? (
                <div className="proposal-error-card">
                  <div className="proposal-error-title">No valid diff produced</div>
                  <p>{msg.content}</p>
                  {msg.metadata.proposal_error_details && (
                    <details>
                      <summary>View details</summary>
                      <pre>{msg.metadata.proposal_error_details}</pre>
                    </details>
                  )}
                  <button className="proposal-btn-reject" type="button" onClick={() => onClarificationSelect?.(msg.content)}>
                    Retry with more detail
                  </button>
                </div>
              ) : msg.role === "assistant" ? (
                <div className="message-bubble message-bubble-md">
                  {msg.progressSteps && msg.progressSteps.length > 0 ? (
                    <AgentActivityTranscript steps={msg.progressSteps} done={true} />
                  ) : null}
                  <ChangedFilesArchive records={msg.metadata?.edit_archive} />
                  {msg.metadata ? (
                    <>
                      <ValidationResultCard metadata={msg.metadata} />
                      <CommitSummaryCard metadata={msg.metadata} />
                    </>
                  ) : null}
                  <MarkdownContent content={msg.content} />
                </div>
              ) : msg.role === "system" ? (
                <div className="message-bubble message-bubble-system">{msg.content}</div>
              ) : (
                <div className="message-bubble">{msg.content}</div>
              )}
              {msg.metadata && <ToolCard metadata={msg.metadata} />}
              <span className="message-timestamp">{formatTime(msg.timestamp)}</span>
            </div>
          ))}

          {questionPending && (
            <div className="message-group message-group-assistant">
              <span className="message-role-label">RepoOperator</span>
              {progressSteps.length > 0 ? (
                <>
                  <AgentActivityTranscript steps={progressSteps} done={false} />
                  {streamedAnswer ? (
                    <div className="message-bubble message-bubble-md">
                      <MarkdownContent content={streamedAnswer} />
                    </div>
                  ) : null}
                </>
              ) : (
                <div className="typing-indicator">
                  <span className="typing-dot" />
                  <span className="typing-dot" />
                  <span className="typing-dot" />
                </div>
              )}
            </div>
          )}
        </>
      )}

      <div ref={bottomRef} />
    </div>
  );
}
