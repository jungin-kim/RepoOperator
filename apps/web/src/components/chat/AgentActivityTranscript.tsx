"use client";

import { useEffect, useState } from "react";
import type { ProgressStep } from "./ProgressTimeline";
import type {
  AgentActivityDetailItem,
  AgentEditSummaryItem,
  AgentTranscriptSection,
} from "./agent-activity-types";
import { buildAgentActivityTranscript } from "./agent-activity-transcript";
import { compactWorkTraceSteps } from "./work-trace-display";

type Props = {
  steps: ProgressStep[];
  done: boolean;
};

function formatDuration(ms: number): string {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  if (totalSeconds < 60) return `${totalSeconds}s`;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}m ${String(seconds).padStart(2, "0")}s`;
}

function dotClass(status: string): string {
  if (status === "running") return "agent-dot-running";
  if (status === "failed") return "agent-dot-failed";
  if (status === "waiting" || status === "waiting_approval") return "agent-dot-waiting";
  return "agent-dot-done";
}

function statusText(status: string, exitCode?: number | null): string {
  if (typeof exitCode === "number") return `exit ${exitCode}`;
  if (status === "waiting_approval") return "waiting";
  if (status === "completed") return "completed";
  return status || "done";
}

function formatEditCounts(additions?: number, deletions?: number): string | null {
  const parts: string[] = [];
  if (typeof additions === "number") parts.push(`+${additions}`);
  if (typeof deletions === "number") parts.push(`-${deletions}`);
  return parts.length ? parts.join(" ") : null;
}

function DetailItemRow({ item }: { item: AgentActivityDetailItem }) {
  let icon: string;
  let desc: string;
  let status = statusText(item.status);

  if (item.kind === "search") {
    icon = "⟳";
    desc = item.query ? `search: ${item.query}` : item.label;
  } else if (item.kind === "read_file") {
    icon = "↳";
    desc = item.files.length > 0 ? item.files.join(", ") : item.label;
  } else if (item.kind === "command") {
    icon = "❯";
    desc = item.command;
    status = statusText(item.status, item.exitCode);
  } else if (item.kind === "edit") {
    icon = "✎";
    desc = item.files.length > 0 ? item.files.join(", ") : item.label;
  } else {
    icon = "·";
    desc = item.path ? `list: ${item.path}` : item.label;
  }

  return (
    <li className={`agent-detail-item agent-detail-${item.kind} agent-detail-${dotClass(item.status)}`}>
      <span className="agent-detail-icon" aria-hidden="true">{icon}</span>
      <span className="agent-detail-desc">{desc}</span>
      <span className="agent-detail-status">{status}</span>
    </li>
  );
}

function EditSummaryRow({ item }: { item: AgentEditSummaryItem }) {
  const counts = formatEditCounts(item.additions, item.deletions);
  return (
    <li className={`agent-detail-item agent-detail-edit agent-detail-${dotClass(item.status)}`}>
      <span className="agent-detail-icon" aria-hidden="true">✎</span>
      <span className="agent-detail-desc">
        {item.path}
        {counts ? ` ${counts}` : ""}
        {item.summary ? ` — ${item.summary}` : ""}
      </span>
      <span className="agent-detail-status">{statusText(item.status)}</span>
      {item.safetyNote ? <span className="agent-detail-safety">{item.safetyNote}</span> : null}
    </li>
  );
}

function TranscriptSectionView({ section }: { section: AgentTranscriptSection }) {
  const [collapsed, setCollapsed] = useState(section.collapsedByDefault);
  const dc = dotClass(section.status);
  const hasRows = section.details.length > 0 || section.edits.length > 0;
  const detailsVisible = section.isCurrent || !collapsed;

  useEffect(() => {
    setCollapsed(section.collapsedByDefault);
  }, [section.id, section.collapsedByDefault]);

  return (
    <div className={`agent-transcript-section agent-transcript-section-${dc}${section.isCurrent ? " agent-transcript-section-current" : ""}`}>
      <div className={`agent-status-note agent-status-note-${dc}`}>
        <span className={`agent-dot ${dc}`} aria-hidden="true" />
        <div className="agent-status-note-body">
          <span className="agent-status-note-text">{section.statusText}</span>
        </div>
      </div>
      {hasRows ? (
        section.collapsible ? (
          <button
            type="button"
            className="agent-section-summary"
            onClick={() => setCollapsed((value) => !value)}
            aria-expanded={!collapsed}
          >
            <span>{section.summaryText}</span>
            <span className="agent-group-caret" aria-hidden="true">{collapsed ? "▼" : "▲"}</span>
          </button>
        ) : (
          <div className="agent-section-summary agent-section-summary-live">
            <span>{section.summaryText}</span>
          </div>
        )
      ) : null}
      {hasRows && detailsVisible ? (
        <ul className="agent-section-details">
          {section.details.map((detail) => (
            <DetailItemRow key={detail.id} item={detail} />
          ))}
          {section.edits.map((edit) => (
            <EditSummaryRow key={edit.id} item={edit} />
          ))}
        </ul>
      ) : null}
    </div>
  );
}

export function AgentActivityTranscript({ steps, done }: Props) {
  const [showTechnicalLog, setShowTechnicalLog] = useState(false);
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    if (done) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [done]);

  if (steps.length === 0) return null;

  const sections = buildAgentActivityTranscript(steps, { finalizeRunning: done });
  const primarySteps = compactWorkTraceSteps(steps);

  if (sections.length === 0 && primarySteps.length === 0) return null;

  const totalMs = runDurationMs(steps, done, now);
  const hasHiddenTechnicalSteps = steps.length > primarySteps.length;

  return (
    <section className="agent-transcript" aria-label="Agent activity">
      <div className="agent-transcript-header" title="Agent activity">
        <span>
          {done
            ? `Worked for ${formatDuration(totalMs)}`
            : `Agent Activity · ${formatDuration(totalMs)}`}
        </span>
      </div>
      {sections.length > 0 && (
        <div className="agent-transcript-items">
          {sections.map((section) => (
            <TranscriptSectionView key={section.id} section={section} />
          ))}
        </div>
      )}
      {hasHiddenTechnicalSteps && (
        <div className="agent-transcript-log">
          <button
            type="button"
            className="agent-transcript-log-toggle"
            onClick={() => setShowTechnicalLog((v) => !v)}
          >
            {showTechnicalLog ? "Hide technical log" : "Show technical log"}
          </button>
          {showTechnicalLog && (
            <div className="agent-transcript-log-events">
              {steps.map((step, index) => (
                <div
                  className="agent-log-event"
                  key={`log-${step.activityId || step.id || index}`}
                >
                  <span>{step.phase || "Step"}</span>
                  <strong>{step.label || step.message || step.eventType || "Event"}</strong>
                  <small>{step.status || "done"}</small>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </section>
  );
}

function runDurationMs(steps: ProgressStep[], done: boolean, now: number): number {
  const starts = steps
    .map((step) => (step.startedAt ? Date.parse(step.startedAt) : Number.NaN))
    .filter(Number.isFinite);
  if (!starts.length) {
    return steps[steps.length - 1]?.elapsedMs ?? steps[steps.length - 1]?.durationMs ?? 0;
  }
  const firstStarted = Math.min(...starts);
  if (!done) return Math.max(0, now - firstStarted);
  const ends = steps
    .map((step) => (step.endedAt ? Date.parse(step.endedAt) : Number.NaN))
    .filter(Number.isFinite);
  if (ends.length) return Math.max(0, Math.max(...ends) - firstStarted);
  return steps[steps.length - 1]?.elapsedMs ?? steps[steps.length - 1]?.durationMs ?? 0;
}
