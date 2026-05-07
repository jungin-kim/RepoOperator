import type {
  AgentActivityEvent,
  AgentRunPayload,
  AgentRunRecord,
} from "@/lib/local-worker-client";
import type { ChatMessage } from "./ChatMessages";
import type { ProgressStep } from "./ProgressTimeline";

export type AgentRunEvent = AgentActivityEvent & {
  type?: string;
  delta?: string;
  result?: AgentRunPayload;
  message?: string;
};

const ACTIVE_RUN_STATUSES = new Set(["pending", "running", "waiting_approval", "cancelling"]);
const TERMINAL_RUN_STATUSES = new Set(["completed", "failed", "cancelled", "timed_out"]);

export function isActiveRunStatus(status?: string | null): boolean {
  return ACTIVE_RUN_STATUSES.has(String(status || ""));
}

export function isTerminalRunStatus(status?: string | null): boolean {
  return TERMINAL_RUN_STATUSES.has(String(status || ""));
}

export function getMessageKeyForRun(runId: string): string {
  return `assistant-run:${runId}`;
}

export function progressStepFromEvent(
  event: AgentRunEvent,
  options: { finalizeRunning?: boolean } = {},
): ProgressStep {
  const status =
    options.finalizeRunning && event.status === "running" ? "completed" : event.status;
  return {
    id: event.id,
    activityId: event.activity_id,
    runId: event.run_id,
    sequence: event.sequence,
    eventType: event.event_type,
    visibility: event.visibility,
    display: event.display,
    phase: event.phase,
    label: event.label ?? event.message,
    detail: event.detail,
    detailDelta: event.detail_delta,
    message: event.message,
    safeReasoningSummary: event.safe_reasoning_summary,
    summaryDelta: event.summary_delta ?? event.safe_reasoning_summary_delta,
    evidenceNeeded: event.evidence_needed,
    uncertainty: event.uncertainty,
    safetyNote: event.safety_note,
    currentAction: event.current_action,
    observation: event.observation,
    observationDelta: event.observation_delta,
    nextAction: event.next_action,
    nextActionDelta: event.next_action_delta,
    relatedSearchQuery: event.related_search_query,
    aggregate: event.aggregate,
    status,
    startedAt: event.started_at,
    endedAt: event.ended_at,
    durationMs: event.duration_ms,
    elapsedMs: event.elapsed_ms,
    files: event.files,
    command: event.command ?? event.related_command,
    proposalId: event.proposal_id,
  };
}

export function mergeRunEventsIntoProgressSteps(
  events?: AgentRunEvent[],
  finalResult?: AgentRunPayload | null,
  options: { finalizeRunning?: boolean } = {},
): ProgressStep[] {
  let steps: ProgressStep[] = [];
  for (const event of events || []) {
    if (event.type !== "progress_delta" || !hasProgressStepContent(event)) continue;
    steps = mergeProgressStep(steps, progressStepFromEvent(event, options));
  }
  if (steps.length === 0 && finalResult?.activity_events?.length) {
    for (const event of finalResult.activity_events) {
      if (event.type !== "progress_delta" || !hasProgressStepContent(event)) continue;
      steps = mergeProgressStep(steps, progressStepFromEvent(event as AgentRunEvent, options));
    }
  }
  steps = attachFinalResultEditArchive(steps, finalResult, options);
  return options.finalizeRunning
    ? steps.map((step) => (step.status === "running" ? { ...step, status: "completed" } : step))
    : steps;
}

export function progressStepsForCompletedRun(
  events?: AgentRunEvent[],
  finalResult?: AgentRunPayload | null,
): ProgressStep[] {
  return mergeRunEventsIntoProgressSteps(events, finalResult, { finalizeRunning: true });
}

export function assistantTextFromRunEvents(
  events?: AgentRunEvent[],
  finalResult?: AgentRunPayload | null,
): string {
  const finalFromEvents = finalResultFromRunEvents(events, finalResult);
  if (finalFromEvents?.response) return finalFromEvents.response;
  return (events || [])
    .filter((event) => event.type === "assistant_delta")
    .map((event) => String(event.delta || ""))
    .join("");
}

export function finalResultFromRunEvents(
  events?: AgentRunEvent[],
  finalResult?: AgentRunPayload | null,
): AgentRunPayload | null {
  if (finalResult?.response) return finalResult;
  for (const event of [...(events || [])].reverse()) {
    if (event.type === "final_message" && event.result) return event.result;
  }
  return finalResult || null;
}

export function upsertAssistantMessageForRun(
  messages: ChatMessage[],
  runId: string,
  patch: Partial<ChatMessage> & { content?: string },
): ChatMessage[] {
  const messageKey = getMessageKeyForRun(runId);
  const existingIndex = messages.findIndex(
    (message) => message.id === messageKey || message.metadata?.run_id === runId,
  );
  if (existingIndex >= 0) {
    return messages.map((message, index) =>
      index === existingIndex
        ? {
            ...message,
            ...patch,
            id: message.id || messageKey,
            role: "assistant",
            content: patch.content ?? message.content,
            timestamp: patch.timestamp ?? message.timestamp,
            metadata: patch.metadata ?? message.metadata,
            progressSteps: patch.progressSteps ?? message.progressSteps,
          }
        : message,
    );
  }
  return [
    ...messages,
    {
      id: messageKey,
      role: "assistant",
      content: patch.content || "",
      timestamp: patch.timestamp || new Date(),
      metadata: patch.metadata,
      proposal: patch.proposal,
      progressSteps: patch.progressSteps,
    },
  ];
}

export function mergeProgressStep(current: ProgressStep[], incoming: ProgressStep): ProgressStep[] {
  if (!hasProgressStepContent(incoming)) return current;
  const incomingKey = progressStepIdentity(incoming, current.length);
  const existingIndex = current.findIndex((step, index) => progressStepIdentity(step, index) === incomingKey);
  if (existingIndex >= 0) {
    return current.map((step, index) => (index === existingIndex ? mergeProgressStepFields(step, incoming) : step));
  }
  return [...current, incoming];
}

function hasProgressStepContent(event: Partial<AgentRunEvent & ProgressStep>): boolean {
  return Boolean(
    event.label
      || event.message
      || event.safe_reasoning_summary
      || event.safeReasoningSummary
      || event.current_action
      || event.currentAction
      || event.observation
      || event.next_action
      || event.nextAction
      || event.related_search_query
      || event.relatedSearchQuery
      || event.safety_note
      || event.safetyNote
      || event.command
      || event.related_command
      || event.files?.length
      || event.proposal_id
      || event.proposalId
      || (event.aggregate && Object.keys(event.aggregate).length > 0),
  );
}

export function maxEventSequence(events?: AgentRunEvent[]): number {
  return Math.max(0, ...(events || []).map((event) => Number(event.sequence || 0)));
}

export function runIdForThread(activeRuns: AgentRunRecord[], threadId: string): string | null {
  return activeRuns.find((run) => run.thread_id === threadId && isActiveRunStatus(run.status))?.id || null;
}

function progressStepIdentity(step: ProgressStep, fallbackIndex: number): string {
  if (step.runId && step.activityId) return `${step.runId}:${step.activityId}`;
  if (step.runId && step.eventType && step.label) return `${step.runId}:${step.eventType}:${step.label}`;
  if (step.runId && step.sequence !== undefined && step.sequence !== null) return `${step.runId}:${step.sequence}`;
  if (step.runId && step.id) return `${step.runId}:${step.id}`;
  if (step.id) return step.id;
  return `${step.runId || "local"}:${step.startedAt || fallbackIndex}:${step.phase || ""}:${step.label || step.message || ""}`;
}

function mergeProgressStepFields(existing: ProgressStep, incoming: ProgressStep): ProgressStep {
  return {
    ...existing,
    ...incoming,
    startedAt: existing.startedAt || incoming.startedAt,
    detail: incoming.detail ?? appendDelta(existing.detail, incoming.detailDelta),
    observation: incoming.observation ?? appendDelta(existing.observation, incoming.observationDelta),
    nextAction: incoming.nextAction ?? appendDelta(existing.nextAction, incoming.nextActionDelta),
    safeReasoningSummary: incoming.safeReasoningSummary ?? appendDelta(existing.safeReasoningSummary, incoming.summaryDelta),
    evidenceNeeded: mergeStringLists(existing.evidenceNeeded, incoming.evidenceNeeded),
    uncertainty: mergeStringLists(existing.uncertainty, incoming.uncertainty),
    safetyNote: incoming.safetyNote ?? existing.safetyNote,
    files: mergeStringLists(existing.files, incoming.files),
    command: incoming.command ?? existing.command,
    proposalId: incoming.proposalId ?? existing.proposalId,
    aggregate: mergeAggregate(existing.aggregate, incoming.aggregate),
  };
}

function appendDelta(base?: string | null, delta?: string | null): string | undefined {
  const current = base || "";
  if (!delta) return current || undefined;
  return `${current}${delta}`;
}

function mergeStringLists(existing?: string[], incoming?: string[]): string[] | undefined {
  const merged: string[] = [];
  for (const item of [...(existing || []), ...(incoming || [])]) {
    if (item && !merged.includes(item)) merged.push(item);
  }
  return merged.length ? merged : undefined;
}

function mergeAggregate(
  existing?: Record<string, unknown> | null,
  incoming?: Record<string, unknown> | null,
): Record<string, unknown> | null | undefined {
  if (!existing && !incoming) return incoming ?? existing;
  return {
    ...(existing || {}),
    ...(incoming || {}),
  };
}

function attachFinalResultEditArchive(
  steps: ProgressStep[],
  finalResult?: AgentRunPayload | null,
  options: { finalizeRunning?: boolean } = {},
): ProgressStep[] {
  const archive = finalResult?.edit_archive || [];
  if (!archive.length) return steps;

  const files = archive.map((record) => record.file_path).filter(Boolean);
  const archiveAggregate = {
    action_type: "generate_edit",
    edit_archive: archive,
    additions: sumEditArchiveField(archive, "additions"),
    deletions: sumEditArchiveField(archive, "deletions"),
    diff_available: archive.some((record) => Boolean(record.diff)),
    status: commonArchiveStatus(archive),
    proposal_id: finalResult?.proposal_relative_path || archive[0]?.proposal_id || null,
  };
  const safetyNote = proposalOnlySafetyNote(finalResult);
  let attached = false;
  const next = steps.map((step) => {
    const aggregate = step.aggregate || {};
    const actionType = typeof aggregate.action_type === "string" ? aggregate.action_type : null;
    const stepFiles = step.files || [];
    const intersectsArchive = stepFiles.some((file) => files.includes(file));
    const isEditStep =
      actionType === "generate_edit"
      || step.eventType === "file_edit"
      || (String(step.phase || "").toLowerCase().includes("editing") && intersectsArchive);
    if (!isEditStep) return step;
    attached = true;
    return {
      ...step,
      status: options.finalizeRunning && step.status === "running" ? "completed" : step.status,
      files: mergeStringLists(step.files, files),
      proposalId: step.proposalId || archiveAggregate.proposal_id || undefined,
      safetyNote: step.safetyNote || safetyNote,
      aggregate: {
        ...archiveAggregate,
        ...aggregate,
        edit_archive: archive,
        action_type: "generate_edit",
      },
    };
  });
  if (attached) return next;
  return [
    ...next,
    {
      id: `edit-archive:${finalResult?.run_id || "completed"}`,
      activityId: `edit-archive:${finalResult?.run_id || "completed"}`,
      runId: finalResult?.run_id || undefined,
      eventType: "work_trace",
      phase: "Editing",
      label: "Prepared proposed changes",
      status: "completed",
      files,
      proposalId: archiveAggregate.proposal_id || undefined,
      safetyNote,
      aggregate: archiveAggregate,
    },
  ];
}

function sumEditArchiveField(
  archive: NonNullable<AgentRunPayload["edit_archive"]>,
  field: "additions" | "deletions",
): number | undefined {
  const values = archive.map((record) => record[field]).filter((value) => typeof value === "number");
  return values.length ? values.reduce((sum, value) => sum + value, 0) : undefined;
}

function commonArchiveStatus(archive: NonNullable<AgentRunPayload["edit_archive"]>): string | undefined {
  const statuses = archive.map((record) => record.status).filter(Boolean);
  return statuses.length > 0 && statuses.every((status) => status === statuses[0]) ? statuses[0] : undefined;
}

function proposalOnlySafetyNote(finalResult?: AgentRunPayload | null): string | null {
  if (
    finalResult?.response_type === "change_proposal"
    || finalResult?.edit_archive?.some((record) => record.status === "proposed")
  ) {
    return "Proposal only. No files were written.";
  }
  return null;
}
