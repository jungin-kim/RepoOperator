import type { ProgressStep } from "./progress-types";
import type {
  AgentActivityDetailItem,
  AgentEditSummaryItem,
  AgentTranscriptSection,
  EditFileSummary,
  ReadFileDetailItem,
} from "./agent-activity-types";
import { isLowValuePrimaryLabel } from "./agent-activity-display";

const SEARCH_TYPES = new Set(["search_files", "search_text"]);
const READ_TYPES = new Set(["read_file"]);
const LIST_TYPES = new Set(["inspect_repo_tree", "analyze_repository"]);
const COMMAND_TYPES = new Set(["preview_command", "inspect_git_state", "run_approved_command"]);
const EDIT_TYPES = new Set(["generate_edit"]);
const SKIP_TYPES = new Set(["final_answer", "ask_clarification"]);

function aggregateOf(step: ProgressStep): Record<string, unknown> {
  return step.aggregate && typeof step.aggregate === "object" ? step.aggregate : {};
}

function getActionType(step: ProgressStep): string | null {
  const agg = aggregateOf(step);
  const actionType = stringValue(agg.action_type) || stringValue(agg.tool);
  if (actionType) return actionType;

  if (step.eventType === "file_read" || isReadEvent(step, agg)) return "read_file";
  if (step.eventType === "file_edit" || isEditEvent(step, agg)) return "generate_edit";
  if (hasSearchAggregate(agg)) {
    return agg.query ? "search_text" : "search_files";
  }
  if (hasListAggregate(step, agg)) return "inspect_repo_tree";
  if (hasCommandAggregate(step, agg)) return "run_approved_command";
  return null;
}

function hasSearchAggregate(agg: Record<string, unknown>): boolean {
  return Boolean(
    nonEmptyString(agg.query)
      || nonEmptyStringList(agg.queries).length
      || nonEmptyStringList(agg.text_queries).length,
  );
}

function hasListAggregate(step: ProgressStep, agg: Record<string, unknown>): boolean {
  return Boolean(
    typeof agg.entries_count === "number"
    || nonEmptyString(agg.path)
    || String(step.label || "").toLowerCase() === "inspect repository tree"
  );
}

function hasCommandAggregate(step: ProgressStep, agg: Record<string, unknown>): boolean {
  return Boolean(
    step.command
      || nonEmptyString(agg.display_command)
      || nonEmptyString(agg.command)
      || typeof agg.exit_code === "number"
      || typeof agg.returncode === "number",
  );
}

function isReadEvent(step: ProgressStep, agg: Record<string, unknown>): boolean {
  return (
    String(step.phase || "").toLowerCase().includes("reading")
    && (Boolean(step.files?.length) || Boolean(nonEmptyString(agg.file_path)))
  );
}

function isEditEvent(step: ProgressStep, agg: Record<string, unknown>): boolean {
  return Boolean(
    nonEmptyStringList(agg.edit_archive).length
      || Array.isArray(agg.edit_archive)
      || nonEmptyString(agg.edit_summary)
      || typeof agg.additions === "number"
      || typeof agg.deletions === "number"
      || typeof agg.diff_available === "boolean"
      || String(step.phase || "").toLowerCase().includes("editing"),
  );
}

function stringValue(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function nonEmptyString(value: unknown): string | null {
  return stringValue(value);
}

function numberValue(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function booleanValue(value: unknown): boolean | undefined {
  return typeof value === "boolean" ? value : undefined;
}

function nonEmptyStringList(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.map((item) => String(item || "").trim()).filter(Boolean);
}

function fmtCommand(cmd: unknown): string {
  if (!cmd) return "";
  if (Array.isArray(cmd)) return cmd.map((item) => String(item)).join(" ");
  return String(cmd);
}

function meaningfulCurrentAction(currentAction: string | null | undefined): string | null {
  if (!currentAction) return null;
  const cleaned = currentAction.replace(/\.$/, "").trim();
  if (!cleaned) return null;
  if (/^Running `[^`]+`$/i.test(cleaned)) return null;
  if (/^Searching repository files by path, basename, extension, or symbol$/i.test(cleaned)) return null;
  return cleaned;
}

function extractReadFiles(step: ProgressStep): string[] {
  const files = [...(step.files || [])];
  const current = step.currentAction || "";
  const readMatch = current.match(/^Reading `([^`]+)`\.?$/);
  if (readMatch?.[1] && !files.includes(readMatch[1])) files.push(readMatch[1]);
  return files;
}

function extractSearchQuery(step: ProgressStep): string | null {
  const agg = aggregateOf(step);
  const structured =
    step.relatedSearchQuery
    || stringValue(agg.query)
    || nonEmptyStringList(agg.text_queries)[0]
    || nonEmptyStringList(agg.queries).join(", ");
  if (structured) return structured;
  const current = meaningfulCurrentAction(step.currentAction);
  if (!current || /^Reading `/.test(current) || /^Checking command through policy:/i.test(current)) return null;
  return current;
}

function extractListPath(step: ProgressStep): string | null {
  const agg = aggregateOf(step);
  return (
    stringValue(agg.path)
    || stringValue(agg.directory)
    || stringValue(agg.repository_path)
    || stringValue(agg.project_path)
    || step.files?.[0]
    || null
  );
}

function extractCommand(step: ProgressStep): string {
  const agg = aggregateOf(step);
  const current = meaningfulCurrentAction(step.currentAction);
  const policyPrefix = "Checking command through policy:";
  return (
    fmtCommand(step.command)
    || stringValue(agg.display_command)
    || fmtCommand(agg.command)
    || (current?.startsWith(policyPrefix) ? current.slice(policyPrefix.length).trim() : current)
    || "Command"
  );
}

function commandExitCode(step: ProgressStep): number | null {
  const agg = aggregateOf(step);
  return numberValue(agg.exit_code) ?? numberValue(agg.returncode) ?? null;
}

export function groupStatus(details: AgentActivityDetailItem[]): string {
  if (details.some((d) => d.status === "failed")) return "failed";
  if (details.some((d) => d.status === "waiting" || d.status === "waiting_approval")) return "waiting";
  if (details.some((d) => d.status === "running")) return "running";
  return "completed";
}

function resolveStatus(status: string | undefined, finalizeRunning?: boolean): string {
  if (finalizeRunning && status === "running") return "completed";
  return status || "completed";
}

function makeDetailItem(
  step: ProgressStep,
  actionType: string,
  options: { finalizeRunning?: boolean } = {},
): AgentActivityDetailItem {
  const id = step.activityId || step.id || String(step.sequence ?? Math.random());
  const status = resolveStatus(step.status, options.finalizeRunning);

  if (SEARCH_TYPES.has(actionType)) {
    const query = extractSearchQuery(step);
    return { kind: "search", id, query, label: query || step.label || "Search repository", status };
  }
  if (READ_TYPES.has(actionType)) {
    return { kind: "read_file", id, files: extractReadFiles(step), label: step.label || "Read files", status };
  }
  if (LIST_TYPES.has(actionType)) {
    const path = extractListPath(step);
    return { kind: "list_files", id, path, label: path ? `List ${path}` : step.label || "Repository listing", status };
  }
  if (COMMAND_TYPES.has(actionType)) {
    return {
      kind: "command",
      id,
      command: extractCommand(step),
      label: step.label || "Run command",
      status,
      exitCode: commandExitCode(step),
    };
  }
  return { kind: "list_files", id, path: extractListPath(step), label: step.label || actionType.replace(/_/g, " "), status };
}

function groupLabel(details: AgentActivityDetailItem[]): string {
  const searches = details.filter((d) => d.kind === "search").length;
  const reads = details.filter((d): d is ReadFileDetailItem => d.kind === "read_file");
  const lists = details.filter((d) => d.kind === "list_files").length;
  const commands = details.filter((d) => d.kind === "command").length;
  const fileCount = reads.flatMap((d) => d.files).length || reads.length;
  const parts: string[] = [];
  if (searches > 0) parts.push(`Searched ${searches > 1 ? `${searches} times` : "once"}`);
  if (reads.length > 0) parts.push(`read ${fileCount} file${fileCount !== 1 ? "s" : ""}`);
  if (lists > 0) parts.push("listed repository");
  if (commands > 0) parts.push(`ran ${commands} command${commands > 1 ? "s" : ""}`);
  return parts.length > 0 ? parts.join(", ") : "Explored repository";
}

function editStatus(step: ProgressStep, finalizeRunning?: boolean): string {
  const base = resolveStatus(step.status, finalizeRunning);
  if (base === "running" || base === "failed" || base === "waiting" || base === "waiting_approval") return base;
  const agg = aggregateOf(step);
  const status = stringValue(agg.status) || commonEditFileStatus(extractEditFiles(step));
  if (status) return status;
  if (booleanValue(agg.applied) === false) return "proposed";
  return base;
}

function commonEditFileStatus(files: EditFileSummary[]): string | null {
  const statuses = files.map((file) => file.status).filter(Boolean);
  return statuses.length > 0 && statuses.every((status) => status === statuses[0]) ? statuses[0] || null : null;
}

function extractEditFiles(step: ProgressStep): EditFileSummary[] {
  const agg = aggregateOf(step);
  const files: EditFileSummary[] = [];
  const archive = Array.isArray(agg.edit_archive) ? agg.edit_archive : [];
  for (const record of archive) {
    if (!record || typeof record !== "object") continue;
    const item = record as Record<string, unknown>;
    const path = stringValue(item.file_path) || stringValue(item.path) || stringValue(item.file);
    if (!path) continue;
    files.push({
      path,
      additions: numberValue(item.additions),
      deletions: numberValue(item.deletions),
      status: stringValue(item.status),
      summary: stringValue(item.summary),
      diffAvailable: Boolean(stringValue(item.diff)) || booleanValue(item.diff_available),
      proposalId: stringValue(item.proposal_id),
    });
  }

  const aggregateFiles = Array.isArray(agg.files) ? agg.files : [];
  for (const file of aggregateFiles) {
    if (typeof file === "string") {
      files.push({ path: file });
    } else if (file && typeof file === "object") {
      const item = file as Record<string, unknown>;
      const path = stringValue(item.file_path) || stringValue(item.path) || stringValue(item.file);
      if (!path) continue;
      files.push({
        path,
        additions: numberValue(item.additions),
        deletions: numberValue(item.deletions),
        status: stringValue(item.status),
        summary: stringValue(item.summary),
        diffAvailable: booleanValue(item.diff_available) ?? booleanValue(item.diffAvailable),
        proposalId: stringValue(item.proposal_id),
      });
    }
  }

  for (const path of step.files || []) {
    files.push({ path });
  }

  const additions = numberValue(agg.additions);
  const deletions = numberValue(agg.deletions);
  const status = stringValue(agg.status);
  const summary = stringValue(agg.edit_summary) || stringValue(agg.summary);
  const diffAvailable = booleanValue(agg.diff_available) ?? booleanValue(agg.diffAvailable);
  const proposalId = step.proposalId || stringValue(agg.proposal_id);
  const merged = mergeEditFiles([], files);
  if (merged.length === 1) {
    merged[0] = {
      ...merged[0],
      additions: merged[0].additions ?? additions,
      deletions: merged[0].deletions ?? deletions,
      status: merged[0].status ?? status,
      summary: merged[0].summary ?? summary,
      diffAvailable: merged[0].diffAvailable ?? diffAvailable,
      proposalId: merged[0].proposalId ?? proposalId,
    };
  }
  return merged;
}

function mergeEditFiles(a: EditFileSummary[], b?: EditFileSummary[] | null): EditFileSummary[] {
  const out = [...a];
  for (const incoming of b || []) {
    if (!incoming.path) continue;
    const idx = out.findIndex((item) => item.path === incoming.path);
    if (idx >= 0) {
      out[idx] = {
        ...out[idx],
        ...incoming,
        additions: incoming.additions ?? out[idx].additions,
        deletions: incoming.deletions ?? out[idx].deletions,
        status: incoming.status ?? out[idx].status,
        summary: incoming.summary ?? out[idx].summary,
        diffAvailable: incoming.diffAvailable ?? out[idx].diffAvailable,
        proposalId: incoming.proposalId ?? out[idx].proposalId,
      };
    } else {
      out.push(incoming);
    }
  }
  return out;
}

function editTotals(
  step: ProgressStep,
  files: EditFileSummary[],
): { additions?: number; deletions?: number; diffAvailable?: boolean } {
  const agg = aggregateOf(step);
  const additions = numberValue(agg.additions)
    ?? sumOptional(files.map((file) => file.additions));
  const deletions = numberValue(agg.deletions)
    ?? sumOptional(files.map((file) => file.deletions));
  const diffAvailable =
    booleanValue(agg.diff_available)
    ?? booleanValue(agg.diffAvailable)
    ?? (files.some((file) => file.diffAvailable) ? true : undefined);
  return { additions, deletions, diffAvailable };
}

function sumOptional(values: Array<number | undefined>): number | undefined {
  const known = values.filter((value): value is number => typeof value === "number");
  return known.length ? known.reduce((sum, value) => sum + value, 0) : undefined;
}

export function buildAgentActivityTranscript(
  steps: ProgressStep[],
  options: { finalizeRunning?: boolean } = {},
): AgentTranscriptSection[] {
  const sections: AgentTranscriptSection[] = [];
  let currentSection: AgentTranscriptSection | null = null;

  for (const step of steps) {
    const actionType = getActionType(step);
    if (actionType && SKIP_TYPES.has(actionType)) continue;

    const statusText = sectionStatusText(step, actionType);
    if (statusText) {
      finalizePreviousSectionForNext(currentSection);
      currentSection = createSection(step, statusText, options);
      sections.push(currentSection);
      continue;
    }

    if (!actionType || !isPrimaryActionStep(step)) continue;

    if (actionType && EDIT_TYPES.has(actionType)) {
      currentSection = ensureSection(currentSection, sections, step, options);
      upsertEditItems(currentSection, step, options);
      updateSectionState(currentSection, options);
      continue;
    }

    currentSection = ensureSection(currentSection, sections, step, options);
    upsertDetailItem(currentSection, makeDetailItem(step, actionType, options), step);
    updateSectionState(currentSection, options);
  }

  return finalizeSections(sections, options);
}

function sectionStatusText(step: ProgressStep, actionType: string | null): string | null {
  if (step.display === "hidden" || step.visibility === "internal") return null;
  if (isLowValuePrimaryLabel(step.label) && !step.safeReasoningSummary && !step.safetyNote) return null;
  if (step.safeReasoningSummary?.trim()) return step.safeReasoningSummary.trim();
  if (!actionType && step.safetyNote?.trim()) return step.safetyNote.trim();
  return null;
}

function isPrimaryActionStep(step: ProgressStep): boolean {
  if (step.display === "hidden" || step.visibility === "internal") return false;
  if (step.display === "secondary" || step.visibility === "debug") return false;
  return true;
}

function createSection(
  step: ProgressStep,
  statusText: string,
  options: { finalizeRunning?: boolean },
): AgentTranscriptSection {
  const status = sectionStatusFromStatuses([resolveStatus(step.status, options.finalizeRunning)]);
  return decorateSection({
    id: `section:${step.activityId || step.id || step.sequence || statusText}`,
    runId: step.runId,
    statusText,
    status,
    startedAt: step.startedAt ?? null,
    endedAt: step.endedAt ?? null,
    details: [],
    edits: [],
    summary: emptySummary(),
    summaryText: "",
    collapsible: false,
    collapsedByDefault: false,
    isCurrent: false,
  });
}

function ensureSection(
  current: AgentTranscriptSection | null,
  sections: AgentTranscriptSection[],
  step: ProgressStep,
  options: { finalizeRunning?: boolean },
): AgentTranscriptSection {
  if (current) return current;
  const section = createSection(step, "Working", options);
  section.id = `section:implicit:${step.activityId || step.id || step.sequence || sections.length}`;
  sections.push(section);
  return section;
}

function finalizePreviousSectionForNext(section: AgentTranscriptSection | null) {
  if (!section) return;
  const childStatuses = sectionChildStatuses(section);
  if (childStatuses.length === 0 || childStatuses.every((status) => status === "completed")) {
    section.status = "completed";
  }
  decorateSection(section);
}

function upsertDetailItem(section: AgentTranscriptSection, detail: AgentActivityDetailItem, step: ProgressStep) {
  const idx = section.details.findIndex((item) => item.id === detail.id);
  if (idx >= 0) section.details[idx] = detail;
  else section.details.push(detail);
  section.endedAt = step.endedAt ?? section.endedAt;
}

function upsertEditItems(
  section: AgentTranscriptSection,
  step: ProgressStep,
  options: { finalizeRunning?: boolean },
) {
  const files = extractEditFiles(step);
  const totals = editTotals(step, files);
  const status = editStatus(step, options.finalizeRunning);
  const proposalId = step.proposalId ?? stringValue(aggregateOf(step).proposal_id) ?? null;
  const rows = files.length > 0 ? files : [{ path: step.label || "Proposed edit" }];

  for (const file of rows) {
    const id = `edit:${step.activityId || step.id || step.sequence}:${file.path}`;
    const next: AgentEditSummaryItem = {
      kind: "edit",
      id,
      label: step.label || "Preparing edit",
      path: file.path,
      additions: file.additions ?? (rows.length === 1 ? totals.additions : undefined),
      deletions: file.deletions ?? (rows.length === 1 ? totals.deletions : undefined),
      status: file.status || status,
      summary: file.summary,
      diffAvailable: file.diffAvailable ?? totals.diffAvailable,
      proposalId: file.proposalId ?? proposalId,
      safetyNote: step.safetyNote ?? null,
      startedAt: step.startedAt ?? null,
      endedAt: step.endedAt ?? null,
    };
    const idx = section.edits.findIndex((item) => item.id === id);
    if (idx >= 0) {
      section.edits[idx] = {
        ...section.edits[idx],
        ...next,
        additions: next.additions ?? section.edits[idx].additions,
        deletions: next.deletions ?? section.edits[idx].deletions,
        status: next.status || section.edits[idx].status,
        safetyNote: next.safetyNote ?? section.edits[idx].safetyNote,
      };
    } else {
      section.edits.push(next);
    }
  }
  section.endedAt = step.endedAt ?? section.endedAt;
}

function updateSectionState(section: AgentTranscriptSection, options: { finalizeRunning?: boolean }) {
  section.status = sectionStatusFromStatuses(sectionChildStatuses(section), options);
  decorateSection(section);
}

function finalizeSections(
  sections: AgentTranscriptSection[],
  options: { finalizeRunning?: boolean },
): AgentTranscriptSection[] {
  return sections.map((section, index) => {
    const isLast = index === sections.length - 1;
    const next = { ...section, details: [...section.details], edits: [...section.edits] };
    next.status = sectionStatusFromStatuses(sectionChildStatuses(next), options, next.status);
    if (!isLast && next.status === "running" && sectionChildStatuses(next).every((status) => status === "completed")) {
      next.status = "completed";
    }
    if (options.finalizeRunning && next.status === "running") next.status = "completed";
    next.isCurrent = !options.finalizeRunning && isLast && next.status === "running";
    return decorateSection(next);
  });
}

function decorateSection(section: AgentTranscriptSection): AgentTranscriptSection {
  section.summary = sectionSummary(section);
  section.summaryText = sectionSummaryText(section.summary);
  section.collapsible = section.status !== "running" && (section.details.length > 0 || section.edits.length > 0);
  section.collapsedByDefault = section.collapsible;
  section.isCurrent = section.status === "running" && section.isCurrent;
  return section;
}

function sectionSummary(section: AgentTranscriptSection): AgentTranscriptSection["summary"] {
  const reads = section.details.filter((detail): detail is ReadFileDetailItem => detail.kind === "read_file");
  return {
    searches: section.details.filter((detail) => detail.kind === "search").length,
    filesRead: reads.reduce((sum, detail) => sum + (detail.files.length || 1), 0),
    filesListed: section.details.filter((detail) => detail.kind === "list_files").length,
    commandsRun: section.details.filter((detail) => detail.kind === "command").length,
    filesEdited: section.edits.length,
  };
}

function sectionSummaryText(summary: AgentTranscriptSection["summary"]): string {
  const parts: string[] = [];
  if (summary.filesEdited > 0) parts.push(`파일 ${summary.filesEdited}개 수정`);
  if (summary.filesRead > 0) parts.push(`파일 ${summary.filesRead}개`);
  if (summary.searches > 0) parts.push(`검색 ${summary.searches}회`);
  if (summary.filesListed > 0) parts.push(`목록 ${summary.filesListed}개 탐색`);
  if (summary.commandsRun > 0) {
    parts.push(`ran ${summary.commandsRun} command${summary.commandsRun === 1 ? "" : "s"}`);
  }
  return parts.join(", ") || "No recorded actions";
}

function emptySummary(): AgentTranscriptSection["summary"] {
  return { searches: 0, filesRead: 0, filesListed: 0, commandsRun: 0, filesEdited: 0 };
}

function sectionChildStatuses(section: AgentTranscriptSection): string[] {
  return [
    ...section.details.map((detail) => detail.status),
    ...section.edits.map((edit) => edit.status || "completed"),
  ];
}

function sectionStatusFromStatuses(
  statuses: string[],
  options: { finalizeRunning?: boolean } = {},
  fallback: AgentTranscriptSection["status"] = "completed",
): AgentTranscriptSection["status"] {
  const normalized = statuses.map((status) => normalizeSectionStatus(resolveStatus(status, options.finalizeRunning)));
  if (normalized.includes("failed")) return "failed";
  if (normalized.includes("waiting")) return "waiting";
  if (normalized.includes("running")) return "running";
  return normalized.length ? "completed" : fallback;
}

function normalizeSectionStatus(status: string): AgentTranscriptSection["status"] {
  if (status === "failed" || status === "cancelled" || status === "timed_out") return "failed";
  if (status === "waiting" || status === "waiting_approval") return "waiting";
  if (status === "running") return "running";
  return "completed";
}
