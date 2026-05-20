export type StatusNoteItem = {
  kind: "status_note";
  id: string;
  text: string;
  status: string;
  startedAt?: string | null;
  endedAt?: string | null;
  durationMs?: number | null;
  safetyNote?: string | null;
};

export type SearchDetailItem = {
  kind: "search";
  id: string;
  query?: string | null;
  label: string;
  status: string;
  resultCount?: number | null;
};

export type ReadFileDetailItem = {
  kind: "read_file";
  id: string;
  files: string[];
  label: string;
  status: string;
};

export type ListFilesDetailItem = {
  kind: "list_files";
  id: string;
  path?: string | null;
  label: string;
  status: string;
};

export type CommandDetailItem = {
  kind: "command";
  id: string;
  command: string;
  label: string;
  status: string;
  exitCode?: number | null;
};

export type WebResearchDetailItem = {
  kind: "web";
  id: string;
  label: string;
  status: string;
  query?: string | null;
  sourceCount?: number | null;
  sources?: Array<{ title?: string; url?: string; source?: string }>;
};

export type EditDetailItem = {
  kind: "edit";
  id: string;
  files: string[];
  label: string;
  status: string;
  proposalId?: string | null;
};

export type AgentActivityDetailItem =
  | SearchDetailItem
  | ReadFileDetailItem
  | ListFilesDetailItem
  | CommandDetailItem
  | WebResearchDetailItem
  | EditDetailItem;

export type ActivityGroupItem = {
  kind: "activity_group";
  id: string;
  label: string;
  details: AgentActivityDetailItem[];
  status: string;
  startedAt?: string | null;
  endedAt?: string | null;
};

export type EditFileSummary = {
  path: string;
  additions?: number;
  deletions?: number;
  status?: string | null;
  summary?: string | null;
  diffAvailable?: boolean;
  proposalId?: string | null;
};

export type AgentEditSummaryItem = EditFileSummary & {
  kind: "edit";
  id: string;
  label: string;
  status: string;
  safetyNote?: string | null;
  startedAt?: string | null;
  endedAt?: string | null;
};

export type AgentTranscriptSection = {
  id: string;
  runId?: string;
  statusText: string;
  status: "running" | "completed" | "failed" | "waiting";
  startedAt?: string | null;
  endedAt?: string | null;
  details: AgentActivityDetailItem[];
  edits: AgentEditSummaryItem[];
  summary: {
    searches: number;
    filesRead: number;
    filesListed: number;
    commandsRun: number;
    filesEdited: number;
    webSources: number;
  };
  summaryText: string;
  collapsible: boolean;
  collapsedByDefault: boolean;
  isCurrent: boolean;
};

export type EditGroupItem = {
  kind: "edit_group";
  id: string;
  label: string;
  files: EditFileSummary[];
  status: string;
  additions?: number;
  deletions?: number;
  diffAvailable?: boolean;
  safetyNote?: string | null;
  proposalId?: string | null;
  startedAt?: string | null;
  endedAt?: string | null;
};

export type AgentActivityTranscriptItem =
  | StatusNoteItem
  | ActivityGroupItem
  | EditGroupItem;
