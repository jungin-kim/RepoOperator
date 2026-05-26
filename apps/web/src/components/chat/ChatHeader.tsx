"use client";

import { type FormEvent, useEffect, useState } from "react";
import type {
  PermissionMode,
  ProviderBranchSummary,
  ProviderProjectSummary,
  RepoOpenPayload,
} from "@/lib/local-worker-client";
import type { RepositoryOpenProgress } from "./ChatApp";

type ConnectionState = "checking" | "connected" | "unavailable";

interface ChatHeaderProps {
  connectionState: ConnectionState;
  configuredModelName: string;
  configuredModelProvider: string;
  writeMode: PermissionMode;
  permissionPending: boolean;
  permissionMessage: string | null;
  permissionError: string | null;
  onPermissionModeChange: (mode: PermissionMode) => void;
  theme: "light" | "dark";
  onThemeToggle: () => void;

  gitProvider: string;
  onGitProviderChange: (value: string) => void;

  projects: ProviderProjectSummary[];
  recentProjects: ProviderProjectSummary[];
  projectsPending: boolean;
  selectedProjectPath: string;
  onProjectChange: (path: string) => void;

  branches: ProviderBranchSummary[];
  branchesPending: boolean;
  selectedBranch: string;
  onBranchChange: (branch: string) => void;
  openedRepository: RepoOpenPayload | null;
  branchActionPending: boolean;
  branchActionError: string | null;
  onCreateBranch: (branchName: string, baseBranch: string) => Promise<void>;

  useAdvanced: boolean;
  manualProjectPath: string;
  manualBranch: string;
  onManualProjectPathChange: (v: string) => void;
  onManualBranchChange: (v: string) => void;
  onToggleAdvanced: () => void;

  repoPending: boolean;
  repositoryOpenProgress: RepositoryOpenProgress | null;
  repoError: string | null;
  onOpenRepo: () => void;
}

const cloneStages = [
  "Connecting to provider",
  "Preparing local workspace",
  "Cloning repository",
  "Checking out branch",
  "Finalizing repository context",
];

const refreshStages = [
  "Connecting to provider",
  "Refreshing existing checkout",
  "Checking out branch",
  "Finalizing repository context",
];

const localStages = [
  "Opening local project",
  "Inspecting git metadata",
  "Finalizing repository context",
];

const permissionModes: Array<{
  mode: PermissionMode;
  label: string;
  description: string;
  disabled?: boolean;
}> = [
  {
    mode: "default",
    label: "Default",
    description: "Work inside the active repository sandbox with approval gates.",
  },
  {
    mode: "accept_edits",
    label: "Accept edits",
    description: "Use approval cards for elevated actions, network, and risky tools.",
  },
  {
    mode: "auto_readonly",
    label: "Read-only",
    description: "Allow repository reads and safe git inspection without writes.",
  },
  {
    mode: "full_access",
    label: "Full access",
    description: "Dangerous broader local access after confirmation.",
  },
];

const BRANCH_RE = /^[a-zA-Z0-9._/\-]+$/;

function validateBranchName(name: string): string | null {
  if (!name.trim()) return "Branch name is required.";
  if (name.startsWith("-")) return "Branch name must not start with a hyphen.";
  if (name.includes("..")) return "Branch name must not contain '..'.";
  if (name.includes(" ")) return "Branch name must not contain spaces.";
  if (!BRANCH_RE.test(name)) return "Branch name contains invalid characters.";
  return null;
}

function permissionLabel(mode: PermissionMode): string {
  return permissionModes.find((item) => item.mode === mode)?.label ?? mode.replaceAll("_", " ");
}

function permissionTone(mode: PermissionMode): "review" | "full" | "default" {
  if (mode === "accept_edits" || mode === "routine_safe" || mode === "headless_safe") return "review";
  if (mode === "full_access") return "full";
  return "default";
}

export function ChatHeader({
  connectionState,
  configuredModelName,
  configuredModelProvider,
  writeMode,
  permissionPending,
  permissionMessage,
  permissionError,
  onPermissionModeChange,
  theme,
  onThemeToggle,
  gitProvider,
  onGitProviderChange,
  projects,
  recentProjects,
  projectsPending,
  selectedProjectPath,
  onProjectChange,
  branches,
  branchesPending,
  selectedBranch,
  onBranchChange,
  openedRepository,
  branchActionPending,
  branchActionError,
  onCreateBranch,
  useAdvanced,
  manualProjectPath,
  manualBranch,
  onManualProjectPathChange,
  onManualBranchChange,
  onToggleAdvanced,
  repoPending,
  repositoryOpenProgress,
  repoError,
  onOpenRepo,
}: ChatHeaderProps) {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showPermissions, setShowPermissions] = useState(false);
  const [showBranchCreate, setShowBranchCreate] = useState(false);
  const [newBranchName, setNewBranchName] = useState("");
  const [newBranchBase, setNewBranchBase] = useState("");
  const [branchFormError, setBranchFormError] = useState<string | null>(null);
  const [openElapsedSeconds, setOpenElapsedSeconds] = useState(0);

  useEffect(() => {
    if (!repositoryOpenProgress) {
      setOpenElapsedSeconds(0);
      return;
    }

    const interval = window.setInterval(() => {
      setOpenElapsedSeconds(
        Math.floor((Date.now() - repositoryOpenProgress.startedAt) / 1000),
      );
    }, 1000);
    setOpenElapsedSeconds(
      Math.floor((Date.now() - repositoryOpenProgress.startedAt) / 1000),
    );

    return () => window.clearInterval(interval);
  }, [repositoryOpenProgress]);

  const connectionLabel =
    connectionState === "connected"
      ? "Connected"
      : connectionState === "checking"
        ? "Checking"
        : "Unavailable";

  const modelLabel = configuredModelName || configuredModelProvider || "No model";

  const allProjects = [
    ...recentProjects,
    ...projects.filter(
      (p) => !recentProjects.some((r) => r.project_path === p.project_path),
    ),
  ];

  const branchRequired = gitProvider !== "local";
  const showBranchControls = branchRequired || Boolean(openedRepository?.is_git_repository);
  const effectiveProject = useAdvanced ? manualProjectPath.trim() : selectedProjectPath;
  const effectiveBranch = useAdvanced ? manualBranch.trim() : selectedBranch;
  const repoOpenMode = repositoryOpenProgress?.mode || "unknown";
  const progressStages =
    repoOpenMode === "clone"
      ? cloneStages
      : repoOpenMode === "refresh"
        ? refreshStages
        : repoOpenMode === "local"
          ? localStages
          : cloneStages;
  const stageIndex = Math.min(
    progressStages.length - 1,
    Math.floor(openElapsedSeconds / 4),
  );
  const stageLabel = progressStages[stageIndex];
  const stagePercent = Math.min(
    92,
    Math.round(((stageIndex + 0.45) / progressStages.length) * 100),
  );
  const openModeLabel =
    repoOpenMode === "clone"
      ? "First-time clone"
      : repoOpenMode === "refresh"
        ? "Existing checkout refresh"
        : repoOpenMode === "local"
          ? "Local project"
          : "Repository open";
  const openHelpText =
    repoOpenMode === "clone"
      ? "First-time clones can take a few minutes. RepoOperator is preparing a local checkout."
      : repoOpenMode === "refresh"
        ? "RepoOperator is updating the existing local checkout and switching to the selected branch."
        : "RepoOperator is opening the local workspace and reading git context.";
  const canOpen =
    connectionState === "connected" &&
    Boolean(effectiveProject) &&
    (!branchRequired || Boolean(effectiveBranch));
  const canCreateBranch =
    connectionState === "connected" &&
    Boolean(openedRepository?.is_git_repository) &&
    Boolean(openedRepository?.project_path);
  const displayedBranch = selectedBranch || openedRepository?.branch || "";
  const branchOptions = branches.some((branch) => branch.name === displayedBranch) || !displayedBranch
    ? branches
    : [{ name: displayedBranch, is_default: false }, ...branches];

  function handleAdvancedToggle() {
    setShowAdvanced((v) => !v);
    onToggleAdvanced();
  }

  function handleBranchSelect(value: string) {
    if (value === "__create_branch__") {
      setShowBranchCreate(true);
      setNewBranchBase(displayedBranch || branches[0]?.name || "HEAD");
      setBranchFormError(null);
      return;
    }
    onBranchChange(value);
  }

  async function handleCreateBranch(e: FormEvent) {
    e.preventDefault();
    const branchName = newBranchName.trim();
    const validationError = validateBranchName(branchName);
    if (validationError) {
      setBranchFormError(validationError);
      return;
    }
    setBranchFormError(null);
    try {
      await onCreateBranch(branchName, newBranchBase || displayedBranch || "HEAD");
      setNewBranchName("");
      setShowBranchCreate(false);
    } catch {
      // The parent owns the detailed git error so it can preserve repository state.
    }
  }

  return (
    <div className="chat-header">
      <div className="chat-header-row">
        <div className="chat-header-selectors">
          <select
            className="header-select"
            value={gitProvider}
            onChange={(e) => onGitProviderChange(e.target.value)}
            aria-label="Repository source"
          >
            <option value="gitlab">GitLab</option>
            <option value="github">GitHub</option>
            <option value="local">Local project</option>
          </select>

          <span className="header-sep">/</span>

          <select
            className="header-select"
            value={selectedProjectPath}
            onChange={(e) => onProjectChange(e.target.value)}
            disabled={
              projectsPending ||
              connectionState !== "connected" ||
              allProjects.length === 0
            }
            aria-label="Project"
          >
            {allProjects.length === 0 ? (
              <option value="">
                {projectsPending
                  ? "Loading…"
                  : connectionState !== "connected"
                    ? "Not connected"
                    : gitProvider === "local"
                      ? "No recent local projects"
                      : "No projects"}
              </option>
            ) : null}
            {allProjects.map((p) => (
              <option key={p.project_path} value={p.project_path}>
                {p.display_name}
              </option>
            ))}
          </select>

          {showBranchControls && (
            <>
              <span className="header-sep">@</span>
              <select
                className="header-select"
                value={displayedBranch}
                onChange={(e) => handleBranchSelect(e.target.value)}
                disabled={
                  branchesPending ||
                  connectionState !== "connected" ||
                  (!selectedProjectPath && !openedRepository?.project_path) ||
                  (branchOptions.length === 0 && !canCreateBranch)
                }
                aria-label="Branch"
              >
                {branchOptions.length === 0 ? (
                  <option value="">
                    {branchesPending ? "Loading…" : canCreateBranch ? "No branch" : "Open repository to create branch"}
                  </option>
                ) : null}
                {branchOptions.map((b) => (
                  <option key={b.name} value={b.name}>
                    {b.name}
                    {b.is_default ? " (default)" : ""}
                  </option>
                ))}
                <option value="__create_branch__" disabled={!canCreateBranch}>
                  + Create new branch
                </option>
              </select>
            </>
          )}

          {gitProvider === "local" && (
            <input
              className="header-advanced-input"
              value={manualProjectPath}
              onChange={(e) => onManualProjectPathChange(e.target.value)}
              placeholder="/path/to/project"
              aria-label="Local project path"
              style={{ flex: 1, minWidth: "160px", maxWidth: "300px" }}
            />
          )}

          <button
            className="header-open-btn"
            type="button"
            onClick={onOpenRepo}
            disabled={!canOpen}
          >
            {repoPending ? "Open selected repository" : "Open repository"}
          </button>

          {gitProvider !== "local" && (
            <button
              className="header-icon-btn"
              type="button"
              onClick={handleAdvancedToggle}
              title="Manual path override"
            >
              {showAdvanced ? "▲" : "▼"} Advanced
            </button>
          )}
        </div>

        <div className="chat-header-status">
          <span
            className={`status-pill-sm${
              connectionState === "connected"
                ? " status-pill-sm-connected"
                : connectionState === "checking"
                  ? " status-pill-sm-checking"
                  : ""
            }`}
          >
            {connectionLabel}
          </span>
          {configuredModelProvider && (
            <span className="model-chip" title={modelLabel}>
              {modelLabel}
            </span>
          )}
          <div className="permission-control">
            <button
              className={`permission-trigger${permissionTone(writeMode) === "review" ? " permission-trigger-review" : ""}`}
              type="button"
              aria-label="Permission mode"
              aria-expanded={showPermissions}
              onClick={() => setShowPermissions((value) => !value)}
              disabled={permissionPending}
              title="Change RepoOperator permission mode"
            >
              <span className="permission-trigger-icon" aria-hidden="true">
                {permissionTone(writeMode) === "review" ? "◇" : permissionTone(writeMode) === "full" ? "!" : "●"}
              </span>
              {permissionPending ? "Updating…" : permissionLabel(writeMode)}
              <span className="permission-trigger-caret" aria-hidden="true">▾</span>
            </button>
            {showPermissions && (
              <div className="permission-menu" role="menu">
                {permissionModes.map((item) => (
                  <button
                    key={item.mode}
                    className={`permission-option${item.mode === writeMode ? " permission-option-selected" : ""}`}
                    type="button"
                    role="menuitemradio"
                    aria-checked={item.mode === writeMode}
                    disabled={permissionPending || item.disabled}
                    onClick={() => {
                      if (item.disabled) return;
                      setShowPermissions(false);
                      onPermissionModeChange(item.mode);
                    }}
                  >
                    <span className="permission-option-check" aria-hidden="true">
                      {item.mode === writeMode ? "✓" : ""}
                    </span>
                    <span className="permission-option-copy">
                      <span className="permission-option-label">
                        {item.label}
                        {item.disabled ? " — Coming soon" : ""}
                      </span>
                      <span className="permission-option-description">{item.description}</span>
                    </span>
                  </button>
                ))}
              </div>
            )}
          </div>
          <button
            className="theme-toggle"
            type="button"
            aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
            onClick={onThemeToggle}
            title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
          >
            {theme === "dark" ? <SunIcon /> : <MoonIcon />}
          </button>
        </div>
      </div>

      {(permissionMessage || permissionError) && (
        <p className={`permission-inline-message${permissionError ? " permission-inline-message-error" : ""}`}>
          {permissionError || permissionMessage}
        </p>
      )}

      {branchActionError && !showBranchCreate && (
        <p className="header-error">{branchActionError}</p>
      )}

      {showBranchCreate && (
        <form className="header-branch-create" onSubmit={(e) => void handleCreateBranch(e)}>
          <div className="header-advanced-field">
            <label className="header-advanced-label" htmlFor="top-new-branch">
              New branch name
            </label>
            <input
              id="top-new-branch"
              className="header-advanced-input"
              value={newBranchName}
              onChange={(e) => {
                setNewBranchName(e.target.value);
                setBranchFormError(null);
              }}
              placeholder="feature/my-change"
              disabled={branchActionPending}
              autoComplete="off"
              spellCheck={false}
            />
          </div>
          <div className="header-advanced-field">
            <label className="header-advanced-label" htmlFor="top-new-branch-base">
              Base branch
            </label>
            <select
              id="top-new-branch-base"
              className="header-select header-branch-base-select"
              value={newBranchBase || displayedBranch || "HEAD"}
              onChange={(e) => setNewBranchBase(e.target.value)}
              disabled={branchActionPending}
            >
              {branchOptions.length > 0 ? (
                branchOptions.map((branch) => (
                  <option key={branch.name} value={branch.name}>
                    {branch.name}
                  </option>
                ))
              ) : (
                <option value="HEAD">HEAD</option>
              )}
            </select>
          </div>
          <button
            className="header-open-btn"
            type="submit"
            disabled={branchActionPending || !newBranchName.trim()}
          >
            {branchActionPending ? "Creating…" : "Create and switch"}
          </button>
          <button
            className="header-icon-btn"
            type="button"
            onClick={() => {
              setShowBranchCreate(false);
              setNewBranchName("");
              setBranchFormError(null);
            }}
            disabled={branchActionPending}
          >
            Cancel
          </button>
          <p className="header-branch-note">
            {canCreateBranch
              ? "RepoOperator creates the branch in the opened local checkout and switches to it."
              : "Open a repository before creating a branch."}
          </p>
          {(branchFormError || branchActionError) && (
            <p className="header-branch-error">{branchFormError || branchActionError}</p>
          )}
        </form>
      )}

      {showAdvanced && gitProvider !== "local" && (
        <div className="chat-header-advanced">
          <div className="header-advanced-field">
            <label className="header-advanced-label" htmlFor="adv-project">
              Manual project path
            </label>
            <input
              id="adv-project"
              className="header-advanced-input"
              value={manualProjectPath}
              onChange={(e) => onManualProjectPathChange(e.target.value)}
              placeholder="group/my-repo"
            />
          </div>
          <div className="header-advanced-field">
            <label className="header-advanced-label" htmlFor="adv-branch">
              Manual branch
            </label>
            <input
              id="adv-branch"
              className="header-advanced-input"
              value={manualBranch}
              onChange={(e) => onManualBranchChange(e.target.value)}
              placeholder="main"
            />
          </div>
          <p style={{ margin: 0, fontSize: "0.82rem", color: "var(--muted)", alignSelf: "flex-end" }}>
            Use only when the project list is missing the repo you need.
          </p>
        </div>
      )}

      {repositoryOpenProgress && (
        <div className="repo-open-progress" role="status" aria-live="polite">
          <div className="repo-open-progress-copy">
            <span className="repo-open-progress-kicker">{openModeLabel}</span>
            <strong>{stageLabel}</strong>
            <span>{openHelpText}</span>
          </div>
          <div className="repo-open-progress-meta">
            <span>{openElapsedSeconds}s</span>
            <span>{repositoryOpenProgress.gitProvider}</span>
            <span>{repositoryOpenProgress.projectPath}</span>
            {repositoryOpenProgress.branch && <span>{repositoryOpenProgress.branch}</span>}
          </div>
          <div className="repo-open-progress-track" aria-hidden="true">
            <div
              className="repo-open-progress-bar"
              style={{ width: `${stagePercent}%` }}
            />
          </div>
        </div>
      )}

      {repoError && <p className="header-error">{repoError}</p>}
    </div>
  );
}

function SunIcon() {
  return (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2" />
      <path d="M12 20v2" />
      <path d="m4.93 4.93 1.41 1.41" />
      <path d="m17.66 17.66 1.41 1.41" />
      <path d="M2 12h2" />
      <path d="M20 12h2" />
      <path d="m6.34 17.66-1.41 1.41" />
      <path d="m19.07 4.93-1.41 1.41" />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M12 3a6 6 0 0 0 9 8 9 9 0 1 1-9-8Z" />
    </svg>
  );
}
