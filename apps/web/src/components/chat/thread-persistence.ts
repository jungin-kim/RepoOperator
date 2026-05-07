export type RepoIdentityLike = {
  git_provider?: string | null;
  project_path?: string | null;
  local_repo_path?: string | null;
  branch?: string | null;
};

export const LEGACY_ACTIVE_THREAD_KEY = "repooperator-active-thread-id";
export const ACTIVE_REPO_IDENTITY_KEY = "repooperator-active-repo-identity";
export const ACTIVE_THREAD_KEY_PREFIX = "repooperator-active-thread:";

type StorageLike = Pick<Storage, "getItem" | "setItem" | "removeItem">;

export type RestoreContext = {
  repoIdentity?: string | null;
  activeRunThreadIds?: string[];
  allowCrossRepoFallback?: boolean;
  storage?: StorageLike;
};

export type RestorableThread = {
  id: string;
  repoResult: RepoIdentityLike;
};

export function repoIdentityKey(repo: RepoIdentityLike): string {
  const provider = normalizeStorageSegment(repo.git_provider || "local");
  const rawPath = repo.project_path || repo.local_repo_path || "unknown";
  const normalizedPath = normalizeStorageSegment(normalizeRepoPath(rawPath));
  const branch = normalizeStorageSegment(repo.branch || "default");
  return `${provider}:${normalizedPath}:${branch}`;
}

export function activeThreadStorageKey(repo: RepoIdentityLike): string {
  return activeThreadStorageKeyForIdentity(repoIdentityKey(repo));
}

export function activeThreadStorageKeyForIdentity(identity: string): string {
  return `${ACTIVE_THREAD_KEY_PREFIX}${identity}`;
}

export function repoMatchesIdentity(repo: RepoIdentityLike, identity: string | null | undefined): boolean {
  return Boolean(identity) && repoIdentityKey(repo) === identity;
}

export function findThreadToRestore<T extends RestorableThread>(
  loadedThreads: T[],
  context: RestoreContext = {},
): T | null {
  if (loadedThreads.length === 0) return null;
  const storage = context.storage ?? (typeof window !== "undefined" ? window.localStorage : null);
  const repoIdentity =
    context.repoIdentity
    || storage?.getItem(ACTIVE_REPO_IDENTITY_KEY)
    || null;
  const activeRunThreadIds = context.activeRunThreadIds || [];

  if (repoIdentity) {
    const scopedKey = activeThreadStorageKeyForIdentity(repoIdentity);
    const scopedThreadId = storage?.getItem(scopedKey) || null;
    const scopedThread = loadedThreads.find(
      (thread) => thread.id === scopedThreadId && repoMatchesIdentity(thread.repoResult, repoIdentity),
    );
    if (scopedThread) return scopedThread;
    if (scopedThreadId) storage?.removeItem(scopedKey);

    const activeRunThread = findActiveRunThread(loadedThreads, activeRunThreadIds, repoIdentity);
    if (activeRunThread) return activeRunThread;

    const sameRepoThread = loadedThreads.find((thread) => repoMatchesIdentity(thread.repoResult, repoIdentity));
    if (sameRepoThread) return sameRepoThread;

    migrateLegacyThread(loadedThreads, repoIdentity, storage);
    return context.allowCrossRepoFallback ? loadedThreads[0] : null;
  }

  const activeRunThread = findActiveRunThread(loadedThreads, activeRunThreadIds, null);
  if (activeRunThread) return activeRunThread;

  const legacyThread = migrateLegacyThread(loadedThreads, null, storage);
  if (legacyThread) return legacyThread;

  return loadedThreads[0];
}

function findActiveRunThread<T extends RestorableThread>(
  loadedThreads: T[],
  activeRunThreadIds: string[],
  repoIdentity: string | null,
): T | null {
  for (const threadId of activeRunThreadIds) {
    const thread = loadedThreads.find(
      (item) => item.id === threadId && (!repoIdentity || repoMatchesIdentity(item.repoResult, repoIdentity)),
    );
    if (thread) return thread;
  }
  return null;
}

function migrateLegacyThread<T extends RestorableThread>(
  loadedThreads: T[],
  repoIdentity: string | null,
  storage: StorageLike | null,
): T | null {
  const legacyThreadId = storage?.getItem(LEGACY_ACTIVE_THREAD_KEY);
  if (!legacyThreadId) return null;
  const legacyThread = loadedThreads.find((thread) => thread.id === legacyThreadId) || null;
  const legacyMatchesCurrentRepo =
    legacyThread && (!repoIdentity || repoMatchesIdentity(legacyThread.repoResult, repoIdentity));
  storage?.removeItem(LEGACY_ACTIVE_THREAD_KEY);
  if (legacyThread && legacyMatchesCurrentRepo) {
    const identity = repoIdentityKey(legacyThread.repoResult);
    storage?.setItem(activeThreadStorageKeyForIdentity(identity), legacyThread.id);
    storage?.setItem(ACTIVE_REPO_IDENTITY_KEY, identity);
    return legacyThread;
  }
  return null;
}

function normalizeRepoPath(path: string): string {
  const normalized = String(path || "")
    .trim()
    .replace(/\\/g, "/")
    .replace(/\/+$/, "");
  return normalized || "unknown";
}

function normalizeStorageSegment(value: string): string {
  return encodeURIComponent(String(value || "unknown").trim() || "unknown");
}
