import { describe, expect, it } from "vitest";
import {
  ACTIVE_REPO_IDENTITY_KEY,
  activeThreadStorageKeyForIdentity,
  findThreadToRestore,
  repoIdentityKey,
  type RepoIdentityLike,
} from "../thread-persistence";

type TestThread = {
  id: string;
  repoResult: RepoIdentityLike;
};

function storage(seed: Record<string, string> = {}) {
  const data = new Map(Object.entries(seed));
  return {
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => data.set(key, value),
    removeItem: (key: string) => data.delete(key),
  };
}

const mainRepo = { git_provider: "local", project_path: "/repo", branch: "main" };
const devRepo = { git_provider: "local", project_path: "/repo", branch: "dev" };
const otherRepo = { git_provider: "local", project_path: "/other", branch: "main" };

function thread(id: string, repoResult: RepoIdentityLike): TestThread {
  return { id, repoResult };
}

describe("findThreadToRestore", () => {
  it("restores the repo-scoped active thread from active repo identity", () => {
    const identity = repoIdentityKey(mainRepo);
    const store = storage({
      [ACTIVE_REPO_IDENTITY_KEY]: identity,
      [activeThreadStorageKeyForIdentity(identity)]: "main-active",
    });

    const restored = findThreadToRestore(
      [thread("other", otherRepo), thread("main-active", mainRepo)],
      { storage: store },
    );

    expect(restored?.id).toBe("main-active");
  });

  it("does not select mismatched loadedThreads[0] when repo identity is known", () => {
    const restored = findThreadToRestore(
      [thread("other", otherRepo)],
      { repoIdentity: repoIdentityKey(mainRepo), storage: storage() },
    );

    expect(restored).toBeNull();
  });

  it("active run thread id wins when it matches repo identity", () => {
    const restored = findThreadToRestore(
      [thread("main-old", mainRepo), thread("main-running", mainRepo)],
      {
        repoIdentity: repoIdentityKey(mainRepo),
        activeRunThreadIds: ["main-running"],
        storage: storage(),
      },
    );

    expect(restored?.id).toBe("main-running");
  });

  it("restores the branch-scoped thread after branch switch", () => {
    const identity = repoIdentityKey(devRepo);
    const store = storage({
      [ACTIVE_REPO_IDENTITY_KEY]: identity,
      [activeThreadStorageKeyForIdentity(identity)]: "dev-thread",
    });

    const restored = findThreadToRestore(
      [thread("main-thread", mainRepo), thread("dev-thread", devRepo)],
      { storage: store },
    );

    expect(restored?.id).toBe("dev-thread");
  });

  it("uses loadedThreads[0] only when repo identity is unknown", () => {
    const restored = findThreadToRestore(
      [thread("first", otherRepo), thread("main", mainRepo)],
      { storage: storage() },
    );

    expect(restored?.id).toBe("first");
  });
});
