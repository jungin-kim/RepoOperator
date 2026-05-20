from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from repooperator_worker.services.json_safe import json_safe


@dataclass(frozen=True)
class GitRemoteTarget:
    remote_url: str | None
    source_branch: str
    target_branch: str
    provider: str

    def model_dump(self) -> dict[str, Any]:
        return json_safe(self)


class LocalGitClient(Protocol):
    def status(self) -> dict[str, Any]:
        ...

    def diff(self, *, staged: bool = False) -> dict[str, Any]:
        ...

    def log(self, *, limit: int = 10) -> dict[str, Any]:
        ...

    def create_branch(self, branch: str, *, from_ref: str = "HEAD") -> dict[str, Any]:
        ...

    def commit(self, message: str, *, stage_all: bool = True) -> dict[str, Any]:
        ...

    def push(self, remote: str, branch: str) -> dict[str, Any]:
        ...


class GitProviderClient(Protocol):
    provider: str

    def create_review_request(self, target: GitRemoteTarget, *, title: str, body: str | None = None) -> dict[str, Any]:
        ...


@dataclass
class GitHubProviderClient:
    provider: str = "github"

    def create_review_request(self, target: GitRemoteTarget, *, title: str, body: str | None = None) -> dict[str, Any]:
        del target, title, body
        raise NotImplementedError("GitHub PR creation is available as an approval-gated service seam but is not configured yet.")


@dataclass
class GitLabProviderClient:
    provider: str = "gitlab"

    def create_review_request(self, target: GitRemoteTarget, *, title: str, body: str | None = None) -> dict[str, Any]:
        del target, title, body
        raise NotImplementedError("GitLab MR creation is available through review_providers once approval is supplied.")
