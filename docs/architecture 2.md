# RepoOperator Architecture

## Overview

RepoOperator separates the user interface, local repository execution, and model inference into distinct components:

- `apps/web`: hosted web application for task entry, status, review, and future collaboration features
- `apps/local-worker`: local machine service that performs repository and command operations
- `packages/shared`: shared schemas, types, and protocol definitions
- `packages/agent-core`: agent loop and orchestration primitives that can be reused across services

This architecture is designed to keep machine-specific operations close to the developer environment while allowing model inference and higher-level orchestration to be centralized.

For a visual overview, see [the architecture diagram](architecture-diagram.md).

## System Components

### Hosted Web UI

The hosted web UI is responsible for:

- authenticating and identifying the user
- presenting tasks and responses
- showing diffs, execution state, and logs
- coordinating with a local worker that is running on the user's machine
- communicating with the central model backend for streaming agent output

The web UI should remain thin. It should not directly execute repository operations or assume access to a checked-out repository on the server.

### Local Worker

The local worker is a trusted local process that exposes a narrow API over `localhost`. It is responsible for:

- opening an existing local repository or preparing a local clone
- reading and writing files
- generating diffs
- running shell commands and tests
- applying patches
- performing git operations on the local clone

The local worker is where environment-specific behavior belongs. It has access to the machine's filesystem, developer tooling, local credentials, and runtime dependencies.

### Central Model Backend

The central model backend is responsible for:

- model inference
- prompt and tool orchestration
- streaming partial responses
- request accounting, throttling, and backend policy enforcement
- provider abstraction for different model vendors or local model deployments

Centralizing model inference makes it easier to manage model access, unify orchestration behavior, and avoid duplicating inference infrastructure across every developer machine.

### Shared Packages

The shared packages exist to keep contracts explicit:

- `packages/shared` should define message schemas, API payloads, capability descriptors, and shared utilities
- `packages/agent-core` should define task execution state machines, tool-call abstractions, and core planning or patch-application logic that is not tied to a specific transport

## Request And Data Flow

The intended request flow is:

1. A user submits a task from the hosted web UI.
2. The web app ensures the relevant local worker is reachable on the user's machine.
3. The local worker gathers narrowly scoped local context such as file excerpts, diffs, repository metadata, or command results.
4. Only the minimum task context required for reasoning is sent to the central model backend.
5. The central backend performs inference and returns structured tool or edit decisions.
6. The local worker applies reads, writes, patch previews, command execution, and git operations locally.
7. Results stream back to the UI for review, approval, and iteration.

This flow intentionally avoids treating the server as the owner of the repository clone.

## Why Repository Operations Are Local

Repository operations should happen locally for several reasons:

- the local machine already contains the user's source tree, branches, uncommitted changes, and toolchain
- shell commands and tests are meaningful only in the user's real development environment
- local execution avoids recreating every repository, dependency cache, and machine-specific runtime in a central system
- keeping file reads and writes local reduces the amount of source code that must leave the machine
- local git state, ignored files, hooks, credentials, and working tree details are often essential to real workflows

In short, local repository ownership preserves fidelity with the developer's actual environment.

## Why Model Inference Is Centralized

Model inference is centralized because it offers practical advantages:

- a single backend can manage model providers, versions, quotas, and routing
- users do not need to install or configure heavyweight inference stacks locally
- streaming, observability, and policy enforcement become easier to implement consistently
- the open-source project can support multiple inference backends behind one stable interface

This keeps local workers lightweight while allowing the project to evolve its orchestration and provider integrations without coupling them to filesystem access.

## Why This Differs From A Central-Clone Architecture

A central-clone architecture typically keeps a server-side copy of the repository and runs reads, writes, and commands in that central environment. RepoOperator intentionally differs in a few ways:

- the source of truth for repository state is the developer's local machine, not a server clone
- command execution happens where dependencies, secrets, and toolchains already exist
- the central backend focuses on reasoning and coordination, not owning the filesystem
- the transport boundary is between local operations and centralized inference, not between UI and a remote repository sandbox

This tradeoff favors accuracy to the developer environment over server-side control of every operation.

## Design Implications

This architecture implies a few important constraints:

- local worker APIs should be explicit, small, and auditable
- the worker should bind to `localhost` and require deliberate trust decisions
- the system should send minimal context upstream
- patch application and command execution should be reviewable
- transports should remain replaceable so the project can support different deployment modes later

## Near-Term Interfaces

Early interfaces should likely include:

- worker health and capability discovery
- repository open or attach
- file read and write
- diff generation
- command execution
- patch preview and apply
- model task submission and streaming response events

The exact shape can evolve, but the architectural boundary should remain stable: local worker for repository operations, central backend for inference.

## Normalized Product Contract

The current public-facing RepoOperator contract is intentionally small and explicit:

- The repository identifier is `project_path` across worker APIs.
- For GitLab and GitHub, `project_path` uses the provider path such as `group/repo` or `owner/repo`.
- For local projects, `project_path` is an absolute filesystem path on the user's machine.
- Supported repository sources for repository open flows are `gitlab`, `github`, and `local`.
- Canonical user configuration lives under `~/.repooperator/config.json`.
- Environment variables remain available as advanced overrides, but they are not the primary product path.

For the first read-only workflow, the main worker APIs are:

- `POST /repo/open` with `project_path`, `branch`, and optional `git_provider`
- `POST /fs/read` with `project_path` and `relative_path`
- `POST /agent/run` with `project_path` and `task`

Provider support is intentionally split by capability:

- GitLab and GitHub are both supported for repository clone and fetch flows.
- Local projects are supported as a first-class source without clone or fetch.
- GitLab merge request creation exists as an additional provider-specific workflow.
- Future provider-specific review features should remain isolated from the core repository contract.
