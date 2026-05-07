# RepoOperator

[![CI](https://img.shields.io/github/actions/workflow/status/jungin-kim/RepoOperator/ci.yml?branch=main&label=CI)](https://github.com/jungin-kim/RepoOperator/actions)
[![Web E2E](https://img.shields.io/github/actions/workflow/status/jungin-kim/RepoOperator/web-e2e.yml?branch=main&label=Web%20E2E)](https://github.com/jungin-kim/RepoOperator/actions)
[![npm](https://img.shields.io/npm/v/repooperator?label=npm)](https://www.npmjs.com/package/repooperator)
[![License](https://img.shields.io/github/license/jungin-kim/RepoOperator)](LICENSE)
[![GitHub repo](https://img.shields.io/badge/GitHub-RepoOperator-181717?logo=github)](https://github.com/jungin-kim/RepoOperator)
[![Issues](https://img.shields.io/github/issues/jungin-kim/RepoOperator)](https://github.com/jungin-kim/RepoOperator/issues)

![RepoOperator CLI Screenshot](/repooperator-screenshot.png)

RepoOperator is a local-first repository assistant for opening private codebases, answering codebase questions, preparing proposed changes, and keeping repository access on your machine.

## Why RepoOperator

Many coding-agent products choose one of two extremes: clone everything into a hosted environment, or run the entire experience inside one local editor.

RepoOperator explores a third path:

- repository access, credentials, tools, and working copies stay local
- the user experience can still be browser-based and product-oriented
- model access can be local through Ollama or remote through an enterprise-compatible API

The current alpha is intentionally focused: onboard a machine, start the local runtime, open a repository, and work through repository questions or proposal-only edits.

## Features

- One-command local product startup with `repooperator up`
- Guided onboarding for repository source and model connection setup
- Local worker that performs repository operations on the developer machine
- Browser UI with project selection, branch selection, and repository-aware chat
- GitLab and GitHub project discovery through stored provider config
- First-class local project support using absolute filesystem paths
- Ollama-first local model runtime support
- Remote model API support for OpenAI-compatible and enterprise-style APIs
- Read-only repository Q&A through the local worker
- Runtime config stored under `~/.repooperator`

## Quickstart

### Install via Homebrew (macOS — recommended)

```bash
brew tap jungin-kim/repooperator
brew install repooperator
```

Homebrew handles Python 3.12 automatically — no manual Python setup needed.

### Install via npm

RepoOperator requires **Python 3.11 or later** for the local worker. If you use the
npm path, install Python first.

```bash
npm install -g repooperator
```

If onboarding fails with a Python version error, set `REPOOPERATOR_PYTHON` to a
compatible interpreter:

```bash
REPOOPERATOR_PYTHON=$(which python3.12) repooperator onboard
```

### First-run setup

Run onboarding once (automatically prepares the local runtime):

```bash
repooperator onboard
```

Start the local product runtime:

```bash
repooperator up
```

Open the printed local web URL, choose a repository, and ask a repository question.

> **No source clone required.** `repooperator onboard` downloads and prepares all runtime
> dependencies into `~/.repooperator/runtime/` automatically. You do not need to clone the
> RepoOperator source repository as a normal user.

If something looks wrong, run the diagnostics command:

```bash
repooperator doctor
```

To reset everything and start fresh:

```bash
rm -rf ~/.repooperator
repooperator onboard
```

### Troubleshooting: Python version errors

If you see an error like:

```
ERROR: Package 'repooperator-local-worker' requires a different Python: 3.9.6 not in '>=3.11'
```

Your system Python is too old. Fix options:

**macOS (Homebrew):**
```bash
brew install python@3.12
REPOOPERATOR_PYTHON="$(brew --prefix python@3.12)/bin/python3.12" repooperator onboard
```

**Linux (apt):**
```bash
sudo apt install python3.12
REPOOPERATOR_PYTHON=$(which python3.12) repooperator onboard
```

**Linux (dnf):**
```bash
sudo dnf install python3.12
REPOOPERATOR_PYTHON=$(which python3.12) repooperator onboard
```

You can also set `REPOOPERATOR_PYTHON` permanently in your shell profile so
onboarding always uses the right interpreter.

## First End-To-End Local Flow

The intended common path is:

1. Install the CLI.
2. Run `repooperator onboard`.
3. Choose a repository source: GitLab, GitHub, or local project.
4. Choose a model connection mode: local runtime or remote API.
5. Run `repooperator up`.
6. Open the printed web URL.
7. Select a project and branch.
8. Open the repository locally.
9. Ask a repository question or request a proposal-only change.
10. Review the answer in the browser.

```bash
npm install -g repooperator
repooperator onboard
repooperator up
repooperator doctor
repooperator status
```

Health can also be checked directly:

```bash
curl http://127.0.0.1:8000/health
```

## Authentication and Token Setup

RepoOperator needs a personal access token to list and clone repositories from GitHub or GitLab.
Tokens are stored locally in `~/.repooperator/config.json` and never sent outside your machine.

### GitHub

**Creating a fine-grained personal access token (recommended)**

1. Go to **GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens**.
2. Click **Generate new token**.
3. Set a token name and expiration date.
4. Under **Repository access**, choose the owner or organization whose repositories you want to open.
   - The **owner/org** value you enter during `repooperator onboard` should be a username or
     organization name such as `jungin-kim`, **not** a full URL.
5. Under **Permissions → Repository permissions**, grant at minimum:
   - **Metadata**: Read (required for repository discovery)
   - **Contents**: Read (required for reading file contents)
6. Click **Generate token** and copy it immediately.

**Public GitHub vs GitHub Enterprise**

| Scenario | Base URL |
|---|---|
| Public GitHub (github.com) | Leave blank — the default `https://api.github.com` is used automatically. |
| GitHub Enterprise Server | Enter your instance URL such as `https://github.example.com`. |

During onboarding, if you see a prompt for **GitHub base URL**, enter your Enterprise server URL or
leave it blank for public GitHub.

**Minimum token permissions for RepoOperator**

| Permission | Access level | Required for |
|---|---|---|
| Metadata | Read | Repository listing and discovery |
| Contents | Read | File reading and repository Q&A |

Write permissions are not required for read-only Q&A. If you plan to use the write-with-approval
workflow, Contents: Write is additionally needed to apply proposed changes via the GitHub API
in future releases.

---

### GitLab

**Creating a personal access token**

1. Go to **GitLab → User settings → Access tokens** (or your organization's GitLab instance).
2. Click **Add new token**.
3. Set a token name and expiration date.
4. Select the scopes:
   - `read_repository` — required for clone and file reading
   - `read_api` — required for project listing and branch discovery
5. Click **Create personal access token** and copy it immediately.

**Self-hosted GitLab**

If you use a self-hosted GitLab instance, enter your instance URL during onboarding, for example:

```text
https://gitlab.example.com
```

Leave blank only if you use public GitLab (`https://gitlab.com`).

**Minimum token scopes for RepoOperator**

| Scope | Required for |
|---|---|
| `read_repository` | Cloning and reading repository contents |
| `read_api` | Project listing, branch listing, and repository metadata |

---

### Security notes

- **Never commit tokens to source control.** RepoOperator stores them in
  `~/.repooperator/config.json`, which is local to your machine and should not be committed.
- **Rotate exposed tokens immediately.** If a token is accidentally shared, revoke it and generate
  a new one.
- **Use least privilege.** Grant only the scopes listed above. Avoid using classic GitHub PATs with
  full `repo` scope unless your provider requires them.
- **Prefer short-lived tokens.** Set a reasonable expiration date (30–90 days) and rotate on
  schedule.
- **Fine-grained tokens are preferred** for GitHub because they can be scoped to specific
  repositories or organizations and allow read-only permissions without write access.

---

## Supported Repository Sources

RepoOperator currently supports these repository sources:

| Source | Status | Notes |
| --- | --- | --- |
| GitLab | Working alpha path | Project listing, branch listing, clone/fetch, repository Q&A, and proposal flows are the most exercised path. |
| GitHub | Supported alpha path | Uses the same provider-oriented flow, with GitHub token and base URL from onboarding. |
| Local project | Supported alpha path | Uses absolute filesystem paths and can work with git repositories or plain directories. |

Provider credentials are stored in `~/.repooperator/config.json` by onboarding. Raw environment variables remain available as advanced overrides.

## Supported Model Connection Modes

RepoOperator supports two model connection modes:

| Mode | Providers | Notes |
| --- | --- | --- |
| Local self-served runtime | Ollama, vLLM | Ollama is first-class for laptop use. vLLM uses an OpenAI-compatible endpoint on this machine or a trusted LAN host. |
| Remote model API | OpenAI-compatible, OpenAI, Anthropic, Gemini | Intended for enterprise API gateways and hosted model providers. |

For a local Ollama setup, onboarding uses:

```text
http://127.0.0.1:11434/v1
```

For vLLM, start your runtime separately and point RepoOperator at its OpenAI-compatible base URL:

```bash
vllm serve <model> --host 0.0.0.0 --port 8001 --api-key <optional-token>
```

Default vLLM base URL:

```text
http://127.0.0.1:8001/v1
```

Only expose unauthenticated vLLM endpoints on trusted local or private networks.

## Web App

The web app is split into two experiences:

- `/` is a lightweight landing page.
- `/app` is the repository-aware chat workspace.

The app screen includes:

- a sidebar for new chat, thread state, and recent repositories
- a top bar for repository source, project, branch, worker status, and model status
- a chat area with repository context, messages, collapsible tool/result cards, and a composer

For local development:

```bash
cd apps/web
npm install
npm run dev
```

Then open:

```text
http://localhost:3000
```

## CLI Commands

Core commands:

```bash
repooperator onboard       # Guided first-run setup (auto-prepares runtime on first install)
repooperator up            # Start worker and web UI (self-heals missing runtime)
repooperator down          # Stop worker and web UI
repooperator doctor        # Run local diagnostics (reports runtime, worker, web, model state)
repooperator status        # Show runtime status
repooperator config show   # Print redacted config
```

Worker maintenance commands:

```bash
repooperator worker start
repooperator worker stop
repooperator worker restart
repooperator worker status
repooperator worker logs
```

The recommended product flow is:

```bash
repooperator onboard
repooperator up
```

Use `worker` commands when you need lower-level runtime inspection or maintenance.

## Source Install (Contributors Only)

Normal users do **not** need to clone this repository. The npm package bundles the local worker
and web sources inside it and installs them automatically.

If you are contributing to RepoOperator, clone the repo and work from source:

```bash
git clone https://github.com/jungin-kim/RepoOperator.git
cd RepoOperator

# Install worker deps
cd apps/local-worker
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Install web deps
cd ../web
npm install
npm run dev

# Run CLI from source
cd ../../packages/cli
node bin/repooperator.js onboard
```

You can override the paths the CLI uses with environment variables:

```bash
export REPOOPERATOR_WORKER_PATH=/path/to/apps/local-worker
export REPOOPERATOR_WEB_PATH=/path/to/apps/web
```

## Architecture Overview

RepoOperator is built from three main pieces:

```text
Browser UI
  |
  | local HTTP proxy
  v
Local worker on the developer machine
  |
  | git, filesystem, command, provider APIs
  v
Local repositories and configured model backend
```

Key directories:

- `packages/cli` contains the `repooperator` CLI.
- `apps/local-worker` contains the Python local worker and API routes.
- `apps/web` contains the Next.js web app.
- `docs` contains architecture, onboarding, demo, security, roadmap, and troubleshooting notes.

Helpful docs:

- [Onboarding guide](docs/onboarding.md)
- [Read-only demo](docs/demo.md)
- [Architecture](docs/architecture.md)
- [Architecture diagram](docs/architecture-diagram.md)
- [Security](docs/security.md)
- [Roadmap](docs/roadmap.md)
- [Troubleshooting](docs/troubleshooting.md)

## Write Mode and Permissions

RepoOperator has a built-in permission model to prevent accidental file modifications.

### Permission modes

| Mode | Behavior |
|---|---|
| `basic` (default) | Repository sandbox mode. RepoOperator can read files, prepare diffs, and run safe in-repo commands with guardrails. |
| `auto_review` | Daily-use mode for sandbox work plus approval cards for elevated actions, network access, and risky local tools. |
| `full_access` | Dangerous advanced mode for broader local access after a strong confirmation. Risky commands are still logged and previewed where practical. |

Auto-apply, auto-commit, and auto-push are never enabled by default.

### Setting permission mode

Open the web app and use the permission selector in the top bar:

- **Basic permissions** maps to `basic`.
- **Auto review** maps to `auto_review`.
- **Full access** maps to `full_access` after a confirmation prompt.

RepoOperator persists the selected mode in `~/.repooperator/config.json` while preserving the rest of your configuration. Advanced users can still override the mode for debugging with `REPOOPERATOR_PERMISSION_MODE`, but normal usage should go through the web UI.

Local tool access is capability-based and visible in the Debug page. RepoOperator currently detects `git` and `glab`, allows safe repository diagnostics, and requires explicit approval cards for mutating, networked, or risky commands.

### File edit workflow

1. Open a repository and switch to a branch you want to work on.
2. Click **Propose change** in the composer area.
3. Enter the relative file path and describe the change.
4. RepoOperator generates a proposed diff and shows it in the chat.
5. Review the diff before doing anything.
6. Click **Apply** to write the change to the file, or **Reject** to discard it.
7. No files are modified until you explicitly click Apply.

Applying changes only modifies files within the current repository and current branch.
Changes never escape the repository root.

### Branch management

When a git repository is open, the current branch is shown as a clickable pill in the repository
banner. Clicking it opens the branch panel where you can:

- see all local branches
- switch to an existing local branch
- create a new branch from any base branch and check it out immediately

Creating and switching branches modifies the actual local git working tree, so GitHub Desktop and
other git tools will reflect the change immediately.

## Current Capabilities

RepoOperator currently supports:

- CLI onboarding with repository provider and model connection setup
- one-command local runtime startup through `repooperator up`
- worker lifecycle management through the CLI
- bounded health checks through `doctor` and `status`
- GitLab and GitHub provider-backed repository open flows
- local project open flows with absolute paths
- visible project lists and branch lists in the web UI
- non-interactive clone/fetch for private repositories when provider credentials are configured
- repository questions through the local worker with persisted activity events
- query-aware repository context and tool-based file evidence gathering
- agent-core orchestration with request understanding, planning, safe tool execution, and final synthesis
- local branch listing, branch creation, and branch switching from the web UI
- approval-based write workflow: propose a change, review the diff, apply with one click
- write permission model with `read-only` (default) and `write-with-approval` modes
- write mode badge in the web UI header showing the active permission level

## Current Limitations

RepoOperator is still alpha-stage software. Important limitations:

- Editing, patch review, validation, commit, and merge-request workflows are still evolving.
- Hosted-to-local worker pairing is currently development-style rather than packaged desktop software.
- GitLab is the most exercised provider path today.
- GitHub and local project flows follow the same direction, but need more real-world hardening.
- Web chat history is currently local UI state, not a durable multi-user collaboration system.
- CI/Web E2E workflows are expected project surfaces, but this repository may not yet include complete workflow definitions.

## Roadmap

Near-term priorities:

- harden `repooperator up` and `repooperator down` across more local environments
- improve repository selection and branch-selection UX
- improve file-aware retrieval for code-specific questions
- show which files were used in each answer more consistently
- expand GitHub provider coverage
- add stronger web end-to-end coverage
- keep the npm-installed local runtime flow reliable across platforms

Longer-term direction:

- richer repository understanding workflows
- review and patch proposal flows
- team-friendly hosted UI pairing with local workers
- safer write workflows with explicit review and confirmation steps

See [docs/roadmap.md](docs/roadmap.md) for more detail.

## Contributing

Contributions are welcome, especially around:

- worker reliability and API contracts
- provider integrations
- web app usability
- retrieval quality
- onboarding and documentation
- security review and threat modeling

Start with [CONTRIBUTING.md](CONTRIBUTING.md). For bugs or feature requests, open an issue on [GitHub](https://github.com/jungin-kim/RepoOperator/issues).

## License

RepoOperator is released under the MIT License. See [LICENSE](LICENSE).
