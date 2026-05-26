# Troubleshooting

This guide covers common problems when running RepoOperator locally.

## Worker Health Check Fails

Symptoms:

- the web UI shows the worker as unavailable
- `GET /health` fails

Checks:

1. Make sure the worker process is running.
2. Confirm it is bound to `127.0.0.1:8000`.
3. Run:

```bash
curl http://127.0.0.1:8000/health
```

If this fails locally, fix the worker before debugging the web app.

## Web App Cannot Reach The Worker

Symptoms:

- the web UI reports connection errors
- repository open or task submission fails immediately

Checks:

1. Confirm `NEXT_PUBLIC_LOCAL_WORKER_BASE_URL` is set correctly.
2. Confirm the worker is listening on the same URL.
3. Restart the web app after changing environment variables.
4. For slow local inference, increase `REPOOPERATOR_WORKER_PROXY_AGENT_TIMEOUT_MS` instead of keeping the default short health-check timeout.

Default local value:

```text
http://127.0.0.1:8000
```

Recommended local inference proxy values:

```text
REPOOPERATOR_WORKER_PROXY_TIMEOUT_MS=5000
REPOOPERATOR_WORKER_PROXY_AGENT_TIMEOUT_MS=60000
```

## Repository Open Fails

Symptoms:

- `/repo/open` returns an error

Checks:

1. Confirm `LOCAL_REPO_BASE_DIR` exists or can be created.
2. For GitLab or GitHub, confirm `project_path` matches the provider path exactly. For local projects, confirm `project_path` is an absolute filesystem path that exists.
3. Confirm the git provider configuration in `~/.repooperator/config.json` matches the selected provider.
4. For advanced override flows, confirm provider environment overrides are correct.
5. Confirm the branch exists on the remote or locally when the selected source is a git repository.

## GitLab Authentication Fails

Symptoms:

- clone or push fails
- merge request creation returns an API error
- git reports it could not read a username or that terminal prompts are disabled

Checks:

1. Confirm `~/.repooperator/config.json` contains the correct GitLab base URL and token.
2. Confirm the token is valid and not expired.
3. Confirm the token has sufficient repository and API permissions.
4. Confirm the `project_path` matches the GitLab project path.
5. If the error says the repository was not found, confirm the token can access that private repository.

## GitHub Authentication Fails

Symptoms:

- clone or push fails for GitHub-backed repositories

Checks:

1. Confirm `~/.repooperator/config.json` contains the correct GitHub base URL and token.
2. Confirm the token is valid and has repository access.
3. Confirm the `project_path` matches the GitHub owner and repository path.

## Model API Request Fails

Symptoms:

- `/agent/run` or `/agent/propose-file` returns an API error

Checks:

1. Confirm `OPENAI_BASE_URL` is correct.
2. Confirm `OPENAI_API_KEY` is valid.
3. Confirm `OPENAI_MODEL` is supported by that endpoint.
4. Confirm the machine can reach the configured API host.

## File Write Fails

Symptoms:

- `/fs/write` returns a path error

Checks:

1. Confirm `project_path` is under `LOCAL_REPO_BASE_DIR`.
2. Confirm `relative_path` is actually relative.
3. Confirm the path does not attempt to escape the repository root.
4. Confirm the write is not targeting `.git` internals.

## Validation Command Fails

Symptoms:

- `/cmd/run` returns a non-zero exit code
- stdout or stderr contains tool failures

Checks:

1. Review the exact command shown in the UI or API response.
2. Review `stdout` and `stderr`.
3. Confirm the required tool is installed locally.
4. Confirm the command works when run manually in the repository.
5. If it times out, increase `timeout_seconds` explicitly in the request.

## Commit Fails

Symptoms:

- `/git/commit` reports that there are no staged changes
- `git commit` fails with author or identity errors

Checks:

1. Confirm there are actual changes in the repository.
2. Confirm your local git identity is configured.
3. Confirm the repository is in a valid git state.

Example:

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

## Duplicate Conflict Files Appear

Symptoms:

- files such as `skills 2.py`, `package-lock 6.json`, or `cache 2` appear after repeated edits, tests, or builds
- git refs such as `origin/HEAD 2` appear under `.git/refs`, `.git/logs/refs`, or `.git/packed-refs`

Checks:

1. Run `python3 scripts/check-workspace-hygiene.py`.
2. If `origin/HEAD 2` or similar git refs appear, delete the broken duplicate ref files, then run `git remote prune origin` and `git fetch origin`.
3. Do not keep the repository inside an iCloud, Dropbox, OneDrive, or other cloud-sync folder if `.git` refs are being duplicated.
4. If duplicate files keep appearing, disable sync conflict handling for this repo or move it to a non-synced directory.

Generated caches such as `apps/web/.next/`, `apps/web/test-results/`, and `packages/cli/runtime/` can be removed and regenerated. Source, test, docs, and package files with conflict suffixes should be investigated before deletion unless they are known generated artifacts.

## Push Fails

Symptoms:

- `/git/push` returns an authentication or remote error

Checks:

1. Confirm the branch exists locally.
2. Confirm the remote name is correct.
3. Confirm credentials or provider token settings are valid.
4. Confirm the remote branch is allowed to be pushed.

## Merge Request Creation Fails

Symptoms:

- `/git/merge-request` returns a GitLab API error

Checks:

1. Confirm the source branch has already been pushed.
2. Confirm the target branch exists.
3. Confirm `project_path` matches the GitLab project namespace and repo name.
4. Review the API error body returned by the worker.

## When In Doubt

Collect these details before opening an issue:

- worker version or current branch
- operating system
- exact request payload
- exact error message
- whether the same command works manually outside RepoOperator

If you file an issue, include sanitized logs and omit secrets or tokens.
