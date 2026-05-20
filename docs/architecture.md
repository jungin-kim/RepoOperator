# RepoOperator Architecture

RepoOperator is a local-first coding agent proxy. The web app is the operator
surface, the local worker owns repository access, and the agent core coordinates
safe primitive actions against the checked-out repository.

## Current Components

- `apps/web`: Next.js chat UI, repository picker, permission controls, command approvals, proposal cards, and run rehydration.
- `apps/local-worker`: FastAPI worker bound to localhost. It opens repositories, persists threads and run events, enforces command/file safety, and executes agent actions.
- `packages/cli`: npm CLI that onboards configuration, starts/stops the local worker and web app, and bundles runtime sources for packaged installs.

## Agent Core

Active agent runs enter through:

- `run_controller_graph`
- `stream_controller_graph`

Despite the historical name, active execution is no longer routed by a classifier
graph. The current backend loop is:

1. `context_service.py` collects repository, branch, thread, skill, and prior-run context.
2. `request_understanding.py` extracts factual request hints: files, symbols, constraints, requested outputs, likely tool hints, and ambiguity.
3. `planner.py` chooses the next safe primitive action from structured evidence and tool specs.
4. `agent_loop.py` runs the plan/act/observe loop and checks cancellation, steering, approval, and budget limits.
5. `tool_orchestrator.py` invokes tools from `tools/registry.py` and `tools/builtin.py`.
6. `events.py` persists user-visible work trace events and technical events for rehydration and debugging.
7. `final_synthesis.py` builds and validates the final answer from gathered evidence.

`agent_core/classifier.py` and `ClassifierResult` remain compatibility-only. They
must not grow workflow-routing fields or drive planner behavior.

## LangGraph Runtime Migration

RepoOperator now has a real LangGraph `StateGraph` runtime under
`agent_core/langgraph_runtime.py`. The graph owns tested routing decisions,
stores JSON-safe checkpoint state, uses a LangGraph checkpointer mirrored into
run event storage, and resumes command-approval interrupts from the saved graph
checkpoint. `ToolOrchestrator` remains the only execution boundary for reads,
commands, and proposal generation.

Runtime selection is controlled by:

- `REPOOPERATOR_AGENT_RUNTIME=legacy|langgraph`
- `REPOOPERATOR_AGENT_RUNTIME_DEFAULT=legacy|langgraph`

Production default intentionally remains `legacy` until the parity matrix covers
the full backend route set under normal packaging: streamed final responses,
completed-while-away rehydrate, broad supervisor decomposition, and complex
multi-file create/modify/delete proposals. LangGraph-specific tests exercise
project summary, explicit routing without the legacy chooser, approval
interrupt/resume, event-service checkpoint restore, supervisor dispatch/reduce,
and first-class `ChangeSetProposal` payloads.

## Frontend Run State

The chat UI uses these current modules for agent activity and rehydration:

- `AgentActivityTranscript.tsx`
- `agent-activity-transcript.ts`
- `agent-activity-types.ts`
- `run-event-state.ts`
- `thread-persistence.ts`

Completed and active runs are reconstructed from persisted run events before
falling back to final-result activity archives. Debug and secondary events stay
persisted even when the primary transcript hides them.

## Safety Boundaries

- File reads and proposal generation stay inside the active repository.
- Generated edits are proposal-only unless an explicit apply path is used.
- Commands are previewed through command policy before execution.
- Mutating or risky commands require approval.
- Secret and artifact safety checks are enforced before responses or proposals are surfaced.
- Hidden/private reasoning is filtered; only safe summaries may enter events or UI.

## Compatibility Paths

These are intentionally small and should not receive new behavior:

- `services/agent_orchestration_graph.py`: old run/stream import path that delegates to `controller_graph.py`.
- `agent_core/action_executor.py`: old executor class that delegates to `ToolOrchestrator`.
- `agent_core/classifier.py`: old classifier import path backed by `RequestUnderstanding`.
- `services/agent_graph.py`: deprecated read-only graph retained for old direct imports, not used by active run endpoints.

Prefer adding tests against the current modules instead of expanding these adapters.
