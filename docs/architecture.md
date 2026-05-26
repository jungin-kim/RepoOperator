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

- `run_langgraph_controller`
- `stream_langgraph_controller`
- `resume_langgraph_controller`

Normal runtime execution is LangGraph-only. The current backend path is:

1. `context_service.py` collects repository, branch, thread, skill, and prior-run context.
2. `request_understanding.py` extracts factual request hints: files, symbols, constraints, requested outputs, likely tool hints, and ambiguity.
3. `agent_core/graph/` owns the LangGraph state graph, route functions, subgraphs, nodes, checkpoint restore, and finalization.
4. `planner.py` chooses safe primitive actions from structured evidence and tool specs.
5. `tool_orchestrator.py` invokes tools from `tools/registry.py` and `tools/builtin.py`.
6. `events.py` persists user-visible work trace events and technical events for rehydration and debugging.
7. `final_synthesis.py` builds and validates the final answer from gathered evidence.

Request understanding is not an authoritative workflow router. It may provide
weak tool hints, but graph routing and planner decisions must stay grounded in
safe primitive actions, gathered evidence, and validator results.

## Graph Support Layout

Graph helper behavior is split by responsibility:

- `repository_support.py`: active repository validation.
- `context_support.py`: context loading and context-pack refresh.
- `understanding_support.py`: request-understanding setup and initial budget wiring.
- `budget_support.py`: loop budgets and continuation checks.
- `cancellation_support.py`: cancellation checkpoints.
- `observation_support.py`: observation recording and plan updates.
- `trace_support.py`: user-visible action trace events.
- `final_answer_support.py`: final answer text and response assembly.
- `support.py`: small re-export surface for graph-internal imports and tests.

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

The normal agent runtime has no older-runtime fallback. Remaining compatibility
objects should be local, small, and unable to drive routing. Tests should target
the LangGraph entry points and focused support modules directly.
