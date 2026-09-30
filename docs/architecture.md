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

## Decision Core: model-driven vs. deterministic

RepoOperator supports two planners behind one decision seam
(`choose_graph_next_action` in `graph_routes.py`):

- **Model-driven agentic loop** (`agent_core/agentic_loop.py`): when native
  tool calling is enabled (`model.toolCalling`/`REPOOPERATOR_AGENTIC_TOOL_CALLING`
  with a tool-calling-capable model), the model sees the real tool schemas plus
  the running think -> act -> observe transcript and emits the next tool call.
  This is the primary planner and is what makes RepoOperator behave like an
  autonomous agent rather than a scripted pipeline.
- **Deterministic choosers**: a priority chain of rule-based proposers
  (explicit targets, symbols, policy evidence, single-step model proposal,
  commands, search candidates, edit, project summary) runs as a safety fallback
  whenever the model is unavailable, declines, or proposes a repeat/ineffective
  action.

Policy checks on the model's choice (answer without evidence, describe a
change instead of applying it, patch during a read-only question, skip a
requested command) are returned to the model as a failed tool result
(`{"ok": false, "code", "problem", "hint"}`) and the model gets one retry.
Only a repeated violation hands the step to the deterministic choosers.

The system prompt lives in `src/repooperator_worker/prompts/agent.md` and is
re-read when the file changes; `~/.repooperator/prompts/agent.md` overrides it.

The transcript is append-only per step so provider prompt caches keep hitting:
the task turn is snapshotted once per run, per-step status goes last, the
action window slides in blocks, and the tool list is fixed for the run.
Read-only calls the model batches in one response are queued and run without
extra model round trips. Token usage (incl. cached input) is tracked per run
(`services/usage_tracker.py`) and returned as `model_usage` on the response.

Both planners produce the same `AgentAction` shape, so execution, permission
gating, secret redaction, and budgets are enforced identically by the tool
orchestrator regardless of which planner chose the action. Native tool calling
is provided by `services/model_client.py` (OpenAI-compatible + native Anthropic)
and `services/model_tools.py` (schema conversion and response parsing).

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

### Live streaming and plan

The whole run streams over SSE, not just the final answer:

- Tool selection and results stream as `progress_delta` events and render live in
  `AgentActivityTranscript.tsx`.
- Final-answer tokens stream as `assistant_delta` events.
- A live plan/todo checklist (`LivePlan.tsx`) is driven by the structured
  `aggregate.plan` on the `langgraph-plan` event (emitted by `emit_plan_update`):
  each subtask renders with a live status (pending/running/completed/failed).
  `livePlanFromEvent` in `run-event-state.ts` extracts it on both the SSE and the
  poll/reconnect paths.

## Multi-agent supervisor

Broad, repository-wide requests fan out through the supervisor subgraph. When
native tool calling is enabled, each work unit runs a real model-backed subagent
(`agent_core/subagent.py`): a bounded think -> act -> observe loop scoped to a
role and file group, using only read-only evidence tools through the shared
ToolOrchestrator, then synthesizing a structured report. When tool calling is
off, the supervisor falls back to its deterministic worker behavior. Mutation,
commands, git, and network stay with the parent graph and its approval gates —
subagents never escalate privileges.

## Plugins via MCP

External tools are provided through the Model Context Protocol. Servers are
declared in `~/.repooperator/mcp.json` (or a repo `.repooperator/mcp.json`) and
surfaced as `mcp_<server>_<tool>` tools in the registry. `services/mcp_client.py`
implements a dependency-free MCP client:

- **stdio**: spawns the server command and exchanges newline-delimited JSON-RPC
  2.0 messages (initialize handshake, `tools/list`, `tools/call`), reading
  responses on a background thread so timeouts are reliable.
- **http/sse**: POSTs JSON-RPC and accepts a JSON body or SSE `data:` frame.

Servers are connected on demand per tool call. MCP tool execution stays behind
the same approval gate as any networked/mutating tool: `check_permission`
returns `ask`, and only after approval does the adapter connect and invoke.

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
