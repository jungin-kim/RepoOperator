# RepoOperator LangGraph Status Taxonomy

RepoOperator is LangGraph-only. The normal runtime entrypoints are
`run_langgraph_controller`, `stream_langgraph_controller`, and
`resume_langgraph_controller`.

This note separates graph control, run lifecycle, action/tool results,
change-set workflow state, validation state, UI progress, and debug metadata.
Graph state exists for routing and checkpointing. It is not a UI rendering
contract.

## Status Sets

### RunStatus

Used only for run lifecycle records:

- `pending`
- `running`
- `waiting_approval`
- `cancelling`
- `completed`
- `failed`
- `cancelled`
- `timed_out`

### GraphNodeName

Node IDs are graph topology/control labels, not user-visible statuses. For
example, `validate_result` is the action-result observation/normalization node.
It is not a validation card source.

Validation-like node names mean:

- `validate_result`: observes and normalizes the latest action/tool result.
- `validate_change_set`: validates a proposal/change-set before approval.
- `post_apply_validation`, `select_validation_commands`,
  `preview_command`, `await_validation_approval`, `run_validation_command`,
  and `parse_validation_result`: post-apply validation flow.

### ActionResultStatus

Used for tool/action results:

- `success`
- `failed`
- `skipped`
- `blocked`
- `waiting_approval`
- `cancelled`
- `timed_out`

### ProgressStatus

Used by UI activity cards:

- `queued`
- `running`
- `completed`
- `failed`
- `waiting`
- `waiting_approval`
- `cancelled`

### ProposalStatus

Used only for change-set/proposal workflow:

- `draft`
- `valid`
- `invalid`
- `repairable`
- `awaiting_approval`
- `approved`
- `applied`
- `rejected`
- `failed`

### ValidationKind

Real validation only:

- `change_set`
- `post_apply`
- `command`
- `git`

### ValidationStatus

Real validation status only:

- `passed`
- `failed`
- `skipped`
- `blocked`
- `warning`

### EventKind

Backend event semantic kind:

- `graph_transition`
- `tool_action`
- `action_result`
- `validation`
- `proposal`
- `approval`
- `git`
- `web`
- `final_answer`
- `debug_rationale`

### EventAudience

UI/debug audience:

- `primary`
- `secondary`
- `debug`
- `internal`

## Rendering Rules

- `visible_rationale_log`, `evidence_basis`,
  `user_understanding_context`, and `context_pack_report` are debug/context
  metadata, not primary normal chat work log.
- `safe_reasoning_summary` must not become a normal transcript title/status by
  itself.
- `validation_result` must include a real `kind` or `source` before the UI may
  render a validation card.
- Final answer quality guard results are not `validation_result`. Use
  `answer_quality` or `final_answer_quality` if this metadata is needed, and
  keep it debug-only.
- GraphState top-level keys are for state control and checkpointing, not UI
  rendering contracts.
- UI progress should be built from concrete action/tool/proposal/validation
  events with fields such as `operation`, `action_type`, `tool_name`, `files`,
  `command`, `proposal_id`, or real validation `kind`/`source`.

## GraphState Inventory

### graph_control

- `request_snapshot`
- `run_id`
- `thread_id`
- `repo`
- `branch`
- `pending_action`
- `next_node`
- `routing_stage`
- `stop_reason`
- `loop_iteration`
- `graph_started_at`
- `checkpoint_sequence`
- `cancellation_requested`
- `stream_final_answer`
- `max_loop_iterations`
- `max_file_reads`
- `max_commands`
- `max_edits`
- `budgets`

### run_context

- `context_packet`
- `ide_context`
- `capability_snapshot`
- `model_profile_snapshot`
- `context_pack_summary`
- `routine_context`
- `skills_used`
- `skills_context`
- `memories_used`

### model_context

- `request_understanding_snapshot`
- `classifier_snapshot`
- `task_frame_snapshot`
- `subtasks`
- `current_subtask_id`
- `plan`
- `evidence_store`
- `evidence_goal`
- `zero_result_queries`
- `failed_action_signatures`
- `strategy_shifts`

### understanding_debug

- `user_understanding_context`
- `understanding_history`
- `short_term_memory`
- `recommendation_context`
- `classifier_snapshot`
- `task_frame_snapshot`

### evidence_debug

- `evidence_basis`
- `visible_rationale_log`
- `evidence_basis_history`
- `context_pack_report`
- `evidence_reports`
- `file_role_reports`
- `risk_notes`

### action_history

- `actions_taken`
- `action_results`
- `files_read`
- `files_changed`
- `commands_run`
- `pending_approval`

### artifact_workflow

- `change_set_proposal`
- `proposed_changes`
- `proposal_errors`
- `repair_attempts`
- `attempts`
- `edit_mode`
- `applied_change_set_id`
- `approval_decision`

### validation_workflow

- `validation_results`
- `validation_command_selection`

### git_workflow

- `git_workflow`

### supervisor_workflow

- `supervisor_mode`
- `current_worker_role`
- `subtask_updates`
- `worker_tasks`
- `worker_reports`

### ui_event_buffer

- `events_to_emit`
- `final_response`
- `response_snapshot`
- `current_step`

### deprecated_or_duplicate

These are kept for compatibility in the current patch but should not become new
UI contracts:

- `proposal_id`
- `proposal_status`
- `apply_status`
- `post_apply_validation_status`
- `validation_done`
- `edit_done`
- `analysis_done`
- `evidence_done`
- `current_step`
- `messages`
- `events`
- `observations`

## Cleanup Notes

- Prefer `change_set_proposal.status` and proposal workflow fields over
  duplicated top-level proposal/apply status keys.
- Prefer `validation_results[]` entries with real `kind`/`source` and normalized
  `status` over bare top-level `post_apply_validation_status`.
- Keep `validate_result` documented as action result normalization until a
  larger graph topology rename can be done safely.
- Keep debug rationale available through Debug Context, but mark rationale and
  evidence events as `kind="debug_rationale"` and `audience="debug"`.
