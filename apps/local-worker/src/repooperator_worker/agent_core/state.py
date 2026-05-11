from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING

from repooperator_worker.agent_core.actions import AgentAction, ActionResult

if TYPE_CHECKING:
    from repooperator_worker.agent_core.request_understanding import RequestUnderstanding


@dataclass
class ClassifierResult:
    """Backward-compatibility struct only.

    Do NOT add old workflow-routing fields here. Those fields are intentionally absent.
    The planner must never branch on these fields.
    Populated exclusively by request_understanding_to_classifier_result().
    """
    intent: str = "ambiguous"
    confidence: float = 0.0
    target_files: list[str] = field(default_factory=list)
    target_symbols: list[str] = field(default_factory=list)
    requested_action: str = ""
    needs_tool: str | None = None
    needs_clarification: bool = False
    clarification_question: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentSubtask:
    id: str
    title: str
    goal: str
    status: Literal["pending", "running", "completed", "blocked", "failed"] = "pending"
    planned_operations: list[str] = field(default_factory=list)
    completed_operations: list[str] = field(default_factory=list)
    evidence_files: list[str] = field(default_factory=list)
    attempts: int = 0
    blocker: str | None = None


@dataclass
class AgentCoreState:
    run_id: str
    thread_id: str | None
    repo: str
    branch: str | None
    user_task: str
    classifier_result: ClassifierResult = field(default_factory=ClassifierResult)
    # Preferred: request understanding facts (populated by understand_request).
    # classifier_result above is kept only for backward compatibility.
    request_understanding: Any | None = None
    plan: list[str] = field(default_factory=list)
    current_step: str | None = None
    observations: list[str] = field(default_factory=list)
    actions_taken: list[AgentAction] = field(default_factory=list)
    action_results: list[ActionResult] = field(default_factory=list)
    files_read: list[str] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)
    commands_run: list[str] = field(default_factory=list)
    pending_approval: dict[str, Any] | None = None
    steering_instructions: list[dict[str, Any]] = field(default_factory=list)
    cancellation_requested: bool = False
    skills_used: list[str] = field(default_factory=list)
    memories_used: list[str] = field(default_factory=list)
    recommendation_context: dict[str, Any] | None = None
    context_packet: dict[str, Any] | None = None
    stop_reason: str | None = None
    final_response: str = ""
    loop_iteration: int = 0
    max_loop_iterations: int = 8
    max_file_reads: int = 40
    max_commands: int = 8
    max_edits: int = 6
    subtasks: list[AgentSubtask] = field(default_factory=list)
    current_subtask_id: str | None = None
    zero_result_queries: list[str] = field(default_factory=list)
    failed_action_signatures: list[str] = field(default_factory=list)
    strategy_shifts: list[str] = field(default_factory=list)
