from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from repooperator_worker.agent_core.events import append_activity_event
from repooperator_worker.agent_core.planner import _existing_target_files
from repooperator_worker.agent_core.state import AgentCoreState
from repooperator_worker.schemas import AgentRunRequest
from repooperator_worker.services.json_safe import json_safe, safe_repr
from repooperator_worker.services.model_client import ModelGenerationRequest, OpenAICompatibleModelClient


SUPPORTED_STEERING_TYPES = {
    "add_target_file",
    "change_output_format",
    "cancel",
    "continue",
    "defer",
    "unknown",
}


STEERING_PROMPT = """\
You are RepoOperator's steering parser. Return JSON only.
Schema:
{
  "steering_type": "add_target_file|change_output_format|cancel|continue|defer|unknown",
  "target_files": [],
  "output_format": null,
  "confidence": 0.0,
  "reason": "short explanation"
}
Extract only a structured steering decision for an already-running agent. Do not route by language keywords.
"""


@dataclass
class SteeringDecision:
    steering_type: str = "unknown"
    target_files: list[str] | None = None
    output_format: str | None = None
    confidence: float = 0.0
    reason: str = ""


def consume_steering_for_state(state: AgentCoreState, request: AgentRunRequest) -> None:
    try:
        from repooperator_worker.services.agent_run_coordinator import consume_steering

        items = consume_steering(state.run_id)
    except Exception:
        items = []
    for item in items:
        content = str(item.get("content") or "").strip()
        state.steering_instructions.append(item)
        applied = False
        decision = parse_steering_instruction(content, request, state)
        target_files = _existing_target_files(request, decision.target_files or [])
        if decision.steering_type == "add_target_file" and target_files:
            existing = list(state.classifier_result.target_files)
            for path in target_files:
                if path not in existing:
                    existing.append(path)
            state.classifier_result.target_files = existing
            applied = True
        if decision.steering_type == "change_output_format" and decision.output_format and decision.confidence >= 0.65:
            state.observations.append(f"Steering requested output format: {decision.output_format}.")
            applied = True
        if decision.steering_type == "cancel" and decision.confidence >= 0.8:
            state.cancellation_requested = True
            state.stop_reason = "cancelled"
            applied = True
        event_type = "steering_applied" if applied else "steering_deferred"
        append_activity_event(
            run_id=state.run_id,
            request=request,
            activity_id=f"controller-steering:{item.get('id') or len(state.steering_instructions)}",
            event_type="activity_completed",
            phase="Planning",
            label="Updated plan from steering" if applied else "Steering deferred",
            status="completed",
            observation=(
                decision.reason or "Steering updated structured run state."
                if applied
                else decision.reason or "Steering was recorded, but it did not safely map to the current action."
            ),
            detail=content[:220],
            aggregate={"steering_event_type": event_type, "decision": json_safe(decision)},
        )


def parse_steering_instruction(content: str, request: AgentRunRequest, state: AgentCoreState) -> SteeringDecision:
    raw_content = (content or "").strip()
    if not raw_content:
        return SteeringDecision(steering_type="defer", target_files=[], confidence=0.0, reason="Empty steering instruction.")
    try:
        raw = OpenAICompatibleModelClient().generate_text(
            ModelGenerationRequest(
                system_prompt=STEERING_PROMPT,
                user_prompt=json.dumps(
                    {
                        "task": request.task,
                        "steering": raw_content,
                        "current_target_files": state.classifier_result.target_files,
                        "files_read": state.files_read,
                        "stop_reason": state.stop_reason,
                    },
                    ensure_ascii=False,
                ),
            )
        )
        decision = _validate_steering_payload(_parse_json(raw))
        if decision.steering_type != "unknown" and decision.confidence >= 0.5:
            return decision
    except Exception as exc:  # noqa: BLE001
        return SteeringDecision(steering_type="defer", target_files=[], confidence=0.0, reason=f"Steering parser unavailable: {safe_repr(exc, limit=160)}")

    file_targets = _file_tokens(raw_content)
    if file_targets:
        return SteeringDecision(
            steering_type="add_target_file",
            target_files=file_targets,
            confidence=0.55,
            reason="Detected explicit file path tokens; paths still require repository containment validation.",
        )
    return SteeringDecision(steering_type="defer", target_files=[], confidence=0.0, reason="No safe structured steering decision was available.")


def _validate_steering_payload(payload: dict[str, Any]) -> SteeringDecision:
    steering_type = str(payload.get("steering_type") or "unknown")
    if steering_type not in SUPPORTED_STEERING_TYPES:
        steering_type = "unknown"
    target_files = [str(item).strip().lstrip("/") for item in payload.get("target_files") or [] if str(item).strip()]
    try:
        confidence = float(payload.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return SteeringDecision(
        steering_type=steering_type,
        target_files=target_files,
        output_format=str(payload.get("output_format")) if payload.get("output_format") else None,
        confidence=max(0.0, min(1.0, confidence)),
        reason=str(payload.get("reason") or ""),
    )


def _parse_json(text: str) -> dict[str, Any]:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines)
    try:
        payload = json.loads(stripped)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        return {}


def _file_tokens(text: str) -> list[str]:
    from repooperator_worker.agent_core.request_parsing import extract_file_tokens

    return extract_file_tokens(text)
