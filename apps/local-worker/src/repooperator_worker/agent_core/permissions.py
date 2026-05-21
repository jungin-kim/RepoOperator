from __future__ import annotations

import fnmatch
import ipaddress
import json
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Literal
from urllib.parse import urlparse


class PermissionMode(str, Enum):
    DEFAULT = "default"
    PLAN_ONLY = "plan_only"
    PROPOSAL_ONLY = "proposal_only"
    ACCEPT_EDITS = "accept_edits"
    AUTO_READONLY = "auto_readonly"
    FULL_ACCESS = "full_access"
    ROUTINE_SAFE = "routine_safe"
    HEADLESS_SAFE = "headless_safe"

    # Backward-compatible aliases for pre-mode-expansion code/tests.
    PLAN = "plan_only"
    AUTO = "accept_edits"
    BYPASS = "full_access"


PermissionDecisionValue = Literal["allow", "deny", "ask"]


class PermissionRuleSource(str, Enum):
    BASE_SAFETY = "base_safety"
    WORKSPACE_PATH = "workspace_path"
    PROVIDER_NETWORK = "provider_network"
    COMMAND_POLICY = "command_policy"
    PROJECT = "project"
    USER = "user"
    SESSION = "session"
    MODE = "mode"
    TOOL_DEFAULT = "tool_default"

    # Compatibility name: old SYSTEM rules are now base-safety rules.
    SYSTEM = "base_safety"


class PermissionMatcherKind(str, Enum):
    NONE = "none"
    PATH = "path"
    COMMAND = "command"
    DOMAIN_URL = "domain_url"
    GIT_REMOTE_BRANCH = "git_remote_branch"
    CHANGE_SET = "change_set"


_SOURCE_PRECEDENCE = {
    PermissionRuleSource.BASE_SAFETY: 1,
    PermissionRuleSource.WORKSPACE_PATH: 2,
    PermissionRuleSource.PROVIDER_NETWORK: 3,
    PermissionRuleSource.COMMAND_POLICY: 4,
    PermissionRuleSource.PROJECT: 5,
    PermissionRuleSource.USER: 6,
    PermissionRuleSource.SESSION: 6,
    PermissionRuleSource.MODE: 7,
    PermissionRuleSource.TOOL_DEFAULT: 8,
}
_DECISION_WEIGHT = {"deny": 3, "ask": 2, "allow": 1}

_PROPOSAL_TOOLS = {"generate_edit", "generate_change_set", "validate_change_set"}
_APPLY_TOOLS = {"apply_change_set"}
_DIRECT_WRITE_TOOLS = {"create_file", "modify_file", "delete_file", "rename_file"}
_GIT_WRITE_TOOLS = {"git_branch_create", "git_commit", "git_push", "github_create_pr", "gitlab_create_mr"}
_REMOTE_WRITE_TOOLS = {"git_push", "github_create_pr", "gitlab_create_mr"}
_COMMAND_TOOLS = {"preview_command", "inspect_git_state", "run_approved_command", "run_validation_command"}
_NETWORK_TOOLS = {"search_web", "fetch_url"}
_READ_SEARCH_TOOLS = {
    "inspect_repo_tree",
    "search_files",
    "search_text",
    "read_file",
    "read_many_files",
    "analyze_repository",
    "summarize_web_evidence",
    "ask_clarification",
    "final_answer",
}
_GIT_READ_TOOLS = {"git_status", "git_diff", "git_log"}
_GIT_STATUS_DIFF_TOOLS = {"git_status", "git_diff"}


@dataclass(frozen=True)
class PermissionDecision:
    decision: PermissionDecisionValue
    reason: str = ""
    approval_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = PermissionRuleSource.TOOL_DEFAULT.value
    priority: int = _SOURCE_PRECEDENCE[PermissionRuleSource.TOOL_DEFAULT]
    mode: str = PermissionMode.DEFAULT.value
    denial_code: str | None = None
    recovery_hint: str | None = None
    approval_payload: dict[str, Any] | None = None
    audit_record: dict[str, Any] | None = None

    @classmethod
    def allow(
        cls,
        reason: str = "",
        *,
        source: PermissionRuleSource | str = PermissionRuleSource.TOOL_DEFAULT,
        priority: int | None = None,
        mode: PermissionMode | str | None = None,
        approval_payload: dict[str, Any] | None = None,
        **metadata: Any,
    ) -> "PermissionDecision":
        source_value = _source_value(source)
        return cls(
            decision="allow",
            reason=reason,
            metadata=metadata,
            source=source_value,
            priority=_priority_value(source_value, priority),
            mode=_mode_value(mode),
            approval_payload=approval_payload,
        )

    @classmethod
    def deny(
        cls,
        reason: str = "",
        *,
        source: PermissionRuleSource | str = PermissionRuleSource.TOOL_DEFAULT,
        priority: int | None = None,
        mode: PermissionMode | str | None = None,
        denial_code: str | None = None,
        recovery_hint: str | None = None,
        approval_payload: dict[str, Any] | None = None,
        **metadata: Any,
    ) -> "PermissionDecision":
        source_value = _source_value(source)
        return cls(
            decision="deny",
            reason=reason,
            metadata=metadata,
            source=source_value,
            priority=_priority_value(source_value, priority),
            mode=_mode_value(mode),
            denial_code=denial_code,
            recovery_hint=recovery_hint,
            approval_payload=approval_payload,
        )

    @classmethod
    def ask(
        cls,
        reason: str = "",
        *,
        approval_id: str | None = None,
        source: PermissionRuleSource | str = PermissionRuleSource.TOOL_DEFAULT,
        priority: int | None = None,
        mode: PermissionMode | str | None = None,
        denial_code: str | None = None,
        recovery_hint: str | None = None,
        approval_payload: dict[str, Any] | None = None,
        **metadata: Any,
    ) -> "PermissionDecision":
        source_value = _source_value(source)
        return cls(
            decision="ask",
            reason=reason,
            approval_id=approval_id,
            metadata=metadata,
            source=source_value,
            priority=_priority_value(source_value, priority),
            mode=_mode_value(mode),
            denial_code=denial_code,
            recovery_hint=recovery_hint,
            approval_payload=approval_payload,
        )

    def model_dump(self) -> dict[str, Any]:
        return _json_safe(
            {
                "decision": self.decision,
                "reason": self.reason,
                "approval_id": self.approval_id,
                "metadata": self.metadata,
                "source": self.source,
                "priority": self.priority,
                "mode": self.mode,
                "denial_code": self.denial_code,
                "recovery_hint": self.recovery_hint,
                "approval_payload": self.approval_payload,
                "audit_record": self.audit_record,
            }
        )


@dataclass(frozen=True)
class ToolPermissionContext:
    request: Any
    run_id: str
    permission_mode: PermissionMode = PermissionMode.DEFAULT
    active_repository: str | None = None
    prior_denials: list[dict[str, Any]] = field(default_factory=list)
    reason: str | None = None


@dataclass(frozen=True)
class PermissionRule:
    id: str
    source: PermissionRuleSource
    tool_name: str
    decision: PermissionDecisionValue
    reason: str
    priority: int
    pattern: str | None = None
    predicate_name: str | None = None
    predicate: Callable[[dict[str, Any], ToolPermissionContext], bool] | None = None
    matcher_kind: PermissionMatcherKind | str = PermissionMatcherKind.NONE

    def matches(self, tool_name: str, payload: dict[str, Any], context: ToolPermissionContext) -> bool:
        if self.tool_name not in {"*", tool_name}:
            return False
        if self.predicate:
            return bool(self.predicate(payload, context))
        if self.pattern:
            return _pattern_matches(str(self.matcher_kind), self.pattern, payload)
        return True

    def model_dump(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source.value,
            "tool_name": self.tool_name,
            "decision": self.decision,
            "reason": self.reason,
            "priority": self.priority,
            "pattern": self.pattern,
            "predicate_name": self.predicate_name,
            "matcher_kind": str(self.matcher_kind.value if isinstance(self.matcher_kind, Enum) else self.matcher_kind),
        }


@dataclass(frozen=True)
class PermissionAuditRecord:
    run_id: str
    tool_name: str
    decision: PermissionDecisionValue
    matched_rules: list[dict[str, Any]]
    base_decision: dict[str, Any] | None = None
    command_preview: dict[str, Any] | None = None
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    reason: str = ""
    source: str = PermissionRuleSource.TOOL_DEFAULT.value
    priority: int = _SOURCE_PRECEDENCE[PermissionRuleSource.TOOL_DEFAULT]
    mode: str = PermissionMode.DEFAULT.value
    denial_code: str | None = None
    recovery_hint: str | None = None
    approval_payload: dict[str, Any] | None = None
    action_signature: str | None = None
    repeated_denial: bool = False

    def model_dump(self) -> dict[str, Any]:
        return _json_safe(
            {
                "run_id": self.run_id,
                "tool_name": self.tool_name,
                "decision": self.decision,
                "matched_rules": self.matched_rules,
                "base_decision": self.base_decision,
                "command_preview": self.command_preview,
                "timestamp": self.timestamp,
                "reason": self.reason,
                "source": self.source,
                "priority": self.priority,
                "mode": self.mode,
                "denial_code": self.denial_code,
                "recovery_hint": self.recovery_hint,
                "approval_payload": self.approval_payload,
                "action_signature": self.action_signature,
                "repeated_denial": self.repeated_denial,
            }
        )


class PermissionPolicy:
    SOURCE_WEIGHT = {source: 10_000 - precedence for source, precedence in _SOURCE_PRECEDENCE.items()}
    DECISION_WEIGHT = _DECISION_WEIGHT

    def __init__(self, rules: list[PermissionRule] | None = None) -> None:
        self.rules = list(rules or [])

    def evaluate(
        self,
        *,
        tool_name: str,
        payload: dict[str, Any],
        context: ToolPermissionContext,
        base_decision: PermissionDecision,
    ) -> tuple[PermissionDecision, PermissionAuditRecord]:
        mode = permission_mode_from_value(context.permission_mode)
        base_rule = PermissionRule(
            id=f"tool_default:{tool_name}",
            source=_source_from_value(base_decision.source),
            tool_name=tool_name,
            decision=base_decision.decision,
            reason=base_decision.reason or "Tool default decision.",
            priority=base_decision.priority,
        )
        matched = [rule for rule in self.rules if rule.matches(tool_name, payload, context)]
        matched.append(base_rule)
        selected = sorted(
            matched,
            key=lambda rule: (
                _source_precedence(rule.source),
                -int(rule.priority),
                -_DECISION_WEIGHT.get(rule.decision, 0),
            ),
        )[0]
        selected = self._enforce_base_safety(base_rule, selected, base_decision)
        decision = self._decision_from_selected(
            selected=selected,
            base_decision=base_decision,
            mode=mode,
            tool_name=tool_name,
            payload=payload,
        )
        decision = apply_mode_rules_to_decision(tool_name, payload, context, decision)
        signature = permission_action_signature(tool_name, payload)
        decision, repeated = self._apply_denial_tracking(decision, context, signature)
        audit = PermissionAuditRecord(
            run_id=context.run_id,
            tool_name=tool_name,
            decision=decision.decision,
            matched_rules=[rule.model_dump() for rule in matched],
            base_decision=base_rule.model_dump(),
            command_preview=base_decision.metadata.get("command_preview"),
            reason=decision.reason,
            source=decision.source,
            priority=decision.priority,
            mode=decision.mode,
            denial_code=decision.denial_code,
            recovery_hint=decision.recovery_hint,
            approval_payload=decision.approval_payload,
            action_signature=signature,
            repeated_denial=repeated,
        )
        return replace(decision, audit_record=audit.model_dump()), audit

    def _decision_from_selected(
        self,
        *,
        selected: PermissionRule,
        base_decision: PermissionDecision,
        mode: PermissionMode,
        tool_name: str,
        payload: dict[str, Any],
    ) -> PermissionDecision:
        metadata = {
            **base_decision.metadata,
            "matched_permission_rule": selected.model_dump(),
        }
        approval_payload = base_decision.approval_payload or _approval_payload_from_metadata(base_decision.metadata) or _approval_payload_for(tool_name, payload)
        denial_code = base_decision.denial_code
        recovery_hint = base_decision.recovery_hint
        if selected.decision == "deny" and not denial_code:
            denial_code = _denial_code_for_rule(selected)
        if selected.decision in {"ask", "deny"} and not recovery_hint:
            recovery_hint = _recovery_hint_for(tool_name, selected.decision)
        return PermissionDecision(
            decision=selected.decision,
            reason=selected.reason,
            approval_id=base_decision.approval_id,
            metadata=metadata,
            source=selected.source.value,
            priority=_source_precedence(selected.source),
            mode=mode.value,
            denial_code=denial_code,
            recovery_hint=recovery_hint,
            approval_payload=approval_payload,
        )

    def _apply_denial_tracking(
        self,
        decision: PermissionDecision,
        context: ToolPermissionContext,
        signature: str,
    ) -> tuple[PermissionDecision, bool]:
        prior = [
            item
            for item in context.prior_denials
            if isinstance(item, dict)
            and item.get("signature") == signature
            and item.get("decision") in {"ask", "deny"}
        ]
        if not prior:
            return decision, False
        if decision.decision == "allow" and _has_explicit_approval(decision.metadata, decision.approval_payload):
            return decision, False
        if decision.decision == "ask":
            return (
                replace(
                    decision,
                    decision="deny",
                    reason="Approval was already requested or denied for this action in this run; blocking repeat prompt.",
                    source=PermissionRuleSource.BASE_SAFETY.value,
                    priority=_SOURCE_PRECEDENCE[PermissionRuleSource.BASE_SAFETY],
                    denial_code="repeated_approval_required",
                    recovery_hint="Wait for the existing approval decision or start a new run with a changed request.",
                ),
                True,
            )
        if decision.decision == "deny":
            return (
                replace(
                    decision,
                    denial_code=decision.denial_code or "repeated_denial",
                    recovery_hint=decision.recovery_hint or "Change the request before retrying this denied action.",
                ),
                True,
            )
        return decision, False

    def _enforce_base_safety(
        self,
        base_rule: PermissionRule,
        selected: PermissionRule,
        base_decision: PermissionDecision,
    ) -> PermissionRule:
        if selected.source == PermissionRuleSource.BASE_SAFETY and selected.decision == "deny":
            return selected
        command_preview = base_decision.metadata.get("command_preview")
        if command_preview and base_decision.decision in {"ask", "deny"}:
            return base_rule
        if base_decision.decision == "deny":
            return base_rule
        if base_decision.decision == "ask" and selected.decision == "allow" and selected.source != PermissionRuleSource.BASE_SAFETY:
            return base_rule
        return selected


def apply_mode_rules_to_decision(
    tool_name: str,
    payload: dict[str, Any],
    context: ToolPermissionContext,
    decision: PermissionDecision,
) -> PermissionDecision:
    mode = permission_mode_from_value(context.permission_mode)
    decision = replace(decision, mode=mode.value)
    if decision.decision == "deny":
        return decision

    explicit_approval = _payload_has_allow_decision(payload) or bool(payload.get("approval_id"))
    if mode == PermissionMode.PLAN_ONLY:
        if tool_name in _PROPOSAL_TOOLS or _is_pure_read_tool(tool_name):
            return decision
        if tool_name in _NETWORK_TOOLS and explicit_approval:
            return decision
        return _mode_deny(
            decision,
            mode,
            "plan_only blocks writes, command execution, and unapproved network access.",
            denial_code="mode_plan_only",
            recovery_hint="Switch to proposal_only or accept_edits after planning, or request explicit approval for the blocked action.",
        )

    if mode == PermissionMode.PROPOSAL_ONLY:
        if tool_name in _APPLY_TOOLS or tool_name in _DIRECT_WRITE_TOOLS or _is_mutating_git_or_command(tool_name, decision):
            return _mode_deny(
                decision,
                mode,
                "proposal_only allows proposal generation and validation but denies apply/write operations.",
                denial_code="mode_proposal_only_write_denied",
                recovery_hint="Review the generated proposal and switch to accept_edits with approval to apply it.",
            )
        return decision

    if mode == PermissionMode.ACCEPT_EDITS:
        if tool_name in _APPLY_TOOLS:
            if explicit_approval and decision.decision == "allow":
                return decision
            if decision.decision == "allow":
                return _mode_ask(
                    decision,
                    mode,
                    "accept_edits requires an approval decision before applying a change set.",
                    approval_payload=_approval_payload_for(tool_name, payload),
                )
        return _guard_remote_write_without_approval(tool_name, payload, decision, mode)

    if mode == PermissionMode.AUTO_READONLY:
        if _is_auto_readonly_allowed(tool_name, decision):
            return decision
        return _mode_deny(
            decision,
            mode,
            "auto_readonly allows read/search/status/diff actions only; writes and remote mutations are denied.",
            denial_code="mode_auto_readonly_write_denied",
            recovery_hint="Switch to proposal_only for proposals or accept_edits with approval for writes.",
        )

    if mode == PermissionMode.FULL_ACCESS:
        return _guard_remote_write_without_approval(tool_name, payload, decision, mode)

    if mode == PermissionMode.ROUTINE_SAFE:
        guarded = _guard_remote_write_without_approval(tool_name, payload, decision, mode)
        if guarded.decision == "deny":
            return guarded
        if guarded.decision == "ask":
            return replace(
                guarded,
                source=PermissionRuleSource.MODE.value,
                priority=_SOURCE_PRECEDENCE[PermissionRuleSource.MODE],
                approval_payload=guarded.approval_payload or _approval_payload_for(tool_name, payload),
                recovery_hint="Queue this approval for a user decision; routine runs must not prompt interactively.",
            )
        return guarded

    if mode == PermissionMode.HEADLESS_SAFE:
        guarded = _guard_remote_write_without_approval(tool_name, payload, decision, mode)
        if guarded.decision == "ask":
            return _mode_deny(
                guarded,
                mode,
                "headless_safe cannot prompt for approval, so approval-gated actions fail closed.",
                denial_code="approval_required_headless",
                recovery_hint="Run this action interactively or provide explicit approval before headless execution.",
            )
        return guarded

    return decision


def permission_mode_from_value(value: str | PermissionMode | None) -> PermissionMode:
    if isinstance(value, PermissionMode):
        return value
    normalized = str(value or PermissionMode.DEFAULT.value).strip().lower().replace("-", "_")
    aliases = {
        "": PermissionMode.DEFAULT,
        "default": PermissionMode.DEFAULT,
        "basic": PermissionMode.DEFAULT,
        "read_only": PermissionMode.AUTO_READONLY,
        "readonly": PermissionMode.AUTO_READONLY,
        "plan": PermissionMode.PLAN_ONLY,
        "plan_only": PermissionMode.PLAN_ONLY,
        "proposal": PermissionMode.PROPOSAL_ONLY,
        "proposal_only": PermissionMode.PROPOSAL_ONLY,
        "auto": PermissionMode.ACCEPT_EDITS,
        "auto_review": PermissionMode.ACCEPT_EDITS,
        "autoreview": PermissionMode.ACCEPT_EDITS,
        "accept_edits": PermissionMode.ACCEPT_EDITS,
        "write_with_approval": PermissionMode.ACCEPT_EDITS,
        "full_access": PermissionMode.FULL_ACCESS,
        "fullaccess": PermissionMode.FULL_ACCESS,
        "bypass": PermissionMode.FULL_ACCESS,
        "routine_safe": PermissionMode.ROUTINE_SAFE,
        "headless_safe": PermissionMode.HEADLESS_SAFE,
        "auto_readonly": PermissionMode.AUTO_READONLY,
    }
    return aliases.get(normalized, PermissionMode.DEFAULT)


def permission_action_signature(tool_name: str, payload: dict[str, Any]) -> str:
    stable_payload = {
        key: _signature_value(value)
        for key, value in payload.items()
        if key
        not in {
            "action_id",
            "reason_summary",
            "expected_output",
            "safety_requirements",
            "approval_decision",
            "remember_for_session",
        }
    }
    try:
        encoded = json.dumps(stable_payload, sort_keys=True, ensure_ascii=False)
    except TypeError:
        encoded = json.dumps(_json_safe(stable_payload), sort_keys=True, ensure_ascii=False)
    return f"{tool_name}:{encoded}"


def is_private_url(value: str) -> bool:
    parsed = urlparse(str(value or "").strip())
    if parsed.scheme and parsed.scheme.lower() not in {"http", "https"}:
        return True
    host = (parsed.hostname or "").strip().lower().rstrip(".")
    if not host:
        return False
    if host in {"localhost", "ip6-localhost", "ip6-loopback"}:
        return True
    if host.endswith(".local") or host.endswith(".internal"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def permission_matcher_kind_for_tool(tool_name: str, operation: str | None = None) -> PermissionMatcherKind:
    if tool_name in _COMMAND_TOOLS or operation == "command":
        return PermissionMatcherKind.COMMAND
    if tool_name in {"fetch_url", "search_web"} or operation in {"web_search", "web_fetch"}:
        return PermissionMatcherKind.DOMAIN_URL
    if tool_name in _GIT_WRITE_TOOLS or tool_name in _GIT_READ_TOOLS:
        return PermissionMatcherKind.GIT_REMOTE_BRANCH
    if tool_name in {"apply_change_set", "validate_change_set", "generate_change_set"}:
        return PermissionMatcherKind.CHANGE_SET
    if tool_name in _DIRECT_WRITE_TOOLS or tool_name in {"read_file", "read_many_files", "search_files", "search_text", "generate_edit"}:
        return PermissionMatcherKind.PATH
    return PermissionMatcherKind.NONE


def _guard_remote_write_without_approval(
    tool_name: str,
    payload: dict[str, Any],
    decision: PermissionDecision,
    mode: PermissionMode,
) -> PermissionDecision:
    if not _is_dangerous_remote_write(tool_name, payload, decision):
        return decision
    if _payload_has_allow_decision(payload) or _remote_write_explicitly_configured(payload, decision):
        return decision
    if decision.decision == "allow":
        return _mode_ask(
            decision,
            mode,
            "Remote writes require explicit approval or a narrow remote-write allow configuration.",
            approval_payload=_approval_payload_for(tool_name, payload),
        )
    return decision


def _mode_deny(
    decision: PermissionDecision,
    mode: PermissionMode,
    reason: str,
    *,
    denial_code: str,
    recovery_hint: str,
) -> PermissionDecision:
    return replace(
        decision,
        decision="deny",
        reason=reason,
        source=PermissionRuleSource.MODE.value,
        priority=_SOURCE_PRECEDENCE[PermissionRuleSource.MODE],
        mode=mode.value,
        denial_code=denial_code,
        recovery_hint=recovery_hint,
    )


def _mode_ask(
    decision: PermissionDecision,
    mode: PermissionMode,
    reason: str,
    *,
    approval_payload: dict[str, Any] | None = None,
) -> PermissionDecision:
    return replace(
        decision,
        decision="ask",
        reason=reason,
        source=PermissionRuleSource.MODE.value,
        priority=_SOURCE_PRECEDENCE[PermissionRuleSource.MODE],
        mode=mode.value,
        approval_payload=approval_payload or decision.approval_payload,
        recovery_hint=decision.recovery_hint or _recovery_hint_for("", "ask"),
    )


def _is_pure_read_tool(tool_name: str) -> bool:
    return tool_name in _READ_SEARCH_TOOLS


def _is_auto_readonly_allowed(tool_name: str, decision: PermissionDecision) -> bool:
    if tool_name in _READ_SEARCH_TOOLS or tool_name in _GIT_STATUS_DIFF_TOOLS:
        return True
    if tool_name in _COMMAND_TOOLS:
        preview = (decision.metadata.get("command_preview") if isinstance(decision.metadata, dict) else {}) or {}
        return bool(preview.get("read_only")) and not bool(preview.get("needs_approval"))
    return False


def _is_mutating_git_or_command(tool_name: str, decision: PermissionDecision) -> bool:
    if tool_name in _GIT_WRITE_TOOLS:
        return True
    if tool_name in _COMMAND_TOOLS:
        preview = (decision.metadata.get("command_preview") if isinstance(decision.metadata, dict) else {}) or {}
        return not bool(preview.get("read_only"))
    return False


def _is_dangerous_remote_write(tool_name: str, payload: dict[str, Any], decision: PermissionDecision) -> bool:
    if tool_name in _REMOTE_WRITE_TOOLS:
        return True
    preview = (decision.metadata.get("command_preview") if isinstance(decision.metadata, dict) else {}) or {}
    command = [str(item).lower() for item in (preview.get("command") or payload.get("command") or [])]
    if tuple(command[:2]) == ("git", "push"):
        return True
    if tuple(command[:3]) in {("glab", "mr", "create"), ("glab", "mr", "update")}:
        return True
    return False


def _payload_has_allow_decision(payload: dict[str, Any]) -> bool:
    decision = payload.get("approval_decision") if isinstance(payload.get("approval_decision"), dict) else {}
    return str(decision.get("decision") or "").strip().lower() == "allow"


def _has_explicit_approval(metadata: dict[str, Any], approval_payload: dict[str, Any] | None) -> bool:
    if approval_payload and str(approval_payload.get("approval") or approval_payload.get("decision") or "").lower() == "allow":
        return True
    decision = metadata.get("approval_decision") if isinstance(metadata, dict) and isinstance(metadata.get("approval_decision"), dict) else {}
    return str(decision.get("decision") or "").lower() == "allow"


def _remote_write_explicitly_configured(payload: dict[str, Any], decision: PermissionDecision) -> bool:
    metadata = decision.metadata if isinstance(decision.metadata, dict) else {}
    return bool(
        payload.get("explicit_remote_write_allow")
        or metadata.get("explicit_remote_write_allow")
        or (decision.approval_payload or {}).get("explicit_remote_write_allow")
    )


def _approval_payload_for(tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    if tool_name in _COMMAND_TOOLS:
        return {"kind": "command_approval", "command": payload.get("command") or []}
    if tool_name == "apply_change_set":
        return {"kind": "change_set_apply", "proposal_id": payload.get("proposal_id"), "change_set_proposal": payload.get("change_set_snapshot")}
    if tool_name in _REMOTE_WRITE_TOOLS:
        return {
            "kind": tool_name,
            "remote": payload.get("remote"),
            "branch": payload.get("branch") or payload.get("source_branch"),
            "target_branch": payload.get("target_branch"),
        }
    if tool_name in _NETWORK_TOOLS:
        return {"kind": tool_name, "url": payload.get("url"), "query": payload.get("query")}
    if tool_name in _DIRECT_WRITE_TOOLS:
        return {"kind": tool_name, "path": payload.get("path") or payload.get("from")}
    return {"kind": tool_name}


def _approval_payload_from_metadata(metadata: dict[str, Any]) -> dict[str, Any] | None:
    payload = metadata.get("approval_payload") if isinstance(metadata, dict) else None
    return payload if isinstance(payload, dict) else None


def _denial_code_for_rule(rule: PermissionRule) -> str:
    if rule.source == PermissionRuleSource.BASE_SAFETY:
        return "base_safety_denied"
    if rule.source == PermissionRuleSource.WORKSPACE_PATH:
        return "workspace_path_denied"
    if rule.source == PermissionRuleSource.PROVIDER_NETWORK:
        return "provider_network_denied"
    if rule.source == PermissionRuleSource.COMMAND_POLICY:
        return "command_policy_denied"
    return "permission_rule_denied"


def _recovery_hint_for(tool_name: str, decision: str) -> str:
    if decision == "ask":
        return "Request explicit approval before retrying this action."
    if tool_name in _NETWORK_TOOLS:
        return "Use a public HTTP(S) URL or disable the network-dependent action."
    if tool_name in _DIRECT_WRITE_TOOLS or tool_name in _APPLY_TOOLS:
        return "Generate a proposal and request explicit approval before applying writes."
    return "Change the request or policy before retrying this action."


def _pattern_matches(matcher_kind: str, pattern: str, payload: dict[str, Any]) -> bool:
    normalized = matcher_kind.lower()
    values: list[str]
    if normalized == PermissionMatcherKind.PATH.value:
        values = _extract_paths(payload)
    elif normalized == PermissionMatcherKind.COMMAND.value:
        command = payload.get("command") or []
        values = [" ".join(str(item) for item in command), *(str(item) for item in command)]
    elif normalized == PermissionMatcherKind.DOMAIN_URL.value:
        values = _extract_urls_and_domains(payload)
    elif normalized == PermissionMatcherKind.GIT_REMOTE_BRANCH.value:
        values = _extract_git_remote_branch(payload)
    elif normalized == PermissionMatcherKind.CHANGE_SET.value:
        values = _extract_change_set_values(payload)
    else:
        values = [json.dumps(_json_safe(payload), sort_keys=True, ensure_ascii=False)]
    return any(fnmatch.fnmatchcase(value, pattern) or pattern in value for value in values)


def _extract_paths(payload: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("path", "relative_path", "from", "to", "target_files", "relative_paths", "files"):
        item = payload.get(key)
        if isinstance(item, str):
            values.append(item)
        elif isinstance(item, list):
            values.extend(str(part) for part in item if str(part))
    values.extend(_extract_change_set_values(payload))
    return values


def _extract_urls_and_domains(payload: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("url", "domain", "base_url"):
        value = str(payload.get(key) or "").strip()
        if not value:
            continue
        values.append(value)
        parsed = urlparse(value)
        if parsed.hostname:
            values.append(parsed.hostname.lower())
    query = str(payload.get("query") or "").strip()
    if query:
        values.append(query)
    return values


def _extract_git_remote_branch(payload: dict[str, Any]) -> list[str]:
    remote = str(payload.get("remote") or "").strip()
    branch = str(payload.get("branch") or payload.get("source_branch") or "").strip()
    target = str(payload.get("target_branch") or "").strip()
    values = [value for value in (remote, branch, target, f"{remote}:{branch}" if remote or branch else "") if value]
    command = payload.get("command") or []
    if command:
        values.append(" ".join(str(item) for item in command))
    return values


def _extract_change_set_values(payload: dict[str, Any]) -> list[str]:
    values: list[str] = []
    proposal = payload.get("change_set_snapshot") or payload.get("change_set_proposal") or payload
    if isinstance(proposal, dict):
        if proposal.get("proposal_id"):
            values.append(str(proposal.get("proposal_id")))
        changes = proposal.get("changes") if isinstance(proposal.get("changes"), list) else []
        for change in changes:
            if not isinstance(change, dict):
                continue
            for key in ("path", "rename_to", "operation"):
                if change.get(key):
                    values.append(str(change.get(key)))
    return values


def _signature_value(value: Any) -> Any:
    if isinstance(value, str):
        return value[:500]
    if isinstance(value, list):
        return [_signature_value(item) for item in value[:40]]
    if isinstance(value, dict):
        if value.get("proposal_id"):
            return {"proposal_id": value.get("proposal_id")}
        return {str(key): _signature_value(item) for key, item in list(value.items())[:40] if key != "proposed_content"}
    return _json_safe(value)


def _source_value(source: PermissionRuleSource | str) -> str:
    return _source_from_value(source).value


def _source_from_value(source: PermissionRuleSource | str) -> PermissionRuleSource:
    if isinstance(source, PermissionRuleSource):
        return source
    normalized = str(source or PermissionRuleSource.TOOL_DEFAULT.value).strip().lower()
    if normalized == "system":
        return PermissionRuleSource.BASE_SAFETY
    for candidate in PermissionRuleSource:
        if normalized == candidate.value or normalized == candidate.name.lower():
            return candidate
    return PermissionRuleSource.TOOL_DEFAULT


def _source_precedence(source: PermissionRuleSource | str) -> int:
    return _SOURCE_PRECEDENCE.get(_source_from_value(source), _SOURCE_PRECEDENCE[PermissionRuleSource.TOOL_DEFAULT])


def _priority_value(source: str, priority: int | None) -> int:
    if priority is not None:
        return int(priority)
    return _source_precedence(source)


def _mode_value(mode: PermissionMode | str | None) -> str:
    return permission_mode_from_value(mode).value


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(_json_safe(key)): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            return _json_safe(value.model_dump())
        except Exception:
            pass
    return str(value)
