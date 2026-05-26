from __future__ import annotations

import difflib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from repooperator_worker.agent_core.apply_change_set import apply_change_set_for_run
from repooperator_worker.agent_core.change_set import (
    ChangePlan,
    ChangeSetProposal,
    ProposedFileChange,
    stable_proposal_id,
    validate_change_set,
)
from repooperator_worker.services.json_safe import json_safe


TRACE_UPDATE_ENV = "REPOOPERATOR_UPDATE_AGENT_TRACE_SNAPSHOTS"
FORBIDDEN_VISIBLE_MARKERS = (
    "Work log",
    "Technical Log",
    "hidden reasoning",
    "<think>",
    "chain-of-thought",
    "chain of thought",
    "private_reasoning",
    "raw reasoning",
    "reasoning_delta",
)
SNAPSHOT_FINAL_RESPONSE_FORBIDDEN_MARKERS = (
    "Work log",
    "Technical Log",
    "hidden reasoning",
    "<think>",
)
GIT_WRITE_TOOLS = {"git_commit", "git_push", "git_branch_create", "github_create_pr", "gitlab_create_mr"}
WRITE_TOOLS = {
    "apply_change_set",
    "create_file",
    "modify_file",
    "delete_file",
    "rename_file",
    "git_commit",
    "git_push",
    "git_branch_create",
    "github_create_pr",
    "gitlab_create_mr",
    "run_validation_command",
    "run_approved_command",
}
WEB_TOOLS = {"search_web", "fetch_url", "summarize_web_evidence"}


@dataclass(frozen=True)
class RepoFixture:
    name: str
    files: dict[str, str]
    description: str = ""


@dataclass(frozen=True)
class ChangeSetSpec:
    summary: str
    path: str
    proposed_content: str
    operation: str = "modify"
    approval_decision: str | None = None

    @property
    def proposal_id(self) -> str:
        return stable_proposal_id(self.summary, [self.path])


@dataclass(frozen=True)
class TraceScenario:
    name: str
    user_request: str
    repo_fixture: str
    runtime: str
    graph_nodes: list[str]
    tool_calls: list[dict[str, Any]]
    permission_decisions: list[dict[str, Any]]
    transcript_sections: list[dict[str, Any]]
    final_response_contract: dict[str, Any]
    artifacts: dict[str, Any] = field(default_factory=dict)
    change_set: ChangeSetSpec | None = None


@dataclass(frozen=True)
class AgentTraceSnapshot:
    name: str
    user_request: str
    repo_fixture: str
    runtime: str
    expected_graph_nodes: list[str]
    expected_tool_calls: list[dict[str, Any]]
    expected_permission_decisions: list[dict[str, Any]]
    expected_transcript_sections: list[dict[str, Any]]
    expected_final_response_contract: dict[str, Any]
    expected_disk_changes_before_approval: dict[str, Any]
    expected_disk_changes_after_approval: dict[str, Any]
    expected_artifacts: dict[str, Any]
    expected_state_contracts: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "name": self.name,
                "user_request": self.user_request,
                "repo_fixture": self.repo_fixture,
                "runtime": self.runtime,
                "expected_graph_nodes": self.expected_graph_nodes,
                "expected_tool_calls": self.expected_tool_calls,
                "expected_permission_decisions": self.expected_permission_decisions,
                "expected_transcript_sections": self.expected_transcript_sections,
                "expected_final_response_contract": self.expected_final_response_contract,
                "expected_disk_changes_before_approval": self.expected_disk_changes_before_approval,
                "expected_disk_changes_after_approval": self.expected_disk_changes_after_approval,
                "expected_artifacts": self.expected_artifacts,
                "expected_state_contracts": self.expected_state_contracts,
            }
        )


@dataclass(frozen=True)
class AgentTraceComparison:
    passed: bool
    snapshot_path: Path
    diff: str = ""
    updated: bool = False

    def assert_matches(self) -> None:
        if not self.passed:
            raise AssertionError(self.diff)


def run_agent_trace(fixture: str | RepoFixture, scenario: str | TraceScenario) -> AgentTraceSnapshot:
    """Run a deterministic behavioral trace fixture and return its normalized snapshot.

    The harness intentionally uses fake model/tool outcomes for repeatability, but it
    still exercises the real change-set validation/apply implementation for approval
    scenarios so disk-write contracts are not hand-waved.
    """
    resolved_scenario = _resolve_scenario(scenario)
    resolved_fixture = _resolve_fixture(fixture or resolved_scenario.repo_fixture)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        repo = root / "repo"
        home = root / ".repooperator"
        repo.mkdir()
        home.mkdir()
        _write_fixture(repo, resolved_fixture)
        config_path = root / "config.json"
        config_path.write_text(json.dumps({"repooperatorHomeDir": str(home)}), encoding="utf-8")
        before_env = os.environ.get("REPOOPERATOR_CONFIG_PATH")
        os.environ["REPOOPERATOR_CONFIG_PATH"] = str(config_path)
        try:
            baseline = _capture_repo(repo)
            proposal_payload = _proposal_payload(repo, resolved_scenario.change_set) if resolved_scenario.change_set else None
            before_approval = _disk_delta(baseline, _capture_repo(repo))
            if resolved_scenario.change_set and resolved_scenario.change_set.approval_decision == "allow":
                assert proposal_payload is not None
                apply_change_set_for_run(
                    run_id=f"trace-{resolved_scenario.name}",
                    project_path=str(repo),
                    proposal_id=resolved_scenario.change_set.proposal_id,
                    approval_decision={"decision": "allow"},
                    fallback_change_set=proposal_payload,
                )
            after_approval = _disk_delta(baseline, _capture_repo(repo))
        finally:
            if before_env is None:
                os.environ.pop("REPOOPERATOR_CONFIG_PATH", None)
            else:
                os.environ["REPOOPERATOR_CONFIG_PATH"] = before_env

    artifacts = _materialize_artifacts(resolved_scenario)
    return AgentTraceSnapshot(
        name=resolved_scenario.name,
        user_request=resolved_scenario.user_request,
        repo_fixture=resolved_fixture.name,
        runtime=resolved_scenario.runtime,
        expected_graph_nodes=list(resolved_scenario.graph_nodes),
        expected_tool_calls=_materialize_tool_calls(resolved_scenario),
        expected_permission_decisions=list(resolved_scenario.permission_decisions),
        expected_transcript_sections=_materialize_transcript_sections(resolved_scenario),
        expected_final_response_contract=_materialize_final_contract(resolved_scenario, after_approval),
        expected_disk_changes_before_approval=before_approval,
        expected_disk_changes_after_approval=after_approval,
        expected_artifacts=artifacts,
        expected_state_contracts=_materialize_state_contracts(resolved_scenario),
    )


def compare_trace(
    snapshot: AgentTraceSnapshot,
    *,
    snapshot_dir: str | Path | None = None,
) -> AgentTraceComparison:
    snapshot_path = _snapshot_path(snapshot.name, snapshot_dir=snapshot_dir)
    actual = snapshot.to_dict()
    if snapshot_update_enabled():
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(_canonical_json(actual), encoding="utf-8")
        return AgentTraceComparison(passed=True, snapshot_path=snapshot_path, updated=True)
    if not snapshot_path.exists():
        return AgentTraceComparison(
            passed=False,
            snapshot_path=snapshot_path,
            diff=(
                f"Golden trace snapshot is missing: {snapshot_path}\n"
                f"Update mode is disabled. Set {TRACE_UPDATE_ENV}=1 to create golden snapshots intentionally."
            ),
        )
    try:
        expected = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return AgentTraceComparison(passed=False, snapshot_path=snapshot_path, diff=f"Invalid snapshot JSON at {snapshot_path}: {exc}")
    if expected == actual:
        return AgentTraceComparison(passed=True, snapshot_path=snapshot_path)
    diff = "\n".join(
        difflib.unified_diff(
            _canonical_json(expected).splitlines(),
            _canonical_json(actual).splitlines(),
            fromfile=f"expected:{snapshot_path.name}",
            tofile=f"actual:{snapshot.name}",
            lineterm="",
        )
    )
    return AgentTraceComparison(passed=False, snapshot_path=snapshot_path, diff=diff)


def validate_trace_contract(snapshot: AgentTraceSnapshot) -> list[str]:
    issues: list[str] = []
    payload = snapshot.to_dict()
    try:
        from repooperator_worker.agent_core.graph import builder

        known_graph_nodes = set()
        for build in (
            builder.build_repooperator_state_graph,
            builder.build_evidence_gathering_graph,
            builder.build_analysis_graph,
            builder.build_edit_graph,
            builder.build_validation_graph,
            builder.build_web_research_graph,
            builder.build_git_workflow_graph,
            builder.build_finalization_graph,
            builder.build_supervisor_graph,
        ):
            known_graph_nodes.update(build().nodes)
    except Exception:
        known_graph_nodes = set()
    try:
        from repooperator_worker.agent_core.tools.registry import get_default_tool_registry

        registry = get_default_tool_registry()
        known_tool_names = set(getattr(registry, "_tools", {}).keys())
    except Exception:
        known_tool_names = set()

    if known_graph_nodes:
        for node in snapshot.expected_graph_nodes:
            if node not in known_graph_nodes:
                issues.append(f"expected graph node is not in the production graph: {node}")
    if known_tool_names:
        for call in snapshot.expected_tool_calls:
            tool_name = str(call.get("name") or "")
            if tool_name and tool_name not in known_tool_names:
                issues.append(f"expected tool call is not registered: {tool_name}")

    visible_text = json.dumps(
        {
            "final_response": snapshot.expected_final_response_contract.get("text"),
            "transcript": snapshot.expected_transcript_sections,
        },
        ensure_ascii=False,
    )
    for marker in FORBIDDEN_VISIBLE_MARKERS:
        if marker.lower() in visible_text.lower():
            issues.append(f"visible trace contains forbidden marker: {marker}")

    final_text = str(snapshot.expected_final_response_contract.get("text") or "")
    for required in snapshot.expected_final_response_contract.get("must_include") or []:
        if str(required) not in final_text:
            issues.append(f"final response is missing required text: {required}")
    for forbidden in snapshot.expected_final_response_contract.get("must_not_include") or []:
        if str(forbidden).lower() in final_text.lower():
            issues.append(f"final response contains forbidden text: {forbidden}")

    if (
        not snapshot.expected_disk_changes_after_approval.get("changed_files")
        and not snapshot.expected_disk_changes_after_approval.get("created_files")
        and not snapshot.expected_disk_changes_after_approval.get("deleted_files")
        and _claims_applied_write(final_text)
    ):
        issues.append("final response claims files were applied even though no disk change occurred")

    if snapshot.expected_disk_changes_before_approval.get("changed_files") or snapshot.expected_disk_changes_before_approval.get("created_files"):
        issues.append("disk changed before approval")

    for call in snapshot.expected_tool_calls:
        if call.get("name") in WEB_TOOLS:
            sources = call.get("source_metadata") or []
            if not sources:
                issues.append(f"{call.get('name')} is missing source metadata")
            for source in sources:
                if not isinstance(source, dict) or not source.get("title") or not source.get("url") or not source.get("source"):
                    issues.append(f"{call.get('name')} has incomplete source metadata")
    if "local" in snapshot.user_request.lower() and "only" in snapshot.user_request.lower():
        web_calls = [call.get("name") for call in snapshot.expected_tool_calls if call.get("name") in WEB_TOOLS]
        if web_calls:
            issues.append(f"local-only trace unexpectedly used web tools: {web_calls}")

    for decision in snapshot.expected_permission_decisions:
        tool = str(decision.get("tool") or "")
        value = str(decision.get("decision") or "")
        if tool in GIT_WRITE_TOOLS and value != "ask":
            issues.append(f"git write tool {tool} must require approval")
        if snapshot.runtime == "routine" and tool in WRITE_TOOLS and value == "allow":
            issues.append(f"routine runtime cannot bypass approval for {tool}")

    for index, section in enumerate(snapshot.expected_transcript_sections):
        if "status" not in section or "status_text" not in section:
            issues.append(f"transcript section {index} is missing status fields")
        if "actions" not in section:
            issues.append(f"transcript section {index} is missing action group")
        if "edits" not in section:
            issues.append(f"transcript section {index} is missing edit group")

    if payload.get("expected_artifacts", {}).get("proposal_ids"):
        proposal_ids = set(payload["expected_artifacts"]["proposal_ids"])
        surfaced = {
            str(edit.get("proposal_id"))
            for section in snapshot.expected_transcript_sections
            for edit in section.get("edits", [])
            if edit.get("proposal_id")
        }
        if not proposal_ids.issubset(surfaced):
            issues.append("proposal ids are not surfaced in transcript edit groups")

    contracts = snapshot.expected_state_contracts
    if contracts.get("user_understanding_context", {}).get("required"):
        if not contracts["user_understanding_context"].get("normalized_goal"):
            issues.append("user_understanding_context is missing normalized goal")
        if "requested_outputs" not in contracts["user_understanding_context"]:
            issues.append("user_understanding_context is missing requested outputs")
    if contracts.get("evidence_basis", {}).get("required"):
        if not contracts["evidence_basis"].get("sources"):
            issues.append("evidence_basis is missing source categories")
        if contracts["evidence_basis"].get("web_untrusted_required") and "web_sources" not in contracts["evidence_basis"].get("sources", []):
            issues.append("web trace evidence_basis must expose untrusted web sources")
    if contracts.get("visible_rationale_log", {}).get("required"):
        if contracts["visible_rationale_log"].get("minimum_entries", 0) < 1:
            issues.append("visible_rationale_log must require at least one public entry")
        for marker in FORBIDDEN_VISIBLE_MARKERS:
            if marker.lower() in json.dumps(contracts["visible_rationale_log"], ensure_ascii=False).lower():
                issues.append(f"visible_rationale_log contract contains forbidden marker: {marker}")
    debug_contract = contracts.get("debug_context_visibility", {})
    for required in ("user_understanding_context", "evidence_basis", "visible_rationale_log"):
        if required not in debug_contract.get("includes", []):
            issues.append(f"debug context contract is missing {required}")
    return issues


def snapshot_update_enabled(env: dict[str, str] | None = None) -> bool:
    value = (os.environ if env is None else env).get(TRACE_UPDATE_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def default_snapshot_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "tests" / "golden_traces"


def _snapshot_path(name: str, *, snapshot_dir: str | Path | None) -> Path:
    directory = Path(snapshot_dir) if snapshot_dir is not None else default_snapshot_dir()
    return directory / f"{name}.json"


def _canonical_json(value: Any) -> str:
    return json.dumps(json_safe(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _resolve_fixture(fixture: str | RepoFixture) -> RepoFixture:
    if isinstance(fixture, RepoFixture):
        return fixture
    if fixture not in TRACE_FIXTURES:
        raise KeyError(f"Unknown trace fixture: {fixture}")
    return TRACE_FIXTURES[fixture]


def _resolve_scenario(scenario: str | TraceScenario) -> TraceScenario:
    if isinstance(scenario, TraceScenario):
        return scenario
    if scenario not in TRACE_SCENARIOS:
        raise KeyError(f"Unknown trace scenario: {scenario}")
    return TRACE_SCENARIOS[scenario]


def _write_fixture(repo: Path, fixture: RepoFixture) -> None:
    for relative_path, content in fixture.files.items():
        target = repo / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _capture_repo(repo: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(repo.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(repo))] = path.read_text(encoding="utf-8")
    return files


def _disk_delta(before: dict[str, str], after: dict[str, str]) -> dict[str, Any]:
    before_paths = set(before)
    after_paths = set(after)
    changed = sorted(path for path in before_paths & after_paths if before[path] != after[path])
    created = sorted(after_paths - before_paths)
    deleted = sorted(before_paths - after_paths)
    content_paths = [*changed, *created]
    return {
        "changed_files": changed,
        "created_files": created,
        "deleted_files": deleted,
        "contents": {path: after[path] for path in content_paths},
    }


def _proposal_payload(repo: Path, spec: ChangeSetSpec | None) -> dict[str, Any] | None:
    if spec is None:
        return None
    original_content = None
    target = repo / spec.path
    if target.exists():
        original_content = target.read_text(encoding="utf-8")
    proposal = ChangeSetProposal(
        proposal_id=spec.proposal_id,
        plan=ChangePlan(summary=spec.summary, target_files=[spec.path], operations=[spec.operation]),  # type: ignore[list-item]
        changes=[
            ProposedFileChange(
                path=spec.path,
                operation=spec.operation,  # type: ignore[arg-type]
                summary=spec.summary,
                original_content=original_content,
                proposed_content=None if spec.operation == "delete" else spec.proposed_content,
            )
        ],
    )
    validation = validate_change_set(proposal, repo=str(repo))
    proposal.validation = validation
    proposal.status = validation.status
    proposal.validation_status = validation.status
    return proposal.model_dump()


def _materialize_tool_calls(scenario: TraceScenario) -> list[dict[str, Any]]:
    proposal_id = scenario.change_set.proposal_id if scenario.change_set else None
    calls: list[dict[str, Any]] = []
    for call in scenario.tool_calls:
        next_call = dict(call)
        if proposal_id and next_call.get("proposal_id") == "$proposal_id":
            next_call["proposal_id"] = proposal_id
        calls.append(next_call)
    return calls


def _materialize_transcript_sections(scenario: TraceScenario) -> list[dict[str, Any]]:
    proposal_id = scenario.change_set.proposal_id if scenario.change_set else None
    sections: list[dict[str, Any]] = []
    for section in scenario.transcript_sections:
        next_section = json.loads(json.dumps(section))
        for edit in next_section.get("edits") or []:
            if proposal_id and edit.get("proposal_id") == "$proposal_id":
                edit["proposal_id"] = proposal_id
        sections.append(next_section)
    return sections


def _materialize_final_contract(scenario: TraceScenario, after_approval: dict[str, Any]) -> dict[str, Any]:
    contract = dict(scenario.final_response_contract)
    text = str(contract.get("text") or "")
    if scenario.change_set and scenario.change_set.approval_decision == "allow":
        changed = ", ".join(after_approval.get("changed_files") or after_approval.get("created_files") or [])
        contract["text"] = text.replace("$changed_files", changed)
        contract["applied_disk_write_exact"] = after_approval
    return contract


def _materialize_artifacts(scenario: TraceScenario) -> dict[str, Any]:
    artifacts = json.loads(json.dumps(scenario.artifacts))
    if scenario.change_set:
        proposal_id = scenario.change_set.proposal_id
        artifacts.setdefault("proposal_ids", [proposal_id])
        artifacts["proposal_ids"] = [proposal_id if item == "$proposal_id" else item for item in artifacts.get("proposal_ids", [])]
        for artifact in artifacts.get("artifacts", []):
            if artifact.get("id") == "$proposal_id":
                artifact["id"] = proposal_id
            if artifact.get("proposal_id") == "$proposal_id":
                artifact["proposal_id"] = proposal_id
    return artifacts


def _materialize_state_contracts(scenario: TraceScenario) -> dict[str, Any]:
    tool_names = [str(call.get("name") or "") for call in scenario.tool_calls]
    evidence_sources: list[str] = []
    if any(name in {"read_file", "inspect_repo_tree", "search_files", "search_text", "analyze_repository"} for name in tool_names):
        evidence_sources.append("files")
    if any(name in WEB_TOOLS for name in tool_names):
        evidence_sources.append("web_sources")
    if any(name in {"run_approved_command", "inspect_git_state", "preview_command"} for name in tool_names):
        evidence_sources.append("commands")
    if any("change_set" in name or name.startswith("validate") for name in tool_names) or scenario.change_set:
        evidence_sources.append("validation")
        evidence_sources.append("active_proposal")
    if "supervisor" in scenario.graph_nodes or "reduce_work_reports" in scenario.graph_nodes:
        evidence_sources.append("worker_reports")
    requested_outputs = _trace_requested_outputs(scenario)
    return {
        "user_understanding_context": {
            "required": True,
            "normalized_goal": scenario.user_request,
            "requested_outputs": requested_outputs,
            "follow_up_expected": bool(scenario.change_set and scenario.change_set.approval_decision),
        },
        "evidence_basis": {
            "required": True,
            "sources": _dedupe(evidence_sources),
            "web_untrusted_required": any(name in WEB_TOOLS for name in tool_names),
            "active_proposal_required": bool(scenario.change_set),
        },
        "visible_rationale_log": {
            "required": True,
            "minimum_entries": 1,
            "safe_only": True,
            "normal_transcript_uses_safe_summary": True,
        },
        "debug_context_visibility": {
            "includes": ["user_understanding_context", "evidence_basis", "visible_rationale_log"],
            "redacted": True,
            "raw_context_packet_hidden": True,
        },
    }


def _trace_requested_outputs(scenario: TraceScenario) -> list[str]:
    text = scenario.user_request.lower()
    outputs: list[str] = []
    if scenario.change_set:
        outputs.extend(["change_set_proposal", "proposal_only"])
        if scenario.change_set.approval_decision == "allow":
            outputs.append("apply_approved")
    elif "how would" in text or "without changing files" in text:
        outputs.append("explanation_only")
    else:
        outputs.append("assistant_answer")
    if any(call.get("name") in WEB_TOOLS for call in scenario.tool_calls):
        outputs.append("web_research")
    if any(call.get("name") in GIT_WRITE_TOOLS or call.get("name") == "inspect_git_state" for call in scenario.tool_calls):
        outputs.append("git_workflow")
    if scenario.runtime == "routine":
        outputs.append("routine")
    if "supervisor" in scenario.graph_nodes:
        outputs.append("broad_analysis")
    return _dedupe(outputs)


def _claims_applied_write(text: str) -> bool:
    lowered = text.lower()
    if "no files were modified" in lowered or "no files were changed" in lowered or "not applied" in lowered:
        return False
    return bool(re.search(r"\b(i|we)\s+(applied|modified|changed|wrote|saved|committed)\b", lowered))


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


PYTHON_SERVICE_FILES = {
    "README.md": "# Demo Service\n\nA small Python service with a CLI entrypoint and a greeting helper.\n",
    "pyproject.toml": "[project]\nname = \"demo-service\"\nversion = \"0.1.0\"\n",
    "src/app.py": "def greeting(name: str) -> str:\n    return f\"Hello, {name}!\"\n\n\ndef main() -> str:\n    return greeting(\"RepoOperator\")\n",
}

BROAD_ANALYSIS_FILES = {
    **PYTHON_SERVICE_FILES,
    "src/storage.py": "class Store:\n    def save(self, key: str, value: str) -> None:\n        pass\n",
    "tests/test_app.py": "from src.app import greeting\n\n\ndef test_greeting():\n    assert greeting(\"Ada\") == \"Hello, Ada!\"\n",
    "docs/architecture.md": "# Architecture\n\nThe service has a CLI, app helper, storage boundary, and tests.\n",
}

TRACE_FIXTURES: dict[str, RepoFixture] = {
    "python_service": RepoFixture(
        name="python_service",
        files=PYTHON_SERVICE_FILES,
        description="Small Python project with README, pyproject, and app module.",
    ),
    "broad_python_service": RepoFixture(
        name="broad_python_service",
        files=BROAD_ANALYSIS_FILES,
        description="Python project with enough files to exercise supervisor-style analysis.",
    ),
}

FEATURE_CHANGE = ChangeSetSpec(
    summary="Add an excited greeting option.",
    path="src/app.py",
    proposed_content=(
        "def greeting(name: str, *, excited: bool = False) -> str:\n"
        "    suffix = \"!!\" if excited else \"!\"\n"
        "    return f\"Hello, {name}{suffix}\"\n\n\n"
        "def main() -> str:\n"
        "    return greeting(\"RepoOperator\")\n"
    ),
)

APPROVED_CHANGE = ChangeSetSpec(
    summary="Return a stable ready message.",
    path="src/app.py",
    proposed_content=(
        "def greeting(name: str) -> str:\n"
        "    return f\"Hello, {name}!\"\n\n\n"
        "def main() -> str:\n"
        "    return \"RepoOperator ready\"\n"
    ),
    approval_decision="allow",
)

REJECTED_CHANGE = ChangeSetSpec(
    summary="Change CLI return text.",
    path="src/app.py",
    proposed_content=(
        "def greeting(name: str) -> str:\n"
        "    return f\"Hello, {name}!\"\n\n\n"
        "def main() -> str:\n"
        "    return \"Rejected change\"\n"
    ),
    approval_decision="deny",
)


TRACE_SCENARIOS: dict[str, TraceScenario] = {
    "simple_project_summary": TraceScenario(
        name="simple_project_summary",
        user_request="Summarize this project.",
        repo_fixture="python_service",
        runtime="langgraph+deterministic-fakes",
        graph_nodes=["load_context", "capability_discovery", "context_pack", "understand_request", "build_task_plan", "gather_evidence", "final_synthesis"],
        tool_calls=[
            {"name": "inspect_repo_tree", "status": "success", "target": ".", "result": {"entries": ["README.md", "pyproject.toml", "src"]}},
            {"name": "read_file", "status": "success", "target_files": ["README.md"], "result": {"files_read": ["README.md"]}},
        ],
        permission_decisions=[
            {"tool": "inspect_repo_tree", "decision": "allow", "reason": "read-only repository listing"},
            {"tool": "read_file", "decision": "allow", "reason": "read-only file evidence"},
        ],
        transcript_sections=[
            {
                "status": "completed",
                "status_text": "Checked the repository shape and README before summarizing.",
                "actions": [{"kind": "list_files", "target": "."}, {"kind": "read_file", "target": "README.md"}],
                "edits": [],
            }
        ],
        final_response_contract={
            "response_type": "assistant_answer",
            "text": "The project is a small Python service with a CLI entrypoint and greeting helper, grounded in README.md.",
            "must_include": ["Python service", "README.md"],
            "must_not_include": list(SNAPSHOT_FINAL_RESPONSE_FORBIDDEN_MARKERS),
            "claims_files_applied": False,
        },
    ),
    "feature_request_no_explicit_file": TraceScenario(
        name="feature_request_no_explicit_file",
        user_request="Add an excited greeting option.",
        repo_fixture="python_service",
        runtime="langgraph+deterministic-fakes",
        graph_nodes=[
            "load_context",
            "understand_request",
            "build_task_plan",
            "gather_evidence",
            "plan_change_set",
            "generate_change_set",
            "validate_change_set",
            "await_change_approval",
            "final_synthesis",
        ],
        tool_calls=[
            {"name": "search_files", "status": "success", "query": "greeting", "result": {"candidates": ["src/app.py"]}},
            {"name": "read_file", "status": "success", "target_files": ["src/app.py"], "result": {"files_read": ["src/app.py"]}},
            {"name": "generate_change_set", "status": "success", "target_files": ["src/app.py"], "proposal_id": "$proposal_id"},
        ],
        permission_decisions=[
            {"tool": "search_files", "decision": "allow", "reason": "read-only target discovery"},
            {"tool": "read_file", "decision": "allow", "reason": "read-only file evidence"},
            {"tool": "generate_change_set", "decision": "allow", "reason": "proposal-only edit generation"},
            {"tool": "apply_change_set", "decision": "ask", "reason": "validated change set would write to disk"},
        ],
        transcript_sections=[
            {
                "status": "completed",
                "status_text": "Resolved the greeting implementation before preparing a proposal.",
                "actions": [{"kind": "search", "query": "greeting"}, {"kind": "read_file", "target": "src/app.py"}],
                "edits": [{"path": "src/app.py", "status": "proposed", "proposal_id": "$proposal_id"}],
            }
        ],
        final_response_contract={
            "response_type": "change_proposal",
            "text": "I prepared a ChangeSetProposal for src/app.py. No files were modified. Review the proposal card before applying it.",
            "must_include": ["ChangeSetProposal", "No files were modified"],
            "must_not_include": list(SNAPSHOT_FINAL_RESPONSE_FORBIDDEN_MARKERS),
            "claims_files_applied": False,
        },
        artifacts={"proposal_ids": ["$proposal_id"], "artifacts": [{"kind": "proposal_card", "id": "$proposal_id", "file": "src/app.py"}]},
        change_set=FEATURE_CHANGE,
    ),
    "explanation_only_request": TraceScenario(
        name="explanation_only_request",
        user_request="How would you add an excited greeting option without changing files?",
        repo_fixture="python_service",
        runtime="langgraph+deterministic-fakes",
        graph_nodes=["load_context", "understand_request", "build_task_plan", "gather_evidence", "final_synthesis"],
        tool_calls=[
            {"name": "read_file", "status": "success", "target_files": ["src/app.py"], "result": {"files_read": ["src/app.py"]}},
        ],
        permission_decisions=[{"tool": "read_file", "decision": "allow", "reason": "read-only explanation evidence"}],
        transcript_sections=[
            {
                "status": "completed",
                "status_text": "Read the implementation and answered with an explanation-only plan.",
                "actions": [{"kind": "read_file", "target": "src/app.py"}],
                "edits": [],
            }
        ],
        final_response_contract={
            "response_type": "assistant_answer",
            "text": "You can add an optional excited flag to greeting in src/app.py and keep main using the default path. No files were modified.",
            "must_include": ["src/app.py", "No files were modified"],
            "must_not_include": list(SNAPSHOT_FINAL_RESPONSE_FORBIDDEN_MARKERS),
            "claims_files_applied": False,
        },
    ),
    "approved_change_set_apply": TraceScenario(
        name="approved_change_set_apply",
        user_request="Apply the approved ready-message change set.",
        repo_fixture="python_service",
        runtime="langgraph+deterministic-fakes",
        graph_nodes=["load_context", "await_change_approval", "apply_change_set", "post_apply_validation", "final_synthesis"],
        tool_calls=[
            {"name": "validate_change_set", "status": "success", "target_files": ["src/app.py"], "proposal_id": "$proposal_id"},
            {"name": "apply_change_set", "status": "success", "target_files": ["src/app.py"], "proposal_id": "$proposal_id"},
        ],
        permission_decisions=[
            {"tool": "apply_change_set", "decision": "ask", "reason": "writing validated change set requires approval"},
            {"tool": "apply_change_set", "decision": "allow", "reason": "user approved this proposal id"},
        ],
        transcript_sections=[
            {
                "status": "completed",
                "status_text": "Applied the approved change set and checked the resulting file.",
                "actions": [{"kind": "validation", "target": "src/app.py"}],
                "edits": [{"path": "src/app.py", "status": "applied", "proposal_id": "$proposal_id"}],
            }
        ],
        final_response_contract={
            "response_type": "edit_applied",
            "text": "Applied the approved change set to $changed_files.",
            "must_include": ["Applied", "src/app.py"],
            "must_not_include": list(SNAPSHOT_FINAL_RESPONSE_FORBIDDEN_MARKERS),
            "claims_files_applied": True,
        },
        artifacts={"proposal_ids": ["$proposal_id"], "artifacts": [{"kind": "proposal_card", "id": "$proposal_id", "file": "src/app.py"}]},
        change_set=APPROVED_CHANGE,
    ),
    "rejected_apply": TraceScenario(
        name="rejected_apply",
        user_request="Reject the proposed CLI text change.",
        repo_fixture="python_service",
        runtime="langgraph+deterministic-fakes",
        graph_nodes=["load_context", "await_change_approval", "final_synthesis"],
        tool_calls=[
            {"name": "validate_change_set", "status": "success", "target_files": ["src/app.py"], "proposal_id": "$proposal_id"},
            {"name": "apply_change_set", "status": "skipped", "target_files": ["src/app.py"], "proposal_id": "$proposal_id"},
        ],
        permission_decisions=[
            {"tool": "apply_change_set", "decision": "ask", "reason": "writing validated change set requires approval"},
            {"tool": "apply_change_set", "decision": "deny", "reason": "user rejected this proposal"},
        ],
        transcript_sections=[
            {
                "status": "completed",
                "status_text": "The apply request was rejected, so the proposal stayed unapplied.",
                "actions": [{"kind": "validation", "target": "src/app.py"}],
                "edits": [{"path": "src/app.py", "status": "rejected", "proposal_id": "$proposal_id"}],
            }
        ],
        final_response_contract={
            "response_type": "change_proposal",
            "text": "The proposed change was rejected and not applied. No files were modified.",
            "must_include": ["not applied", "No files were modified"],
            "must_not_include": list(SNAPSHOT_FINAL_RESPONSE_FORBIDDEN_MARKERS),
            "claims_files_applied": False,
        },
        artifacts={"proposal_ids": ["$proposal_id"], "artifacts": [{"kind": "proposal_card", "id": "$proposal_id", "file": "src/app.py"}]},
        change_set=REJECTED_CHANGE,
    ),
    "web_research_needed": TraceScenario(
        name="web_research_needed",
        user_request="Look up the latest Python packaging guidance and summarize what matters for this repo.",
        repo_fixture="python_service",
        runtime="langgraph+deterministic-fakes",
        graph_nodes=["load_context", "understand_request", "web_research_graph", "final_synthesis"],
        tool_calls=[
            {
                "name": "search_web",
                "status": "success",
                "query": "Python packaging guidance pyproject latest",
                "source_metadata": [{"title": "Python Packaging User Guide", "url": "https://packaging.python.org/", "source": "mock_web"}],
            },
            {
                "name": "fetch_url",
                "status": "success",
                "url": "https://packaging.python.org/",
                "source_metadata": [{"title": "Python Packaging User Guide", "url": "https://packaging.python.org/", "source": "mock_web"}],
            },
            {
                "name": "summarize_web_evidence",
                "status": "success",
                "source_metadata": [{"title": "Python Packaging User Guide", "url": "https://packaging.python.org/", "source": "mock_web"}],
            },
        ],
        permission_decisions=[
            {"tool": "search_web", "decision": "ask", "reason": "web research requires user permission"},
            {"tool": "search_web", "decision": "allow", "reason": "user granted web research for this run"},
            {"tool": "fetch_url", "decision": "allow", "reason": "same approved source domain"},
        ],
        transcript_sections=[
            {
                "status": "completed",
                "status_text": "Searched and read approved web evidence, then tied it back to pyproject.toml.",
                "actions": [{"kind": "web", "query": "Python packaging guidance pyproject latest", "sources": 1}, {"kind": "read_file", "target": "pyproject.toml"}],
                "edits": [],
            }
        ],
        final_response_contract={
            "response_type": "assistant_answer",
            "text": "Current packaging guidance supports keeping metadata in pyproject.toml. Source notes: Python Packaging User Guide https://packaging.python.org/.",
            "must_include": ["Source notes", "https://packaging.python.org/"],
            "must_not_include": list(SNAPSHOT_FINAL_RESPONSE_FORBIDDEN_MARKERS),
            "claims_files_applied": False,
        },
    ),
    "local_only_no_web": TraceScenario(
        name="local_only_no_web",
        user_request="Using only local repo files, explain the packaging setup.",
        repo_fixture="python_service",
        runtime="langgraph+deterministic-fakes",
        graph_nodes=["load_context", "understand_request", "gather_evidence", "final_synthesis"],
        tool_calls=[
            {"name": "read_file", "status": "success", "target_files": ["pyproject.toml"], "result": {"files_read": ["pyproject.toml"]}},
        ],
        permission_decisions=[{"tool": "read_file", "decision": "allow", "reason": "local-only file evidence"}],
        transcript_sections=[
            {
                "status": "completed",
                "status_text": "Stayed local and read the packaging metadata.",
                "actions": [{"kind": "read_file", "target": "pyproject.toml"}],
                "edits": [],
            }
        ],
        final_response_contract={
            "response_type": "assistant_answer",
            "text": "The local pyproject.toml declares the demo-service package name and version. No web sources were used.",
            "must_include": ["pyproject.toml", "No web sources"],
            "must_not_include": [*SNAPSHOT_FINAL_RESPONSE_FORBIDDEN_MARKERS, "https://"],
            "claims_files_applied": False,
        },
    ),
    "git_commit_approval": TraceScenario(
        name="git_commit_approval",
        user_request="Commit the current changes.",
        repo_fixture="python_service",
        runtime="langgraph+deterministic-fakes",
        graph_nodes=["load_context", "understand_request", "git_workflow_graph", "await_commit_approval", "final_synthesis"],
        tool_calls=[
            {"name": "inspect_git_state", "status": "success", "result": {"dirty": True}},
            {"name": "git_commit", "status": "waiting_approval", "message": "Update demo service"},
        ],
        permission_decisions=[
            {"tool": "inspect_git_state", "decision": "allow", "reason": "read-only git inspection"},
            {"tool": "git_commit", "decision": "ask", "reason": "git commit mutates repository history"},
        ],
        transcript_sections=[
            {
                "status": "waiting",
                "status_text": "Prepared the commit request and stopped for approval.",
                "actions": [{"kind": "command", "target": "git status"}, {"kind": "git_write", "target": "git_commit"}],
                "edits": [],
            }
        ],
        final_response_contract={
            "response_type": "assistant_answer",
            "text": "I prepared a git commit request and am waiting for approval before writing repository history.",
            "must_include": ["waiting for approval"],
            "must_not_include": list(SNAPSHOT_FINAL_RESPONSE_FORBIDDEN_MARKERS),
            "claims_files_applied": False,
        },
    ),
    "routine_run": TraceScenario(
        name="routine_run",
        user_request="Run the nightly validation routine.",
        repo_fixture="python_service",
        runtime="routine",
        graph_nodes=["load_context", "routine_enqueue_node", "execute_tool", "final_synthesis"],
        tool_calls=[
            {"name": "inspect_repo_tree", "status": "success", "target": "."},
            {"name": "run_validation_command", "status": "waiting_approval", "command": ["python3", "-m", "pytest"]},
        ],
        permission_decisions=[
            {"tool": "inspect_repo_tree", "decision": "allow", "reason": "routine can inspect local files"},
            {"tool": "run_validation_command", "decision": "ask", "reason": "routine cannot bypass command approval"},
        ],
        transcript_sections=[
            {
                "status": "waiting",
                "status_text": "Routine queued validation but stopped at the approval boundary.",
                "actions": [{"kind": "list_files", "target": "."}, {"kind": "command", "command": "python3 -m pytest"}],
                "edits": [],
            }
        ],
        final_response_contract={
            "response_type": "assistant_answer",
            "text": "The routine is waiting for approval before running python3 -m pytest.",
            "must_include": ["waiting for approval"],
            "must_not_include": list(SNAPSHOT_FINAL_RESPONSE_FORBIDDEN_MARKERS),
            "claims_files_applied": False,
        },
    ),
    "broad_analysis_supervisor": TraceScenario(
        name="broad_analysis_supervisor",
        user_request="Analyze the whole codebase and summarize each major file group.",
        repo_fixture="broad_python_service",
        runtime="langgraph+deterministic-fakes",
        graph_nodes=[
            "load_context",
            "understand_request",
            "supervisor",
            "decompose_task",
            "dispatch_work_units",
            "reduce_work_reports",
            "final_synthesis",
        ],
        tool_calls=[
            {"name": "analyze_repository", "status": "success", "target": ".", "result": {"files_read": ["README.md", "src/app.py", "src/storage.py", "tests/test_app.py", "docs/architecture.md"]}},
            {"name": "read_file", "status": "success", "target_files": ["docs/architecture.md"], "result": {"files_read": ["docs/architecture.md"]}},
        ],
        permission_decisions=[
            {"tool": "analyze_repository", "decision": "allow", "reason": "read-only broad analysis"},
            {"tool": "read_file", "decision": "allow", "reason": "read-only architecture evidence"},
        ],
        transcript_sections=[
            {
                "status": "completed",
                "status_text": "Delegated broad read-only analysis and reduced the worker reports.",
                "actions": [{"kind": "analysis", "target": "."}, {"kind": "read_file", "target": "docs/architecture.md"}],
                "edits": [],
            }
        ],
        final_response_contract={
            "response_type": "assistant_answer",
            "text": "The codebase groups into app logic, storage boundary, tests, and architecture docs based on read-only supervisor evidence.",
            "must_include": ["app logic", "storage boundary", "tests"],
            "must_not_include": list(SNAPSHOT_FINAL_RESPONSE_FORBIDDEN_MARKERS),
            "claims_files_applied": False,
        },
    ),
}


TRACE_SCENARIO_NAMES = tuple(TRACE_SCENARIOS.keys())


__all__ = [
    "AgentTraceComparison",
    "AgentTraceSnapshot",
    "RepoFixture",
    "TRACE_FIXTURES",
    "TRACE_SCENARIOS",
    "TRACE_SCENARIO_NAMES",
    "TRACE_UPDATE_ENV",
    "TraceScenario",
    "compare_trace",
    "default_snapshot_dir",
    "run_agent_trace",
    "snapshot_update_enabled",
    "validate_trace_contract",
]
