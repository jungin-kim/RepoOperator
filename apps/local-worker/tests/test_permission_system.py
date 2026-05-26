import json
import sys
import tempfile
import unittest
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repooperator_worker.agent_core.actions import AgentAction  # noqa: E402
from repooperator_worker.agent_core.permissions import PermissionDecision, PermissionMode, PermissionPolicy, PermissionRule, PermissionRuleSource, ToolPermissionContext  # noqa: E402
from repooperator_worker.agent_core.tool_orchestrator import ToolOrchestrator  # noqa: E402
from repooperator_worker.agent_core.tools.builtin import (  # noqa: E402
    ApplyChangeSetTool,
    DeleteFileTool,
    FetchUrlTool,
    GenerateEditTool,
    GitCommitTool,
    GitDiffTool,
    GitPushTool,
    GitStatusTool,
    ReadFileTool,
    RunApprovedCommandTool,
)
from repooperator_worker.schemas import AgentRunRequest  # noqa: E402
from repooperator_worker.services.command_service import preview_command, run_command_with_policy  # noqa: E402


class PermissionSystemTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name) / "repo"
        self.repo.mkdir()
        (self.repo / "README.md").write_text("# Demo\n", encoding="utf-8")
        self.request = AgentRunRequest(project_path=str(self.repo), git_provider="local", branch="main", task="Test permissions")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _context(self, mode: PermissionMode = PermissionMode.DEFAULT) -> ToolPermissionContext:
        return ToolPermissionContext(request=self.request, run_id="run-permission", permission_mode=mode, active_repository=str(self.repo))

    def test_read_only_tool_allowed_in_plan_and_default(self) -> None:
        tool = ReadFileTool()
        self.assertEqual(tool.check_permission({}, self._context(PermissionMode.PLAN)).decision, "allow")
        self.assertEqual(tool.check_permission({}, self._context(PermissionMode.DEFAULT)).decision, "allow")

    def test_generate_edit_proposal_allowed_in_plan(self) -> None:
        decision = GenerateEditTool().check_permission({"target_files": ["README.md"]}, self._context(PermissionMode.PLAN))
        self.assertEqual(decision.decision, "allow")
        self.assertIn("proposal-only", decision.reason)

    def test_run_approved_command_allows_read_only_preview(self) -> None:
        decision = RunApprovedCommandTool().check_permission({"command": ["git", "status", "--short"]}, self._context())
        self.assertEqual(decision.decision, "allow")
        self.assertTrue(decision.metadata["command_preview"]["read_only"])

    def test_run_approved_command_asks_for_mutating_command(self) -> None:
        decision = RunApprovedCommandTool().check_permission({"command": ["git", "commit", "-m", "test"]}, self._context())
        self.assertEqual(decision.decision, "ask")
        self.assertTrue(decision.approval_id)
        self.assertTrue(decision.metadata["command_preview"]["needs_approval"])

    def test_command_service_policy_matches_tool_permission_preview(self) -> None:
        status_preview = preview_command(["git", "status", "--short"], project_path=str(self.repo))
        commit_preview = preview_command(["git", "commit", "-m", "test"], project_path=str(self.repo))

        self.assertTrue(status_preview["read_only"])
        self.assertFalse(status_preview["needs_approval"])
        self.assertFalse(commit_preview["read_only"])
        self.assertTrue(commit_preview["needs_approval"])
        with self.assertRaises(PermissionError):
            run_command_with_policy(["git", "commit", "-m", "test"], project_path=str(self.repo))

    def test_bypass_mode_exists_but_does_not_bypass_command_policy(self) -> None:
        decision = RunApprovedCommandTool().check_permission({"command": ["git", "commit", "-m", "test"]}, self._context(PermissionMode.BYPASS))
        self.assertEqual(decision.decision, "ask")

    def test_permission_policy_priority_and_audit(self) -> None:
        policy = PermissionPolicy(
            [
                PermissionRule(
                    id="user-allow",
                    source=PermissionRuleSource.USER,
                    tool_name="read_file",
                    decision="allow",
                    reason="user allows",
                    priority=100_000,
                ),
                PermissionRule(
                    id="system-deny",
                    source=PermissionRuleSource.SYSTEM,
                    tool_name="read_file",
                    decision="deny",
                    reason="system denies",
                    priority=0,
                ),
            ]
        )
        decision, audit = policy.evaluate(
            tool_name="read_file",
            payload={"target_files": ["README.md"]},
            context=self._context(),
            base_decision=PermissionDecision.allow("tool default"),
        )
        self.assertEqual(decision.decision, "deny")
        self.assertEqual(audit.decision, "deny")
        self.assertEqual(audit.matched_rules[0]["id"], "user-allow")
        self.assertEqual(audit.base_decision["id"], "tool_default:read_file")

    def test_ask_overrides_allow_at_equal_priority(self) -> None:
        policy = PermissionPolicy(
            [
                PermissionRule(id="allow", source=PermissionRuleSource.SESSION, tool_name="read_file", decision="allow", reason="allow", priority=1),
                PermissionRule(id="ask", source=PermissionRuleSource.SESSION, tool_name="read_file", decision="ask", reason="ask", priority=1),
            ]
        )
        decision, _audit = policy.evaluate(
            tool_name="read_file",
            payload={},
            context=self._context(),
            base_decision=PermissionDecision.allow("tool default"),
        )
        self.assertEqual(decision.decision, "ask")

    def test_base_deny_cannot_be_upgraded_by_user_allow(self) -> None:
        policy = PermissionPolicy(
            [
                PermissionRule(id="user-allow", source=PermissionRuleSource.USER, tool_name="run_approved_command", decision="allow", reason="allow", priority=100),
            ]
        )
        decision, audit = policy.evaluate(
            tool_name="run_approved_command",
            payload={"command": ["bash", "-lc", "cat README.md"]},
            context=self._context(),
            base_decision=PermissionDecision.deny("command security denied"),
        )
        self.assertEqual(decision.decision, "deny")
        self.assertEqual(audit.base_decision["decision"], "deny")

    def test_base_ask_cannot_be_upgraded_by_user_allow_without_approval(self) -> None:
        policy = PermissionPolicy(
            [
                PermissionRule(id="user-allow", source=PermissionRuleSource.USER, tool_name="run_approved_command", decision="allow", reason="allow", priority=100),
            ]
        )
        decision, _audit = policy.evaluate(
            tool_name="run_approved_command",
            payload={"command": ["git", "commit", "-m", "test"]},
            context=self._context(),
            base_decision=PermissionDecision.ask("command policy asks", command_preview={"needs_approval": True}),
        )
        self.assertEqual(decision.decision, "ask")

    def test_mutating_command_user_allow_still_asks(self) -> None:
        policy = PermissionPolicy(
            [
                PermissionRule(id="user-allow", source=PermissionRuleSource.USER, tool_name="run_approved_command", decision="allow", reason="allow", priority=100),
            ]
        )
        base = RunApprovedCommandTool().check_permission({"command": ["git", "commit", "-m", "test"]}, self._context())
        decision, _audit = policy.evaluate(
            tool_name="run_approved_command",
            payload={"command": ["git", "commit", "-m", "test"]},
            context=self._context(),
            base_decision=base,
        )
        self.assertEqual(decision.decision, "ask")

    def test_proposal_only_denies_apply_change_set(self) -> None:
        decision = ApplyChangeSetTool().check_permission({"proposal_id": "proposal-1"}, self._context(PermissionMode.PROPOSAL_ONLY))
        self.assertEqual(decision.decision, "deny")
        self.assertEqual(decision.denial_code, "mode_proposal_only_write_denied")

    def test_accept_edits_allows_approved_apply_but_not_git_push(self) -> None:
        context = self._context(PermissionMode.ACCEPT_EDITS)
        apply_decision = ApplyChangeSetTool().check_permission(
            {"proposal_id": "proposal-1", "approval_decision": {"decision": "allow"}},
            context,
        )
        push_decision = GitPushTool().check_permission({"remote": "origin", "branch": "main"}, context)
        self.assertEqual(apply_decision.decision, "allow")
        self.assertEqual(push_decision.decision, "ask")

    def test_auto_readonly_allows_git_status_diff_and_denies_git_writes(self) -> None:
        context = self._context(PermissionMode.AUTO_READONLY)
        self.assertEqual(GitStatusTool().check_permission({}, context).decision, "allow")
        self.assertEqual(GitDiffTool().check_permission({}, context).decision, "allow")
        self.assertEqual(GitCommitTool().check_permission({"message": "test"}, context).decision, "deny")
        self.assertEqual(GitPushTool().check_permission({"remote": "origin", "branch": "main"}, context).decision, "deny")

    def test_routine_safe_queues_pending_approval_for_write(self) -> None:
        orchestrator = ToolOrchestrator(run_id="run-routine-safe", request=self.request, permission_mode=PermissionMode.ROUTINE_SAFE)
        result = orchestrator.execute_action(AgentAction(type="git_commit", reason_summary="commit", payload={"message": "test"}))
        self.assertEqual(result.status, "waiting_approval")
        self.assertEqual(result.next_recommended_action, "queue_approval")
        self.assertEqual(result.payload["permission_decision"]["mode"], "routine_safe")

    def test_headless_safe_fails_closed_when_approval_required(self) -> None:
        orchestrator = ToolOrchestrator(run_id="run-headless-safe", request=self.request, permission_mode=PermissionMode.HEADLESS_SAFE)
        result = orchestrator.execute_action(AgentAction(type="git_commit", reason_summary="commit", payload={"message": "test"}))
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.payload["permission_decision"]["decision"], "deny")
        self.assertEqual(result.payload["permission_decision"]["denial_code"], "approval_required_headless")

    def test_repeated_approval_request_does_not_loop(self) -> None:
        orchestrator = ToolOrchestrator(run_id="run-repeat-denial", request=self.request)
        action = AgentAction(type="git_commit", reason_summary="commit", payload={"message": "test"})
        first = orchestrator.execute_action(action)
        second = orchestrator.execute_action(action)
        self.assertEqual(first.status, "waiting_approval")
        self.assertEqual(second.status, "failed")
        self.assertEqual(second.payload["permission_decision"]["denial_code"], "repeated_approval_required")

    def test_web_private_url_denied_even_when_network_approved(self) -> None:
        decision = FetchUrlTool().check_permission(
            {"url": "http://127.0.0.1:8000/private", "approval_decision": {"decision": "allow"}},
            self._context(PermissionMode.FULL_ACCESS),
        )
        self.assertEqual(decision.decision, "deny")
        self.assertEqual(decision.denial_code, "private_network_url")

    def test_protected_delete_requires_explicit_approval(self) -> None:
        decision = DeleteFileTool().check_permission(
            {"path": "README.md", "justification": "cleanup"},
            self._context(PermissionMode.FULL_ACCESS),
        )
        self.assertEqual(decision.decision, "ask")
        self.assertEqual(decision.approval_payload["kind"], "delete_file")

    def test_permission_audit_record_is_json_safe(self) -> None:
        policy = PermissionPolicy()
        decision, audit = policy.evaluate(
            tool_name="read_file",
            payload={"target_files": ["README.md"]},
            context=self._context(),
            base_decision=PermissionDecision.allow("tool default"),
        )
        json.dumps(decision.model_dump(), ensure_ascii=False)
        json.dumps(audit.model_dump(), ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
