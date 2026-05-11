import json
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch


TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repooperator_worker.agent_core.actions import AgentAction  # noqa: E402
from repooperator_worker.agent_core.hooks import HookEvent, HookManager, HookResult  # noqa: E402
from repooperator_worker.agent_core.permissions import PermissionDecision, ToolPermissionContext  # noqa: E402
from repooperator_worker.agent_core.tool_orchestrator import ToolOrchestrator  # noqa: E402
from repooperator_worker.agent_core.tools.base import BaseTool, ToolExecutionContext, ToolResult, ToolSpec  # noqa: E402
from repooperator_worker.agent_core.tools.registry import ToolRegistry, get_default_tool_registry  # noqa: E402
from repooperator_worker.schemas import AgentRunRequest  # noqa: E402
from repooperator_worker.services.event_service import list_run_events  # noqa: E402


class ToolOrchestratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name) / "repo"
        self.repo.mkdir()
        (self.repo / "README.md").write_text("# Demo\n", encoding="utf-8")
        (self.repo / "cache.sqlite").write_bytes(b"SQLite format 3\x00binary")
        self.request = AgentRunRequest(project_path=str(self.repo), git_provider="local", branch="main", task="Read README.md")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _orchestrator(self, hook_manager: HookManager | None = None, registry: ToolRegistry | None = None, run_id: str = "run-orchestrator") -> ToolOrchestrator:
        return ToolOrchestrator(
            run_id=run_id,
            request=self.request,
            registry=registry or get_default_tool_registry(),
            hook_manager=hook_manager,
        )

    def test_read_file_executes_through_orchestrator(self) -> None:
        result = self._orchestrator().execute_action(
            AgentAction(type="read_file", reason_summary="read", target_files=["README.md"])
        )
        self.assertEqual(result.status, "success")
        self.assertEqual(result.files_read, ["README.md"])
        self.assertIn("README.md", result.payload["contents"])

    def test_read_file_trace_contains_structured_metadata(self) -> None:
        run_id = "run-orchestrator-read-metadata"
        self._orchestrator(run_id=run_id).execute_action(
            AgentAction(type="read_file", reason_summary="read", target_files=["README.md"])
        )
        event = _completed_trace(run_id, "read_file")
        self.assertEqual(event.get("operation"), "read_file")
        self.assertEqual(event.get("action_type"), "read_file")
        self.assertEqual(event.get("tool_name"), "read_file")
        self.assertEqual(event.get("files"), ["README.md"])
        self.assertEqual(event.get("aggregate", {}).get("action_type"), "read_file")
        self.assertEqual(event.get("aggregate", {}).get("file_path"), "README.md")
        json.dumps(event, ensure_ascii=False)

    def test_unsupported_binary_file_is_skipped(self) -> None:
        result = self._orchestrator().execute_action(
            AgentAction(type="read_file", reason_summary="read", target_files=["cache.sqlite"])
        )
        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.files_read, [])
        self.assertIn("cache.sqlite", result.payload["skipped_files"])

    def test_command_preview_goes_through_command_policy(self) -> None:
        result = self._orchestrator().execute_action(
            AgentAction(type="preview_command", reason_summary="preview", command=["git", "status", "--short"])
        )
        self.assertEqual(result.status, "success")
        self.assertTrue(result.command_result["read_only"])
        self.assertFalse(result.command_result["needs_approval"])

    def test_command_trace_contains_command_metadata(self) -> None:
        run_id = "run-orchestrator-command-metadata"
        self._orchestrator(run_id=run_id).execute_action(
            AgentAction(type="preview_command", reason_summary="preview", command=["git", "status", "--short"])
        )
        event = _completed_trace(run_id, "preview_command")
        self.assertEqual(event.get("operation"), "command")
        self.assertEqual(event.get("command"), ["git", "status", "--short"])
        self.assertEqual(event.get("aggregate", {}).get("display_command"), "git status --short")
        self.assertTrue(event.get("aggregate", {}).get("read_only"))
        json.dumps(event, ensure_ascii=False)

    def test_inspect_and_search_traces_contain_counts(self) -> None:
        run_id = "run-orchestrator-search-metadata"
        orchestrator = self._orchestrator(run_id=run_id)
        orchestrator.execute_action(AgentAction(type="inspect_repo_tree", reason_summary="inspect"))
        orchestrator.execute_action(AgentAction(type="search_files", reason_summary="search", payload={"queries": ["README.md"]}))
        orchestrator.execute_action(AgentAction(type="search_text", reason_summary="search", payload={"query": "Demo", "path_globs": ["README.md"]}))

        inspect = _completed_trace(run_id, "inspect_repo_tree")
        search_files = _completed_trace(run_id, "search_files")
        search_text = _completed_trace(run_id, "search_text")
        self.assertEqual(inspect.get("operation"), "list_files")
        self.assertEqual(inspect.get("aggregate", {}).get("entries_count"), 2)
        self.assertEqual(search_files.get("operation"), "search")
        self.assertEqual(search_files.get("aggregate", {}).get("query"), "README.md")
        self.assertGreaterEqual(search_files.get("aggregate", {}).get("result_count"), 1)
        self.assertEqual(search_text.get("aggregate", {}).get("query"), "Demo")
        self.assertGreaterEqual(search_text.get("aggregate", {}).get("result_count"), 1)
        json.dumps([inspect, search_files, search_text], ensure_ascii=False)

    def test_generate_edit_trace_contains_proposal_metadata(self) -> None:
        (self.repo / "main.py").write_text("def main():\n    return 'hello'\n", encoding="utf-8")
        self.request.task = "Add named message support."
        run_id = "run-orchestrator-edit-metadata"
        with patch(
            "repooperator_worker.agent_core.tools.builtin.model_generate_edit_proposal",
            return_value={
                "summary": "Add a named message helper.",
                "proposed_content": "def main():\n    return 'hello'\n\ndef named_message(name, message):\n    return f'{name}: {message}'\n",
                "unified_diff": "@@\n+def named_message(name, message):\n+    return f'{name}: {message}'\n",
                "risk_notes": [],
                "preserves_existing_behavior": True,
            },
        ):
            result = self._orchestrator(run_id=run_id).execute_action(
                AgentAction(type="generate_edit", reason_summary="edit", target_files=["main.py"])
            )
        self.assertEqual(result.status, "success")
        event = _completed_trace(run_id, "generate_edit")
        aggregate = event.get("aggregate", {})
        self.assertEqual(event.get("operation"), "edit")
        self.assertEqual(event.get("proposal_id"), "proposal:main.py")
        self.assertTrue(aggregate.get("edit_archive"))
        self.assertEqual(aggregate.get("files"), ["main.py"])
        self.assertFalse(aggregate.get("applied"))
        self.assertTrue(aggregate.get("diff_available"))
        self.assertNotIn("proposed_content", json.dumps(event, ensure_ascii=False))
        json.dumps(event, ensure_ascii=False)

    def test_mutating_command_does_not_run_automatically(self) -> None:
        result = self._orchestrator().execute_action(
            AgentAction(type="run_approved_command", reason_summary="commit", command=["git", "commit", "-m", "test"])
        )
        self.assertEqual(result.status, "waiting_approval")
        self.assertIsNone(result.command_result.get("exit_code"))
        self.assertTrue(result.command_result["needs_approval"])

    def test_pre_hook_can_block_tool(self) -> None:
        hooks = HookManager()
        hooks.register_pre_tool_hook(lambda event: HookResult(continue_=False, decision="deny", reason="blocked by test"))
        result = self._orchestrator(hook_manager=hooks).execute_action(
            AgentAction(type="read_file", reason_summary="read", target_files=["README.md"])
        )
        self.assertEqual(result.status, "skipped")
        self.assertIn("blocked by test", result.observation)

    def test_pre_hook_updated_input_is_revalidated(self) -> None:
        hooks = HookManager()
        hooks.register_pre_tool_hook(lambda event: HookResult(updated_input={"target_files": ["cache.sqlite"]}, source="test-hook"))
        result = self._orchestrator(hook_manager=hooks).execute_action(
            AgentAction(type="read_file", reason_summary="read", target_files=["README.md"])
        )
        self.assertEqual(result.status, "skipped")
        self.assertTrue(result.payload["hook_updated_input"])
        self.assertTrue(result.payload["hook_revalidated"])
        self.assertIn("cache.sqlite", result.payload["skipped_files"])

    def test_pre_hook_invalid_updated_input_fails_safely(self) -> None:
        hooks = HookManager()
        hooks.register_pre_tool_hook(lambda event: HookResult(updated_input=["not", "an", "object"], source="bad-hook"))
        result = self._orchestrator(hook_manager=hooks).execute_action(
            AgentAction(type="read_file", reason_summary="read", target_files=["README.md"])
        )
        self.assertEqual(result.status, "failed")
        self.assertIn("invalid updated input", result.observation)

    def test_pre_hook_command_mutation_still_requires_permission(self) -> None:
        hooks = HookManager()
        hooks.register_pre_tool_hook(lambda event: HookResult(updated_input={"command": ["git", "commit", "-m", "test"]}, source="command-hook"))
        result = self._orchestrator(hook_manager=hooks).execute_action(
            AgentAction(type="run_approved_command", reason_summary="status", command=["git", "status", "--short"])
        )
        self.assertEqual(result.status, "waiting_approval")
        self.assertIsNone(result.command_result.get("exit_code"))

    def test_post_hook_observes_result(self) -> None:
        seen: list[str] = []

        def observe(event: HookEvent) -> HookResult:
            seen.append(event.result.status)
            return HookResult()

        hooks = HookManager()
        hooks.register_post_tool_hook(observe)
        result = self._orchestrator(hook_manager=hooks).execute_action(
            AgentAction(type="read_file", reason_summary="read", target_files=["README.md"])
        )
        self.assertEqual(result.status, "success")
        self.assertEqual(seen, ["success"])

    def test_oversized_payload_is_marked_with_artifact_metadata(self) -> None:
        registry = ToolRegistry([LargePayloadTool()])
        result = self._orchestrator(registry=registry).execute_action(
            AgentAction(type="large_payload", reason_summary="large")
        )
        json.dumps(result.model_dump(), ensure_ascii=False)
        self.assertTrue(result.payload["_artifact"]["payload_truncated"])
        self.assertEqual(result.payload["_artifact"]["artifact_store"], "local")
        self.assertTrue(result.payload["_artifact"]["artifact_id"])
        self.assertNotIn("path", json.dumps(result.payload["_artifact"], ensure_ascii=False))


@dataclass
class LargePayloadTool(BaseTool):
    spec = ToolSpec(
        name="large_payload",
        description="Return large payload for truncation tests.",
        input_schema={"type": "object"},
        read_only=True,
        concurrency_safe=True,
        max_result_chars=100,
    )

    def check_permission(self, payload: dict[str, Any], context: ToolPermissionContext) -> PermissionDecision:
        return PermissionDecision.allow("test")

    def call(self, payload: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        return ToolResult(tool_name=self.spec.name, status="success", observation="ok", payload={"content": "x" * 1000})


def _completed_trace(run_id: str, action_type: str) -> dict[str, Any]:
    events = [
        event
        for event in list_run_events(run_id)
        if event.get("event_type") == "work_trace"
        and event.get("status") == "completed"
        and event.get("aggregate", {}).get("action_type") == action_type
    ]
    if not events:
        raise AssertionError(f"No completed trace for {action_type}")
    return events[-1]


if __name__ == "__main__":
    unittest.main()
