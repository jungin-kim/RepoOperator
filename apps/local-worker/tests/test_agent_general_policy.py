import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repooperator_worker.agent_core.action_executor import ActionExecutor  # noqa: E402
from repooperator_worker.agent_core.actions import AgentAction, ActionResult  # noqa: E402
from repooperator_worker.agent_core.final_synthesis import validate_or_repair_final_answer  # noqa: E402
from repooperator_worker.agent_core.request_understanding import RequestUnderstanding  # noqa: E402
from repooperator_worker.agent_core.state import AgentCoreState  # noqa: E402
from repooperator_worker.agent_core.task_policy import (  # noqa: E402
    ensure_subtasks,
    next_recovery_action,
    update_subtasks_after_action,
)
from repooperator_worker.agent_core.langgraph_runtime import run_langgraph_controller  # noqa: E402
from repooperator_worker.agent_core.planner import build_task_frame  # noqa: E402
from repooperator_worker.schemas import AgentRunRequest, ConversationMessage  # noqa: E402
from repooperator_worker.services.event_service import list_run_events  # noqa: E402


class _QuietClient:
    @property
    def model_name(self) -> str:
        return "test-model"

    def generate_text(self, request):
        if "bounded next-action planner" in request.system_prompt:
            return "{}"
        return "Grounded answer from gathered evidence."

    def stream_text(self, request):
        if "bounded next-action planner" in request.system_prompt:
            return iter(())
        yield {"type": "assistant_delta", "delta": "Grounded answer from gathered evidence."}


class GeneralAgentPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name) / "repo"
        self.repo.mkdir()
        (self.repo / "README.md").write_text("# Generic App\n\nA small service for messages.\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _request(self, task: str) -> AgentRunRequest:
        return AgentRunRequest(
            project_path=str(self.repo),
            git_provider="local",
            branch="main",
            thread_id="thread-general-policy",
            task=task,
            conversation_history=[],
        )

    def test_generic_feature_request_gathers_evidence_before_clarification(self) -> None:
        (self.repo / "src").mkdir()
        (self.repo / "src" / "index.ts").write_text("import { createMessage } from './messages';\ncreateMessage('hello');\n", encoding="utf-8")
        (self.repo / "src" / "messages.ts").write_text(
            "export function createMessage(body: string) {\n  return { body };\n}\n",
            encoding="utf-8",
        )
        request = self._request("Add named message support.")
        understanding = RequestUnderstanding(
            user_goal=request.task,
            requested_outputs=["code_change_proposal"],
            likely_needed_tools=["search_files", "read_file", "generate_edit"],
        )

        def proposal(relative_path, content, task, context):
            return {
                "file": relative_path,
                "summary": "Add sender name to message creation.",
                "proposed_content": content.replace("return { body };", "return { name: 'anonymous', body };"),
                "risk_notes": [],
                "preserves_existing_behavior": True,
            }

        with patch("repooperator_worker.agent_core.request_understanding.understand_request", return_value=understanding), patch(
            "repooperator_worker.agent_core.graph.support.OpenAICompatibleModelClient", return_value=_QuietClient()
        ), patch("repooperator_worker.agent_core.tools.builtin.model_generate_edit_proposal", side_effect=proposal), patch(
            "repooperator_worker.agent_core.graph.support.get_active_repository", return_value=None
        ):
            result = run_langgraph_controller(request, run_id="generic-feature-evidence")

        actions = [event["action"]["type"] for event in list_run_events("generic-feature-evidence") if event.get("type") == "action_result"]
        self.assertGreaterEqual(actions.index("inspect_repo_tree"), 0)
        self.assertIn("search_files", actions)
        self.assertIn("read_file", actions)
        self.assertNotEqual(actions[:1], ["ask_clarification"])
        self.assertIn("proposed patch only", result.response)
        self.assertIn("No files were modified", result.response)

    def test_generic_followup_reads_prior_source_before_clarification(self) -> None:
        (self.repo / "src").mkdir()
        (self.repo / "src" / "domain.ts").write_text("export function score(value: number) { return value * 2; }\n", encoding="utf-8")
        request = self._request("What does that domain layer do?")
        request.conversation_history = [
            ConversationMessage(role="assistant", content="I read src/domain.ts.", metadata={"files_read": ["src/domain.ts"]})
        ]
        understanding = RequestUnderstanding(
            user_goal=request.task,
            requested_outputs=["explanation"],
            likely_needed_tools=["read_file"],
        )
        with patch("repooperator_worker.agent_core.request_understanding.understand_request", return_value=understanding), patch(
            "repooperator_worker.agent_core.graph.support.OpenAICompatibleModelClient", return_value=_QuietClient()
        ), patch("repooperator_worker.agent_core.graph.support.get_active_repository", return_value=None):
            result = run_langgraph_controller(request, run_id="generic-followup-evidence")

        self.assertIn("src/domain.ts", result.files_read)
        actions = [event["action"]["type"] for event in list_run_events("generic-followup-evidence") if event.get("type") == "action_result"]
        self.assertNotIn("ask_clarification", actions[:1])

    def test_generic_broad_analysis_runs_inventory_and_bounded_batch(self) -> None:
        (self.repo / "package.json").write_text('{"scripts":{"test":"node test.js"}}\n', encoding="utf-8")
        (self.repo / "src").mkdir()
        (self.repo / "src" / "main.ts").write_text("import { serve } from './service';\nserve();\n", encoding="utf-8")
        (self.repo / "src" / "service.ts").write_text("export function serve() { return true; }\n", encoding="utf-8")
        (self.repo / "tests").mkdir()
        (self.repo / "tests" / "service.test.ts").write_text("import { serve } from '../src/service';\nserve();\n", encoding="utf-8")
        request = self._request("Analyze all source files and explain the directory structure.")
        understanding = RequestUnderstanding(
            user_goal=request.task,
            requested_outputs=["all_source_analysis", "directory_structure"],
            likely_needed_tools=["search_files", "read_file"],
        )
        with patch("repooperator_worker.agent_core.request_understanding.understand_request", return_value=understanding), patch(
            "repooperator_worker.agent_core.graph.support.OpenAICompatibleModelClient", return_value=_QuietClient()
        ), patch("repooperator_worker.agent_core.graph.support.get_active_repository", return_value=None):
            result = run_langgraph_controller(request, run_id="generic-broad-batch")

        actions = [event["action"]["type"] for event in list_run_events("generic-broad-batch") if event.get("type") == "action_result"]
        self.assertIn("inspect_repo_tree", actions)
        self.assertIn("search_files", actions)
        self.assertIn("read_file", actions)
        self.assertIn("Analyzed batch", result.response)
        self.assertIn("File role table", result.response)
        self.assertIn("Remaining groups/files", result.response)
        self.assertNotIn("max_loop_iterations", result.response)

    def test_clarification_happens_after_missing_file_search_is_exhausted(self) -> None:
        request = self._request("Explain MissingWidget.ts")
        understanding = RequestUnderstanding(user_goal=request.task, mentioned_files=["MissingWidget.ts"], likely_needed_tools=["read_file"])
        with patch("repooperator_worker.agent_core.request_understanding.understand_request", return_value=understanding), patch(
            "repooperator_worker.agent_core.graph.support.get_active_repository", return_value=None
        ):
            result = run_langgraph_controller(request, run_id="generic-missing-file")
        actions = [event["action"]["type"] for event in list_run_events("generic-missing-file") if event.get("type") == "action_result"]
        self.assertEqual(actions[:1], ["search_files"])
        self.assertEqual(result.stop_reason, "needs_clarification")
        self.assertEqual(result.files_read, [])

    def test_subtasks_are_generic_and_update_after_success(self) -> None:
        request = self._request("Add a retry option.")
        state = AgentCoreState(run_id="subtasks", thread_id="t", repo=str(self.repo), branch="main", user_task=request.task)
        state.request_understanding = RequestUnderstanding(
            user_goal=request.task,
            requested_outputs=["code_change_proposal"],
            likely_needed_tools=["search_files", "read_file", "generate_edit"],
        )
        frame = build_task_frame(request, state)
        ensure_subtasks(state, request, frame)
        self.assertEqual([item.title for item in state.subtasks[:3]], [
            "Locate relevant implementation area",
            "Understand current behavior/data/API flow",
            "Prepare proposal-only change",
        ])
        action = AgentAction(type="inspect_repo_tree", reason_summary="inspect")
        update_subtasks_after_action(state, action, ActionResult(action_id=action.action_id, status="success"), "list_files")
        self.assertEqual(state.subtasks[0].status, "completed")
        self.assertIn("list_files", state.subtasks[0].completed_operations)

    def test_failed_read_recovers_with_basename_search_before_clarification(self) -> None:
        request = self._request("Explain src/widget.ts")
        state = AgentCoreState(run_id="recover-read", thread_id="t", repo=str(self.repo), branch="main", user_task=request.task)
        action = AgentAction(type="read_file", reason_summary="read", target_files=["src/widget.ts"])
        result = ActionResult(action_id=action.action_id, status="failed", observation="missing")
        recovery = next_recovery_action(state, request, build_task_frame(request, state), action, result)
        self.assertIsNotNone(recovery)
        self.assertEqual(recovery.type, "search_files")
        self.assertIn("widget.ts", recovery.payload["queries"])

    def test_invalid_generated_proposal_is_rejected_with_proposal_error(self) -> None:
        (self.repo / "main.py").write_text("def main():\n    return 1\n", encoding="utf-8")
        request = self._request("Update main.")
        with patch(
            "repooperator_worker.agent_core.tools.builtin.model_generate_edit_proposal",
            return_value={
                "file": "main.py",
                "summary": "Broken",
                "proposed_content": "```python\ndef main(:\n    return 2\n```",
                "risk_notes": [],
            },
        ), patch("repooperator_worker.agent_core.tools.builtin.OpenAICompatibleModelClient", side_effect=RuntimeError("no repair model")):
            result = ActionExecutor(run_id="invalid-proposal", request=request).execute(
                AgentAction(type="generate_edit", reason_summary="edit", target_files=["main.py"])
            )
        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.payload["edit_proposals"], [])
        self.assertIn("proposal_error", result.payload)
        self.assertIn("markdown fences", result.payload["proposal_error"])

    def test_korean_internal_planning_final_answer_is_repaired(self) -> None:
        request = self._request("이 파일이 어떤 역할인지 설명해줘.")
        state = AgentCoreState(run_id="ko-repair", thread_id="t", repo=str(self.repo), branch="main", user_task=request.task)
        state.files_read = ["README.md"]
        state.action_results.append(
            ActionResult(
                action_id="a1",
                status="success",
                files_read=["README.md"],
                payload={"contents": {"README.md": "# Generic App\n\n메시지 서비스를 설명한다.\n"}},
            )
        )
        with patch("repooperator_worker.agent_core.final_synthesis._compat_model_client", side_effect=RuntimeError("offline")):
            repaired = validate_or_repair_final_answer("The user asks what the file does. I need to answer.", state, request)
        self.assertNotIn("The user asks", repaired)
        self.assertIn("근거 파일", repaired)
        json.dumps({"answer": repaired}, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
