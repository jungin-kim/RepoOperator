import sys
import tempfile
import time
import unittest
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repooperator_worker.agent_core.context_packer import pack_context  # noqa: E402
from repooperator_worker.agent_core.model_profile import ModelProfile  # noqa: E402
from repooperator_worker.agent_core.planner import _format_edit_proposal, build_task_frame, current_edit_target_files  # noqa: E402
from repooperator_worker.agent_core.request_understanding import RequestUnderstanding  # noqa: E402
from repooperator_worker.agent_core.state import AgentCoreState  # noqa: E402
from repooperator_worker.schemas import AgentRunRequest  # noqa: E402
from repooperator_worker.services.ide_bridge_service import clear_ide_context, get_ide_context, update_ide_context  # noqa: E402


SMALL = ModelProfile(
    provider="test",
    model_name="tiny",
    context_window=16_000,
    max_output_tokens=2_000,
    supports_streaming=True,
    supports_tool_calls=False,
    supports_json_schema=False,
    supports_reasoning_signal=False,
    tokenizer_hint="unknown",
    compression_strategy="aggressive",
)


class IDEBridgeContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name) / "repo"
        self.repo.mkdir()
        (self.repo / "app.py").write_text("def main():\n    return 1\n", encoding="utf-8")
        (self.repo / "other.py").write_text("def other():\n    return 3\n", encoding="utf-8")
        clear_ide_context()

    def tearDown(self) -> None:
        clear_ide_context()
        self.tmp.cleanup()

    def _request(self, task: str = "Fix this selection") -> AgentRunRequest:
        return AgentRunRequest(project_path=str(self.repo), git_provider="local", branch="main", task=task)

    def _state(self, request: AgentRunRequest) -> AgentCoreState:
        state = AgentCoreState(run_id="run-ide", thread_id=None, repo=str(self.repo), branch="main", user_task=request.task)
        state.request_understanding = RequestUnderstanding(
            user_goal=request.task,
            mentioned_files=[],
            requested_outputs=["code_change_proposal"],
            likely_needed_tools=["read_file", "generate_edit"],
            safety_notes=[],
            uncertainties=[],
        )
        return state

    def test_active_file_is_included_in_edit_context(self) -> None:
        update_ide_context(
            {
                "workspace_root": str(self.repo),
                "active_file": str(self.repo / "app.py"),
                "selected_text": "return 1",
                "open_files": [str(self.repo / "app.py"), str(self.repo / "other.py")],
                "diagnostics": [{"path": str(self.repo / "app.py"), "message": "Expected return value 2", "severity": "error"}],
                "cursor_position": {"line": 2, "column": 5},
                "branch": "main",
                "editor": "vscode",
            }
        )

        packet = pack_context("edit", self._request(), state={}, profile=SMALL)

        self.assertEqual(packet["ide_context"]["active_file"], "app.py")
        self.assertEqual(packet["ide_context"]["selected_text"], "return 1")
        self.assertEqual(packet["ide_context"]["diagnostics"][0]["path"], "app.py")
        self.assertIn("ide_context", packet["context_pack_report"]["included_sections"])

    def test_selected_text_narrows_edit_target_to_active_file(self) -> None:
        update_ide_context(
            {
                "workspace_root": str(self.repo),
                "active_file": str(self.repo / "app.py"),
                "selected_text": "return 1",
                "open_files": [str(self.repo / "app.py"), str(self.repo / "other.py")],
                "branch": "main",
            }
        )
        request = self._request("Update the selected code")
        state = self._state(request)
        state.context_packet = pack_context("edit", request, state={}, profile=SMALL)
        state.files_read = ["app.py", "other.py"]
        frame = build_task_frame(request, state)

        self.assertEqual(current_edit_target_files(state, frame, request), ["app.py"])

    def test_diagnostics_inform_bugfix_workflow(self) -> None:
        update_ide_context(
            {
                "workspace_root": str(self.repo),
                "active_file": str(self.repo / "app.py"),
                "diagnostics": [{"path": str(self.repo / "app.py"), "message": "NameError: missing symbol", "severity": "error"}],
                "branch": "main",
            }
        )
        request = self._request("Fix the bug")
        state = self._state(request)
        state.context_packet = pack_context("edit", request, state={}, profile=SMALL)

        frame = build_task_frame(request, state)

        self.assertTrue(any("diagnostics" in item for item in frame.constraints))
        self.assertEqual(state.context_packet["ide_context"]["diagnostics"][0]["message"], "NameError: missing symbol")

    def test_stale_ide_context_is_ignored_after_ttl(self) -> None:
        now = time.time()
        update_ide_context(
            {
                "workspace_root": str(self.repo),
                "active_file": str(self.repo / "app.py"),
                "selected_text": "return 1",
                "branch": "main",
                "timestamp": now - 1_000,
            }
        )

        self.assertIsNone(get_ide_context(project_path=str(self.repo), branch="main", now=now, ttl_seconds=300))
        packet = pack_context("edit", self._request(), state={}, profile=SMALL)
        self.assertIsNone(packet["ide_context"])
        self.assertIn("ide_context", packet["context_pack_report"]["excluded_sections"])

    def test_no_ide_context_keeps_normal_repo_flow(self) -> None:
        request = self._request("Fix the behavior")
        state = self._state(request)
        state.files_read = ["app.py"]
        frame = build_task_frame(request, state)
        packet = pack_context("edit", request, state={}, profile=SMALL)

        self.assertIsNone(packet["ide_context"])
        self.assertEqual(current_edit_target_files(state, frame, request), [])

    def test_final_answer_can_mention_active_editor_context(self) -> None:
        text = _format_edit_proposal(
            {
                "proposals": [
                    {
                        "file": "app.py",
                        "summary": "Return two.",
                        "before_summary": "return one",
                        "after_summary": "return two",
                        "proposed_content": "def main():\n    return 2\n",
                    }
                ]
            },
            ide_context={"active_file": "app.py"},
        )

        self.assertIn("Used active editor context for `app.py`", text)


if __name__ == "__main__":
    unittest.main()
