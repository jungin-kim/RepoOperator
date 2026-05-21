import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repooperator_worker.agent_core.controller_graph import run_controller_graph  # noqa: E402
from repooperator_worker.schemas import AgentRunRequest, AgentRunResponse  # noqa: E402


class AgentLoopCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name) / "repo"
        self.repo.mkdir()
        (self.repo / "README.md").write_text("# Demo\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_run_controller_graph_delegates_to_agent_loop(self) -> None:
        request = AgentRunRequest(project_path=str(self.repo), git_provider="local", branch="main", task="Summarize")
        response = AgentRunResponse(
            project_path=str(self.repo),
            git_provider="local",
            task=request.task,
            model="test-model",
            branch="main",
            repo_root_name="repo",
            context_summary="",
            top_level_entries=[],
            readme_included=False,
            diff_included=False,
            is_git_repository=False,
            files_read=[],
            response="Done.",
            graph_path="agent_core:test-loop",
            agent_flow="agent_core_controller",
            run_id="run-loop",
        )
        with patch("repooperator_worker.agent_core.controller_graph.get_active_repository", return_value=None), patch(
            "repooperator_worker.agent_core.controller_graph.AgentLoop"
        ) as loop_cls:
            loop_cls.return_value.run.return_value = response
            result = run_controller_graph(request, run_id="run-loop")
        self.assertEqual(result.response, "Done.")
        self.assertTrue(loop_cls.called)
        self.assertTrue(loop_cls.return_value.run.called)


if __name__ == "__main__":
    unittest.main()
