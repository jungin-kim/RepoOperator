import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repooperator_worker.schemas import AgentRunRequest  # noqa: E402
from repooperator_worker.services.agent_service import run_agent_task  # noqa: E402
from repooperator_worker.services.debug_service import get_debug_runtime_status  # noqa: E402


class _QuietClient:
    @property
    def model_name(self) -> str:
        return "test-model"

    def generate_text(self, request):
        return "README.md evidence reached the final answer."

    def stream_text(self, request):
        yield {"type": "assistant_delta", "delta": "README.md evidence reached the final answer."}


class LangGraphOnlyRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self._old_config_path = os.environ.get("REPOOPERATOR_CONFIG_PATH")
        os.environ["REPOOPERATOR_CONFIG_PATH"] = str(Path(self.tmp.name) / ".repooperator" / "config.json")
        self.repo = Path(self.tmp.name) / "repo"
        self.repo.mkdir()
        (self.repo / "README.md").write_text("# Demo\n", encoding="utf-8")

    def tearDown(self) -> None:
        if self._old_config_path is None:
            os.environ.pop("REPOOPERATOR_CONFIG_PATH", None)
        else:
            os.environ["REPOOPERATOR_CONFIG_PATH"] = self._old_config_path
        self.tmp.cleanup()

    def _request(self) -> AgentRunRequest:
        return AgentRunRequest(project_path=str(self.repo), git_provider="local", branch="main", task="Summarize README.md")

    def test_run_agent_task_uses_langgraph_by_default(self) -> None:
        with patch("repooperator_worker.agent_core.graph.support.get_active_repository", return_value=None), patch(
            "repooperator_worker.agent_core.graph.support.OpenAICompatibleModelClient", return_value=_QuietClient()
        ):
            response = run_agent_task(self._request())
        self.assertEqual(response.agent_flow, "langgraph")
        self.assertIn("README.md", response.files_read)

    def test_runtime_debug_reports_langgraph(self) -> None:
        status = get_debug_runtime_status()
        self.assertEqual(status["agent"]["runtime"], "langgraph")
        self.assertEqual(status["agent"]["default_runtime"], "langgraph")


if __name__ == "__main__":
    unittest.main()
