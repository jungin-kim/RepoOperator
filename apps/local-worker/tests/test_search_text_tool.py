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
from repooperator_worker.agent_core.tool_orchestrator import ToolOrchestrator  # noqa: E402
from repooperator_worker.agent_core.tools.registry import get_default_tool_registry  # noqa: E402
from repooperator_worker.schemas import AgentRunRequest  # noqa: E402


class SearchTextToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name) / "repo"
        self.repo.mkdir()
        (self.repo / "main.py").write_text("def foo():\n    return 'bar'\n", encoding="utf-8")
        (self.repo / "secret.txt").write_text("token = ghp_" + "A" * 36 + "\n", encoding="utf-8")
        (self.repo / "cache.sqlite").write_bytes(b"SQLite format 3\x00foo")
        self.request = AgentRunRequest(project_path=str(self.repo), git_provider="local", branch="main", task="search")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run(self, payload: dict) -> dict:
        orchestrator = ToolOrchestrator(run_id="run-search-text", request=self.request, registry=get_default_tool_registry())
        result = orchestrator.execute_action(AgentAction(type="search_text", reason_summary="search", payload=payload))
        self.assertEqual(result.status, "success")
        json.dumps(result.model_dump(), ensure_ascii=False)
        return result.payload

    def test_search_text_finds_plain_matches(self) -> None:
        payload = self._run({"query": "return", "path_globs": ["*.py"]})
        self.assertEqual(payload["matches"][0]["path"], "main.py")
        self.assertEqual(payload["matches"][0]["line"], 2)

    def test_search_text_supports_safe_regex_alternation(self) -> None:
        payload = self._run({"query": "foo|bar", "regex": True, "path_globs": ["*.py"]})
        self.assertEqual(payload["matches"][0]["path"], "main.py")

    def test_search_text_skips_binary_and_redacts_previews(self) -> None:
        payload = self._run({"query": "ghp_", "path_globs": ["*"], "max_results": 20})
        text = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("cache.sqlite", text)
        self.assertNotIn("ghp_" + "A" * 36, text)
        self.assertIn("[REDACTED:github_token]", text)


if __name__ == "__main__":
    unittest.main()
