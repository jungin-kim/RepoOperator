import json
import sys
import tempfile
import unittest
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repooperator_worker.agent_core.context_service import ContextService  # noqa: E402
from repooperator_worker.schemas import AgentRunRequest, ConversationMessage  # noqa: E402


class ContextServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name) / "repo"
        self.repo.mkdir()
        (self.repo / "README.md").write_text("# Demo\n\nFixture project.\n", encoding="utf-8")
        (self.repo / "AGENTS.md").write_text("Use focused tests.\n", encoding="utf-8")
        (self.repo / "package.json").write_text('{"scripts":{"test":"true"}}\n', encoding="utf-8")
        self.service = ContextService()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _request(self) -> AgentRunRequest:
        return AgentRunRequest(
            project_path=str(self.repo),
            git_provider="local",
            branch="main",
            thread_id="thread-context",
            task="Summarize",
            conversation_history=[
                ConversationMessage(
                    role="assistant",
                    content="Previously read files.",
                    metadata={"files_read": ["README.md"], "commands_run": ["git status --short"]},
                )
            ],
        )

    def test_collects_high_signal_and_instruction_files(self) -> None:
        packet = self.service.collect(self._request())
        self.assertIn("README.md", packet.high_signal_files)
        self.assertIn("package.json", packet.high_signal_files)
        self.assertIn("AGENTS.md", packet.project_instructions)

    def test_extracts_prior_thread_metadata(self) -> None:
        packet = self.service.collect(self._request())
        self.assertEqual(packet.prior_files_read, ["README.md"])
        self.assertEqual(packet.prior_commands_run, ["git status --short"])

    def test_returns_json_safe_packet(self) -> None:
        packet = self.service.collect(self._request())
        json.dumps(packet.model_dump(), ensure_ascii=False)

    def test_refreshes_when_high_signal_fingerprint_changes(self) -> None:
        request = self._request()
        first = self.service.collect(request)
        (self.repo / "README.md").write_text("# Changed\n", encoding="utf-8")
        second = self.service.collect(request)
        self.assertNotEqual(first.high_signal_fingerprint, second.high_signal_fingerprint)
        self.assertIn("# Changed", second.high_signal_files["README.md"])

    def test_cached_packet_reports_cache_hit_within_ttl(self) -> None:
        request = self._request()
        first = self.service.collect(request)
        second = self.service.collect(request)
        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(first.cache_key, second.cache_key)

    def test_force_refresh_returns_new_packet_after_readme_changes(self) -> None:
        request = self._request()
        first = self.service.collect(request)
        (self.repo / "README.md").write_text("# Forced\n", encoding="utf-8")
        second = self.service.collect(request, force_refresh=True)
        self.assertFalse(second.cache_hit)
        self.assertEqual(second.invalidation_reason, "force_refresh")
        self.assertNotEqual(first.high_signal_files["README.md"], second.high_signal_files["README.md"])

    def test_changing_branch_returns_different_packet(self) -> None:
        first = self.service.collect(self._request())
        request = self._request()
        request.branch = "feature/test"
        second = self.service.collect(request)
        self.assertNotEqual(first.cache_key, second.cache_key)

    def test_invalidate_repo_clears_cached_packet(self) -> None:
        request = self._request()
        first = self.service.collect(request)
        self.service.invalidate(repo_path=str(self.repo))
        second = self.service.collect(request)
        self.assertFalse(second.cache_hit)
        self.assertEqual(first.cache_key, second.cache_key)

    def test_does_not_fail_when_git_info_unavailable(self) -> None:
        packet = self.service.collect(self._request())
        self.assertIsNone(packet.git_status_summary)
        self.assertIsNone(packet.recent_commits_summary)


if __name__ == "__main__":
    unittest.main()
