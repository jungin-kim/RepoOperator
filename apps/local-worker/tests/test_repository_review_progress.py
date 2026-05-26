import sys
import tempfile
import unittest
import inspect
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch


TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repooperator_worker.schemas import AgentRunRequest  # noqa: E402
from repooperator_worker.agent_core.repository_review import (  # noqa: E402
    MAX_REPOSITORY_REVIEW_BYTES,
    REPOSITORY_REVIEW_BINARY_SUFFIXES,
    REPOSITORY_REVIEW_SUFFIXES,
    inventory_repository_review_files,
    review_progress_labels,
    run_repository_review,
    should_use_repository_wide_review,
)
from repooperator_worker.services.event_service import list_run_events  # noqa: E402


class _ReviewClient:
    @property
    def model_name(self) -> str:
        return "review-test-model"

    def generate_text(self, request):
        if "slow_module.py" in request.user_prompt:
            raise RuntimeError("Model API request timed out after 120 seconds")
        if "server.py" in request.user_prompt:
            return "Purpose: exposes a small server helper. Confirmed issues: none from the shown code."
        if "Client.kt" in request.user_prompt:
            return "Purpose: contains a client entry point. Improvement: add error handling around network calls if present."
        return "Purpose: documentation or configuration. Confirmed issues: none from the shown content."


class _InspectingReviewClient(_ReviewClient):
    home: Path

    def generate_text(self, request):
        if "server.py" in request.user_prompt:
            events = list_run_events("run_review_test")
            server_events = [event for event in events if event.get("activity_id") == "review-file:server.py"]
            if not any(event.get("event_type") == "activity_updated" for event in server_events):
                raise AssertionError("server.py activity update was not persisted before model review returned")
        return super().generate_text(request)


class RepositoryReviewProgressTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name) / "fixture"
        self.repo.mkdir()
        (self.repo / "README.md").write_text("# Fixture\n\nSmall mixed repository.\n", encoding="utf-8")
        docs = self.repo / "docs"
        docs.mkdir()
        (docs / "README.md").write_text("# Docs\n", encoding="utf-8")
        (self.repo / "server.py").write_text("def handle():\n    return {'ok': True}\n", encoding="utf-8")
        (self.repo / "test_agent_routing_graph 2.py").write_text("def stale():\n    return True\n", encoding="utf-8")
        (self.repo / "slow_module.py").write_text("def slow():\n    return 'needs review'\n", encoding="utf-8")
        (self.repo / "Client.kt").write_text("fun main() { println(\"hi\") }\n", encoding="utf-8")
        (self.repo / "diagram.pdf").write_bytes(b"%PDF-1.4\x00binary")
        node_modules = self.repo / "node_modules" / "pkg"
        node_modules.mkdir(parents=True)
        (node_modules / "index.js").write_text("console.log('generated')\n", encoding="utf-8")
        self.home = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.home.cleanup()
        self.tmp.cleanup()

    def _request(self, task: str) -> AgentRunRequest:
        return AgentRunRequest(
            project_path=str(self.repo),
            git_provider="local",
            branch="main",
            thread_id="thread-review",
            task=task,
            conversation_history=[],
        )

    def _run_review(self, task: str):
        request = self._request(task)
        with patch(
            "repooperator_worker.agent_core.repository_review.OpenAICompatibleModelClient",
            return_value=_ReviewClient(),
        ), patch(
            "repooperator_worker.services.event_service.get_repooperator_home_dir",
            return_value=Path(self.home.name),
        ):
            return run_repository_review(
                request=request,
                run_id="run_review_test",
                classifier=SimpleNamespace(intent="repo_analysis"),
            )

    def test_repository_review_uses_per_file_progress_events(self) -> None:
        result = self._run_review("Please review the whole repository and summarize confirmed file-level findings.")

        event_types = [event.get("event_type") for event in result.activity_events]
        self.assertIn("activity_started", event_types)
        self.assertIn("activity_updated", event_types)
        self.assertIn("activity_completed", event_types)
        self.assertTrue(any(event.get("files") for event in result.activity_events))
        self.assertNotIn("context", " ".join(str(event.get("label", "")) for event in result.activity_events).lower())

    def test_file_review_uses_one_stable_activity_id_for_updates(self) -> None:
        result = self._run_review("Please perform a broad repository review.")

        server_events = [
            event for event in result.activity_events
            if event.get("activity_id") == "review-file:server.py"
        ]
        self.assertGreaterEqual(len(server_events), 3)
        self.assertEqual(
            [event.get("event_type") for event in server_events],
            ["activity_started", "activity_updated", "activity_completed"],
        )
        self.assertEqual({event.get("label") for event in server_events}, {"server.py"})

    def test_per_file_timeout_is_partial_and_not_confirmed(self) -> None:
        result = self._run_review("Perform a repository-wide file review.")

        self.assertIn("slow_module.py", result.response)
        self.assertIn("timed out", result.response.lower())
        self.assertNotIn("slow_module.py` was read successfully", result.response)
        self.assertIn("server.py", result.response)
        self.assertIn("Confirmed File-Level Results", result.response)
        timeout_events = [
            event for event in result.activity_events
            if event.get("activity_id") == "review-file:slow_module.py" and event.get("event_type") == "activity_failed"
        ]
        self.assertEqual(len(timeout_events), 1)
        self.assertEqual(timeout_events[0].get("files"), ["slow_module.py"])
        self.assertIn("continuing", str(timeout_events[0].get("detail")).lower())

    def test_unsupported_files_are_skipped_with_reason(self) -> None:
        result = self._run_review("Assess every readable file in this project.")

        skipped_event = next(event for event in result.activity_events if event.get("activity_id") == "repository-review-aggregate")
        aggregate = skipped_event.get("aggregate") or {}
        self.assertGreaterEqual(int(aggregate.get("files_skipped_count") or 0), 1)
        self.assertIn("diagram.pdf", result.response)
        self.assertNotIn("node_modules/pkg/index.js", result.files_read)

    def test_progress_events_do_not_expose_hidden_reasoning(self) -> None:
        result = self._run_review("Review the repository.")

        serialized = "\n".join(str(event) for event in result.activity_events)
        self.assertNotIn("<think>", serialized)
        self.assertNotIn("system_prompt", serialized)
        self.assertNotIn("raw model prompt", serialized.lower())

    def test_progress_event_is_persisted_before_long_model_call_finishes(self) -> None:
        request = self._request("Please review the repository.")
        with patch(
            "repooperator_worker.agent_core.repository_review.OpenAICompatibleModelClient",
            return_value=_InspectingReviewClient(),
        ), patch(
            "repooperator_worker.services.event_service.get_repooperator_home_dir",
            return_value=Path(self.home.name),
        ):
            result = run_repository_review(
                request=request,
                run_id="run_review_test",
                classifier=SimpleNamespace(intent="repo_analysis"),
            )
        self.assertTrue(result.activity_events)

    def test_safe_summary_generator_is_file_specific(self) -> None:
        result = self._run_review("Please review the repository.")
        server = next(
            event for event in result.activity_events
            if event.get("activity_id") == "review-file:server.py" and event.get("event_type") == "activity_updated"
        )
        readme = next(
            event for event in result.activity_events
            if event.get("activity_id") == "review-file:README.md" and event.get("event_type") == "activity_updated"
        )
        self.assertIn("Python source", str(server.get("observation")))
        self.assertIn("documents", str(readme.get("safe_reasoning_summary_delta")))

    def test_no_completed_summary_when_no_file_review_succeeds(self) -> None:
        class _TimeoutClient(_ReviewClient):
            def generate_text(self, request):
                raise RuntimeError("request timed out")

        request = self._request("Review the entire codebase.")
        with patch(
            "repooperator_worker.agent_core.repository_review.OpenAICompatibleModelClient",
            return_value=_TimeoutClient(),
        ), patch(
            "repooperator_worker.services.event_service.get_repooperator_home_dir",
            return_value=Path(self.home.name),
        ):
            result = run_repository_review(
                request=request,
                run_id="run_review_timeout",
                classifier=SimpleNamespace(intent="repo_analysis"),
            )

        self.assertIn("did not complete", result.response)
        self.assertNotIn("Confirmed File-Level Results", result.response)
        self.assertEqual(result.files_read, [])

    def test_repository_wide_review_selection_uses_target_evidence(self) -> None:
        classifier = SimpleNamespace(target_files=[], mentioned_files=[])
        self.assertTrue(should_use_repository_wide_review(classifier))

    def test_duplicate_basename_labels_are_disambiguated(self) -> None:
        labels = review_progress_labels(["README.md", "docs/README.md", "package.json"])
        self.assertEqual(labels["README.md"], "README.md · root")
        self.assertEqual(labels["docs/README.md"], "README.md · docs")
        self.assertEqual(labels["package.json"], "package.json")

    def test_stale_duplicate_copy_is_skipped_from_review_selection(self) -> None:
        inventory = inventory_repository_review_files(self.repo)
        self.assertNotIn("test_agent_routing_graph 2.py", inventory["selected"])
        stale = [item for item in inventory["skipped"] if item["file"] == "test_agent_routing_graph 2.py"]
        self.assertEqual(stale[0]["reason"], "stale duplicate copy")

    def test_stale_duplicate_copy_detection_covers_conflict_suffixes(self) -> None:
        from repooperator_worker.agent_core.repository_review import is_stale_duplicate_copy

        for name in ("skills 6.py", "debug page 10.tsx", "settings copy.json", "module.py.bak", "module.py.orig"):
            with self.subTest(name=name):
                self.assertTrue(is_stale_duplicate_copy(Path(name)))
        self.assertFalse(is_stale_duplicate_copy(Path("package-lock.json")))

    def test_selected_files_override_repository_wide_classifier_fields(self) -> None:
        classifier = SimpleNamespace(target_files=["server.py"], mentioned_files=[])
        self.assertFalse(should_use_repository_wide_review(classifier))

    def test_specific_file_hint_does_not_select_repository_wide_review(self) -> None:
        classifier = SimpleNamespace(target_files=[], mentioned_files=["server.py"])
        self.assertFalse(should_use_repository_wide_review(classifier))

    def test_repository_wide_review_gate_has_no_natural_language_phrase_lists(self) -> None:
        source = inspect.getsource(should_use_repository_wide_review)
        self.assertNotIn("review" + "_signals", source)
        self.assertNotIn("broad" + "_scope_signals", source)
        self.assertNotIn("request.task", source)
        self.assertNotIn(".lower()", source)

    def test_repository_review_safety_constants_remain(self) -> None:
        self.assertIn(".py", REPOSITORY_REVIEW_SUFFIXES)
        self.assertIn(".pdf", REPOSITORY_REVIEW_BINARY_SUFFIXES)
        self.assertGreater(MAX_REPOSITORY_REVIEW_BYTES, 0)


if __name__ == "__main__":
    unittest.main()
