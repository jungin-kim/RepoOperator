import json
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

from repooperator_worker.agent_core.actions import AgentAction, ActionResult  # noqa: E402
from repooperator_worker.agent_core.events import merge_activity_states  # noqa: E402
from repooperator_worker.agent_core.langgraph_runtime import run_langgraph_controller  # noqa: E402
from repooperator_worker.agent_core.repository_review import run_repository_review  # noqa: E402
from repooperator_worker.schemas import AgentRunRequest  # noqa: E402
from repooperator_worker.services.retrieval_service import StructuredRetrievalIntent, retrieve_context  # noqa: E402
from repooperator_worker.services.skills_service import discover_skills  # noqa: E402


class _Client:
    @property
    def model_name(self):
        return "test-model"

    def generate_text(self, request):
        if "task-understanding layer" in request.system_prompt.lower():
            return json.dumps(
                {
                    "user_goal": "Explain README.md",
                    "mentioned_files": ["README.md"],
                    "mentioned_symbols": [],
                    "constraints": [],
                    "requested_outputs": ["explanation"],
                    "likely_needed_tools": ["read_file"],
                    "safety_notes": [],
                    "uncertainties": [],
                    "needs_clarification": False,
                    "clarification_question": None,
                }
            )
        return "README.md documents the fixture repository. No hidden reasoning."

    def stream_text(self, request):
        yield {"type": "assistant_delta", "delta": "README.md documents "}
        yield {"type": "assistant_delta", "delta": "the fixture repository."}


class AgentCoreRefactorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name) / "repo"
        self.repo.mkdir()
        (self.repo / "README.md").write_text("# Fixture\n", encoding="utf-8")
        (self.repo / "app.py").write_text("def main():\n    return 1\n", encoding="utf-8")
        self.home = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.home.cleanup()
        self.tmp.cleanup()

    def _request(self, task="Explain README.md"):
        return AgentRunRequest(project_path=str(self.repo), git_provider="local", branch="main", thread_id="t1", task=task)

    def test_action_protocol_has_required_result_fields(self):
        action = AgentAction(type="read_file", reason_summary="read evidence", target_files=["README.md"])
        result = ActionResult(action_id=action.action_id, status="success", files_read=["README.md"])
        self.assertEqual(action.model_dump()["type"], "read_file")
        self.assertEqual(result.model_dump()["status"], "success")
        self.assertIn("duration_ms", result.model_dump())

    def test_controller_uses_agent_core_without_legacy_read_only_graph(self):
        with patch(
            "repooperator_worker.agent_core.graph.support.OpenAICompatibleModelClient",
            return_value=_Client(),
        ), patch(
            "repooperator_worker.agent_core.graph.repository_support.get_active_repository",
            return_value=None,
        ):
            response = run_langgraph_controller(self._request(), run_id="run_core_test")
        self.assertEqual(response.agent_flow, "langgraph")
        self.assertEqual(response.files_read, ["README.md"])
        self.assertIn("README.md", response.response)

    def test_activity_state_merges_stable_activity_id(self):
        merged = merge_activity_states(
            [
                {"activity_id": "review-file:README.md", "status": "running", "observation": "reading"},
                {"activity_id": "review-file:README.md", "status": "completed", "observation": "reviewed"},
            ]
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["status"], "completed")
        self.assertEqual(merged[0]["observation"], "reviewed")

    def test_structured_retrieval_drives_selection(self):
        result = retrieve_context(
            self.repo,
            "natural language should not decide broad routing",
            StructuredRetrievalIntent(repository_wide=True),
        )
        self.assertEqual(result.query_type, "project_review")
        self.assertTrue(result.files)

    def test_repository_review_returns_evidence_events(self):
        with patch(
            "repooperator_worker.agent_core.repository_review.OpenAICompatibleModelClient",
            return_value=_Client(),
        ), patch(
            "repooperator_worker.services.event_service.get_repooperator_home_dir",
            return_value=Path(self.home.name),
        ):
            response = run_repository_review(request=self._request("Review the repository"), run_id="run_review_core")
        self.assertTrue(response.activity_events)
        self.assertIn("Confirmed File-Level Results", response.response)
        self.assertTrue(all("<think>" not in json.dumps(event) for event in response.activity_events))

    def test_source_aware_skills_keep_builtin_visible(self):
        skills_file = Path(self.home.name) / ".repooperator" / "skills.md"
        skills_file.parent.mkdir(parents=True)
        skills_file.write_text("# Git Workflow\nUser refinement\n", encoding="utf-8")
        config = Path(self.home.name) / ".repooperator" / "config.json"
        config.write_text(json.dumps({"repooperatorHomeDir": str(Path(self.home.name) / ".repooperator")}), encoding="utf-8")
        with patch.dict(os.environ, {"REPOOPERATOR_CONFIG_PATH": str(config)}, clear=True):
            payload = discover_skills()
        names = [(item["source_type"], item["name"]) for item in payload["skills"]]
        self.assertIn(("builtin", "Git Workflow"), names)
        self.assertIn(("user", "Git Workflow"), names)
        self.assertTrue(all(item.get("identity") for item in payload["skills"]))


if __name__ == "__main__":
    unittest.main()
