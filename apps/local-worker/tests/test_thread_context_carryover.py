import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repooperator_worker.schemas import AgentRunRequest, AgentRunResponse  # noqa: E402
from repooperator_worker.services.thread_context_service import build_thread_context, update_thread_context  # noqa: E402


class ThreadContextCarryoverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.home = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name) / "repo"
        self.repo.mkdir()
        (self.repo / "main.py").write_text("def create_message(body):\n    return {'body': body}\n", encoding="utf-8")
        self.config = Path(self.tmp.name) / "config.json"
        self.config.write_text(
            json.dumps({"repooperatorHomeDir": self.home.name, "openai": {"model": "test-model"}}),
            encoding="utf-8",
        )
        self.previous_config_env = os.environ.get("REPOOPERATOR_CONFIG_PATH")
        os.environ["REPOOPERATOR_CONFIG_PATH"] = str(self.config)

    def tearDown(self) -> None:
        if self.previous_config_env is None:
            os.environ.pop("REPOOPERATOR_CONFIG_PATH", None)
        else:
            os.environ["REPOOPERATOR_CONFIG_PATH"] = self.previous_config_env
        self.home.cleanup()
        self.tmp.cleanup()

    def _request(self) -> AgentRunRequest:
        return AgentRunRequest(
            project_path=str(self.repo),
            git_provider="local",
            branch="main",
            thread_id="thread-carryover-test",
            task="Prepare the proposal.",
            conversation_history=[],
        )

    def test_response_writes_structured_target_candidates_for_next_turn(self) -> None:
        request = self._request()
        response = AgentRunResponse(
            project_path=request.project_path,
            git_provider="local",
            active_repository_source="local",
            active_repository_path=request.project_path,
            active_branch="main",
            task=request.task,
            model="test-model",
            branch="main",
            repo_root_name="repo",
            context_summary="",
            top_level_entries=[],
            readme_included=False,
            diff_included=False,
            is_git_repository=False,
            files_read=["main.py"],
            response="Use main.py for the proposal.",
            change_set_proposal={
                "proposal_id": "proposal-test",
                "plan": {"summary": "Update message creation.", "target_files": ["main.py"], "operations": ["modify"]},
                "changes": [{"path": "main.py", "operation": "modify", "summary": "Update create_message."}],
                "status": "valid",
            },
            recommendation_context={
                "target_selection": {
                    "selected_target_files": ["main.py"],
                    "prior_evidence_reused": False,
                    "candidates": [{"path": "main.py", "score": 93, "role": "entrypoints", "sources": ["read_file"]}],
                }
            },
        )
        update_thread_context(request, response)

        restored = build_thread_context(request)
        self.assertEqual(restored.last_proposed_target_file, "main.py")
        self.assertEqual(restored.last_proposal_id, "proposal-test")
        self.assertEqual(restored.last_implementation_plan["target_files"], ["main.py"])
        self.assertEqual(restored.last_target_candidates[0]["path"], "main.py")
        self.assertIn("create_message", restored.symbols)

    def test_explanatory_response_writes_understanding_and_evidence_memory(self) -> None:
        request = self._request()
        response = AgentRunResponse(
            project_path=request.project_path,
            git_provider="local",
            active_repository_source="local",
            active_repository_path=request.project_path,
            active_branch="main",
            task=request.task,
            model="test-model",
            branch="main",
            repo_root_name="repo",
            context_summary="",
            top_level_entries=[],
            readme_included=False,
            diff_included=False,
            is_git_repository=False,
            files_read=["main.py"],
            response="main.py contains create_message, so that is the implementation target.",
            recommendation_context={
                "user_understanding_context": {
                    "normalized_goal": "Prepare a proposal.",
                    "requested_outputs": ["code_change_proposal"],
                },
                "evidence_basis": {
                    "files": [{"path": "main.py", "role": "implementation"}],
                    "target_selection": {
                        "selected_target_files": ["main.py"],
                        "prior_evidence_reused": False,
                        "candidates": [{"path": "main.py", "score": 91}],
                    },
                },
                "target_selection": {
                    "selected_target_files": ["main.py"],
                    "candidates": [{"path": "main.py", "score": 91, "role": "entrypoints"}],
                },
            },
        )
        update_thread_context(request, response)

        restored = build_thread_context(request)
        self.assertEqual(restored.last_user_understanding_context["normalized_goal"], "Prepare a proposal.")
        self.assertEqual(restored.last_evidence_basis[0]["selected_target_files"], ["main.py"])
        self.assertEqual(restored.last_target_candidates[0]["path"], "main.py")


if __name__ == "__main__":
    unittest.main()
