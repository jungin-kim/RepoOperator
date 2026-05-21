import sys
import unittest
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repooperator_worker.agent_core.planner import TaskFrame, validate_model_next_action, validate_visible_work_note  # noqa: E402
from repooperator_worker.agent_core.steering import SteeringDecision  # noqa: E402
from repooperator_worker.agent_core.state import AgentCoreState, ClassifierResult  # noqa: E402
from repooperator_worker.schemas import AgentRunRequest  # noqa: E402


class PlannerSteeringModuleTests(unittest.TestCase):
    def test_planner_and_steering_types_are_direct_imports(self) -> None:
        frame = TaskFrame(user_goal="Explain README.md", mentioned_files=["README.md"])
        decision = SteeringDecision(steering_type="defer")
        self.assertEqual(frame.mentioned_files, ["README.md"])
        self.assertEqual(decision.steering_type, "defer")

    def test_planner_validates_search_text_action_directly(self) -> None:
        request = AgentRunRequest(project_path=".", git_provider="local", branch="main", task="search text")
        state = AgentCoreState(run_id="run-planner", thread_id=None, repo=".", branch="main", user_task=request.task)
        state.classifier_result = ClassifierResult(intent="read_only_question", confidence=0.8)
        action = validate_model_next_action(
            {
                "action_type": "search_text",
                "reason_summary": "Search text safely.",
                "query": "foo|bar",
                "regex": True,
                "confidence": 0.9,
            },
            request,
            state,
            TaskFrame(user_goal=request.task),
        )
        self.assertIsNotNone(action)
        self.assertEqual(action.type, "search_text")
        self.assertTrue(action.payload["regex"])

    def test_visible_work_note_is_validated_and_truncated(self) -> None:
        note = validate_visible_work_note(
            {
                "goal": "x" * 500,
                "why_this_action": "Search first because the file is not named.",
                "evidence_needed": ["candidate files"] * 10,
                "uncertainty": "not-a-list",
                "safety_note": None,
            }
        )
        self.assertIsNotNone(note)
        self.assertLessEqual(len(note["goal"]), 160)
        self.assertEqual(len(note["evidence_needed"]), 6)
        self.assertNotIn("uncertainty", note)

    def test_invalid_visible_work_note_is_ignored(self) -> None:
        self.assertIsNone(validate_visible_work_note("plain text"))


if __name__ == "__main__":
    unittest.main()
