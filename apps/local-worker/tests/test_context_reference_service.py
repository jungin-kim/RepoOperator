import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from repooperator_worker.services.context_reference_service import resolve_context_reference


class _FakeClient:
    response: dict

    def generate_text(self, request):
        return json.dumps(self.response)


class ContextReferenceServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        (self.repo / "trim_videos.py").write_text(
            "def split_video(input_path):\n    return input_path\n",
            encoding="utf-8",
        )
        (self.repo / "README.md").write_text("# Demo\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _resolve(self, response: dict, **overrides):
        _FakeClient.response = response
        defaults = {
            "task": "current message",
            "conversation_history": [
                {"role": "user", "content": "Analyze trim_videos.py"},
                {"role": "assistant", "content": "split_video is in trim_videos.py."},
            ],
            "project_path": str(self.repo),
            "recent_files": ["trim_videos.py"],
            "last_analyzed_file": "trim_videos.py",
            "symbols": {"split_video": "trim_videos.py"},
            "suggestion_summary": "Refactor split_video in trim_videos.py.",
            "proposal_file": None,
            "candidate_files": [],
        }
        defaults.update(overrides)
        with patch(
            "repooperator_worker.services.context_reference_service.OpenAICompatibleModelClient",
            return_value=_FakeClient(),
        ):
            return resolve_context_reference(**defaults)

    def test_resolves_previous_file_from_llm(self) -> None:
        result = self._resolve(
            {
                "refers_to_previous_context": True,
                "reference_type": "file",
                "target_files": ["trim_videos.py"],
                "target_symbols": [],
                "confidence": 0.92,
                "needs_clarification": False,
                "clarification_question": None,
            },
            task="please update the file we just reviewed",
        )
        self.assertEqual(result.resolver, "llm")
        self.assertEqual(result.target_files, ["trim_videos.py"])

    def test_resolves_previous_symbol_to_file(self) -> None:
        result = self._resolve(
            {
                "refers_to_previous_context": True,
                "reference_type": "symbol",
                "target_files": [],
                "target_symbols": ["split_video"],
                "confidence": 0.9,
                "needs_clarification": False,
                "clarification_question": None,
            },
            task="refactor the function from the recent analysis",
        )
        self.assertEqual(result.reference_type, "symbol")
        self.assertEqual(result.target_files, ["trim_videos.py"])
        self.assertEqual(result.target_symbols, ["split_video"])

    def test_resolves_previous_change_suggestion(self) -> None:
        result = self._resolve(
            {
                "refers_to_previous_context": True,
                "reference_type": "change_suggestion",
                "target_files": ["trim_videos.py"],
                "target_symbols": ["split_video"],
                "confidence": 0.88,
                "needs_clarification": False,
                "clarification_question": None,
            },
            task="prepare a patch from the prior suggestion",
        )
        self.assertEqual(result.reference_type, "change_suggestion")
        self.assertEqual(result.target_files, ["trim_videos.py"])

    def test_ambiguous_context_asks_clarification(self) -> None:
        result = self._resolve(
            {
                "refers_to_previous_context": True,
                "reference_type": "file",
                "target_files": [],
                "target_symbols": [],
                "confidence": 0.45,
                "needs_clarification": True,
                "clarification_question": "Which recent file should I use?",
            },
            recent_files=["trim_videos.py", "README.md"],
            candidate_files=["trim_videos.py", "README.md"],
        )
        self.assertTrue(result.needs_clarification)
        self.assertEqual(result.target_files, [])

    def test_no_recent_context_returns_no_reference(self) -> None:
        result = self._resolve(
            {
                "refers_to_previous_context": False,
                "reference_type": "none",
                "target_files": [],
                "target_symbols": [],
                "confidence": 0.0,
                "needs_clarification": False,
                "clarification_question": None,
            },
            recent_files=[],
            last_analyzed_file=None,
            symbols={},
            suggestion_summary=None,
            conversation_history=[],
        )
        self.assertFalse(result.refers_to_previous_context)
        self.assertEqual(result.reference_type, "none")


if __name__ == "__main__":
    unittest.main()
