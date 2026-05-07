"""
Tests for request_understanding.py.

Covers:
- RequestUnderstanding dataclass fields and defaults
- understand_request falls back to deterministic extraction on model failure
- Deterministic file token extraction from task text
- Malformed model JSON still extracts file tokens correctly
- request_understanding_to_classifier_result adapter: intent="ambiguous", no routing fields
- likely_needed_tools only contains allowed tool names
- needs_clarification propagates to ClassifierResult
- Adding a new user task does NOT require a new intent category
"""
from __future__ import annotations

import json
import inspect
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repooperator_worker.agent_core.request_understanding import (  # noqa: E402
    RequestUnderstanding,
    _ALLOWED_TOOL_HINTS,
    _build_understanding,
    _parse_json,
    request_understanding_to_classifier_result,
    understand_request,
)
from repooperator_worker.agent_core.request_parsing import extract_file_tokens  # noqa: E402
from repooperator_worker.agent_core.planner import TaskFrame, edit_requested, edit_requested_text  # noqa: E402
from repooperator_worker.schemas import AgentRunRequest  # noqa: E402


def _req(task: str = "what does this repo do?", **kwargs) -> AgentRunRequest:
    return AgentRunRequest(
        task=task,
        project_path="/tmp/mock",
        git_provider="local",
        **kwargs,
    )


class TestRequestUnderstandingDataclass(unittest.TestCase):

    def test_default_fields_are_empty(self):
        ru = RequestUnderstanding()
        self.assertEqual(ru.user_goal, "")
        self.assertEqual(ru.mentioned_files, [])
        self.assertEqual(ru.mentioned_symbols, [])
        self.assertEqual(ru.constraints, [])
        self.assertEqual(ru.requested_outputs, [])
        self.assertEqual(ru.likely_needed_tools, [])
        self.assertEqual(ru.safety_notes, [])
        self.assertEqual(ru.uncertainties, [])
        self.assertFalse(ru.needs_clarification)
        self.assertIsNone(ru.clarification_question)

    def test_has_no_routing_fields(self):
        ru = RequestUnderstanding()
        self.assertFalse(hasattr(ru, "requested_workflow"))
        self.assertFalse(hasattr(ru, "retrieval_goal"))
        self.assertFalse(hasattr(ru, "requires_repository_wide_review"))
        self.assertFalse(hasattr(ru, "analysis_scope"))
        self.assertFalse(hasattr(ru, "intent"))

    def test_model_dump_returns_dict(self):
        ru = RequestUnderstanding(user_goal="explain the repo", mentioned_files=["README.md"])
        d = ru.model_dump()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["user_goal"], "explain the repo")
        self.assertEqual(d["mentioned_files"], ["README.md"])


class TestDeterministicExtraction(unittest.TestCase):

    def test_file_token_helper_is_shared_without_planner_dependency(self):
        self.assertEqual(extract_file_tokens("Check README.md and src/app.py"), ["README.md", "src/app.py"])
        import repooperator_worker.agent_core.request_understanding as module

        source = inspect.getsource(module)
        self.assertNotIn("agent_core.planner import", source)

    def test_file_tokens_extracted_from_task(self):
        ru = _build_understanding({}, _req(task="Please explain README.md and app.py"))
        self.assertIn("README.md", ru.mentioned_files)
        self.assertIn("app.py", ru.mentioned_files)

    def test_no_files_in_generic_task(self):
        ru = _build_understanding({}, _req(task="what does this project do?"))
        self.assertEqual(ru.mentioned_files, [])

    def test_model_files_merged_with_deterministic_tokens(self):
        payload = {"mentioned_files": ["src/core.py"], "user_goal": "check core"}
        ru = _build_understanding(payload, _req(task="Also look at README.md"))
        self.assertIn("src/core.py", ru.mentioned_files)
        self.assertIn("README.md", ru.mentioned_files)

    def test_deduplication_preserves_order(self):
        payload = {"mentioned_files": ["README.md", "app.py"]}
        ru = _build_understanding(payload, _req(task="check README.md"))
        self.assertEqual(ru.mentioned_files.count("README.md"), 1)

    def test_malformed_model_json_still_extracts_file_tokens(self):
        ru = _build_understanding({}, _req(task="Explain Border.cs and check utils.py"))
        self.assertIn("Border.cs", ru.mentioned_files)
        self.assertIn("utils.py", ru.mentioned_files)


class TestLikelyNeededToolsConstraint(unittest.TestCase):

    def test_only_allowed_tool_hints_are_kept(self):
        payload = {
            "likely_needed_tools": [
                "read_file",
                "run_command",
                "unknown_tool",
                "delete_files",
                "inspect_repo_tree",
            ]
        }
        ru = _build_understanding(payload, _req())
        for tool in ru.likely_needed_tools:
            self.assertIn(tool, _ALLOWED_TOOL_HINTS, f"Disallowed tool hint: {tool}")
        self.assertNotIn("unknown_tool", ru.likely_needed_tools)
        self.assertNotIn("delete_files", ru.likely_needed_tools)

    def test_empty_tool_hints_allowed(self):
        ru = _build_understanding({}, _req())
        self.assertEqual(ru.likely_needed_tools, [])


class TestStructuredEditDetection(unittest.TestCase):

    def test_korean_recover_task_is_edit_like_from_tool_hint(self):
        frame = TaskFrame(
            user_goal="세이브 파일 깨졌을 때 복구 가능하게 해줘.",
            likely_needed_tools=["search_text", "read_file", "generate_edit"],
            requested_outputs=["code_change_proposal"],
        )
        self.assertTrue(edit_requested(frame))
        self.assertFalse(edit_requested_text(frame.user_goal))

    def test_korean_improve_task_is_edit_like_from_requested_output(self):
        frame = TaskFrame(
            user_goal="저장 쪽 위험한 코드 찾아서 개선안 줘.",
            likely_needed_tools=["search_files", "read_file"],
            requested_outputs=["code_review", "edit_proposal"],
        )
        self.assertTrue(edit_requested(frame))
        self.assertFalse(edit_requested_text(frame.user_goal))

    def test_edit_requested_text_fallback_is_not_expanded_keyword_list(self):
        source = inspect.getsource(edit_requested_text)
        for token in ("복구", "개선", "recover", "improve", "harden", "stabilize", "cleanup"):
            self.assertNotIn(token, source)


class TestUnderstandRequestFallback(unittest.TestCase):

    def test_understand_request_falls_back_on_model_error(self):
        with patch(
            "repooperator_worker.agent_core.request_understanding.OpenAICompatibleModelClient",
        ) as mock_cls:
            mock_client = MagicMock()
            mock_client.generate_text.side_effect = RuntimeError("no model")
            mock_cls.return_value = mock_client
            ru = understand_request(_req(task="Explain README.md"))
        self.assertIsInstance(ru, RequestUnderstanding)
        self.assertIn("README.md", ru.mentioned_files)

    def test_understand_request_uses_model_output_when_valid(self):
        model_payload = {
            "user_goal": "check the README",
            "mentioned_files": ["README.md"],
            "mentioned_symbols": ["main"],
            "constraints": ["read-only"],
            "requested_outputs": ["explanation"],
            "likely_needed_tools": ["read_file"],
            "safety_notes": [],
            "uncertainties": [],
            "needs_clarification": False,
            "clarification_question": None,
        }
        with patch(
            "repooperator_worker.agent_core.request_understanding.OpenAICompatibleModelClient",
        ) as mock_cls:
            mock_client = MagicMock()
            mock_client.generate_text.return_value = json.dumps(model_payload)
            mock_cls.return_value = mock_client
            ru = understand_request(_req(task="Explain README.md"))
        self.assertEqual(ru.user_goal, "check the README")
        self.assertIn("README.md", ru.mentioned_files)
        self.assertEqual(ru.mentioned_symbols, ["main"])
        self.assertEqual(ru.constraints, ["read-only"])

    def test_understand_request_malformed_json_uses_deterministic_fallback(self):
        with patch(
            "repooperator_worker.agent_core.request_understanding.OpenAICompatibleModelClient",
        ) as mock_cls:
            mock_client = MagicMock()
            mock_client.generate_text.return_value = "not json at all"
            mock_cls.return_value = mock_client
            ru = understand_request(_req(task="check app.py for issues"))
        self.assertIn("app.py", ru.mentioned_files)


class TestAdapterToClassifierResult(unittest.TestCase):

    def _adapter(self, **ru_kwargs):
        req = _req()
        ru = RequestUnderstanding(**ru_kwargs)
        return request_understanding_to_classifier_result(ru, req)

    def test_adapter_sets_intent_to_ambiguous(self):
        result = self._adapter()
        self.assertEqual(result.intent, "ambiguous")

    def test_adapter_sets_confidence_to_zero(self):
        result = self._adapter()
        self.assertEqual(result.confidence, 0.0)

    def test_adapter_maps_mentioned_files_to_target_files(self):
        result = self._adapter(mentioned_files=["src/main.py", "README.md"])
        self.assertEqual(result.target_files, ["src/main.py", "README.md"])

    def test_adapter_maps_mentioned_symbols_to_target_symbols(self):
        result = self._adapter(mentioned_symbols=["MyClass", "do_thing"])
        self.assertEqual(result.target_symbols, ["MyClass", "do_thing"])

    def test_adapter_does_not_populate_routing_fields(self):
        result = self._adapter()
        self.assertFalse(hasattr(result, "requested_workflow"))
        self.assertFalse(hasattr(result, "retrieval_goal"))
        self.assertFalse(hasattr(result, "requires_repository_wide_review"))
        self.assertFalse(hasattr(result, "analysis_scope"))

    def test_adapter_propagates_needs_clarification(self):
        result = self._adapter(needs_clarification=True, clarification_question="Which file?")
        self.assertTrue(result.needs_clarification)
        self.assertEqual(result.clarification_question, "Which file?")

    def test_adapter_stores_understanding_in_raw(self):
        ru = RequestUnderstanding(user_goal="explain", mentioned_files=["README.md"])
        result = request_understanding_to_classifier_result(ru, _req())
        self.assertIn("request_understanding", result.raw)

    def test_adapter_requested_action_truncated_to_120(self):
        long_goal = "x" * 200
        result = self._adapter(user_goal=long_goal)
        self.assertLessEqual(len(result.requested_action), 120)


class TestNewTaskDoesNotRequireNewIntentCategory(unittest.TestCase):
    """Adding new user tasks must not require new intent categories."""

    def test_any_task_returns_ambiguous_without_code_change(self):
        new_tasks = [
            "Generate a changelog from recent git commits",
            "Add OpenTelemetry tracing to the API layer",
            "Migrate from pytest to unittest",
            "설명해줘",
            "Give this codebase a security audit",
        ]
        for task in new_tasks:
            with self.subTest(task=task):
                with patch(
                    "repooperator_worker.agent_core.request_understanding.OpenAICompatibleModelClient",
                ) as mock_cls:
                    mock_client = MagicMock()
                    mock_client.generate_text.return_value = json.dumps({
                        "user_goal": task,
                        "mentioned_files": [],
                        "mentioned_symbols": [],
                        "constraints": [],
                        "requested_outputs": [],
                        "likely_needed_tools": [],
                        "safety_notes": [],
                        "uncertainties": [],
                        "needs_clarification": False,
                        "clarification_question": None,
                    })
                    mock_cls.return_value = mock_client
                    ru = understand_request(_req(task=task))
                    result = request_understanding_to_classifier_result(ru, _req(task=task))
                self.assertEqual(result.intent, "ambiguous", f"task: {task!r}")


class TestParseJson(unittest.TestCase):

    def test_strips_markdown_fences(self):
        raw = '```json\n{"user_goal": "explain"}\n```'
        self.assertEqual(_parse_json(raw).get("user_goal"), "explain")

    def test_plain_json(self):
        self.assertEqual(_parse_json('{"a": 1}'), {"a": 1})

    def test_non_object_returns_empty(self):
        self.assertEqual(_parse_json("[1, 2]"), {})

    def test_invalid_returns_empty(self):
        self.assertEqual(_parse_json("not json"), {})


if __name__ == "__main__":
    unittest.main()
