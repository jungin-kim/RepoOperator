"""
Tests for the classifier shim (agent_core/classifier.py) and the
request_understanding adapter it wraps.

Covers:
- classify_intent returns ClassifierResult with intent="ambiguous" (no routing fields)
- controller_graph no longer re-exports the classifier shim
- Malformed model JSON falls back gracefully (intent="ambiguous")
- _parse_classifier_json strips markdown fences
- validate_classifier_payload returns ClassifierResult
- ClassifierResult result is JSON-serialisable
- Wrong classifier result does not block the planner
"""
from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from repooperator_worker.agent_core.classifier import (
    _parse_classifier_json,
    classify_intent,
    validate_classifier_payload,
)
from repooperator_worker.schemas import AgentRunRequest


def _make_request(task: str = "what does this repo do?") -> AgentRunRequest:
    return AgentRunRequest(
        task=task,
        project_path="/tmp/mock",
        git_provider="local",
    )


class TestClassifierModule(unittest.TestCase):

    # ── Shim: no routing constants ────────────────────────────────────────────

    def test_classifier_has_no_supported_intents_constant(self):
        import repooperator_worker.agent_core.classifier as mod
        self.assertFalse(hasattr(mod, "SUPPORTED_INTENTS"), "SUPPORTED_INTENTS must not exist on the shim")

    def test_classifier_has_no_classifier_prompt_constant(self):
        import repooperator_worker.agent_core.classifier as mod
        self.assertFalse(hasattr(mod, "CLASSIFIER_PROMPT"), "CLASSIFIER_PROMPT must not exist on the shim")

    # ── controller_graph boundary ────────────────────────────────────────────

    def test_controller_graph_does_not_export_classifier_shim(self):
        import repooperator_worker.agent_core.controller_graph as controller_graph
        self.assertFalse(hasattr(controller_graph, "classify_intent"))

    # ── classify_intent always returns intent="ambiguous" ────────────────────

    def test_classify_intent_returns_ambiguous_intent(self):
        req = _make_request()
        with patch(
            "repooperator_worker.agent_core.request_understanding.OpenAICompatibleModelClient",
        ) as mock_cls:
            mock_client = MagicMock()
            mock_client.generate_text.return_value = json.dumps({
                "user_goal": "test",
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
            result = classify_intent(req)
        self.assertEqual(result.intent, "ambiguous")

    def test_classify_intent_does_not_populate_routing_fields(self):
        req = _make_request()
        with patch(
            "repooperator_worker.agent_core.request_understanding.OpenAICompatibleModelClient",
        ) as mock_cls:
            mock_client = MagicMock()
            mock_client.generate_text.return_value = "{}"
            mock_cls.return_value = mock_client
            result = classify_intent(req)
        self.assertFalse(hasattr(result, "requested_workflow"))
        self.assertFalse(hasattr(result, "retrieval_goal"))
        self.assertFalse(hasattr(result, "requires_repository_wide_review"))
        self.assertFalse(hasattr(result, "analysis_scope"))

    # ── Malformed JSON falls back to ambiguous ────────────────────────────────

    def test_malformed_json_returns_empty_dict(self):
        result = _parse_classifier_json("not valid json {{{")
        self.assertEqual(result, {})

    def test_malformed_json_in_classify_returns_ambiguous(self):
        req = _make_request()
        with patch(
            "repooperator_worker.agent_core.request_understanding.OpenAICompatibleModelClient",
        ) as mock_cls:
            mock_client = MagicMock()
            mock_client.generate_text.return_value = "INVALID JSON }{{"
            mock_cls.return_value = mock_client
            result = classify_intent(req)
        self.assertEqual(result.intent, "ambiguous")

    # ── validate_classifier_payload backwards compat ─────────────────────────

    def test_validate_with_empty_payload_returns_ambiguous(self):
        req = _make_request()
        result = validate_classifier_payload({}, req)
        self.assertEqual(result.intent, "ambiguous")

    def test_validate_classifier_payload_returns_classifier_result(self):
        from repooperator_worker.agent_core.state import ClassifierResult
        req = _make_request()
        result = validate_classifier_payload({"mentioned_files": ["src/main.py"]}, req)
        self.assertIsInstance(result, ClassifierResult)

    # ── ClassifierResult is JSON-safe ─────────────────────────────────────────

    def test_classifier_result_is_json_safe(self):
        from dataclasses import asdict
        req = _make_request()
        result = validate_classifier_payload({"mentioned_files": ["src/main.py"]}, req)
        serialised = json.dumps(asdict(result), ensure_ascii=False)
        self.assertIsInstance(serialised, str)

    # ── Wrong intent does not block planner ───────────────────────────────────

    def test_wrong_intent_does_not_block_planner(self):
        from repooperator_worker.agent_core.planner import build_task_frame
        from repooperator_worker.agent_core.state import AgentCoreState

        req = _make_request(task="what does main.py do?")
        state = AgentCoreState(
            run_id="test-run",
            thread_id="t1",
            repo="/tmp/mock",
            branch=None,
            user_task=req.task,
        )
        state.classifier_result = validate_classifier_payload({"intent": "write_request", "confidence": 0.5}, req)
        frame = build_task_frame(req, state)
        self.assertIsNotNone(frame)

    # ── _parse_classifier_json strips markdown fences ─────────────────────────

    def test_parse_json_strips_markdown_fences(self):
        raw = '```json\n{"user_goal": "explain"}\n```'
        result = _parse_classifier_json(raw)
        self.assertEqual(result.get("user_goal"), "explain")

    def test_parse_json_non_object_returns_empty(self):
        result = _parse_classifier_json("[1, 2, 3]")
        self.assertEqual(result, {})

    # ── needs_clarification propagates ───────────────────────────────────────

    def test_needs_clarification_propagates_through_adapter(self):
        req = _make_request()
        with patch(
            "repooperator_worker.agent_core.request_understanding.OpenAICompatibleModelClient",
        ) as mock_cls:
            mock_client = MagicMock()
            mock_client.generate_text.return_value = json.dumps({
                "user_goal": "unclear",
                "mentioned_files": [],
                "mentioned_symbols": [],
                "constraints": [],
                "requested_outputs": [],
                "likely_needed_tools": [],
                "safety_notes": [],
                "uncertainties": ["scope is unclear"],
                "needs_clarification": True,
                "clarification_question": "Which files should I focus on?",
            })
            mock_cls.return_value = mock_client
            result = classify_intent(req)
        self.assertTrue(result.needs_clarification)
        self.assertEqual(result.clarification_question, "Which files should I focus on?")


if __name__ == "__main__":
    unittest.main()
