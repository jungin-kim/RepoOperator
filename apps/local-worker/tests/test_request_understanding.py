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
from repooperator_worker.agent_core.actions import AgentAction  # noqa: E402
from repooperator_worker.agent_core.graph.state import append_items, initial_graph_state  # noqa: E402
from repooperator_worker.agent_core.understanding_context import (  # noqa: E402
    append_visible_rationale,
    build_evidence_basis,
    build_user_understanding_context,
    debug_context_payload,
    evidence_basis_update,
    redact_context_for_user,
)
from repooperator_worker.agent_core.request_parsing import extract_file_tokens  # noqa: E402
from repooperator_worker.agent_core.planner import TaskFrame, build_task_frame, edit_requested, edit_requested_text  # noqa: E402
from repooperator_worker.schemas import AgentRunRequest  # noqa: E402
from repooperator_worker.agent_core.state import AgentCoreState  # noqa: E402
from repooperator_worker.services.debug_service import get_debug_context_status  # noqa: E402


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


class TestRuntimeBoundary(unittest.TestCase):
    def test_langgraph_runtime_exports_no_old_understanding_entrypoints(self):
        import repooperator_worker.agent_core.langgraph_runtime as runtime

        public = set(getattr(runtime, "__all__", []))
        self.assertFalse(any(name.endswith("_intent") for name in public))
        self.assertFalse(any(name.startswith("validate_") and name.endswith("_payload") for name in public))

    def test_unhelpful_understanding_does_not_block_planning(self):
        request = _req(task="what does main.py do?")
        state = AgentCoreState(
            run_id="test-run",
            thread_id="t1",
            repo="/tmp/mock",
            branch=None,
            user_task=request.task,
        )
        state.request_understanding = RequestUnderstanding(user_goal="irrelevant", likely_needed_tools=[])
        state.classifier_result = request_understanding_to_classifier_result(RequestUnderstanding(user_goal="wrong bucket"), request)

        frame = build_task_frame(request, state)

        self.assertEqual(frame.user_goal, request.task)
        self.assertIn("main.py", frame.mentioned_files)


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


class TestPublicUnderstandingEvidenceContracts(unittest.TestCase):
    def test_initial_state_has_contract_defaults(self):
        state = initial_graph_state(_req("Summarize README.md"), run_id="run-contract-defaults")
        self.assertIsNone(state["user_understanding_context"])
        self.assertIsNone(state["evidence_basis"])
        self.assertEqual(state["visible_rationale_log"], [])
        self.assertEqual(state["evidence_basis_history"], [])
        self.assertEqual(state["understanding_history"], [])
        json.dumps(state, ensure_ascii=False, default=str)

    def test_user_understanding_context_contains_request_shape(self):
        request = _req("세이브 파일 깨졌을 때 복구 가능하게 해줘.")
        state = {
            "request_understanding_snapshot": {
                "user_goal": "세이브 파일 복구 기능을 추가한다.",
                "requested_outputs": ["code_change_proposal"],
                "likely_needed_tools": ["search_text", "read_file", "generate_edit"],
                "constraints": ["No direct file writes before approval."],
            },
            "task_frame_snapshot": {
                "requested_outputs": ["edit_proposal"],
                "likely_capabilities": ["edit_proposal"],
                "uncertainty": ["Need to locate the save-file implementation."],
            },
            "edit_mode": "proposal_only",
        }
        context = build_user_understanding_context(request, state)
        self.assertEqual(context["language"], "ko")
        self.assertIn("change_set_proposal", context["requested_outputs"])
        self.assertIn("proposal_only", context["requested_outputs"])
        self.assertIn("generate_edit", context["likely_needed_tools"])
        self.assertIn("Need to locate", " ".join(context["ambiguities"]))
        self.assertNotIn("requested_workflow", json.dumps(context))
        self.assertNotIn("retrieval_goal", json.dumps(context))

    def test_evidence_basis_summarizes_sources_without_raw_contents(self):
        state = {
            "files_read": ["README.md", "main.py"],
            "evidence_store": {
                "contents": {
                    "README.md": "# Demo\nRAW_FILE_CONTENT",
                    "main.py": "print('hello')\n" * 100,
                },
                "web_evidence": [
                    {
                        "title": "Docs",
                        "url": "https://docs.example.com",
                        "source": "docs.example.com",
                        "text": "RAW_WEB_TEXT",
                    }
                ],
            },
            "context_pack_report": {
                "retained_files": ["README.md"],
                "omitted_files": [{"path": "main.py", "reason": "budget"}],
            },
            "short_term_memory": {"files_read_summaries": [{"path": "README.md", "summary": "Project overview."}]},
            "validation_results": [{"kind": "change_set", "status": "invalid", "errors": ["syntax error"]}],
            "change_set_proposal": {
                "proposal_id": "p1",
                "status": "invalid",
                "changes": [{"path": "main.py", "operation": "modify", "original_content": "old", "proposed_content": "new"}],
            },
            "worker_reports": [{"task_id": "w1", "role": "AnalysisAgent", "summary": "Checked app files.", "files_analyzed": ["main.py"]}],
        }
        basis = build_evidence_basis(state, "test")
        self.assertEqual({item["path"] for item in basis["files"]}, {"README.md", "main.py"})
        omitted = [item for item in basis["files"] if item["path"] == "main.py"][0]
        self.assertFalse(omitted["retained"])
        self.assertEqual(basis["web_sources"][0]["untrusted"], True)
        self.assertEqual(basis["validation"][0]["errors"], ["syntax error"])
        self.assertEqual(basis["active_proposal"]["proposal_id"], "p1")
        self.assertEqual(basis["worker_reports"][0]["worker_task_id"], "w1")
        encoded = json.dumps(basis, ensure_ascii=False)
        self.assertNotIn("RAW_FILE_CONTENT", encoded)
        self.assertNotIn("RAW_WEB_TEXT", encoded)
        self.assertNotIn("original_content", encoded)
        self.assertNotIn("proposed_content", encoded)

    def test_visible_rationale_log_is_append_safe_and_public(self):
        state = {}
        action = AgentAction(type="read_file", reason_summary="Read README.", target_files=["README.md"])
        first = append_visible_rationale(
            state,
            node="route_next",
            action=action,
            summary="README.md is explicitly named, so I am reading it before answering.",
            basis_refs=[{"kind": "file", "path": "README.md"}],
            safety_note=None,
            uncertainty=[],
        )
        second = append_visible_rationale(
            state,
            node="final_synthesis",
            action=None,
            summary="I am preparing the final answer from gathered evidence.",
            basis_refs=[],
            safety_note="No raw context dump is included.",
            uncertainty=[],
        )
        combined = append_items(first["visible_rationale_log"], second["visible_rationale_log"])
        self.assertEqual(len(combined), 2)
        encoded = json.dumps(combined)
        self.assertNotIn("<think>", encoded)
        self.assertNotIn("chain of thought", encoded.lower())
        self.assertTrue(combined[0]["visible"])
        self.assertEqual(combined[0]["basis_refs"][0]["path"], "README.md")

    def test_redaction_removes_forbidden_and_raw_keys(self):
        payload = {
            "reasoning": "do not show",
            "private_reasoning": "do not show",
            "chain_of_thought": "do not show",
            "safe_reasoning_summary": "safe public note",
            "contents": {"README.md": "raw content"},
        }
        redacted = redact_context_for_user(payload)
        self.assertNotIn("reasoning", redacted)
        self.assertNotIn("private_reasoning", redacted)
        self.assertNotIn("chain_of_thought", redacted)
        self.assertEqual(redacted["safe_reasoning_summary"], "safe public note")
        self.assertTrue(redacted["contents"]["redacted"])

    def test_debug_context_payload_caps_and_redacts(self):
        state = {
            "user_understanding_context": {"normalized_goal": "Explain repo", "raw": {"contents": "secret"}},
            "evidence_basis": {
                "files": [{"path": f"file_{index}.py", "summary": "x"} for index in range(60)],
                "web_sources": [{"url": "https://example.com", "text": "raw"}],
            },
            "visible_rationale_log": [{"id": str(index), "summary": "safe"} for index in range(60)],
            "context_pack_report": {"budget_usage": {"context_window": 8000, "estimated_total_tokens": 2000}},
            "short_term_memory": {"target_candidate_summaries": [{"path": "main.py", "score": 90}]},
            "target_selection_diagnostics": {"selected_target_files": ["main.py"], "prior_evidence_reused": True},
            "edit_target_candidates": [{"path": "main.py", "score": 90}],
        }
        payload = debug_context_payload(state)
        self.assertEqual(len(payload["evidence_basis"]["files"]), 40)
        self.assertEqual(len(payload["visible_rationale_log"]), 30)
        self.assertEqual(payload["context_pack_report"]["budget_usage"]["context_window"], 8000)
        self.assertEqual(payload["short_term_memory"]["target_candidate_summaries"][0]["path"], "main.py")
        self.assertTrue(payload["target_selection"]["prior_evidence_reused"])
        self.assertEqual(payload["edit_target_candidates"][0]["path"], "main.py")
        self.assertNotIn("secret", json.dumps(payload))

    def test_evidence_basis_update_history_is_json_safe(self):
        update = evidence_basis_update({"files_read": ["README.md"]}, trigger_node="read_files")
        json.dumps(update, ensure_ascii=False)
        self.assertEqual(update["evidence_basis"]["files"][0]["path"], "README.md")
        self.assertEqual(update["evidence_basis_history"][0]["last_updated_by"], "read_files")

    def test_debug_endpoint_payload_includes_new_fields_from_checkpoint(self):
        checkpoint_event = {
            "type": "langgraph_checkpoint",
            "checkpoint": {
                "channel_values": {
                    "user_understanding_context": {"normalized_goal": "Explain README.md"},
                    "evidence_basis": {
                        "files": [{"path": "README.md", "retained": True, "contents": "raw"}],
                        "memory_carryover": {
                            "thread_target_candidates": [{"path": "README.md", "score": 88}],
                            "last_implementation_plan": {"summary": "Update README.", "target_files": ["README.md"]},
                        },
                    },
                    "visible_rationale_log": [{"id": "r1", "summary": "Read README.md first."}],
                    "context_pack_report": {"budget_usage": {"context_window": 16000}},
                    "short_term_memory": {"carryover_summaries": [{"kind": "prior_edit_target_evidence"}]},
                    "target_selection_diagnostics": {"selected_target_files": ["README.md"]},
                    "edit_target_candidates": [{"path": "README.md", "score": 88}],
                }
            },
        }
        model_profile = type("Profile", (), {"model_dump": lambda self: {"model_name": "test"}})()
        with patch("repooperator_worker.services.debug_service.detect_model_profile", return_value=model_profile), patch(
            "repooperator_worker.services.debug_service.get_active_runs", return_value=[]
        ), patch("repooperator_worker.services.debug_service.list_recent_runs", return_value=[{"id": "run-debug"}]), patch(
            "repooperator_worker.services.debug_service.list_run_events", return_value=[checkpoint_event]
        ):
            payload = get_debug_context_status()
        self.assertIn("user_understanding_context", payload)
        self.assertIn("evidence_basis", payload)
        self.assertIn("visible_rationale_log", payload)
        self.assertIn("context_pack_report", payload)
        self.assertIn("short_term_memory", payload)
        self.assertIn("target_selection", payload)
        self.assertIn("edit_target_candidates", payload)
        self.assertEqual(payload["user_understanding_context"]["normalized_goal"], "Explain README.md")
        self.assertEqual(payload["edit_target_candidates"][0]["path"], "README.md")
        self.assertEqual(payload["evidence_basis"]["memory_carryover"]["thread_target_candidates"][0]["path"], "README.md")
        self.assertNotIn('"contents": "raw"', json.dumps(payload["evidence_basis"]))


if __name__ == "__main__":
    unittest.main()
