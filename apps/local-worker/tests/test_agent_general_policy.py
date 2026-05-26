import json
import inspect
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

from repooperator_worker.agent_core.action_executor import ActionExecutor  # noqa: E402
from repooperator_worker.agent_core.actions import AgentAction, ActionResult  # noqa: E402
from repooperator_worker.agent_core.final_synthesis import validate_or_repair_final_answer  # noqa: E402
from repooperator_worker.agent_core.request_understanding import RequestUnderstanding  # noqa: E402
from repooperator_worker.agent_core.state import AgentCoreState  # noqa: E402
from repooperator_worker.agent_core.task_policy import (  # noqa: E402
    ensure_subtasks,
    next_recovery_action,
    update_subtasks_after_action,
)
from repooperator_worker.agent_core.langgraph_runtime import run_langgraph_controller  # noqa: E402
from repooperator_worker.agent_core.planner import build_task_frame  # noqa: E402
from repooperator_worker.agent_core.planner import likely_edit_file_queries  # noqa: E402
from repooperator_worker.agent_core.edit_target_selection import (  # noqa: E402
    language_aware_edit_discovery,
    select_edit_target_candidates,
)
from repooperator_worker.schemas import AgentRunRequest, ConversationMessage  # noqa: E402
from repooperator_worker.services.event_service import list_run_events  # noqa: E402


class _QuietClient:
    @property
    def model_name(self) -> str:
        return "test-model"

    def generate_text(self, request):
        if "bounded next-action planner" in request.system_prompt:
            return "{}"
        return "Grounded answer from gathered evidence."

    def stream_text(self, request):
        if "bounded next-action planner" in request.system_prompt:
            return iter(())
        yield {"type": "assistant_delta", "delta": "Grounded answer from gathered evidence."}


class GeneralAgentPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.home = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name) / "repo"
        self.repo.mkdir()
        (self.repo / "README.md").write_text("# Generic App\n\nA small service for messages.\n", encoding="utf-8")
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

    def _request(self, task: str) -> AgentRunRequest:
        return AgentRunRequest(
            project_path=str(self.repo),
            git_provider="local",
            branch="main",
            thread_id="thread-general-policy",
            task=task,
            conversation_history=[],
        )

    def _proposal_for_any_text_file(self, relative_path: str, content: str, task: str, context: dict) -> dict:
        suffix = Path(relative_path).suffix
        marker = "# proposal marker\n" if suffix == ".py" else "// proposal marker\n"
        return {
            "file": relative_path,
            "summary": "Prepare a proposal-only implementation update.",
            "proposed_content": content.rstrip() + "\n" + marker,
            "risk_notes": [],
            "preserves_existing_behavior": True,
        }

    def _edit_understanding(self, request: AgentRunRequest) -> RequestUnderstanding:
        return RequestUnderstanding(
            user_goal=request.task,
            requested_outputs=["code_change_proposal"],
            likely_needed_tools=["search_files", "read_file", "generate_edit"],
            safety_notes=["No disk writes before approval."],
        )

    def test_generic_python_single_entry_promotes_read_implementation_to_proposal(self) -> None:
        (self.repo / "requirements.txt").write_text("fastapi\n", encoding="utf-8")
        original = (
            "class MessageStore:\n"
            "    def create_message(self, body: str) -> dict:\n"
            "        return {\"body\": body}\n\n"
            "def render_message(message: dict) -> str:\n"
            "    return message[\"body\"]\n"
        )
        (self.repo / "main.py").write_text(original, encoding="utf-8")
        request = self._request("Add sender name support to messages.")
        with patch("repooperator_worker.agent_core.request_understanding.understand_request", return_value=self._edit_understanding(request)), patch(
            "repooperator_worker.agent_core.graph.support.OpenAICompatibleModelClient", return_value=_QuietClient()
        ), patch("repooperator_worker.agent_core.tools.builtin.model_generate_edit_proposal", side_effect=self._proposal_for_any_text_file), patch(
            "repooperator_worker.agent_core.graph.repository_support.get_active_repository", return_value=None
        ):
            result = run_langgraph_controller(request, run_id="generic-python-single-entry")

        self.assertEqual(result.agent_flow, "langgraph")
        self.assertIn("main.py", result.files_read)
        self.assertIsNotNone(result.change_set_proposal)
        proposal_files = [change["path"] for change in result.change_set_proposal["changes"]]
        self.assertEqual(proposal_files, ["main.py"])
        self.assertNotEqual(result.stop_reason, "needs_clarification")
        self.assertNotIn("could not identify", result.response.lower())
        self.assertIn("ChangeSetProposal", result.response)
        self.assertIn("No files were modified", result.response)
        self.assertEqual((self.repo / "main.py").read_text(encoding="utf-8"), original)

    def test_generic_multiturn_continuation_consumes_structured_prior_targets(self) -> None:
        (self.repo / "src").mkdir()
        original = (
            "class MessageService:\n"
            "    def create_message(self, body: str) -> dict:\n"
            "        return {\"body\": body}\n"
        )
        (self.repo / "src" / "domain.py").write_text(original, encoding="utf-8")
        continuation_utterance = "Please turn that implementation outline into a patch proposal."
        request = self._request(continuation_utterance)
        request.conversation_history = [
            ConversationMessage(
                role="assistant",
                content="Implementation outline: update src/domain.py around MessageService.create_message.",
                metadata={
                    "edit_target_candidates": [
                        {
                            "path": "src/domain.py",
                            "score": 94,
                            "role": "app/service modules",
                            "sources": ["assistant_visible_plan"],
                            "symbols": ["MessageService", "create_message"],
                        }
                    ],
                    "implementation_plan": {"summary": "Update message creation.", "target_files": ["src/domain.py"]},
                },
            )
        ]
        with patch("repooperator_worker.agent_core.request_understanding.understand_request", return_value=self._edit_understanding(request)), patch(
            "repooperator_worker.agent_core.graph.support.OpenAICompatibleModelClient", return_value=_QuietClient()
        ), patch("repooperator_worker.agent_core.tools.builtin.model_generate_edit_proposal", side_effect=self._proposal_for_any_text_file), patch(
            "repooperator_worker.agent_core.graph.repository_support.get_active_repository", return_value=None
        ):
            result = run_langgraph_controller(request, run_id="generic-multiturn-carryover")

        self.assertIn("src/domain.py", result.files_read)
        proposal_files = [change["path"] for change in result.change_set_proposal["changes"]]
        self.assertEqual(proposal_files, ["src/domain.py"])
        self.assertNotEqual(result.stop_reason, "needs_clarification")
        actions = [event["action"]["type"] for event in list_run_events("generic-multiturn-carryover") if event.get("type") == "action_result"]
        self.assertLessEqual(actions.count("search_files"), 1)
        target_selection = (result.recommendation_context or {}).get("target_selection") or {}
        self.assertTrue(target_selection.get("prior_evidence_reused"))
        self.assertEqual((self.repo / "src" / "domain.py").read_text(encoding="utf-8"), original)
        import repooperator_worker.agent_core.edit_target_selection as selector_module
        import repooperator_worker.agent_core.request_understanding as understanding_module

        source = inspect.getsource(selector_module) + inspect.getsource(understanding_module)
        self.assertNotIn(continuation_utterance, source)

    def test_continuation_reaches_proposal_without_patched_request_understanding(self) -> None:
        (self.repo / "src").mkdir()
        original = (
            "class MessageService:\n"
            "    def create_message(self, body: str) -> dict:\n"
            "        return {\"body\": body}\n"
        )
        (self.repo / "src" / "domain.py").write_text(original, encoding="utf-8")
        continuation_utterance = "Please turn the repository change outline into a reviewable patch."
        request = self._request(continuation_utterance)
        request.conversation_history = [
            ConversationMessage(
                role="assistant",
                content="Repository change outline: update src/domain.py around MessageService.create_message.",
                metadata={
                    "edit_target_candidates": [
                        {
                            "path": "src/domain.py",
                            "score": 96,
                            "role": "app/service modules",
                            "sources": ["assistant_visible_plan"],
                            "symbols": ["MessageService", "create_message"],
                        }
                    ],
                    "implementation_plan": {
                        "summary": "Update message creation.",
                        "target_files": ["src/domain.py"],
                        "operations": ["modify MessageService.create_message"],
                    },
                    "user_understanding_context": {
                        "normalized_goal": "Prepare a repository change proposal.",
                        "requested_outputs": ["code_change_proposal"],
                        "mentioned_symbols": ["MessageService", "create_message"],
                    },
                },
            )
        ]
        with patch("repooperator_worker.agent_core.request_understanding.OpenAICompatibleModelClient", side_effect=RuntimeError("offline understanding")), patch(
            "repooperator_worker.agent_core.graph.support.OpenAICompatibleModelClient", return_value=_QuietClient()
        ), patch("repooperator_worker.agent_core.tools.builtin.model_generate_edit_proposal", side_effect=self._proposal_for_any_text_file), patch(
            "repooperator_worker.agent_core.graph.repository_support.get_active_repository", return_value=None
        ):
            result = run_langgraph_controller(request, run_id="generic-continuation-real-understanding")

        self.assertIsNotNone(result.change_set_proposal)
        proposal_files = [change["path"] for change in result.change_set_proposal["changes"]]
        self.assertEqual(proposal_files, ["src/domain.py"])
        self.assertNotEqual(result.stop_reason, "needs_clarification")
        actions = [event["action"]["type"] for event in list_run_events("generic-continuation-real-understanding") if event.get("type") == "action_result"]
        self.assertNotIn("ask_clarification", actions)
        self.assertLessEqual(actions.count("search_files"), 1)
        self.assertEqual((self.repo / "src" / "domain.py").read_text(encoding="utf-8"), original)

        import repooperator_worker.agent_core.edit_target_selection as selector_module
        import repooperator_worker.agent_core.request_understanding as understanding_module

        source = inspect.getsource(selector_module) + inspect.getsource(understanding_module)
        self.assertNotIn(continuation_utterance, source)

    def test_prior_target_does_not_override_explicit_current_file(self) -> None:
        (self.repo / "src").mkdir()
        (self.repo / "src" / "domain.py").write_text("class DomainService:\n    pass\n", encoding="utf-8")
        (self.repo / "src" / "api.py").write_text("class ApiService:\n    pass\n", encoding="utf-8")
        request = self._request("Update src/api.py to expose the new endpoint.")
        request.conversation_history = [
            ConversationMessage(
                role="assistant",
                content="Prior plan targeted src/domain.py.",
                metadata={
                    "edit_target_candidates": [
                        {"path": "src/domain.py", "score": 98, "role": "app/service modules", "symbols": ["DomainService"]}
                    ],
                    "implementation_plan": {"summary": "Update domain layer.", "target_files": ["src/domain.py"]},
                },
            )
        ]
        state = AgentCoreState(run_id="prior-conflict", thread_id="t", repo=str(self.repo), branch="main", user_task=request.task)
        state.request_understanding = RequestUnderstanding(
            user_goal=request.task,
            mentioned_files=["src/api.py"],
            requested_outputs=["code_change_proposal"],
            likely_needed_tools=["read_file", "generate_edit"],
        )
        state.context_packet = {
            "thread_context": {
                "last_target_candidates": [{"path": "src/domain.py", "score": 98, "role": "app/service modules", "symbols": ["DomainService"]}],
                "last_implementation_plan": {"summary": "Update domain layer.", "target_files": ["src/domain.py"]},
            },
            "prior_target_candidates": [{"path": "src/domain.py", "score": 98, "role": "app/service modules", "symbols": ["DomainService"]}],
        }
        state.files_read = ["src/domain.py", "src/api.py"]
        state.action_results.extend(
            [
                ActionResult(action_id="read-domain", status="success", files_read=["src/domain.py"], payload={"contents": {"src/domain.py": (self.repo / "src" / "domain.py").read_text(encoding="utf-8")}}),
                ActionResult(action_id="read-api", status="success", files_read=["src/api.py"], payload={"contents": {"src/api.py": (self.repo / "src" / "api.py").read_text(encoding="utf-8")}}),
            ]
        )
        frame = build_task_frame(request, state)
        selection = select_edit_target_candidates(state, frame, request)
        self.assertEqual(selection.selected_target_files, ["src/api.py"])
        self.assertNotEqual(selection.selected_target_files, ["src/domain.py"])

    def test_compatible_continuation_reuses_prior_candidate(self) -> None:
        (self.repo / "src").mkdir()
        (self.repo / "src" / "domain.py").write_text(
            "class MessageService:\n"
            "    def create_message(self, body: str):\n"
            "        return {\"body\": body}\n",
            encoding="utf-8",
        )
        request = self._request("Turn the message creation outline into a patch proposal.")
        request.conversation_history = [
            ConversationMessage(
                role="assistant",
                content="Message creation outline: update src/domain.py around MessageService.create_message.",
                metadata={
                    "edit_target_candidates": [
                        {"path": "src/domain.py", "score": 96, "role": "app/service modules", "symbols": ["MessageService", "create_message"]}
                    ],
                    "implementation_plan": {"summary": "Update message creation.", "target_files": ["src/domain.py"]},
                },
            )
        ]
        state = AgentCoreState(run_id="prior-compatible", thread_id="t", repo=str(self.repo), branch="main", user_task=request.task)
        state.request_understanding = RequestUnderstanding(
            user_goal=request.task,
            requested_outputs=["code_change_proposal"],
            likely_needed_tools=["read_file", "generate_edit"],
        )
        state.context_packet = {
            "thread_context": {
                "last_target_candidates": [{"path": "src/domain.py", "score": 96, "role": "app/service modules", "symbols": ["MessageService", "create_message"]}],
                "last_implementation_plan": {"summary": "Update message creation.", "target_files": ["src/domain.py"]},
            },
            "prior_target_candidates": [{"path": "src/domain.py", "score": 96, "role": "app/service modules", "symbols": ["MessageService", "create_message"]}],
        }
        state.files_read = ["src/domain.py"]
        state.action_results.append(
            ActionResult(action_id="read-domain", status="success", files_read=["src/domain.py"], payload={"contents": {"src/domain.py": (self.repo / "src" / "domain.py").read_text(encoding="utf-8")}})
        )
        frame = build_task_frame(request, state)
        selection = select_edit_target_candidates(state, frame, request)
        self.assertEqual(selection.selected_target_files, ["src/domain.py"])
        self.assertTrue(selection.prior_evidence_reused)

    def test_multi_file_python_selects_stronger_semantic_target_not_main(self) -> None:
        (self.repo / "requirements.txt").write_text("click\n", encoding="utf-8")
        (self.repo / "main.py").write_text("def main():\n    return None\n", encoding="utf-8")
        original = (
            "class NotificationService:\n"
            "    def send_notification(self, payload: dict) -> dict:\n"
            "        return {\"status\": \"sent\", \"payload\": payload}\n"
        )
        (self.repo / "app.py").write_text(original, encoding="utf-8")
        package = self.repo / "package"
        package.mkdir()
        (package / "helpers.py").write_text("def helper():\n    return 'ok'\n", encoding="utf-8")
        request = self._request("Change notification sending to include retry metadata.")
        with patch("repooperator_worker.agent_core.request_understanding.understand_request", return_value=self._edit_understanding(request)), patch(
            "repooperator_worker.agent_core.graph.support.OpenAICompatibleModelClient", return_value=_QuietClient()
        ), patch("repooperator_worker.agent_core.tools.builtin.model_generate_edit_proposal", side_effect=self._proposal_for_any_text_file), patch(
            "repooperator_worker.agent_core.graph.repository_support.get_active_repository", return_value=None
        ):
            result = run_langgraph_controller(request, run_id="generic-python-multifile")

        proposal_files = [change["path"] for change in result.change_set_proposal["changes"]]
        self.assertEqual(proposal_files, ["app.py"])
        self.assertNotIn("main.py", proposal_files)
        self.assertEqual((self.repo / "app.py").read_text(encoding="utf-8"), original)

    def test_different_language_fallback_uses_detected_typescript_project(self) -> None:
        (self.repo / "package.json").write_text('{"dependencies":{"typescript":"latest"}}\n', encoding="utf-8")
        (self.repo / "src").mkdir()
        original = "export function createMessage(body: string) {\n  return { body };\n}\n"
        (self.repo / "src" / "messages.ts").write_text(original, encoding="utf-8")
        request = self._request("Add sender name support to message creation.")
        with patch("repooperator_worker.agent_core.request_understanding.understand_request", return_value=self._edit_understanding(request)), patch(
            "repooperator_worker.agent_core.graph.support.OpenAICompatibleModelClient", return_value=_QuietClient()
        ), patch("repooperator_worker.agent_core.tools.builtin.model_generate_edit_proposal", side_effect=self._proposal_for_any_text_file), patch(
            "repooperator_worker.agent_core.graph.repository_support.get_active_repository", return_value=None
        ):
            result = run_langgraph_controller(request, run_id="generic-typescript-fallback")

        self.assertIn("src/messages.ts", result.files_read)
        proposal_files = [change["path"] for change in result.change_set_proposal["changes"]]
        self.assertEqual(proposal_files, ["src/messages.ts"])
        search_events = [event for event in list_run_events("generic-typescript-fallback") if event.get("type") == "action_result" and event["action"]["type"] == "search_files"]
        search_payload = json.dumps([event["result"]["payload"] for event in search_events], ensure_ascii=False)
        self.assertIn("*.ts", search_payload)
        self.assertNotIn("*.py", search_payload)

    def test_wrong_language_fallback_for_python_is_bounded_and_python_first(self) -> None:
        (self.repo / "requirements.txt").write_text("pytest\n", encoding="utf-8")
        (self.repo / "main.py").write_text("def process_message(body: str):\n    return {\"body\": body}\n", encoding="utf-8")
        request = self._request("Add sender name support.")
        frame = build_task_frame(request, AgentCoreState(run_id="lang-detect", thread_id="t", repo=str(self.repo), branch="main", user_task=request.task))
        discovery = language_aware_edit_discovery(request, frame)
        encoded = json.dumps(discovery)
        self.assertIn("*.py", encoded)
        self.assertIn("main.py", encoded)
        self.assertNotIn("*.ts", encoded)
        self.assertNotIn("*.go", encoded)

    def test_likely_edit_queries_use_request_repo_not_process_cwd(self) -> None:
        cwd_repo = Path(self.tmp.name) / "cwd-python"
        cwd_repo.mkdir()
        (cwd_repo / "main.py").write_text("def main():\n    return None\n", encoding="utf-8")
        (self.repo / "package.json").write_text('{"dependencies":{"typescript":"latest"}}\n', encoding="utf-8")
        (self.repo / "src").mkdir()
        (self.repo / "src" / "messages.ts").write_text("export function createMessage() { return true; }\n", encoding="utf-8")
        previous_cwd = os.getcwd()
        try:
            os.chdir(cwd_repo)
            request = self._request("Prepare a message update patch.")
            state = AgentCoreState(run_id="query-repo", thread_id="t", repo=str(self.repo), branch="main", user_task=request.task)
            state.request_understanding = self._edit_understanding(request)
            frame = build_task_frame(request, state)
            queries = likely_edit_file_queries(frame, request)
        finally:
            os.chdir(previous_cwd)
        encoded = json.dumps(queries)
        self.assertIn("*.ts", encoded)
        self.assertNotIn("main.py", queries)
        self.assertEqual(likely_edit_file_queries(frame), ["*.py", "*.js", "*.ts", "*.tsx", "*.jsx", "*.cs", "*.go", "*.rs", "*.java", "*.kt", "*.swift", "*.rb", "*.php"])

    def test_already_read_evidence_outranks_later_zero_result_search(self) -> None:
        (self.repo / "main.py").write_text(
            "class MessageService:\n"
            "    def create_message(self, body: str):\n"
            "        return {\"body\": body}\n",
            encoding="utf-8",
        )
        request = self._request("Add sender name support to message creation.")
        state = AgentCoreState(run_id="read-outranks-zero", thread_id="t", repo=str(self.repo), branch="main", user_task=request.task)
        state.files_read = ["main.py"]
        state.action_results.append(
            ActionResult(action_id="read", status="success", files_read=["main.py"], payload={"contents": {"main.py": (self.repo / "main.py").read_text(encoding="utf-8")}})
        )
        state.actions_taken.append(AgentAction(type="search_files", reason_summary="empty", payload={"queries": ["missing"], "source": "edit_discovery"}))
        state.action_results.append(ActionResult(action_id="search", status="success", payload={"candidates": []}))
        state.zero_result_queries.append("missing")
        state.request_understanding = self._edit_understanding(request)
        frame = build_task_frame(request, state)
        selection = select_edit_target_candidates(state, frame, request)
        self.assertEqual(selection.selected_target_files, ["main.py"])
        self.assertTrue(selection.strong_read_targets)

    def test_target_selection_static_guards_against_fixture_cherrypicks(self) -> None:
        import repooperator_worker.agent_core.edit_target_selection as selector_module
        import repooperator_worker.agent_core.task_policy as policy_module

        source = inspect.getsource(selector_module) + inspect.getsource(policy_module)
        for forbidden in (
            "Please turn that implementation outline into a patch proposal.",
            "Please turn the repository change outline into a reviewable patch.",
            "sender name support",
            "generic-python-single-entry",
            "EldersNiceShot",
        ):
            self.assertNotIn(forbidden, source)

    def test_generic_feature_request_gathers_evidence_before_clarification(self) -> None:
        (self.repo / "src").mkdir()
        (self.repo / "src" / "index.ts").write_text("import { createMessage } from './messages';\ncreateMessage('hello');\n", encoding="utf-8")
        (self.repo / "src" / "messages.ts").write_text(
            "export function createMessage(body: string) {\n  return { body };\n}\n",
            encoding="utf-8",
        )
        request = self._request("Add named message support.")
        understanding = RequestUnderstanding(
            user_goal=request.task,
            requested_outputs=["code_change_proposal"],
            likely_needed_tools=["search_files", "read_file", "generate_edit"],
        )

        def proposal(relative_path, content, task, context):
            return {
                "file": relative_path,
                "summary": "Add sender name to message creation.",
                "proposed_content": content.replace("return { body };", "return { name: 'anonymous', body };"),
                "risk_notes": [],
                "preserves_existing_behavior": True,
            }

        with patch("repooperator_worker.agent_core.request_understanding.understand_request", return_value=understanding), patch(
            "repooperator_worker.agent_core.graph.support.OpenAICompatibleModelClient", return_value=_QuietClient()
        ), patch("repooperator_worker.agent_core.tools.builtin.model_generate_edit_proposal", side_effect=proposal), patch(
            "repooperator_worker.agent_core.graph.repository_support.get_active_repository", return_value=None
        ):
            result = run_langgraph_controller(request, run_id="generic-feature-evidence")

        actions = [event["action"]["type"] for event in list_run_events("generic-feature-evidence") if event.get("type") == "action_result"]
        self.assertGreaterEqual(actions.index("inspect_repo_tree"), 0)
        self.assertIn("search_files", actions)
        self.assertIn("read_file", actions)
        self.assertNotEqual(actions[:1], ["ask_clarification"])
        self.assertIn("proposed patch only", result.response)
        self.assertIn("No files were modified", result.response)

    def test_generic_followup_reads_prior_source_before_clarification(self) -> None:
        (self.repo / "src").mkdir()
        (self.repo / "src" / "domain.ts").write_text("export function score(value: number) { return value * 2; }\n", encoding="utf-8")
        request = self._request("What does that domain layer do?")
        request.conversation_history = [
            ConversationMessage(role="assistant", content="I read src/domain.ts.", metadata={"files_read": ["src/domain.ts"]})
        ]
        understanding = RequestUnderstanding(
            user_goal=request.task,
            requested_outputs=["explanation"],
            likely_needed_tools=["read_file"],
        )
        with patch("repooperator_worker.agent_core.request_understanding.understand_request", return_value=understanding), patch(
            "repooperator_worker.agent_core.graph.support.OpenAICompatibleModelClient", return_value=_QuietClient()
        ), patch("repooperator_worker.agent_core.graph.repository_support.get_active_repository", return_value=None):
            result = run_langgraph_controller(request, run_id="generic-followup-evidence")

        self.assertIn("src/domain.ts", result.files_read)
        actions = [event["action"]["type"] for event in list_run_events("generic-followup-evidence") if event.get("type") == "action_result"]
        self.assertNotIn("ask_clarification", actions[:1])

    def test_generic_broad_analysis_runs_inventory_and_bounded_batch(self) -> None:
        (self.repo / "package.json").write_text('{"scripts":{"test":"node test.js"}}\n', encoding="utf-8")
        (self.repo / "src").mkdir()
        (self.repo / "src" / "main.ts").write_text("import { serve } from './service';\nserve();\n", encoding="utf-8")
        (self.repo / "src" / "service.ts").write_text("export function serve() { return true; }\n", encoding="utf-8")
        (self.repo / "tests").mkdir()
        (self.repo / "tests" / "service.test.ts").write_text("import { serve } from '../src/service';\nserve();\n", encoding="utf-8")
        request = self._request("Analyze all source files and explain the directory structure.")
        understanding = RequestUnderstanding(
            user_goal=request.task,
            requested_outputs=["all_source_analysis", "directory_structure"],
            likely_needed_tools=["search_files", "read_file"],
        )
        with patch("repooperator_worker.agent_core.request_understanding.understand_request", return_value=understanding), patch(
            "repooperator_worker.agent_core.graph.support.OpenAICompatibleModelClient", return_value=_QuietClient()
        ), patch("repooperator_worker.agent_core.graph.repository_support.get_active_repository", return_value=None):
            result = run_langgraph_controller(request, run_id="generic-broad-batch")

        actions = [event["action"]["type"] for event in list_run_events("generic-broad-batch") if event.get("type") == "action_result"]
        self.assertIn("inspect_repo_tree", actions)
        self.assertIn("search_files", actions)
        self.assertIn("read_file", actions)
        self.assertIn("Analyzed batch", result.response)
        self.assertIn("File role table", result.response)
        self.assertIn("Remaining groups/files", result.response)
        self.assertNotIn("max_loop_iterations", result.response)

    def test_clarification_happens_after_missing_file_search_is_exhausted(self) -> None:
        request = self._request("Explain MissingWidget.ts")
        understanding = RequestUnderstanding(user_goal=request.task, mentioned_files=["MissingWidget.ts"], likely_needed_tools=["read_file"])
        with patch("repooperator_worker.agent_core.request_understanding.understand_request", return_value=understanding), patch(
            "repooperator_worker.agent_core.graph.repository_support.get_active_repository", return_value=None
        ):
            result = run_langgraph_controller(request, run_id="generic-missing-file")
        actions = [event["action"]["type"] for event in list_run_events("generic-missing-file") if event.get("type") == "action_result"]
        self.assertEqual(actions[:1], ["search_files"])
        self.assertEqual(result.stop_reason, "needs_clarification")
        self.assertEqual(result.files_read, [])

    def test_subtasks_are_generic_and_update_after_success(self) -> None:
        request = self._request("Add a retry option.")
        state = AgentCoreState(run_id="subtasks", thread_id="t", repo=str(self.repo), branch="main", user_task=request.task)
        state.request_understanding = RequestUnderstanding(
            user_goal=request.task,
            requested_outputs=["code_change_proposal"],
            likely_needed_tools=["search_files", "read_file", "generate_edit"],
        )
        frame = build_task_frame(request, state)
        ensure_subtasks(state, request, frame)
        self.assertEqual([item.title for item in state.subtasks[:3]], [
            "Locate relevant implementation area",
            "Understand current behavior/data/API flow",
            "Prepare proposal-only change",
        ])
        action = AgentAction(type="inspect_repo_tree", reason_summary="inspect")
        update_subtasks_after_action(state, action, ActionResult(action_id=action.action_id, status="success"), "list_files")
        self.assertEqual(state.subtasks[0].status, "completed")
        self.assertIn("list_files", state.subtasks[0].completed_operations)

    def test_generate_change_set_payload_completes_proposal_subtask(self) -> None:
        request = self._request("Add a retry option.")
        state = AgentCoreState(run_id="subtasks-proposal", thread_id="t", repo=str(self.repo), branch="main", user_task=request.task)
        state.request_understanding = self._edit_understanding(request)
        frame = build_task_frame(request, state)
        ensure_subtasks(state, request, frame)
        state.subtasks[0].status = "completed"
        state.subtasks[1].status = "completed"
        state.subtasks[2].status = "running"
        state.current_subtask_id = state.subtasks[2].id
        action = AgentAction(type="generate_change_set", reason_summary="proposal", target_files=["main.py"])
        result = ActionResult(
            action_id=action.action_id,
            status="success",
            payload={"change_set_proposal": {"proposal_id": "p1", "changes": [{"path": "main.py", "operation": "modify"}], "status": "valid"}},
        )
        update_subtasks_after_action(state, action, result, "edit")
        self.assertEqual(state.subtasks[2].status, "completed")
        self.assertEqual(state.current_subtask_id, state.subtasks[3].id)

        state = AgentCoreState(run_id="subtasks-proposal-empty", thread_id="t", repo=str(self.repo), branch="main", user_task=request.task)
        state.request_understanding = self._edit_understanding(request)
        ensure_subtasks(state, request, frame)
        state.subtasks[2].status = "running"
        state.current_subtask_id = state.subtasks[2].id
        empty_result = ActionResult(action_id=action.action_id, status="success", payload={})
        update_subtasks_after_action(state, action, empty_result, "edit")
        self.assertEqual(state.subtasks[2].status, "running")

    def test_failed_read_recovers_with_basename_search_before_clarification(self) -> None:
        request = self._request("Explain src/widget.ts")
        state = AgentCoreState(run_id="recover-read", thread_id="t", repo=str(self.repo), branch="main", user_task=request.task)
        action = AgentAction(type="read_file", reason_summary="read", target_files=["src/widget.ts"])
        result = ActionResult(action_id=action.action_id, status="failed", observation="missing")
        recovery = next_recovery_action(state, request, build_task_frame(request, state), action, result)
        self.assertIsNotNone(recovery)
        self.assertEqual(recovery.type, "search_files")
        self.assertIn("widget.ts", recovery.payload["queries"])

    def test_invalid_generated_proposal_is_rejected_with_proposal_error(self) -> None:
        (self.repo / "main.py").write_text("def main():\n    return 1\n", encoding="utf-8")
        request = self._request("Update main.")
        with patch(
            "repooperator_worker.agent_core.tools.builtin.model_generate_edit_proposal",
            return_value={
                "file": "main.py",
                "summary": "Broken",
                "proposed_content": "```python\ndef main(:\n    return 2\n```",
                "risk_notes": [],
            },
        ), patch("repooperator_worker.agent_core.tools.builtin.OpenAICompatibleModelClient", side_effect=RuntimeError("no repair model")):
            result = ActionExecutor(run_id="invalid-proposal", request=request).execute(
                AgentAction(type="generate_edit", reason_summary="edit", target_files=["main.py"])
            )
        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.payload["edit_proposals"], [])
        self.assertIn("proposal_error", result.payload)
        self.assertIn("markdown fences", result.payload["proposal_error"])

    def test_korean_internal_planning_final_answer_is_repaired(self) -> None:
        request = self._request("이 파일이 어떤 역할인지 설명해줘.")
        state = AgentCoreState(run_id="ko-repair", thread_id="t", repo=str(self.repo), branch="main", user_task=request.task)
        state.files_read = ["README.md"]
        state.action_results.append(
            ActionResult(
                action_id="a1",
                status="success",
                files_read=["README.md"],
                payload={"contents": {"README.md": "# Generic App\n\n메시지 서비스를 설명한다.\n"}},
            )
        )
        with patch("repooperator_worker.agent_core.final_synthesis._compat_model_client", side_effect=RuntimeError("offline")):
            repaired = validate_or_repair_final_answer("The user asks what the file does. I need to answer.", state, request)
        self.assertNotIn("The user asks", repaired)
        self.assertIn("근거 파일", repaired)
        json.dumps({"answer": repaired}, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
