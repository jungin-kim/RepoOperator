import inspect
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime
from enum import Enum
from pathlib import Path
from unittest.mock import patch


TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repooperator_worker.agent_core.action_executor import ActionExecutor  # noqa: E402
from repooperator_worker.agent_core.actions import AgentAction, ActionResult  # noqa: E402
from repooperator_worker.agent_core.controller_graph import build_final_answer_text, determine_loop_budget, run_controller_graph, stream_controller_graph  # noqa: E402
from repooperator_worker.agent_core.final_synthesis import _answer_with_model, validate_or_repair_final_answer  # noqa: E402
from repooperator_worker.agent_core.planner import _existing_target_files, build_task_frame  # noqa: E402
from repooperator_worker.agent_core.request_understanding import RequestUnderstanding  # noqa: E402
from repooperator_worker.agent_core.state import AgentCoreState, ClassifierResult  # noqa: E402
from repooperator_worker.agent_core.repository_review import review_single_file  # noqa: E402
from repooperator_worker.agent_core.steering import SteeringDecision, consume_steering_for_state, parse_steering_instruction  # noqa: E402
from repooperator_worker.agent_core.tools.builtin import validate_edit_proposal  # noqa: E402
from repooperator_worker.schemas import AgentRunRequest, AgentRunResponse, ConversationMessage  # noqa: E402
from repooperator_worker.services.agent_orchestration_graph import (  # noqa: E402
    run_agent_orchestration_graph,
    stream_agent_orchestration_graph,
)
from repooperator_worker.services.agent_run_coordinator import start_run, stream_run  # noqa: E402
from repooperator_worker.services.agent_service import run_agent_task  # noqa: E402
from repooperator_worker.services.event_service import append_run_event, list_run_events  # noqa: E402
from repooperator_worker.services.json_safe import json_safe, safe_agent_response_payload  # noqa: E402


class _StreamingReviewClient:
    @property
    def model_name(self) -> str:
        return "test-model"

    def stream_text(self, _request):
        yield {"type": "assistant_delta", "delta": "Purpose: checks the fixture. "}
        yield {"type": "assistant_delta", "delta": "Confirmed issues: none."}

    def generate_text(self, _request):
        raise AssertionError("review_single_file should prefer stream_text")


class _LoopClient:
    @property
    def model_name(self) -> str:
        return "test-model"

    def stream_text(self, _request):
        yield {"type": "assistant_delta", "delta": "README.md evidence reached the final answer."}

    def generate_text(self, _request):
        return "README.md evidence reached the final answer."


class _PlannerClient:
    def __init__(self, *responses: dict, answer: str = "Grounded final answer."):
        self.responses = list(responses)
        self.answer = answer

    @property
    def model_name(self) -> str:
        return "test-model"

    def generate_text(self, request):
        if "bounded next-action planner" in request.system_prompt and self.responses:
            return json.dumps(self.responses.pop(0), ensure_ascii=False)
        return self.answer

    def stream_text(self, _request):
        yield {"type": "assistant_delta", "delta": self.answer}


class _SynthesisClient:
    def __init__(self, answer: str):
        self.answer = answer

    @property
    def model_name(self) -> str:
        return "test-model"

    def generate_text(self, request):
        if "bounded next-action planner" in request.system_prompt:
            return "{}"
        return self.answer

    def stream_text(self, request):
        if "bounded next-action planner" in request.system_prompt:
            return iter(())
        yield {"type": "assistant_delta", "delta": self.answer}


class _EditProposalClient:
    @property
    def model_name(self) -> str:
        return "test-model"

    def generate_text(self, request):
        if "edit proposal generator" not in request.system_prompt:
            return "{}"
        payload = json.loads(request.user_prompt)
        content = payload["content"]
        proposed = content.replace("&", "&&")
        return json.dumps(
            {
                "file": payload["file"],
                "summary": "Use short-circuit boolean checks.",
                "proposed_content": proposed,
                "risk_notes": [],
                "preserves_existing_behavior": True,
            }
        )


class _JsonSafeEnum(Enum):
    SAMPLE = "sample"


def _edit_understanding(
    request: AgentRunRequest,
    *,
    files: list[str] | None = None,
    outputs: list[str] | None = None,
    tools: list[str] | None = None,
) -> RequestUnderstanding:
    return RequestUnderstanding(
        user_goal=request.task,
        mentioned_files=files or [],
        requested_outputs=outputs or ["code_change_proposal"],
        likely_needed_tools=tools or ["search_files", "read_file", "generate_edit"],
        safety_notes=["Do not write files unless explicitly approved."],
        uncertainties=[] if files else ["Need to locate the implementation file first."],
    )


class ActivePathMigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.home = tempfile.TemporaryDirectory()
        self.repo_base = Path(self.tmp.name) / "repos"
        self.repo = self.repo_base / "jungin-kim" / "EldersNiceShot"
        self.repo.mkdir(parents=True)
        (self.repo / "README.md").write_text("# Fixture\n", encoding="utf-8")
        (self.repo / "app.py").write_text("def main():\n    return 1\n", encoding="utf-8")
        self.config = Path(self.tmp.name) / "config.json"
        self.config.write_text(
            json.dumps(
                {
                    "repooperatorHomeDir": self.home.name,
                    "localRepoBaseDir": str(self.repo_base),
                    "openai": {"baseUrl": "http://127.0.0.1:11434/v1", "apiKey": "test", "model": "test-model"},
                }
            ),
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

    def _request(self, project_path: str | None = None) -> AgentRunRequest:
        return AgentRunRequest(
            project_path=project_path or str(self.repo),
            git_provider="local",
            branch="main",
            thread_id="thread-active-path",
            task="Explain README.md",
            conversation_history=[],
        )

    def _response(self, request: AgentRunRequest, run_id: str | None = None) -> AgentRunResponse:
        return AgentRunResponse(
            project_path=request.project_path,
            git_provider=request.git_provider,
            active_repository_source=request.git_provider,
            active_repository_path=request.project_path,
            active_branch=request.branch,
            task=request.task,
            model="test-model",
            branch=request.branch,
            repo_root_name="EldersNiceShot",
            context_summary="",
            top_level_entries=[],
            readme_included=False,
            diff_included=False,
            is_git_repository=True,
            files_read=["README.md"],
            response="README.md describes the fixture.",
            response_type="assistant_answer",
            intent_classification="read_only_question",
            graph_path="agent_core:test",
            agent_flow="agent_core_controller",
            run_id=run_id,
        )

    def _write_unity_fixture(self) -> None:
        scripts = self.repo / "Assets" / "Scripts"
        scripts.mkdir(parents=True, exist_ok=True)
        (self.repo / "README.md").write_text(
            "# Elders Nice Shot\nPlayers ready up, take turns striking balls, and score before the timer ends.\n",
            encoding="utf-8",
        )
        (scripts / "GameManager.cs").write_text(
            "public class GameManager {\n"
            "  public void ReadyPhase() {}\n"
            "  public void StrikePhase() {}\n"
            "  public void ScoreAndEndTurn() {}\n"
            "}\n",
            encoding="utf-8",
        )
        (scripts / "Ball.cs").write_text(
            "public class Ball {\n"
            "  public GameManager manager;\n"
            "  public void OnCollisionEnter() { manager.ScoreAndEndTurn(); }\n"
            "}\n",
            encoding="utf-8",
        )
        (scripts / "Border.cs").write_text(
            "public class Border {\n"
            "  void Update() {}\n"
            "  bool IsOut(bool left, bool right) { return left & right; }\n"
            "}\n",
            encoding="utf-8",
        )
        (scripts / "DataHandler.cs").write_text(
            "using System.IO;\n"
            "using System.Runtime.Serialization.Formatters.Binary;\n"
            "using UnityEngine;\n"
            "public class DataHandler : MonoBehaviour {\n"
            "  public void Save(PlayerData data) { var formatter = new BinaryFormatter(); }\n"
            "}\n",
            encoding="utf-8",
        )

    def _write_satellite_fixture(self) -> None:
        (self.repo / "README.md").write_text(
            "# Satellite Simulation\n\n"
            "A Python satellite orbit simulation project for modelling satellites, orbital motion, and propagation algorithms.\n\n"
            "Run `python main.py` to configure satellites and simulate orbital movement.\n",
            encoding="utf-8",
        )
        (self.repo / "main.py").write_text(
            "from satellite import Satellite\n"
            "from orbit import Orbit\n"
            "from algorithms import propagate_orbit\n\n"
            "def main():\n"
            "    orbit = Orbit(altitude=500)\n"
            "    satellite = Satellite('demo', orbit)\n"
            "    propagate_orbit(satellite)\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n",
            encoding="utf-8",
        )
        (self.repo / "algorithms.py").write_text(
            "def propagate_orbit(satellite):\n"
            "    return satellite.step()\n",
            encoding="utf-8",
        )
        (self.repo / "orbit.py").write_text(
            "class Orbit:\n"
            "    def __init__(self, altitude):\n"
            "        self.altitude = altitude\n",
            encoding="utf-8",
        )
        (self.repo / "satellite.py").write_text(
            "class Satellite:\n"
            "    def __init__(self, name, orbit):\n"
            "        self.name = name\n"
            "        self.orbit = orbit\n"
            "    def step(self):\n"
            "        return self.orbit\n",
            encoding="utf-8",
        )
        (self.repo / "http_cache.sqlite").write_bytes(b"SQLite format 3\x00binary-cache")

    def _classifier(
        self,
        *,
        intent: str = "ambiguous",
        target_files: list[str] | None = None,
    ) -> ClassifierResult:
        return ClassifierResult(
            intent=intent,
            confidence=0.1,
            target_files=target_files or [],
        )

    def test_agent_service_calls_agent_core_controller(self) -> None:
        request = self._request()
        called: list[str] = []

        def fake_controller(req):
            called.append(req.task)
            return self._response(req)

        with patch(
            "repooperator_worker.agent_core.controller_graph.run_controller_graph",
            side_effect=fake_controller,
        ), patch(
            "repooperator_worker.services.agent_orchestration_graph.run_agent_orchestration_graph",
            side_effect=AssertionError("old orchestration graph must not run"),
        ):
            result = run_agent_task(request)

        self.assertEqual(called, [request.task])
        self.assertEqual(result.agent_flow, "agent_core_controller")

    def test_agent_run_coordinator_sync_calls_agent_core_controller(self) -> None:
        request = self._request()
        called: list[str] = []

        def fake_controller(req, *, run_id=None):
            called.append(str(run_id))
            return self._response(req, run_id)

        with patch.dict(os.environ, {"REPOOPERATOR_CONFIG_PATH": str(self.config)}, clear=False), patch(
            "repooperator_worker.agent_core.controller_graph.run_controller_graph",
            side_effect=fake_controller,
        ), patch(
            "repooperator_worker.services.agent_service.run_agent_task",
            side_effect=AssertionError("agent_service must not be on coordinator sync path"),
        ):
            result = start_run(request)

        self.assertEqual(len(called), 1)
        self.assertEqual(result.run_id, called[0])

    def test_agent_run_coordinator_stream_calls_agent_core_stream(self) -> None:
        request = self._request()
        called: list[str] = []

        def fake_stream(req, *, run_id=None):
            called.append(str(run_id))
            yield {"type": "progress_delta", "run_id": run_id, "activity_id": "test", "label": "Working", "status": "completed"}
            yield {"type": "assistant_delta", "delta": "Done."}
            yield {"type": "final_message", "result": self._response(req, run_id).model_dump(mode="json")}

        with patch.dict(os.environ, {"REPOOPERATOR_CONFIG_PATH": str(self.config)}, clear=False), patch(
            "repooperator_worker.agent_core.controller_graph.stream_controller_graph",
            side_effect=fake_stream,
        ), patch(
            "repooperator_worker.services.agent_orchestration_graph.stream_agent_orchestration_graph",
            side_effect=AssertionError("old stream graph must not run"),
        ):
            run_id, stream = stream_run(request)
            chunks: list[str] = []
            deadline = time.time() + 3
            while time.time() < deadline:
                chunk = next(stream)
                chunks.append(chunk)
                if "[DONE]" in chunk:
                    break
            sequences = [event["sequence"] for event in list_run_events(run_id)]

        self.assertEqual(called, [run_id])
        self.assertTrue(any("progress_delta" in chunk for chunk in chunks))
        self.assertEqual(sequences, sorted(set(sequences)))

    def test_agent_orchestration_graph_is_adapter(self) -> None:
        request = self._request()
        with patch(
            "repooperator_worker.services.agent_orchestration_graph.run_controller_graph",
            return_value=self._response(request),
        ) as run_controller:
            result = run_agent_orchestration_graph(request)
        self.assertTrue(run_controller.called)
        self.assertEqual(result.graph_path, "agent_core:test")

        with patch(
            "repooperator_worker.services.agent_orchestration_graph.stream_controller_graph",
            return_value=iter([{"type": "final_message", "result": self._response(request).model_dump(mode="json")}]),
        ) as stream_controller:
            events = list(stream_agent_orchestration_graph(request, run_id="run-adapter"))
        self.assertTrue(stream_controller.called)
        self.assertEqual(events[0]["type"], "final_message")

    def test_old_agent_graph_is_not_imported_by_active_services(self) -> None:
        import repooperator_worker.services.agent_run_coordinator as coordinator
        import repooperator_worker.services.agent_service as service

        combined = inspect.getsource(coordinator) + "\n" + inspect.getsource(service)
        self.assertNotIn("agent_graph", combined)
        self.assertNotIn("run_agent_graph", combined)

    def test_existing_target_files_resolves_provider_style_project_path(self) -> None:
        request = self._request("jungin-kim/EldersNiceShot")
        with patch.dict(
            os.environ,
            {"REPOOPERATOR_CONFIG_PATH": str(self.config), "LOCAL_REPO_BASE_DIR": str(self.repo_base)},
            clear=False,
        ), patch(
            "pathlib.Path.cwd",
            side_effect=AssertionError("current working directory must not be used"),
        ):
            self.assertEqual(_existing_target_files(request, ["README.md", "../outside.py"]), ["README.md"])

    def test_visible_reasoning_is_removed_from_final_answer(self) -> None:
        request = self._request()

        class _Client:
            def stream_text(self, _prompt):
                yield {"type": "assistant_delta", "delta": "<think>private notes</think>\nFinal answer"}

            def generate_text(self, _prompt):
                raise AssertionError("stream result should be used")

        with patch("repooperator_worker.agent_core.controller_graph.OpenAICompatibleModelClient", return_value=_Client()):
            answer = _answer_with_model(request, {"README.md": "# Fixture\n"})
        self.assertEqual(answer, "Final answer")
        self.assertNotIn("<think>", answer)
        self.assertNotIn("private notes", answer)

    def test_repository_review_streams_file_deltas_on_same_activity(self) -> None:
        deltas: list[str] = []
        result = review_single_file(
            request=self._request(),
            relative_path="app.py",
            content="def main():\n    return 1\n",
            truncated=False,
            client=_StreamingReviewClient(),
            on_delta=deltas.append,
        )
        self.assertIn("Confirmed issues", result["summary"])
        self.assertEqual(deltas, ["Purpose: checks the fixture. ", "Confirmed issues: none."])

    def test_repository_review_streaming_honors_cancellation(self) -> None:
        deltas: list[str] = []
        result = review_single_file(
            request=self._request(),
            relative_path="app.py",
            content="def main():\n    return 1\n",
            truncated=False,
            client=_StreamingReviewClient(),
            on_delta=deltas.append,
            should_cancel=lambda: bool(deltas),
        )
        self.assertTrue(result["cancelled"])
        self.assertEqual(deltas, ["Purpose: checks the fixture. "])

    def test_controller_loop_reads_target_file_then_answers(self) -> None:
        request = self._request()
        with patch(
            "repooperator_worker.agent_core.controller_graph.OpenAICompatibleModelClient",
            return_value=_LoopClient(),
        ), patch(
            "repooperator_worker.agent_core.controller_graph.get_active_repository",
            return_value=None,
        ):
            result = run_controller_graph(request, run_id="loop-target-file")
        self.assertGreater(result.loop_iteration, 1)
        self.assertEqual(result.files_read, ["README.md"])
        self.assertEqual(result.graph_path, "agent_core:read_file_answer")
        self.assertIn("README.md evidence", result.response)

    def test_explicit_target_files_are_resolved_and_read(self) -> None:
        self._write_unity_fixture()
        request = self._request()
        request.task = "README.md랑 GameManager.cs만 읽고, 이 게임의 플레이 흐름을 설명해줘."
        with patch(
            "repooperator_worker.agent_core.controller_graph.OpenAICompatibleModelClient",
            return_value=_LoopClient(),
        ), patch(
            "repooperator_worker.agent_core.controller_graph.get_active_repository",
            return_value=None,
        ):
            result = run_controller_graph(request, run_id="explicit-target-files")
        self.assertIn("README.md", result.files_read)
        self.assertIn("Assets/Scripts/GameManager.cs", result.files_read)
        action_events = [event for event in list_run_events("explicit-target-files") if event.get("type") == "action_result"]
        self.assertNotIn("analyze_repository", [event["action"]["type"] for event in action_events])
        trace_events = [event for event in list_run_events("explicit-target-files") if event.get("event_type") == "work_trace"]
        self.assertTrue(any(event.get("display") == "primary" and "Read" in str(event.get("current_action")) for event in trace_events))

    def test_follow_up_file_role_uses_prior_context_and_reads_target(self) -> None:
        self._write_unity_fixture()
        request = self._request()
        request.task = "방금 말한 플레이 흐름 기준으로 Ball.cs는 어떤 역할이야?"
        request.conversation_history = [
            ConversationMessage(
                role="assistant",
                content="Read README.md and GameManager.cs.",
                metadata={"files_read": ["README.md", "Assets/Scripts/GameManager.cs"]},
            )
        ]
        with patch(
            "repooperator_worker.agent_core.controller_graph.OpenAICompatibleModelClient",
            return_value=_LoopClient(),
        ), patch(
            "repooperator_worker.agent_core.controller_graph.get_active_repository",
            return_value=None,
        ):
            result = run_controller_graph(request, run_id="follow-up-ball")
        self.assertIn("Assets/Scripts/Ball.cs", result.files_read)

    def test_edit_target_file_is_resolved_and_reported_as_proposal(self) -> None:
        self._write_unity_fixture()
        request = self._request()
        request.task = "Border.cs에서 &를 &&로 바꾸고 빈 Update 제거해줘. 변경 전후 설명도 해줘."
        def generic_proposal(relative_path, content, task, context):
            return {
                "file": relative_path,
                "summary": "Prepare a small proposal for the requested target file.",
                "proposed_content": content + "\n// proposal marker\n",
                "risk_notes": [],
                "preserves_existing_behavior": True,
            }
        with patch("repooperator_worker.agent_core.request_understanding.understand_request", return_value=_edit_understanding(request, files=["Border.cs"])), patch(
            "repooperator_worker.agent_core.tools.builtin.model_generate_edit_proposal",
            side_effect=generic_proposal,
        ), patch(
            "repooperator_worker.agent_core.controller_graph.get_active_repository",
            return_value=None,
        ):
            result = run_controller_graph(request, run_id="border-edit-proposal")
        self.assertIn("Assets/Scripts/Border.cs", result.files_read)
        self.assertIn("proposed patch only", result.response)
        self.assertIn("No files were modified", result.response)
        self.assertNotIn("where is Border.cs", result.response)

    def test_safe_save_logic_proposal_removes_binary_formatter(self) -> None:
        self._write_unity_fixture()
        request = self._request()
        request.task = "저장 로직을 안전하게 고쳐줘."
        with patch("repooperator_worker.agent_core.request_understanding.understand_request", return_value=_edit_understanding(request)), patch(
            "repooperator_worker.agent_core.controller_graph.get_active_repository",
            return_value=None,
        ):
            result = run_controller_graph(request, run_id="safe-save-proposal")
        self.assertIn("Assets/Scripts/DataHandler.cs", result.files_read)
        self.assertNotIn("max_loop_iterations", result.response)
        self.assertNotIn("I stopped because", result.response)

    def test_task_aware_loop_budget_gives_feature_discovery_more_room(self) -> None:
        summary_request = self._request()
        summary_state = AgentCoreState(
            run_id="budget-summary",
            thread_id=summary_request.thread_id,
            repo=summary_request.project_path,
            branch=summary_request.branch,
            user_task=summary_request.task,
        )
        summary_state.request_understanding = RequestUnderstanding(user_goal="Summarize the project.", likely_needed_tools=["read_file"])
        summary_budget = determine_loop_budget(build_task_frame(summary_request, summary_state), summary_request, {})

        edit_request = self._request()
        edit_request.task = "이 프로젝트에 기명 메세지 기능을 넣고싶어."
        edit_state = AgentCoreState(
            run_id="budget-edit",
            thread_id=edit_request.thread_id,
            repo=edit_request.project_path,
            branch=edit_request.branch,
            user_task=edit_request.task,
        )
        edit_state.request_understanding = _edit_understanding(edit_request)
        edit_budget = determine_loop_budget(build_task_frame(edit_request, edit_state), edit_request, {})

        self.assertLess(summary_budget.max_loop_iterations, edit_budget.max_loop_iterations)
        self.assertGreaterEqual(edit_budget.max_loop_iterations, 10)
        self.assertLessEqual(edit_budget.max_loop_iterations, 18)

    def test_feature_request_reads_readme_and_main_before_clarifying(self) -> None:
        (self.repo / "main.py").write_text("def send_message(name, body):\n    return body\n", encoding="utf-8")
        (self.repo / "requirements.txt").write_text("fastapi\n", encoding="utf-8")
        request = self._request()
        request.task = "이 프로젝트에 기명 메세지 기능을 넣고싶어."
        with patch("repooperator_worker.agent_core.request_understanding.understand_request", return_value=_edit_understanding(request)), patch(
            "repooperator_worker.agent_core.controller_graph.get_active_repository",
            return_value=None,
        ):
            result = run_controller_graph(request, run_id="feature-named-message")
        self.assertIn("README.md", result.files_read)
        self.assertIn("main.py", result.files_read)
        self.assertNotIn("max_loop_iterations", result.response)
        self.assertNotIn("I stopped because", result.response)
        actions = [event["action"]["type"] for event in list_run_events("feature-named-message") if event.get("type") == "action_result"]
        self.assertIn("inspect_repo_tree", actions)
        trace_events = [event for event in list_run_events("feature-named-message") if event.get("event_type") == "work_trace"]
        self.assertTrue(any("README.md" in event.get("files", []) for event in trace_events))
        self.assertTrue(any("main.py" in event.get("files", []) for event in trace_events))

    def test_repeated_zero_result_search_is_not_repeated(self) -> None:
        request = self._request()
        request.task = "Find MissingSymbol usage."
        planner = _PlannerClient(
            {"action_type": "search_text", "reason_summary": "Search missing symbol.", "query": "MissingSymbol", "confidence": 0.9},
            {"action_type": "search_text", "reason_summary": "Search missing symbol again.", "query": " missingsymbol ", "confidence": 0.9},
        )
        with patch(
            "repooperator_worker.agent_core.controller_graph.OpenAICompatibleModelClient",
            return_value=planner,
        ), patch(
            "repooperator_worker.agent_core.controller_graph.get_active_repository",
            return_value=None,
        ):
            run_controller_graph(request, run_id="zero-search-repeat")
        search_text_queries = [
            (event["action"].get("payload") or {}).get("query", "").strip().lower()
            for event in list_run_events("zero-search-repeat")
            if event.get("type") == "action_result" and event["action"]["type"] == "search_text"
        ]
        self.assertEqual(len(search_text_queries), len(set(search_text_queries)))

    def test_limit_fallback_does_not_expose_raw_stop_reasons(self) -> None:
        request = self._request()
        for reason in ("max_loop_iterations", "max_file_reads", "max_commands", "timed_out"):
            state = AgentCoreState(
                run_id=f"fallback-{reason}",
                thread_id=request.thread_id,
                repo=request.project_path,
                branch=request.branch,
                user_task=request.task,
                stop_reason=reason,
            )
            answer = build_final_answer_text(state, request)
            self.assertNotIn(reason, answer)
            self.assertNotIn("I stopped because", answer)
            self.assertTrue("Next safe step:" in answer or "Missing evidence:" in answer)

    def test_recent_commits_uses_read_only_git_log_despite_wrong_intent(self) -> None:
        subprocess.run(["git", "init"], cwd=self.repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=self.repo, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=self.repo, check=True)
        subprocess.run(["git", "add", "README.md"], cwd=self.repo, check=True)
        subprocess.run(["git", "commit", "-m", "Initial fixture"], cwd=self.repo, check=True, capture_output=True)
        request = self._request()
        request.task = "최근 커밋 보여줘."
        with patch(
            "repooperator_worker.agent_core.controller_graph.get_active_repository",
            return_value=None,
        ):
            result = run_controller_graph(request, run_id="recent-commits")
        self.assertIn("git log --oneline -n 5", result.response)
        self.assertIn("Initial fixture", result.response)
        action_events = [event for event in list_run_events("recent-commits") if event.get("type") == "action_result"]
        self.assertIn("run_approved_command", [event["action"]["type"] for event in action_events])
        self.assertNotIn("analyze_repository", [event["action"]["type"] for event in action_events])

    def test_combined_git_request_does_not_commit_without_approval(self) -> None:
        subprocess.run(["git", "init"], cwd=self.repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=self.repo, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=self.repo, check=True)
        subprocess.run(["git", "add", "README.md"], cwd=self.repo, check=True)
        subprocess.run(["git", "commit", "-m", "Initial fixture"], cwd=self.repo, check=True, capture_output=True)
        (self.repo / "app.py").write_text("def main():\n    return 2\n", encoding="utf-8")
        request = self._request()
        request.task = "최근 커밋 보여줘. 커밋해줘."
        with patch(
            "repooperator_worker.agent_core.controller_graph.get_active_repository",
            return_value=None,
        ):
            result = run_controller_graph(request, run_id="combined-git")
        action_events = [event for event in list_run_events("combined-git") if event.get("type") == "action_result"]
        commands = [event["action"].get("command") for event in action_events]
        self.assertIn(["git", "log", "--oneline", "-n", "5"], commands)
        self.assertIn(["git", "status", "--short"], commands)
        self.assertNotIn(["git", "commit"], commands)
        self.assertIn("did not create a commit", result.response)

    def test_missing_requested_file_asks_clarification_without_speculation(self) -> None:
        request = self._request()
        request.task = "MissingManager.cs만 읽고 설명해줘."
        with patch(
            "repooperator_worker.agent_core.controller_graph.get_active_repository",
            return_value=None,
        ):
            result = run_controller_graph(request, run_id="missing-file")
        self.assertEqual(result.stop_reason, "needs_clarification")
        self.assertIn("MissingManager.cs", result.response)
        self.assertEqual(result.files_read, [])

    def test_llm_planner_can_choose_git_log_without_phrase_fallback(self) -> None:
        subprocess.run(["git", "init"], cwd=self.repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=self.repo, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=self.repo, check=True)
        subprocess.run(["git", "add", "README.md"], cwd=self.repo, check=True)
        subprocess.run(["git", "commit", "-m", "Initial fixture"], cwd=self.repo, check=True, capture_output=True)
        request = self._request()
        request.task = "지난 작업 이력 보여줘."
        planner = _PlannerClient(
            {
                "action_type": "inspect_git_state",
                "reason_summary": "Inspect recent git history.",
                "command": ["git", "log", "--oneline", "-n", "5"],
                "confidence": 0.9,
            }
        )
        with patch(
            "repooperator_worker.agent_core.controller_graph.OpenAICompatibleModelClient",
            return_value=planner,
        ), patch(
            "repooperator_worker.agent_core.controller_graph.get_active_repository",
            return_value=None,
        ):
            result = run_controller_graph(request, run_id="planner-git-history")
        self.assertIn("git log --oneline -n 5", result.response)
        self.assertIn("Initial fixture", result.response)
        action_types = [event["action"]["type"] for event in list_run_events("planner-git-history") if event.get("type") == "action_result"]
        self.assertNotIn("analyze_repository", action_types)

    def test_planner_visible_work_note_becomes_work_trace_event(self) -> None:
        request = self._request()
        request.task = "find risky serialization usage."
        planner = _PlannerClient(
            {
                "action_type": "search_text",
                "reason_summary": "Search for risky serialization usage.",
                "query": "BinaryFormatter",
                "confidence": 0.9,
                "visible_work_note": {
                    "goal": "Locate risky serialization evidence.",
                    "why_this_action": "A text search is the fastest safe way to confirm whether BinaryFormatter appears before reading files.",
                    "evidence_needed": ["Text matches for BinaryFormatter"],
                    "uncertainty": ["The relevant file is not named."],
                    "safety_note": None,
                },
            }
        )
        with patch("repooperator_worker.agent_core.controller_graph.OpenAICompatibleModelClient", return_value=planner), patch(
            "repooperator_worker.agent_core.controller_graph.get_active_repository",
            return_value=None,
        ):
            run_controller_graph(request, run_id="planner-work-note")
        trace_events = [event for event in list_run_events("planner-work-note") if event.get("event_type") == "work_trace"]
        decision = next(event for event in trace_events if event.get("display") == "primary" and event.get("aggregate", {}).get("action_type") == "search_text")
        self.assertIn("fastest safe way", str(decision.get("safe_reasoning_summary")))
        self.assertEqual(decision.get("evidence_needed"), ["Text matches for BinaryFormatter"])
        self.assertEqual(decision.get("visibility"), "user")
        json.dumps(decision, ensure_ascii=False)

    def test_low_level_activity_events_are_not_primary_work_trace(self) -> None:
        request = self._request()
        with patch("repooperator_worker.agent_core.controller_graph.OpenAICompatibleModelClient", return_value=_LoopClient()), patch(
            "repooperator_worker.agent_core.controller_graph.get_active_repository",
            return_value=None,
        ):
            run_controller_graph(request, run_id="low-level-visibility")
        low_labels = {"Loaded context", "Framed request", "Recorded observation", "Updated plan", "Created initial plan"}
        low_events = [event for event in list_run_events("low-level-visibility") if event.get("label") in low_labels]
        self.assertTrue(low_events)
        self.assertTrue(all(event.get("display") != "primary" and event.get("visibility") != "user" for event in low_events))

    def test_llm_planner_searches_persistence_file_for_non_explicit_edit(self) -> None:
        self._write_unity_fixture()
        for index in range(8):
            (self.repo / "Assets" / "Scripts" / f"Unrelated{index}.cs").write_text(f"public class Unrelated{index} {{}}\n", encoding="utf-8")
        request = self._request()
        request.task = "세이브 파일 깨졌을 때 복구 가능하게 해줘."
        planner = _PlannerClient(
            {
                "action_type": "search_files",
                "reason_summary": "Find persistence code by implementation evidence.",
                "search_queries": ["*.cs"],
                "text_queries": ["Save", "Load", "BinaryFormatter", "persistentDataPath", "PlayerData"],
                "confidence": 0.9,
            }
        )
        understanding = _edit_understanding(
            request,
            outputs=["code_change_proposal"],
            tools=["search_text", "read_file", "generate_edit"],
        )
        with patch("repooperator_worker.agent_core.request_understanding.understand_request", return_value=understanding), patch(
            "repooperator_worker.agent_core.controller_graph.OpenAICompatibleModelClient",
            return_value=planner,
        ), patch(
            "repooperator_worker.agent_core.tools.builtin.OpenAICompatibleModelClient",
            side_effect=RuntimeError("force deterministic validated fallback"),
        ), patch(
            "repooperator_worker.agent_core.controller_graph.get_active_repository",
            return_value=None,
        ):
            result = run_controller_graph(request, run_id="planner-persistence-search")
        self.assertIn("Assets/Scripts/DataHandler.cs", result.files_read)
        self.assertNotIn("max_loop_iterations", result.response)
        self.assertNotIn("Assets/Scripts/Unrelated0.cs", result.files_read[:1])

    def test_search_files_ranking_prefers_code_evidence(self) -> None:
        self._write_unity_fixture()
        for index in range(12):
            (self.repo / "Assets" / "Scripts" / f"Noise{index}.cs").write_text(f"public class Noise{index} {{ public void Move() {{}} }}\n", encoding="utf-8")
        executor = ActionExecutor(run_id="ranking-search", request=self._request())
        result = executor.execute(
            AgentAction(
                type="search_files",
                reason_summary="Find persistence implementation.",
                payload={"queries": ["*.cs"], "text_queries": ["Save", "Load", "BinaryFormatter", "PlayerData"], "max_results": 6},
            )
        )
        self.assertEqual(result.payload["candidates"][0], "Assets/Scripts/DataHandler.cs")
        detail = result.payload["candidate_details"][0]
        self.assertGreater(detail["score"], 20)
        self.assertIn("BinaryFormatter", " ".join(detail["matched_queries"]))

    def test_border_proposal_avoids_bitwise_flag_replacement(self) -> None:
        scripts = self.repo / "Assets" / "Scripts"
        scripts.mkdir(parents=True, exist_ok=True)
        (scripts / "Border.cs").write_text(
            "public class Border {\n"
            "  void Update() {}\n"
            "  int Mask(int left, int right) { return left & right; }\n"
            "  bool Ready(bool isLeft, bool isReady) { return isLeft & isReady; }\n"
            "}\n",
            encoding="utf-8",
        )
        request = self._request()
        request.task = "Border.cs에서 &를 &&로 바꾸고 빈 Update 제거해줘."
        def bitwise_safe_proposal(relative_path, content, task, context):
            proposed = content.replace("bool Ready(bool isLeft, bool isReady) { return isLeft & isReady; }", "bool Ready(bool isLeft, bool isReady) { return isLeft && isReady; }")
            proposed = proposed.replace("  void Update() {}\n", "")
            return {
                "file": relative_path,
                "summary": "Update the boolean branch without changing bitwise flag logic.",
                "proposed_content": proposed,
                "risk_notes": [],
                "preserves_existing_behavior": True,
            }
        with patch("repooperator_worker.agent_core.request_understanding.understand_request", return_value=_edit_understanding(request, files=["Border.cs"])), patch(
            "repooperator_worker.agent_core.tools.builtin.model_generate_edit_proposal",
            side_effect=bitwise_safe_proposal,
        ), patch(
            "repooperator_worker.agent_core.controller_graph.get_active_repository",
            return_value=None,
        ):
            result = run_controller_graph(request, run_id="border-bitwise-safe")
        self.assertIn("return isLeft && isReady", result.response)
        self.assertIn("return left & right", result.response)
        self.assertNotIn("return left && right", result.response)

    def test_datahandler_proposal_preserves_class_structure(self) -> None:
        scripts = self.repo / "Assets" / "Scripts"
        scripts.mkdir(parents=True, exist_ok=True)
        (scripts / "DataHandler.cs").write_text(
            "using System.IO;\n"
            "using System.Runtime.Serialization.Formatters.Binary;\n"
            "using UnityEngine;\n"
            "public class DataHandler : MonoBehaviour {\n"
            "  public static DataHandler Instance;\n"
            "  public PlayerData currentData;\n"
            "  void Awake() { Instance = this; }\n"
            "  void Start() { currentData = Load(); }\n"
            "  public void Save(PlayerData data) { var formatter = new BinaryFormatter(); }\n"
            "  public PlayerData Load() { return new PlayerData(); }\n"
            "}\n",
            encoding="utf-8",
        )
        request = self._request()
        request.task = "DataHandler.cs 저장 쪽 위험한 코드 찾아서 개선안 줘."
        def preserve_structure_proposal(relative_path, content, task, context):
            return {
                "file": relative_path,
                "summary": "Prepare a structure-preserving proposal for the target file.",
                "proposed_content": content.replace(
                    "public void Save(PlayerData data) { var formatter = new BinaryFormatter(); }",
                    "public void Save(PlayerData data) { /* safer persistence strategy goes here */ }",
                ),
                "risk_notes": ["Manual review is still required before applying."],
                "preserves_existing_behavior": True,
            }
        with patch("repooperator_worker.agent_core.request_understanding.understand_request", return_value=_edit_understanding(request, files=["DataHandler.cs"], outputs=["code_review", "edit_proposal"])), patch(
            "repooperator_worker.agent_core.tools.builtin.model_generate_edit_proposal",
            side_effect=preserve_structure_proposal,
        ), patch(
            "repooperator_worker.agent_core.controller_graph.get_active_repository",
            return_value=None,
        ):
            result = run_controller_graph(request, run_id="datahandler-preserve")
        self.assertIn("Awake()", result.response)
        self.assertIn("Start()", result.response)
        self.assertIn("No files were modified", result.response)

    def test_project_summary_answer_is_synthesized_not_file_dump(self) -> None:
        self._write_unity_fixture()
        (self.repo / "manifest.json").write_text('{"unity": true}', encoding="utf-8")
        request = self._request()
        request.task = "이 프로젝트가 뭐 하는 프로젝트인지 알아내줘."
        answer = "This project is a Unity turn-based ball-striking game.\n\nPurpose: players ready up and score through timed strike phases.\nTech stack: Unity/C#.\nKey modules: GameManager coordinates flow; DataHandler persists player data."
        with patch(
            "repooperator_worker.agent_core.controller_graph.OpenAICompatibleModelClient",
            return_value=_SynthesisClient(answer),
        ), patch(
            "repooperator_worker.agent_core.controller_graph.get_active_repository",
            return_value=None,
        ):
            result = run_controller_graph(request, run_id="project-synthesis")
        self.assertTrue(result.response.startswith("This project is"))
        self.assertIn("Purpose:", result.response)
        self.assertIn("Tech stack:", result.response)
        self.assertLess(result.response.count("Reviewed "), 2)
        self.assertNotRegex(result.response, r"[�]{2,}")

    def test_no_cannot_read_file_after_successful_read(self) -> None:
        self._write_unity_fixture()
        request = self._request()
        request.task = "README.md랑 GameManager.cs만 읽고 플레이 흐름 설명해줘."
        with patch(
            "repooperator_worker.agent_core.controller_graph.OpenAICompatibleModelClient",
            return_value=_SynthesisClient("I cannot read the files because the files object is empty."),
        ), patch(
            "repooperator_worker.agent_core.controller_graph.get_active_repository",
            return_value=None,
        ):
            result = run_controller_graph(request, run_id="no-cannot-read")
        self.assertIn("README.md", result.files_read)
        self.assertIn("Assets/Scripts/GameManager.cs", result.files_read)
        self.assertNotIn("cannot read", result.response.lower())
        self.assertNotIn("files object is empty", result.response.lower())

    def test_planner_overrides_prior_read_files_before_edit_generation(self) -> None:
        self._write_unity_fixture()
        request = self._request()
        request.task = "저장 로직 쪽을 고쳐줘."
        request.conversation_history = [
            ConversationMessage(
                role="assistant",
                content="I previously read README.md and GameManager.cs.",
                metadata={"files_read": ["README.md", "Assets/Scripts/GameManager.cs"]},
            )
        ]
        planner = _PlannerClient(
            {
                "action_type": "search_files",
                "reason_summary": "Find persistence implementation before editing.",
                "search_queries": ["*.cs"],
                "text_queries": ["Save", "Load", "BinaryFormatter", "PlayerData"],
                "confidence": 0.9,
            }
        )
        with patch("repooperator_worker.agent_core.request_understanding.understand_request", return_value=_edit_understanding(request)), patch(
            "repooperator_worker.agent_core.controller_graph.OpenAICompatibleModelClient",
            return_value=planner,
        ), patch(
            "repooperator_worker.agent_core.tools.builtin.OpenAICompatibleModelClient",
            side_effect=RuntimeError("force deterministic validated fallback"),
        ), patch(
            "repooperator_worker.agent_core.controller_graph.get_active_repository",
            return_value=None,
        ):
            result = run_controller_graph(request, run_id="planner-overrides-last-read")
        self.assertIn("Assets/Scripts/DataHandler.cs", result.files_read)
        self.assertNotIn("max_loop_iterations", result.response)

    def test_planner_final_answer_without_evidence_is_rejected(self) -> None:
        request = self._request()
        request.task = "Summarize this repository."
        planner = _PlannerClient(
            {
                "action_type": "final_answer",
                "reason_summary": "Answer immediately.",
                "confidence": 0.9,
                "enough_evidence": False,
            }
        )
        with patch(
            "repooperator_worker.agent_core.controller_graph.OpenAICompatibleModelClient",
            return_value=planner,
        ), patch(
            "repooperator_worker.agent_core.controller_graph.get_active_repository",
            return_value=None,
        ):
            result = run_controller_graph(request, run_id="reject-final-no-evidence")
        action_types = [event["action"]["type"] for event in list_run_events("reject-final-no-evidence") if event.get("type") == "action_result"]
        self.assertTrue({"inspect_repo_tree", "search_files", "read_file"} & set(action_types))
        self.assertNotEqual(action_types[:1], ["final_answer"])
        self.assertTrue(result.response)

    def test_planner_mutating_command_is_previewed_not_run(self) -> None:
        subprocess.run(["git", "init"], cwd=self.repo, check=True, capture_output=True)
        request = self._request()
        request.task = "Create a commit."
        planner = _PlannerClient(
            {
                "action_type": "run_approved_command",
                "reason_summary": "Commit changes.",
                "command": ["git", "commit", "-m", "test"],
                "confidence": 0.9,
            }
        )
        with patch(
            "repooperator_worker.agent_core.controller_graph.OpenAICompatibleModelClient",
            return_value=planner,
        ), patch(
            "repooperator_worker.agent_core.controller_graph.get_active_repository",
            return_value=None,
        ):
            result = run_controller_graph(request, run_id="planner-mutating-preview")
        action_events = [event for event in list_run_events("planner-mutating-preview") if event.get("type") == "action_result"]
        action_types = [event["action"]["type"] for event in action_events]
        self.assertIn("inspect_git_state", action_types)
        self.assertNotIn("run_approved_command", action_types)
        self.assertEqual(result.stop_reason, "waiting_approval")
        trace_events = [event for event in list_run_events("planner-mutating-preview") if event.get("event_type") == "work_trace"]
        self.assertTrue(any(event.get("phase") == "Safety" and event.get("safety_note") for event in trace_events))

    def test_edit_validation_rejects_unjustified_awake_removal(self) -> None:
        original = (
            "class Service:\n"
            "    def start(self):\n"
            "        return True\n"
            "\n"
            "def build_service():\n"
            "    return Service()\n"
        )
        proposed = (
            "class Service:\n"
            "    def start(self):\n"
            "        return True\n"
        )
        self.assertIsNone(
            validate_edit_proposal(
                "src/service.py",
                original,
                {"file": "src/service.py", "proposed_content": proposed, "risk_notes": []},
                "Update the service.",
            )
        )

    def test_final_answer_guard_repairs_false_write_claim(self) -> None:
        state = AgentCoreState(run_id="quality-guard", thread_id="t", repo=str(self.repo), branch="main", user_task="edit")
        state.action_results.append(
            ActionResult(
                action_id="a1",
                status="success",
                payload={
                    "applied": False,
                    "edit_proposals": [
                        {
                            "file": "Assets/Scripts/Border.cs",
                            "before_summary": "contains single ampersand boolean checks",
                            "after_summary": "uses short-circuit boolean checks",
                            "diff_summary": "--- before\n+++ after\n",
                        }
                    ],
                },
            )
        )
        repaired = validate_or_repair_final_answer("I applied the change to the file.", state, self._request())
        self.assertIn("proposed patch only", repaired)
        self.assertIn("No files were modified", repaired)

    def test_final_answer_guard_repairs_garbage_tokens(self) -> None:
        state = AgentCoreState(run_id="garbage-guard", thread_id="t", repo=str(self.repo), branch="main", user_task="read")
        state.files_read = ["README.md"]
        repaired = validate_or_repair_final_answer("abc한글xyz ��", state, self._request())
        self.assertNotIn("��", repaired)
        self.assertIn("README.md", repaired)

    def test_satellite_project_summary_fallback_produces_actual_answer(self) -> None:
        self._write_satellite_fixture()
        request = self._request()
        request.task = "이 레포가 뭐 하는 프로젝트인지 알아내줘."
        bad = "Purpose and architecture should be synthesized from those files..."
        with patch(
            "repooperator_worker.agent_core.controller_graph.OpenAICompatibleModelClient",
            return_value=_SynthesisClient(bad),
        ), patch(
            "repooperator_worker.agent_core.controller_graph.get_active_repository",
            return_value=None,
        ):
            result = run_controller_graph(request, run_id="satellite-project-summary")
        self.assertNotIn("should be synthesized", result.response)
        self.assertNotIn("I inspected the gathered project evidence", result.response)
        self.assertIn("satellite", result.response.lower())
        self.assertIn("orbit", result.response.lower())
        self.assertIn("README.md", result.response)

    def test_satellite_execution_flow_fallback_from_readme_and_main(self) -> None:
        self._write_satellite_fixture()
        request = self._request()
        request.task = "README.md랑 main.py만 읽고, 실행 흐름을 설명해줘."
        bad = "I can answer from those files, but the model answer needed repair..."
        with patch(
            "repooperator_worker.agent_core.controller_graph.OpenAICompatibleModelClient",
            return_value=_SynthesisClient(bad),
        ), patch(
            "repooperator_worker.agent_core.controller_graph.get_active_repository",
            return_value=None,
        ):
            result = run_controller_graph(request, run_id="satellite-flow")
        self.assertNotIn("model answer needed repair", result.response)
        self.assertNotIn("I can answer from those files", result.response)
        self.assertIn("main.py", result.response)
        self.assertIn("satellite", result.response.lower())
        self.assertTrue("propagate_orbit" in result.response or "algorithms" in result.response)

    def test_satellite_architecture_followup_excludes_sqlite(self) -> None:
        self._write_satellite_fixture()
        request = self._request()
        request.task = "방금 읽은 파일들 기준으로, 이 프로젝트의 아키텍처를 짧게 정리해줘."
        request.conversation_history = [
            ConversationMessage(
                role="assistant",
                content="Previously read README.md and main.py.",
                metadata={"files_read": ["README.md", "main.py"]},
            )
        ]
        planner = _PlannerClient(
            {
                "action_type": "read_file",
                "reason_summary": "Read core modules for architecture evidence.",
                "target_files": ["README.md", "main.py", "algorithms.py", "orbit.py", "satellite.py", "http_cache.sqlite"],
                "confidence": 0.9,
            },
            answer="Ask for a narrower change or review focus and I can continue from those files.",
        )
        with patch(
            "repooperator_worker.agent_core.controller_graph.OpenAICompatibleModelClient",
            return_value=planner,
        ), patch(
            "repooperator_worker.agent_core.controller_graph.get_active_repository",
            return_value=None,
        ):
            result = run_controller_graph(request, run_id="satellite-architecture")
        self.assertIn("아키텍처", result.response)
        self.assertIn("main.py", result.response)
        self.assertIn("algorithms.py", result.response)
        self.assertNotIn("Ask for a narrower change", result.response)
        self.assertNotIn("http_cache.sqlite", result.files_read)
        self.assertNotIn("http_cache.sqlite", result.response)

    def test_read_file_rejects_sqlite_cache_file(self) -> None:
        self._write_satellite_fixture()
        executor = ActionExecutor(run_id="sqlite-read", request=self._request())
        result = executor.execute(AgentAction(type="read_file", reason_summary="Read cache", target_files=["http_cache.sqlite"]))
        self.assertEqual(result.status, "skipped")
        self.assertNotIn("http_cache.sqlite", result.files_read)
        self.assertEqual(result.payload["contents"], {})
        self.assertIn("http_cache.sqlite", result.payload["skipped_files"])

    def test_final_answer_validation_ignores_generic_observations(self) -> None:
        self._write_satellite_fixture()
        request = self._request()
        request.task = "이 레포가 뭐 하는 프로젝트인지 알아내줘."
        planner = _PlannerClient(
            {"action_type": "final_answer", "reason_summary": "Answer now.", "confidence": 0.9, "enough_evidence": True}
        )
        with patch(
            "repooperator_worker.agent_core.controller_graph.OpenAICompatibleModelClient",
            return_value=planner,
        ), patch(
            "repooperator_worker.agent_core.controller_graph.get_active_repository",
            return_value=None,
        ):
            result = run_controller_graph(request, run_id="generic-observation-not-enough")
        actions = [event["action"]["type"] for event in list_run_events("generic-observation-not-enough") if event.get("type") == "action_result"]
        self.assertNotEqual(actions[:1], ["final_answer"])
        self.assertIn("README.md", result.files_read)

    def test_no_generic_progress_card(self) -> None:
        self._write_satellite_fixture()
        request = self._request()
        request.task = "이 레포가 뭐 하는 프로젝트인지 알아내줘."
        with patch(
            "repooperator_worker.agent_core.controller_graph.OpenAICompatibleModelClient",
            return_value=_SynthesisClient("This is a satellite orbit simulation project."),
        ), patch(
            "repooperator_worker.agent_core.controller_graph.get_active_repository",
            return_value=None,
        ):
            run_controller_graph(request, run_id="no-worked-progress")
        events = list_run_events("no-worked-progress")
        forbidden_phase = "Act" + "ivity"
        forbidden_label = "Work" + "ed"
        self.assertFalse(any(event.get("phase") == forbidden_phase and event.get("label") == forbidden_label for event in events))

    def test_stream_final_message_omits_streamed_activity_metadata(self) -> None:
        request = self._request()
        with patch.dict(os.environ, {"REPOOPERATOR_CONFIG_PATH": str(self.config)}, clear=False), patch(
            "repooperator_worker.agent_core.controller_graph.OpenAICompatibleModelClient",
            return_value=_LoopClient(),
        ), patch(
            "repooperator_worker.agent_core.controller_graph.get_active_repository",
            return_value=None,
        ):
            events = list(stream_controller_graph(request, run_id="stream-no-duplicate"))
        final = next(event for event in events if event.get("type") == "final_message")
        self.assertEqual(final["result"]["activity_events"], [])

    def test_analyze_repository_action_with_classifier_payload_is_json_safe(self) -> None:
        request = self._request()
        classifier = ClassifierResult(
            intent="repo_analysis",
            confidence=0.9,
        )
        action = AgentAction(
            type="analyze_repository",
            reason_summary="Review repo",
            payload={"classifier": classifier},
        )
        json.dumps(action.model_dump(), ensure_ascii=False)
        with patch.dict(os.environ, {"REPOOPERATOR_CONFIG_PATH": str(self.config)}, clear=False), patch(
            "repooperator_worker.agent_core.repository_review.OpenAICompatibleModelClient",
            return_value=_LoopClient(),
        ), patch(
            "repooperator_worker.services.event_service.get_repooperator_home_dir",
            return_value=Path(self.home.name),
        ):
            result = ActionExecutor(run_id="run-json-safe-action", request=request).execute(action)
            event = append_run_event(
                "run-json-safe-action",
                {
                    "type": "action_result",
                    "action": action.model_dump(),
                    "result": result.model_dump(),
                },
            )
        json.dumps(event, ensure_ascii=False)
        self.assertEqual(result.status, "success")
        self.assertIsInstance(result.payload.get("response"), dict)

    def test_analyze_repository_preserves_success_when_response_metadata_needs_sanitizing(self) -> None:
        request = self._request()
        response = self._response(request, "run-bad-metadata").model_copy(
            update={"activity_events": [{"bad": object(), "classifier": ClassifierResult()}]}
        )
        action = AgentAction(type="analyze_repository", reason_summary="Review repo", payload={"classifier": ClassifierResult()})
        with patch(
            "repooperator_worker.agent_core.tools.builtin.run_repository_review",
            return_value=response,
        ):
            result = ActionExecutor(run_id="run-bad-metadata", request=request).execute(action)
        self.assertEqual(result.status, "success")
        payload = result.model_dump()
        json.dumps(payload, ensure_ascii=False)
        self.assertEqual(payload["payload"]["response"]["response"], response.response)
        self.assertEqual(payload["payload"]["response"]["files_read"], response.files_read)
        self.assertTrue(payload["payload"]["response"]["metadata_serialization_error"])

    def test_json_safe_handles_core_boundary_values(self) -> None:
        response = self._response(self._request(), "run-json-safe-values").model_copy(
            update={"activity_events": [{"decision": SteeringDecision(steering_type="defer"), "when": datetime(2026, 5, 6), "kind": _JsonSafeEnum.SAMPLE}]}
        )
        action = AgentAction(type="analyze_repository", reason_summary="Review repo", payload={"classifier": ClassifierResult(), "paths": {Path("README.md")}})
        result = ActionResult(action_id=action.action_id, status="success", payload={"response": safe_agent_response_payload(response)})
        event = {"type": "action_result", "aggregate": {"steering": SteeringDecision(steering_type="defer")}, "action": action, "result": result}
        for value in [ClassifierResult(), SteeringDecision(), action, result, event, response]:
            json.dumps(json_safe(value), ensure_ascii=False)

    def test_stream_controller_graph_final_message_result_is_json_safe(self) -> None:
        request = self._request()
        bad_response = self._response(request, "run-stream-safe").model_copy(update={"activity_events": [{"bad": object()}]})
        with patch.dict(os.environ, {"REPOOPERATOR_CONFIG_PATH": str(self.config)}, clear=False), patch(
            "repooperator_worker.services.event_service.get_repooperator_home_dir",
            return_value=Path(self.home.name),
        ), patch(
            "repooperator_worker.agent_core.controller_graph.run_controller_graph",
            return_value=bad_response,
        ):
            events = list(stream_controller_graph(request, run_id="run-stream-safe"))
        final = next(event for event in events if event.get("type") == "final_message")
        json.dumps(final["result"], ensure_ascii=False)
        self.assertEqual(final["result"]["response"], bad_response.response)

    def test_stream_run_does_not_reappend_persisted_assistant_delta(self) -> None:
        request = self._request()

        def fake_stream(req, *, run_id=None):
            persisted = append_run_event(
                str(run_id),
                {"type": "assistant_delta", "delta": "Hello once.", "streaming_mode": "model_stream"},
            )
            yield persisted
            yield {"type": "final_message", "result": self._response(req, run_id).model_dump(mode="json")}

        with patch.dict(os.environ, {"REPOOPERATOR_CONFIG_PATH": str(self.config)}, clear=False), patch(
            "repooperator_worker.services.event_service.get_repooperator_home_dir",
            return_value=Path(self.home.name),
        ), patch(
            "repooperator_worker.agent_core.controller_graph.stream_controller_graph",
            side_effect=fake_stream,
        ):
            run_id, stream = stream_run(request)
            deadline = time.time() + 3
            while time.time() < deadline:
                if "[DONE]" in next(stream):
                    break
            assistant_events = [event for event in list_run_events(run_id) if event.get("type") == "assistant_delta"]
            sequences = [event["sequence"] for event in list_run_events(run_id)]
        self.assertEqual(len(assistant_events), 1)
        self.assertEqual(sequences, sorted(set(sequences)))

    def test_steering_parser_unknown_defers_without_direct_cancel_keyword_routing(self) -> None:
        request = self._request()
        state = ClassifierResult()
        source = inspect.getsource(consume_steering_for_state)
        self.assertNotIn('{"stop", "cancel"}', source)
        with patch("repooperator_worker.agent_core.steering.OpenAICompatibleModelClient", side_effect=RuntimeError("offline")):
            decision = parse_steering_instruction("please decide something later", request, self._state_for_steering(state))
        self.assertEqual(decision.steering_type, "defer")

    def test_cancel_steering_works_via_structured_parser_output(self) -> None:
        request = self._request()

        class _SteeringClient:
            def generate_text(self, _prompt):
                return json.dumps({"steering_type": "cancel", "target_files": [], "confidence": 0.95, "reason": "user requested cancellation"})

        with patch("repooperator_worker.agent_core.steering.OpenAICompatibleModelClient", return_value=_SteeringClient()):
            decision = parse_steering_instruction("irrelevant content", request, self._state_for_steering(ClassifierResult()))
        self.assertEqual(decision.steering_type, "cancel")
        self.assertGreaterEqual(decision.confidence, 0.8)

    def test_consume_steering_emits_applied_and_deferred_from_structured_parser(self) -> None:
        request = self._request()
        state = self._state_for_steering(ClassifierResult())
        with patch.dict(os.environ, {"REPOOPERATOR_CONFIG_PATH": str(self.config)}, clear=False), patch(
            "repooperator_worker.services.event_service.get_repooperator_home_dir",
            return_value=Path(self.home.name),
        ), patch(
            "repooperator_worker.services.agent_run_coordinator.consume_steering",
            return_value=[{"id": "one", "content": "README.md"}, {"id": "two", "content": "unclear"}],
        ), patch(
            "repooperator_worker.agent_core.steering.parse_steering_instruction",
            side_effect=[
                SteeringDecision(steering_type="add_target_file", target_files=["README.md"], confidence=0.9, reason="file target"),
                SteeringDecision(steering_type="unknown", target_files=[], confidence=0.0, reason="unknown"),
            ],
        ):
            consume_steering_for_state(state, request)
            events = list_run_events("run-steering-test")
        self.assertIn("README.md", state.classifier_result.target_files)
        steering_events = [event for event in events if str(event.get("activity_id", "")).startswith("controller-steering:")]
        self.assertEqual([event.get("aggregate", {}).get("steering_event_type") for event in steering_events], ["steering_applied", "steering_deferred"])

    def test_frontend_progress_merge_does_not_autocomplete_unrelated_running_activity(self) -> None:
        source = (TESTS_DIR.parents[2] / "apps" / "web" / "src" / "components" / "chat" / "run-event-state.ts").read_text(encoding="utf-8")
        merge_body = source.split("export function mergeProgressStep(", 1)[1].split("export function maxEventSequence", 1)[0]
        self.assertNotIn("completedPrev", merge_body)
        self.assertNotIn("index === current.length - 1 && step.status === \"running\"", merge_body)

    def test_frontend_rehydrate_uses_stored_events_before_final_activity_events(self) -> None:
        helper_source = (TESTS_DIR.parents[2] / "apps" / "web" / "src" / "components" / "chat" / "run-event-state.ts").read_text(encoding="utf-8")
        app_source = (TESTS_DIR.parents[2] / "apps" / "web" / "src" / "components" / "chat" / "ChatApp.tsx").read_text(encoding="utf-8")
        helper_body = helper_source.split("export function progressStepsForCompletedRun(", 1)[1].split("export function assistantTextFromRunEvents", 1)[0]
        self.assertIn("mergeRunEventsIntoProgressSteps(events", helper_body)
        self.assertIn("finalResult?.activity_events", helper_source)
        self.assertIn("progressStepsForCompletedRun(events", app_source)
        self.assertIn("progressStepsForCompletedRun(completedEvents", app_source)

    def test_repository_review_final_response_json_safe(self) -> None:
        request = self._request()
        with patch.dict(os.environ, {"REPOOPERATOR_CONFIG_PATH": str(self.config)}, clear=False), patch(
            "repooperator_worker.agent_core.repository_review.OpenAICompatibleModelClient",
            return_value=_LoopClient(),
        ), patch(
            "repooperator_worker.agent_core.controller_graph.get_active_repository",
            return_value=None,
        ), patch(
            "repooperator_worker.services.event_service.get_repooperator_home_dir",
            return_value=Path(self.home.name),
        ):
            response = run_controller_graph(request, run_id="run-repo-review-json-safe")
        json.dumps(response.model_dump(mode="json"), ensure_ascii=False)
        self.assertNotIn("ClassifierResult(", response.response)

    def _state_for_steering(self, classifier: ClassifierResult):
        from repooperator_worker.agent_core.state import AgentCoreState

        return AgentCoreState(
            run_id="run-steering-test",
            thread_id="thread-active-path",
            repo=str(self.repo),
            branch="main",
            user_task="Analyze",
            classifier_result=classifier,
        )

    def test_agent_service_error_uses_agent_core_metadata(self) -> None:
        request = self._request()
        with patch(
            "repooperator_worker.agent_core.controller_graph.run_controller_graph",
            side_effect=RuntimeError("boom"),
        ), patch(
            "repooperator_worker.services.agent_service.logger.exception",
        ):
            result = run_agent_task(request)
        self.assertEqual(result.response_type, "agent_error")
        self.assertEqual(result.agent_flow, "agent_core_controller")
        self.assertEqual(result.graph_path, "agent_core:error")


if __name__ == "__main__":
    unittest.main()
