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
from repooperator_worker.agent_core.langgraph_runtime import (  # noqa: E402
    GraphCheckpointIdentity,
    InMemoryGraphCheckpointAdapter,
    append_items,
    build_analysis_graph,
    build_compiled_repooperator_graph,
    build_edit_graph,
    build_evidence_gathering_graph,
    build_finalization_graph,
    build_repooperator_state_graph,
    build_validation_graph,
    initial_graph_state,
    route_after_change_plan,
    route_after_tool_result,
    route_after_understanding,
    route_to_final_or_continue,
)
from repooperator_worker.agent_core.controller_graph import run_controller_graph  # noqa: E402
from repooperator_worker.agent_core.tool_orchestrator import ToolOrchestrator  # noqa: E402
from repooperator_worker.schemas import AgentRunRequest  # noqa: E402


class _QuietClient:
    @property
    def model_name(self) -> str:
        return "test-model"

    def generate_text(self, request):
        if "bounded next-action planner" in request.system_prompt:
            return "{}"
        return "README.md evidence reached the final answer."

    def stream_text(self, request):
        yield {"type": "assistant_delta", "delta": "README.md evidence reached the final answer."}


class LangGraphRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name) / "repo"
        self.repo.mkdir()
        (self.repo / "README.md").write_text("# Demo\n", encoding="utf-8")
        (self.repo / "app.py").write_text("def main():\n    return 1\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _request(self, task: str = "Summarize README.md") -> AgentRunRequest:
        return AgentRunRequest(
            project_path=str(self.repo),
            git_provider="local",
            branch="main",
            thread_id="thread-langgraph",
            task=task,
        )

    def test_graph_compiles_with_expected_nodes(self) -> None:
        graph = build_repooperator_state_graph()
        expected = {
            "load_context",
            "understand_request",
            "build_task_plan",
            "route_next",
            "gather_evidence",
            "execute_tool",
            "validate_result",
            "plan_change_set",
            "generate_change_set",
            "validate_change_set",
            "repair_change_set",
            "ask_clarification",
            "await_approval",
            "final_synthesis",
            "supervisor",
        }
        self.assertTrue(expected.issubset(set(graph.nodes)))
        self.assertIsNotNone(build_compiled_repooperator_graph())

    def test_major_work_subgraphs_compile(self) -> None:
        subgraphs = [
            build_evidence_gathering_graph(),
            build_analysis_graph(),
            build_edit_graph(),
            build_validation_graph(),
            build_finalization_graph(),
        ]
        for graph in subgraphs:
            self.assertIsNotNone(graph.compile())

    def test_route_after_understanding_maps_actions_to_work_modes(self) -> None:
        state = initial_graph_state(self._request(), run_id="run-route")
        state["pending_action"] = AgentAction(type="read_file", reason_summary="Read evidence.", target_files=["README.md"])
        self.assertEqual(route_after_understanding(state), "gather_evidence")

        state["pending_action"] = AgentAction(type="generate_edit", reason_summary="Prepare patch.", target_files=["app.py"])
        self.assertEqual(route_after_understanding(state), "plan_change_set")

        state["pending_action"] = AgentAction(type="final_answer", reason_summary="Finish.")
        self.assertEqual(route_after_understanding(state), "final_synthesis")

    def test_append_reducer_keeps_action_and_result_history(self) -> None:
        action = AgentAction(type="inspect_repo_tree", reason_summary="Inspect.")
        result = ActionResult(action_id=action.action_id, status="success")
        self.assertEqual(append_items([], [action]), [action])
        self.assertEqual(append_items([result], []), [result])

    def test_approval_and_cancellation_routes_stop_at_safe_boundaries(self) -> None:
        state = initial_graph_state(self._request("Run git status"), run_id="run-approval")
        action = AgentAction(type="preview_command", reason_summary="Preview command.", command=["git", "status"])
        state["actions_taken"] = [action]
        state["action_results"] = [ActionResult(action_id=action.action_id, status="waiting_approval", command_result={"command": ["git", "status"]})]
        self.assertEqual(route_after_tool_result(state), "await_approval")

        state["stop_reason"] = "cancelled"
        self.assertEqual(route_to_final_or_continue(state), "final_synthesis")

    def test_failed_change_set_routes_to_repair_once(self) -> None:
        state = initial_graph_state(self._request("Edit app.py"), run_id="run-repair")
        action = AgentAction(type="generate_edit", reason_summary="Generate edit.", target_files=["app.py"])
        state["actions_taken"] = [action]
        state["action_results"] = [ActionResult(action_id=action.action_id, status="failed", errors=["invalid proposal"])]
        self.assertEqual(route_after_change_plan(state), "repair_change_set")

    def test_langgraph_runtime_project_summary_flow_uses_tool_orchestrator(self) -> None:
        request = self._request("Summarize README.md")
        calls: list[str] = []
        original_execute_action = ToolOrchestrator.execute_action

        def tracking_execute_action(self, action):
            calls.append(action.type)
            return original_execute_action(self, action)

        with patch.dict(os.environ, {"REPOOPERATOR_AGENT_RUNTIME": "langgraph"}, clear=False), patch(
            "repooperator_worker.agent_core.controller_graph.get_active_repository", return_value=None
        ), patch("repooperator_worker.agent_core.controller_graph.OpenAICompatibleModelClient", return_value=_QuietClient()), patch(
            "repooperator_worker.agent_core.langgraph_runtime.ToolOrchestrator.execute_action", tracking_execute_action
        ):
            response = run_controller_graph(request, run_id="run-langgraph-summary")
        self.assertIn("README.md", response.files_read)
        self.assertIn("README.md evidence", response.response)
        self.assertIn("read_file", calls)

    def test_broad_task_dispatches_supervisor_reports(self) -> None:
        request = self._request("Analyze the whole codebase and explain every file group.")
        state = initial_graph_state(request, run_id="run-supervisor")
        from repooperator_worker.agent_core.langgraph_runtime import supervisor_node

        update = supervisor_node(state)
        self.assertTrue(update["supervisor_mode"])
        self.assertTrue(update["file_role_reports"])
        workers = {item["worker"] for item in update["file_role_reports"]}
        self.assertIn("AnalysisAgent", workers)

    def test_checkpoint_adapter_round_trips_waiting_approval_state(self) -> None:
        adapter = InMemoryGraphCheckpointAdapter()
        request = self._request("Run git status")
        state = initial_graph_state(request, run_id="run-checkpoint")
        state["pending_approval"] = {"command": ["git", "status"], "needs_approval": True}
        state["stop_reason"] = "waiting_approval"
        identity = GraphCheckpointIdentity("run-checkpoint", request.thread_id, request.project_path, request.branch)
        adapter.save(identity, 7, "await_approval", state)
        record = adapter.load_latest(identity)
        self.assertIsNotNone(record)
        self.assertEqual(record.sequence, 7)
        self.assertEqual(record.state["stop_reason"], "waiting_approval")
        self.assertEqual(record.state["pending_approval"]["command"], ["git", "status"])


if __name__ == "__main__":
    unittest.main()
