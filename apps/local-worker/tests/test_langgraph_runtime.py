import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repooperator_worker.agent_core.actions import AgentAction, ActionResult  # noqa: E402
from repooperator_worker.agent_core.langgraph_runtime import (  # noqa: E402
    append_items,
    build_analysis_graph,
    build_compiled_repooperator_graph,
    build_edit_graph,
    build_evidence_gathering_graph,
    build_finalization_graph,
    build_git_workflow_graph,
    build_repooperator_state_graph,
    build_supervisor_graph,
    build_validation_graph,
    build_web_research_graph,
    final_emit_message_node,
    graph_config_for_request,
    initial_graph_state,
    resume_langgraph_controller,
    run_langgraph_controller,
    route_after_change_plan,
    route_after_tool_result,
    route_after_understanding,
    route_to_final_or_continue,
)
from repooperator_worker.agent_core.events import (  # noqa: E402
    EVENT_AUDIENCE_DEBUG,
    EVENT_KIND_DEBUG_RATIONALE,
    activity_event,
)
from repooperator_worker.agent_core.graph.adapters import _graph_transition_event  # noqa: E402
from repooperator_worker.agent_core.graph.nodes.finalization import _workflow_response_updates  # noqa: E402
from repooperator_worker.agent_core.graph_checkpoints import EventServiceLangGraphSaver  # noqa: E402
from repooperator_worker.agent_core.graph_state import (  # noqa: E402
    action_from_snapshot,
    action_to_snapshot,
    request_from_snapshot,
    request_to_snapshot,
    response_from_snapshot,
    result_from_snapshot,
    result_to_snapshot,
)
from repooperator_worker.agent_core.change_set import (  # noqa: E402
    ProposedFileChange,
    ChangePlan,
    ChangeSetProposal,
    validate_change_set,
)
from repooperator_worker.agent_core.tool_orchestrator import ToolOrchestrator  # noqa: E402
from repooperator_worker.schemas import AgentRunRequest  # noqa: E402
from repooperator_worker.services.agent_run_coordinator import resume_approval, wait_for_approval  # noqa: E402
from repooperator_worker.services.agent_service import run_agent_task  # noqa: E402
from repooperator_worker.services.event_service import get_run, list_run_events, start_active_run  # noqa: E402
from repooperator_worker.services.json_safe import json_safe  # noqa: E402


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
        self._old_config_path = os.environ.get("REPOOPERATOR_CONFIG_PATH")
        os.environ["REPOOPERATOR_CONFIG_PATH"] = str(Path(self.tmp.name) / ".repooperator" / "config.json")
        self.repo = Path(self.tmp.name) / "repo"
        self.repo.mkdir()
        (self.repo / "README.md").write_text("# Demo\n", encoding="utf-8")
        (self.repo / "app.py").write_text("def main():\n    return 1\n", encoding="utf-8")

    def tearDown(self) -> None:
        if self._old_config_path is None:
            os.environ.pop("REPOOPERATOR_CONFIG_PATH", None)
        else:
            os.environ["REPOOPERATOR_CONFIG_PATH"] = self._old_config_path
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
            "await_change_approval",
            "apply_change_set",
            "post_apply_validation",
            "final_synthesis",
            "supervisor",
            "capability_discovery",
            "context_pack",
            "web_research_graph",
            "git_workflow_graph",
            "routine_enqueue_node",
            "decompose_task",
            "dispatch_work_units",
            "reduce_work_reports",
        }
        self.assertTrue(expected.issubset(set(graph.nodes)))
        self.assertIsNotNone(build_compiled_repooperator_graph())

    def test_split_graph_modules_reexport_compatibility_entrypoints(self) -> None:
        from repooperator_worker.agent_core import langgraph_runtime as facade
        from repooperator_worker.agent_core.graph import builder as graph_builder
        from repooperator_worker.agent_core.graph import routes as graph_routes
        from repooperator_worker.agent_core.graph import runtime as graph_runtime
        from repooperator_worker.agent_core.graph import state as graph_state
        from repooperator_worker.agent_core.graph.nodes import finalization

        self.assertIs(facade.RepoOperatorGraphState, graph_state.RepoOperatorGraphState)
        self.assertIs(facade.build_repooperator_state_graph, graph_builder.build_repooperator_state_graph)
        self.assertIs(facade.build_compiled_repooperator_graph, graph_runtime.build_compiled_repooperator_graph)
        self.assertIs(facade.run_langgraph_controller, graph_runtime.run_langgraph_controller)
        self.assertIs(facade.resume_langgraph_controller, graph_runtime.resume_langgraph_controller)
        self.assertIs(facade.route_after_tool_result, graph_routes.route_after_tool_result)
        self.assertIs(facade.final_emit_message_node, finalization.final_emit_message_node)
        self.assertFalse(hasattr(facade, "ToolOrchestrator"))
        self.assertFalse(any(name.startswith("_") for name in getattr(facade, "__all__", [])))

    def test_major_work_subgraphs_compile(self) -> None:
        subgraphs = [
            build_evidence_gathering_graph(),
            build_analysis_graph(),
            build_edit_graph(),
            build_validation_graph(),
            build_web_research_graph(),
            build_git_workflow_graph(),
            build_finalization_graph(),
            build_supervisor_graph(),
        ]
        for graph in subgraphs:
            self.assertIsNotNone(graph.compile())

    def test_run_langgraph_controller_direct_entrypoint_works(self) -> None:
        request = self._request("Summarize README.md")
        with patch("repooperator_worker.agent_core.graph.repository_support.get_active_repository", return_value=None), patch(
            "repooperator_worker.agent_core.graph.support.OpenAICompatibleModelClient", return_value=_QuietClient()
        ):
            response = run_langgraph_controller(request, run_id="run-direct-langgraph")
        self.assertEqual(response.agent_flow, "langgraph")
        self.assertIn("README.md", response.files_read)
        self.assertIn("README.md evidence", response.response)

    def test_graph_compiles_with_langgraph_checkpointer(self) -> None:
        checkpointer = InMemorySaver()
        compiled = build_compiled_repooperator_graph(checkpoint_adapter=checkpointer)
        self.assertIs(compiled.checkpointer, checkpointer)

    def test_initial_graph_state_and_snapshots_are_json_safe(self) -> None:
        request = self._request()
        state = initial_graph_state(request, run_id="run-json-safe")
        self.assertIn("visible_rationale_log", state)
        self.assertEqual(state["visible_rationale_log"], [])
        self.assertIsNone(state["user_understanding_context"])
        self.assertIsNone(state["evidence_basis"])
        action = AgentAction(type="read_file", reason_summary="Read.", target_files=["README.md"])
        result = ActionResult(action_id=action.action_id, status="success", files_read=["README.md"])
        action_snapshot = action_to_snapshot(action)
        result_snapshot = result_to_snapshot(result)
        self.assertEqual(action_from_snapshot(action_snapshot).type, "read_file")
        self.assertEqual(result_from_snapshot(result_snapshot).files_read, ["README.md"])
        self.assertEqual(request_from_snapshot(request_to_snapshot(request)).project_path, request.project_path)
        json.dumps(json_safe({**state, "actions_taken": [action_snapshot], "action_results": [result_snapshot]}))

    def test_safe_reasoning_only_event_is_debug_rationale(self) -> None:
        event = activity_event(
            run_id="taxonomy-safe-summary",
            request=self._request(),
            activity_id="rationale-only",
            event_type="work_trace",
            phase="Decision",
            label="Planning",
            safe_reasoning_summary="I separated " + "the request into internal steps.",
        )

        self.assertEqual(event.get("kind"), EVENT_KIND_DEBUG_RATIONALE)
        self.assertEqual(event.get("audience"), EVENT_AUDIENCE_DEBUG)
        self.assertEqual(event.get("visibility"), "debug")
        self.assertEqual(event.get("display"), "secondary")

    def test_graph_transition_bare_success_is_not_real_validation(self) -> None:
        state = initial_graph_state(self._request(), run_id="run-graph-transition-taxonomy")
        event = _graph_transition_event(
            state,
            "validate_result",
            subgraph="validation_graph",
            operation="validate_result",
            validation_result={"status": "success"},
        )

        self.assertNotEqual(event.get("kind"), "validation")
        self.assertNotIn("validation_result", event)
        self.assertEqual(event.get("action_result", {}).get("status"), "success")

    def test_response_updates_ignore_action_result_validation_and_normalize_real_validation(self) -> None:
        state = initial_graph_state(self._request(), run_id="run-validation-taxonomy")
        state["validation_results"] = [{"kind": "action_result", "status": "success"}]
        self.assertNotIn("validation_result", _workflow_response_updates(state))

        state["validation_results"] = [{"kind": "command", "status": "success", "display_command": "npm test"}]
        updates = _workflow_response_updates(state)
        self.assertEqual(updates["validation_result"]["kind"], "command")
        self.assertEqual(updates["validation_result"]["source"], "command")
        self.assertEqual(updates["validation_result"]["status"], "passed")

        state["validation_results"] = [{"kind": "change_set", "status": "valid"}]
        updates = _workflow_response_updates(state)
        self.assertEqual(updates["validation_result"]["kind"], "change_set")
        self.assertEqual(updates["validation_result"]["source"], "change_set")
        self.assertEqual(updates["validation_result"]["status"], "passed")

    def test_route_after_understanding_maps_actions_to_work_modes(self) -> None:
        state = initial_graph_state(self._request(), run_id="run-route")
        state["pending_action"] = AgentAction(type="read_file", reason_summary="Read evidence.", target_files=["README.md"])
        self.assertEqual(route_after_understanding(state), "gather_evidence")

        state["pending_action"] = AgentAction(type="generate_edit", reason_summary="Prepare patch.", target_files=["app.py"])
        self.assertEqual(route_after_understanding(state), "plan_change_set")

        state["pending_action"] = AgentAction(type="generate_change_set", reason_summary="Prepare change set.", target_files=["app.py"])
        self.assertEqual(route_after_understanding(state), "plan_change_set")

        state["pending_action"] = AgentAction(type="final_answer", reason_summary="Finish.")
        self.assertEqual(route_after_understanding(state), "final_synthesis")

    def test_route_evidence_next_snapshots_policy_action(self) -> None:
        from repooperator_worker.agent_core.graph.nodes.evidence import route_evidence_next_node

        state = initial_graph_state(self._request("Summarize README.md"), run_id="run-evidence-route")
        update = route_evidence_next_node(state)
        self.assertIsInstance(update.get("pending_action"), dict)
        action = action_from_snapshot(update.get("pending_action"))
        self.assertIsNotNone(action)
        self.assertEqual(action.type, "inspect_repo_tree")
        self.assertTrue(update.get("events_to_emit"))

    def test_append_reducer_keeps_action_and_result_history(self) -> None:
        action = AgentAction(type="inspect_repo_tree", reason_summary="Inspect.")
        result = ActionResult(action_id=action.action_id, status="success")
        self.assertEqual(append_items([], [action]), [action])
        self.assertEqual(append_items([result], []), [result])
        self.assertEqual(append_items([{"id": "a"}], [{"id": "b"}]), [{"id": "a"}, {"id": "b"}])

    def test_langgraph_runtime_has_no_legacy_chooser_dependency(self) -> None:
        import importlib.util
        import inspect
        import repooperator_worker.agent_core.graph.routes as graph_routes

        old_module = "repooperator_worker.agent_core." + "controller_" + "graph"
        old_symbol = "controller_" + "choose_next_action"
        self.assertIsNone(importlib.util.find_spec(old_module))
        self.assertNotIn(old_symbol, inspect.getsource(graph_routes))
        request = self._request("Summarize README.md")
        with patch.dict(os.environ, {"REPOOPERATOR_AGENT_RUNTIME": "legacy"}, clear=False), patch(
            "repooperator_worker.agent_core.graph.repository_support.get_active_repository", return_value=None
        ), patch("repooperator_worker.agent_core.graph.support.OpenAICompatibleModelClient", return_value=_QuietClient()):
            response = run_langgraph_controller(request, run_id="run-no-legacy-chooser")
        self.assertIn("README.md", response.files_read)
        self.assertEqual(response.agent_flow, "langgraph")

    def test_agent_service_uses_langgraph_even_with_legacy_runtime_env(self) -> None:
        request = self._request("Summarize README.md")
        with patch.dict(
            os.environ,
            {"REPOOPERATOR_AGENT_RUNTIME": "legacy", "REPOOPERATOR_AGENT_RUNTIME_DEFAULT": "legacy"},
            clear=False,
        ), patch(
            "repooperator_worker.agent_core.graph.repository_support.get_active_repository", return_value=None
        ), patch("repooperator_worker.agent_core.graph.support.OpenAICompatibleModelClient", return_value=_QuietClient()):
            response = run_agent_task(request)
        self.assertIn("README.md", response.files_read)
        self.assertEqual(response.agent_flow, "langgraph")

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
            "repooperator_worker.agent_core.graph.repository_support.get_active_repository", return_value=None
        ), patch("repooperator_worker.agent_core.graph.support.OpenAICompatibleModelClient", return_value=_QuietClient()), patch(
            "repooperator_worker.agent_core.graph.adapters.ToolOrchestrator.execute_action", tracking_execute_action
        ):
            response = run_langgraph_controller(request, run_id="run-langgraph-summary")
        self.assertIn("README.md", response.files_read)
        self.assertIn("README.md evidence", response.response)
        self.assertIn("read_file", calls)

    def test_broad_task_dispatches_supervisor_reports(self) -> None:
        request = self._request("Analyze the whole codebase and explain every file group.")
        state = initial_graph_state(request, run_id="run-supervisor")
        from repooperator_worker.agent_core.langgraph_runtime import supervisor_node

        calls: list[str] = []
        original_execute_action = ToolOrchestrator.execute_action

        def tracking_execute_action(self, action):
            calls.append(action.type)
            return original_execute_action(self, action)

        with patch("repooperator_worker.agent_core.graph.nodes.supervisor.ToolOrchestrator.execute_action", tracking_execute_action):
            update = supervisor_node(state)
        self.assertTrue(update["supervisor_mode"])
        self.assertGreater(len(update["worker_tasks"]), 1)
        self.assertGreater(len(update["worker_reports"]), 1)
        self.assertTrue(update["file_role_reports"])
        workers = {item["worker"] for item in update["file_role_reports"]}
        self.assertIn("AnalysisAgent", workers)
        self.assertIn("read_file", calls)
        self.assertIn("input_files", update["worker_tasks"][0])
        self.assertIn("files_analyzed", update["worker_reports"][0])

    def test_complex_task_decomposition_creates_work_units(self) -> None:
        from repooperator_worker.agent_core.langgraph_runtime import decompose_task_node, dispatch_work_units_node, reduce_work_reports_node

        state = initial_graph_state(self._request("Analyze the whole codebase and prepare validation guidance."), run_id="run-work-units")
        decomposed = decompose_task_node(state)
        self.assertGreaterEqual(len(decomposed["worker_tasks"]), 3)
        self.assertIn("capability_needed", decomposed["worker_tasks"][0])
        dispatched = dispatch_work_units_node({**state, **decomposed})
        self.assertTrue(dispatched["worker_reports"])
        reduced = reduce_work_reports_node({**state, **decomposed, **dispatched})
        self.assertTrue(reduced["evidence_reports"] or reduced["file_role_reports"])

    def test_interrupt_resume_allow_runs_without_restarting_evidence(self) -> None:
        request = self._request("Run git status")
        state = initial_graph_state(request, run_id="run-interrupt")
        state["pending_approval"] = {"command": ["git", "status", "--short"], "approval_id": "", "reason": "test"}
        state["stop_reason"] = "waiting_approval"
        state["files_read"] = ["README.md"]
        checkpointer = InMemorySaver()
        compiled = build_compiled_repooperator_graph(checkpoint_adapter=checkpointer)
        config = graph_config_for_request(request, "run-interrupt")
        interrupted = compiled.invoke(state, config=config)
        self.assertIn("__interrupt__", interrupted)

        resumed = compiled.invoke(Command(resume={"decision": "allow"}), config=config)
        self.assertIn("README.md", resumed.get("files_read") or [])
        run_actions = [action.get("type") for action in resumed.get("actions_taken") or []]
        self.assertIn("run_approved_command", run_actions)

    def test_interrupt_resume_deny_finalizes_safely(self) -> None:
        request = self._request("Run git status")
        state = initial_graph_state(request, run_id="run-deny")
        state["pending_approval"] = {"command": ["git", "status", "--short"], "approval_id": "", "reason": "test"}
        state["stop_reason"] = "waiting_approval"
        checkpointer = InMemorySaver()
        compiled = build_compiled_repooperator_graph(checkpoint_adapter=checkpointer)
        config = graph_config_for_request(request, "run-deny")
        compiled.invoke(state, config=config)
        denied = compiled.invoke(Command(resume={"decision": "deny"}), config=config)
        self.assertEqual(denied.get("stop_reason"), "approval_denied")
        self.assertIn("approval was denied", denied.get("final_response") or "")

    def test_change_set_validation_rejects_unapproved_delete_and_accepts_modify(self) -> None:
        proposal = ChangeSetProposal(
            plan=ChangePlan(summary="Modify app.py", target_files=["app.py"], operations=["modify"]),
            changes=[
                ProposedFileChange(
                    path="app.py",
                    operation="modify",
                    summary="Return two.",
                    original_content=(self.repo / "app.py").read_text(encoding="utf-8"),
                    proposed_content="def main():\n    return 2\n",
                )
            ],
        )
        self.assertEqual(validate_change_set(proposal, repo=str(self.repo)).status, "valid")

        delete = ChangeSetProposal(
            plan=ChangePlan(summary="Delete app.py", target_files=["app.py"], operations=["delete"]),
            changes=[ProposedFileChange(path="app.py", operation="delete", summary="Delete file")],
        )
        result = validate_change_set(delete, repo=str(self.repo))
        self.assertEqual(result.status, "invalid")
        self.assertIn("delete proposals require explicit", "; ".join(result.errors))

    def test_change_set_validation_supports_create_and_explicit_delete(self) -> None:
        create = ChangeSetProposal(
            plan=ChangePlan(summary="Create a helper module", target_files=["helper.py"], operations=["create"]),
            changes=[
                ProposedFileChange(
                    path="helper.py",
                    operation="create",
                    summary="Add helper.",
                    proposed_content="def helper():\n    return 1\n",
                )
            ],
        )
        self.assertEqual(validate_change_set(create, repo=str(self.repo)).status, "valid")

        collision = ChangeSetProposal(
            plan=ChangePlan(summary="Create colliding file", target_files=["app.py"], operations=["create"]),
            changes=[ProposedFileChange(path="app.py", operation="create", summary="Collide.", proposed_content="print(1)\n")],
        )
        self.assertEqual(validate_change_set(collision, repo=str(self.repo)).status, "invalid")

        explicit_delete = ChangeSetProposal(
            plan=ChangePlan(summary="Explicitly delete obsolete app.py as requested", target_files=["app.py"], operations=["delete"]),
            changes=[
                ProposedFileChange(
                    path="app.py",
                    operation="delete",
                    summary="Delete obsolete app.py.",
                    delete_justification="User explicitly requested deleting this obsolete file.",
                    original_content=(self.repo / "app.py").read_text(encoding="utf-8"),
                )
            ],
        )
        self.assertEqual(validate_change_set(explicit_delete, repo=str(self.repo)).status, "valid")

    def test_change_set_validation_rejects_binary_syntax_and_fenced_source(self) -> None:
        binary = ChangeSetProposal(
            plan=ChangePlan(summary="Create image", target_files=["image.png"], operations=["create"]),
            changes=[ProposedFileChange(path="image.png", operation="create", summary="Binary.", proposed_content="not really png")],
        )
        self.assertEqual(validate_change_set(binary, repo=str(self.repo)).status, "invalid")

        invalid_python = ChangeSetProposal(
            plan=ChangePlan(summary="Modify app.py", target_files=["app.py"], operations=["modify"]),
            changes=[
                ProposedFileChange(
                    path="app.py",
                    operation="modify",
                    summary="Break syntax.",
                    original_content=(self.repo / "app.py").read_text(encoding="utf-8"),
                    proposed_content="def main(:\n    return 2\n",
                )
            ],
        )
        self.assertIn("Python syntax is invalid", "; ".join(validate_change_set(invalid_python, repo=str(self.repo)).errors))

        fenced_source = ChangeSetProposal(
            plan=ChangePlan(summary="Modify app.py", target_files=["app.py"], operations=["modify"]),
            changes=[
                ProposedFileChange(
                    path="app.py",
                    operation="modify",
                    summary="Fenced source.",
                    original_content=(self.repo / "app.py").read_text(encoding="utf-8"),
                    proposed_content="```python\ndef main():\n    return 2\n```\n",
                )
            ],
        )
        self.assertIn("markdown fences", "; ".join(validate_change_set(fenced_source, repo=str(self.repo)).errors))

    def test_final_response_includes_change_set_payload_and_archive(self) -> None:
        request = self._request("Create helper.py")
        state = initial_graph_state(request, run_id="run-change-set-response")
        proposal = ChangeSetProposal(
            plan=ChangePlan(summary="Create helper.py", target_files=["helper.py"], operations=["create"]),
            changes=[
                ProposedFileChange(
                    path="helper.py",
                    operation="create",
                    summary="Add helper.",
                    proposed_content="def helper():\n    return 1\n",
                )
            ],
        )
        validation = validate_change_set(proposal, repo=str(self.repo))
        proposal.validation = validation
        proposal.status = validation.status
        state["change_set_proposal"] = proposal.model_dump()
        state["final_response"] = "Prepared a proposal-only change set."

        update = final_emit_message_node(state)
        response = response_from_snapshot(update["response_snapshot"])
        self.assertEqual(response.response_type, "change_proposal")
        self.assertEqual(response.change_set_proposal["status"], "valid")
        self.assertEqual(response.edit_archive[0]["operation"], "create")
        self.assertIn("+def helper", response.edit_archive[0]["diff"])

    def test_change_set_apply_resume_writes_after_approval_only(self) -> None:
        request = self._request("Apply app.py proposal")
        original = (self.repo / "app.py").read_text(encoding="utf-8")
        proposal = ChangeSetProposal(
            plan=ChangePlan(summary="Modify app.py", target_files=["app.py"], operations=["modify"]),
            changes=[
                ProposedFileChange(
                    path="app.py",
                    operation="modify",
                    summary="Return two.",
                    original_content=original,
                    proposed_content="def main():\n    return 2\n",
                )
            ],
        )
        validation = validate_change_set(proposal, repo=str(self.repo))
        proposal.validation = validation
        proposal.status = validation.status
        proposal.validation_status = validation.status
        state = initial_graph_state(request, run_id="run-apply-change-set")
        state["change_set_proposal"] = proposal.model_dump()
        state["pending_approval"] = {
            "kind": "change_set_apply",
            "proposal_id": proposal.model_dump()["proposal_id"],
            "change_set_proposal": proposal.model_dump(),
            "reason": "test",
        }
        state["stop_reason"] = "waiting_approval"
        checkpointer = InMemorySaver()
        compiled = build_compiled_repooperator_graph(checkpoint_adapter=checkpointer)
        config = graph_config_for_request(request, "run-apply-change-set")
        interrupted = compiled.invoke(state, config=config)
        self.assertIn("__interrupt__", interrupted)
        self.assertEqual((self.repo / "app.py").read_text(encoding="utf-8"), original)

        resumed = compiled.invoke(Command(resume={"decision": "allow", "kind": "change_set_apply", "proposal_id": proposal.model_dump()["proposal_id"]}), config=config)
        self.assertEqual((self.repo / "app.py").read_text(encoding="utf-8"), "def main():\n    return 2\n")
        self.assertEqual(resumed.get("apply_status"), "applied")
        self.assertIn("app.py", resumed.get("files_changed") or [])

    def test_change_set_apply_denial_leaves_disk_unchanged(self) -> None:
        request = self._request("Reject app.py proposal")
        original = (self.repo / "app.py").read_text(encoding="utf-8")
        proposal = ChangeSetProposal(
            plan=ChangePlan(summary="Modify app.py", target_files=["app.py"], operations=["modify"]),
            changes=[
                ProposedFileChange(
                    path="app.py",
                    operation="modify",
                    summary="Return two.",
                    original_content=original,
                    proposed_content="def main():\n    return 2\n",
                )
            ],
        )
        validation = validate_change_set(proposal, repo=str(self.repo))
        proposal.validation = validation
        proposal.status = validation.status
        proposal.validation_status = validation.status
        state = initial_graph_state(request, run_id="run-deny-change-set")
        state["change_set_proposal"] = proposal.model_dump()
        state["pending_approval"] = {
            "kind": "change_set_apply",
            "proposal_id": proposal.model_dump()["proposal_id"],
            "change_set_proposal": proposal.model_dump(),
            "reason": "test",
        }
        state["stop_reason"] = "waiting_approval"
        checkpointer = InMemorySaver()
        compiled = build_compiled_repooperator_graph(checkpoint_adapter=checkpointer)
        config = graph_config_for_request(request, "run-deny-change-set")
        compiled.invoke(state, config=config)
        denied = compiled.invoke(Command(resume={"decision": "deny", "kind": "change_set_apply"}), config=config)
        self.assertEqual((self.repo / "app.py").read_text(encoding="utf-8"), original)
        self.assertEqual(denied.get("apply_status"), "rejected")
        self.assertIn("not applied", denied.get("final_response") or "")

    def test_event_service_checkpointer_round_trips_waiting_approval_state(self) -> None:
        request = self._request("Run git status")
        state = initial_graph_state(request, run_id="run-checkpoint")
        state["pending_approval"] = {"command": ["git", "status"], "needs_approval": True}
        state["stop_reason"] = "waiting_approval"
        config = graph_config_for_request(request, "run-checkpoint")
        saver = EventServiceLangGraphSaver()
        compiled = build_compiled_repooperator_graph(checkpoint_adapter=saver)
        compiled.invoke(state, config=config)

        events = [event for event in list_run_events("run-checkpoint") if event.get("type") == "langgraph_checkpoint"]
        self.assertTrue(events)
        restored = EventServiceLangGraphSaver().get_tuple(config)
        self.assertIsNotNone(restored)
        values = restored.checkpoint.get("channel_values") or {}
        self.assertEqual(values.get("stop_reason"), "waiting_approval")
        self.assertEqual(values.get("pending_approval", {}).get("command"), ["git", "status"])

    def test_production_approval_resume_denial_uses_langgraph_checkpoint(self) -> None:
        request = self._request("Run git status")
        run_id = "run-production-deny"
        start_active_run(run_id=run_id, request=request, thread_id=request.thread_id)
        state = initial_graph_state(request, run_id=run_id)
        state["pending_approval"] = {"command": ["git", "status", "--short"], "approval_id": "cmd_test", "reason": "test"}
        state["stop_reason"] = "waiting_approval"
        checkpointer = EventServiceLangGraphSaver()
        compiled = build_compiled_repooperator_graph(checkpoint_adapter=checkpointer)
        compiled.invoke(state, config=graph_config_for_request(request, run_id))
        wait_for_approval(
            run_id,
            {
                "runtime": "langgraph",
                "request_snapshot": request_to_snapshot(request),
                "command": ["git", "status", "--short"],
                "approval_id": "cmd_test",
            },
        )

        final_payload = resume_approval(run_id, {"decision": "no_explain"})
        self.assertEqual(get_run(run_id)["status"], "completed")
        self.assertEqual(final_payload["stop_reason"], "approval_denied")
        self.assertIn("approval was denied", final_payload["response"])


if __name__ == "__main__":
    unittest.main()
