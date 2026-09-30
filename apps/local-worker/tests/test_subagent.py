"""Phase 3: model-backed worker subagents."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from repooperator_worker.agent_core import subagent
from repooperator_worker.agent_core.actions import ActionResult
from repooperator_worker.schemas import AgentRunRequest
from repooperator_worker.services.model_tools import ToolCall, ToolCallResponse


def _settings(agentic=True):
    return SimpleNamespace(
        agentic_tool_calling=agentic,
        openai_base_url="http://127.0.0.1:11434/v1",
        openai_model="llama3",
        openai_api_key=None,
        configured_model_provider="ollama",
        configured_model_name="llama3",
        configured_model_connection_mode="local-runtime",
        model_request_timeout_seconds=30,
    )


def _request():
    return AgentRunRequest(project_path="/tmp/repo", task="Summarize the codebase")


class SequencedClient:
    """Returns queued responses in order; records calls."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def generate_with_tools(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0) if self._responses else ToolCallResponse(text="done")


class FakeOrchestrator:
    def __init__(self):
        self.executed = []

    def execute_action(self, action):
        self.executed.append(action)
        return ActionResult(action_id=action.action_id, status="success", observation="read it", files_read=["src/app.py"])


class SubagentTests(unittest.TestCase):
    def test_parallel_calls_run_in_one_step(self) -> None:
        responses = [
            ToolCallResponse(
                tool_calls=(
                    ToolCall(id="c1", name="read_file", arguments={"target_files": ["a.py"]}),
                    ToolCall(id="c2", name="read_file", arguments={"target_files": ["b.py"]}),
                )
            ),
            ToolCallResponse(text="a and b are the entrypoints."),
        ]
        client = SequencedClient(responses)
        orch = FakeOrchestrator()
        subagent.run_worker_subagent(
            {"role": "AnalysisAgent", "files": ["a.py", "b.py"]},
            request=_request(),
            run_id="r1",
            settings=_settings(),
            client_factory=lambda s: client,
            orchestrator=orch,
        )
        self.assertEqual([a.target_files for a in orch.executed], [["a.py"], ["b.py"]])
        self.assertEqual(len(client.calls), 2, "both reads in one step, then the report")
        second = client.calls[1]["messages"]
        self.assertEqual(len(second[1]["tool_calls"]), 2)
        self.assertEqual([m["tool_call_id"] for m in second[2:4]], ["c1", "c2"])

    def test_disabled_returns_none(self) -> None:
        report = subagent.run_worker_subagent(
            {"role": "AnalysisAgent", "files": ["a.py"]},
            request=_request(),
            run_id="r1",
            settings=_settings(agentic=False),
            client_factory=lambda s: SequencedClient([]),
            orchestrator=FakeOrchestrator(),
        )
        self.assertIsNone(report)

    def test_tool_call_then_report(self) -> None:
        responses = [
            ToolCallResponse(tool_calls=(ToolCall(id="c1", name="read_file", arguments={"target_files": ["src/app.py"]}),)),
            ToolCallResponse(text="app.py defines the entrypoint and wires routes."),
        ]
        client = SequencedClient(responses)
        orch = FakeOrchestrator()
        report = subagent.run_worker_subagent(
            {"role": "AnalysisAgent", "group": "app", "scope": "app", "files": ["src/app.py"], "task_id": "AnalysisAgent:app"},
            request=_request(),
            run_id="r1",
            settings=_settings(),
            client_factory=lambda s: client,
            orchestrator=orch,
        )
        self.assertIsNotNone(report)
        self.assertTrue(report["subagent"])
        self.assertEqual(report["worker"], "AnalysisAgent")
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["files_analyzed"], ["src/app.py"])
        self.assertIn("entrypoint", report["summary"])
        # The read_file tool was actually executed through the orchestrator.
        self.assertEqual(orch.executed[0].type, "read_file")
        self.assertEqual(orch.executed[0].target_files, ["src/app.py"])

    def test_immediate_text_report_without_tools(self) -> None:
        client = SequencedClient([ToolCallResponse(text="Nothing to inspect; scope is trivial.")])
        report = subagent.run_worker_subagent(
            {"role": "AnalysisAgent", "files": ["a.py"]},
            request=_request(),
            run_id="r1",
            settings=_settings(),
            client_factory=lambda s: client,
            orchestrator=FakeOrchestrator(),
        )
        self.assertEqual(report["summary"], "Nothing to inspect; scope is trivial.")
        self.assertEqual(report["files_analyzed"], ["a.py"])  # falls back to scope files

    def test_edit_planning_role_carries_risk_note(self) -> None:
        client = SequencedClient([ToolCallResponse(text="Candidate edit files identified.")])
        report = subagent.run_worker_subagent(
            {"role": "EditPlanningAgent", "files": ["src/x.py"]},
            request=_request(),
            run_id="r1",
            settings=_settings(),
            client_factory=lambda s: client,
            orchestrator=FakeOrchestrator(),
        )
        self.assertIn("risk_notes", report)
        self.assertTrue(report["risk_notes"])

    def test_stops_after_max_steps(self) -> None:
        # Always returns a tool call; loop must stop after MAX_SUBAGENT_STEPS.
        forever = [
            ToolCallResponse(tool_calls=(ToolCall(id=f"c{i}", name="read_file", arguments={"target_files": ["a.py"]}),))
            for i in range(subagent.MAX_SUBAGENT_STEPS + 3)
        ]
        client = SequencedClient(forever)
        orch = FakeOrchestrator()
        report = subagent.run_worker_subagent(
            {"role": "AnalysisAgent", "files": ["a.py"]},
            request=_request(),
            run_id="r1",
            settings=_settings(),
            client_factory=lambda s: client,
            orchestrator=orch,
        )
        self.assertIsNotNone(report)
        self.assertLessEqual(len(orch.executed), subagent.MAX_SUBAGENT_STEPS)


if __name__ == "__main__":
    unittest.main()
