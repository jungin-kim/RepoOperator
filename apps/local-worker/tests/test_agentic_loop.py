"""Phase 1: model-driven native tool-calling planner."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from repooperator_worker.agent_core import agentic_loop
from repooperator_worker.agent_core.planner import TaskFrame
from repooperator_worker.services.model_tools import ToolCall, ToolCallResponse
from repooperator_worker.schemas import AgentRunRequest


def _settings(*, agentic=True, base_url="http://127.0.0.1:11434/v1", model="llama3", provider="ollama", api_key=None):
    return SimpleNamespace(
        agentic_tool_calling=agentic,
        openai_base_url=base_url,
        openai_model=model,
        openai_api_key=api_key,
        configured_model_provider=provider,
        configured_model_name=model,
        configured_model_connection_mode="local-runtime",
        model_request_timeout_seconds=30,
    )


def _state(**overrides):
    base = dict(
        context_packet={},
        files_read=[],
        max_file_reads=10,
        max_commands=5,
        loop_iteration=0,
        actions_taken=[],
        action_results=[],
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _request():
    return AgentRunRequest(project_path="/tmp/repo", task="Explain what read_file does")


def _frame():
    return TaskFrame(user_goal="explain", likely_needed_tools=["read_file"], likely_capabilities=[])


class FakeClient:
    def __init__(self, response):
        self._response = response
        self.calls = []

    def generate_with_tools(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class ToolCallingAvailabilityTests(unittest.TestCase):
    def test_disabled_when_flag_off(self) -> None:
        self.assertFalse(agentic_loop.tool_calling_available(_settings(agentic=False)))

    def test_disabled_when_no_endpoint(self) -> None:
        self.assertFalse(agentic_loop.tool_calling_available(_settings(base_url=None, provider="openai-compatible")))

    def test_enabled_when_configured(self) -> None:
        self.assertTrue(agentic_loop.tool_calling_available(_settings()))

    def test_anthropic_endpoint_uses_api_key(self) -> None:
        self.assertTrue(
            agentic_loop.tool_calling_available(
                _settings(provider="anthropic", base_url=None, api_key="k", model="claude-3.5")
            )
        )


class ActionMappingTests(unittest.TestCase):
    def test_tool_call_maps_to_action(self) -> None:
        response = ToolCallResponse(
            text="reading the file",
            tool_calls=(ToolCall(id="c1", name="read_file", arguments={"target_files": ["src/a.py"], "reason_summary": "look at a"}),),
        )
        client = FakeClient(response)
        action = agentic_loop.propose_next_action_with_tool_calling(
            _request(), _state(), _frame(), client_factory=lambda s: client, settings=_settings()
        )
        self.assertIsNotNone(action)
        self.assertEqual(action.type, "read_file")
        self.assertEqual(action.target_files, ["src/a.py"])
        self.assertEqual(action.reason_summary, "look at a")
        self.assertEqual(action.payload["target_files"], ["src/a.py"])
        # Tools were actually offered to the model.
        self.assertTrue(client.calls[0]["tools"])

    def test_path_alias_maps_to_target_files(self) -> None:
        response = ToolCallResponse(
            tool_calls=(ToolCall(id="c1", name="delete_file", arguments={"path": "/x/y.py", "justification": "dead"}),),
        )
        # An edit-shaped request: the read-only inverse gate must not block the
        # edit action whose alias mapping this test exercises.
        edit_request = AgentRunRequest(project_path="/tmp/repo", task="delete the dead file y.py")
        edit_frame = TaskFrame(user_goal="delete the dead file y.py", likely_needed_tools=["delete_file"], likely_capabilities=[])
        action = agentic_loop.propose_next_action_with_tool_calling(
            edit_request, _state(), edit_frame, client_factory=lambda s: FakeClient(response), settings=_settings()
        )
        self.assertEqual(action.type, "delete_file")
        self.assertEqual(action.target_files, ["x/y.py"])  # leading slash stripped

    def test_unknown_tool_returns_none(self) -> None:
        response = ToolCallResponse(tool_calls=(ToolCall(id="c1", name="not_a_real_tool", arguments={}),))
        action = agentic_loop.propose_next_action_with_tool_calling(
            _request(), _state(), _frame(), client_factory=lambda s: FakeClient(response), settings=_settings()
        )
        self.assertIsNone(action)

    def test_text_only_response_becomes_final_answer(self) -> None:
        # Text-only reply becomes final_answer once evidence exists.
        response = ToolCallResponse(text="It reads a repository file.")
        state = _state(files_read=["a.py"])
        action = agentic_loop.propose_next_action_with_tool_calling(
            _request(), state, _frame(), client_factory=lambda s: FakeClient(response), settings=_settings()
        )
        self.assertEqual(action.type, "final_answer")
        self.assertEqual(action.payload["model_answer"], "It reads a repository file.")

    def test_final_answer_blocked_without_evidence(self) -> None:
        # Model tries to answer immediately with no files read -> must defer.
        response = ToolCallResponse(text="This is a local-first coding agent.")
        action = agentic_loop.propose_next_action_with_tool_calling(
            _request(), _state(), _frame(), client_factory=lambda s: FakeClient(response), settings=_settings()
        )
        self.assertIsNone(action)

    def test_final_answer_allowed_with_evidence(self) -> None:
        response = ToolCallResponse(text="It reads repository files and answers questions.")
        state = _state(files_read=["README.md"])
        action = agentic_loop.propose_next_action_with_tool_calling(
            _request(), state, _frame(), client_factory=lambda s: FakeClient(response), settings=_settings()
        )
        self.assertIsNotNone(action)
        self.assertEqual(action.type, "final_answer")

    def test_tool_call_allowed_without_evidence(self) -> None:
        # A tool call (evidence gathering) is always fine even with no evidence yet.
        response = ToolCallResponse(tool_calls=(ToolCall(id="c1", name="inspect_repo_tree", arguments={}),))
        action = agentic_loop.propose_next_action_with_tool_calling(
            _request(), _state(), _frame(), client_factory=lambda s: FakeClient(response), settings=_settings()
        )
        self.assertEqual(action.type, "inspect_repo_tree")

    def test_empty_response_returns_none(self) -> None:
        response = ToolCallResponse()
        action = agentic_loop.propose_next_action_with_tool_calling(
            _request(), _state(), _frame(), client_factory=lambda s: FakeClient(response), settings=_settings()
        )
        self.assertIsNone(action)

    def test_disabled_flag_returns_none_without_calling_model(self) -> None:
        called = {"n": 0}

        def factory(_s):
            called["n"] += 1
            return FakeClient(ToolCallResponse(text="x"))

        action = agentic_loop.propose_next_action_with_tool_calling(
            _request(), _state(), _frame(), client_factory=factory, settings=_settings(agentic=False)
        )
        self.assertIsNone(action)
        self.assertEqual(called["n"], 0)

    def test_model_exception_falls_back_to_none(self) -> None:
        class Boom:
            def generate_with_tools(self, **kwargs):
                raise RuntimeError("network down")

        action = agentic_loop.propose_next_action_with_tool_calling(
            _request(), _state(), _frame(), client_factory=lambda s: Boom(), settings=_settings()
        )
        self.assertIsNone(action)


class TranscriptTests(unittest.TestCase):
    def test_history_becomes_tool_call_transcript(self) -> None:
        from repooperator_worker.agent_core.actions import AgentAction, ActionResult

        action = AgentAction(type="read_file", reason_summary="r", target_files=["a.py"])
        result = ActionResult(action_id=action.action_id, status="success", observation="file body", files_read=["a.py"])
        state = _state(actions_taken=[action], action_results=[result], files_read=["a.py"])
        messages = agentic_loop._build_transcript(_request(), state, _frame())
        # user turn + assistant tool_call + tool result
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[1]["tool_calls"][0]["function"]["name"], "read_file")
        self.assertEqual(messages[2]["role"], "tool")
        self.assertIn("file body", messages[2]["content"])
        self.assertIn("a.py", messages[2]["content"])


if __name__ == "__main__":
    unittest.main()


def _read(path):
    from repooperator_worker.agent_core.actions import AgentAction, ActionResult

    action = AgentAction(type="read_file", reason_summary="r", target_files=[path])
    result = ActionResult(action_id=action.action_id, status="success", observation=f"{path} body", files_read=[path])
    return action, result


class CacheFriendlyTranscriptTests(unittest.TestCase):
    """Each step must only APPEND to the previous request so prompt caches hit."""

    def setUp(self) -> None:
        agentic_loop._RUN_STABLE_PAYLOAD.clear()
        agentic_loop._RUN_TOOL_NAMES.clear()
        agentic_loop._RUN_ACTION_QUEUE.clear()

    def test_step_n_plus_one_extends_step_n(self) -> None:
        a1, r1 = _read("a.py")
        a2, r2 = _read("b.py")
        step1 = agentic_loop._build_transcript(
            _request(), _state(run_id="cache-1", actions_taken=[a1], action_results=[r1], files_read=["a.py"], loop_iteration=1), _frame()
        )
        # The task frame drifts as evidence arrives; the opening turn must not.
        drifted = TaskFrame(user_goal="explain", likely_needed_tools=["read_file", "search_text"], likely_capabilities=["x"])
        step2 = agentic_loop._build_transcript(
            _request(),
            _state(run_id="cache-1", actions_taken=[a1, a2], action_results=[r1, r2], files_read=["a.py", "b.py"], loop_iteration=2),
            drifted,
        )
        # Everything except the trailing run status is a strict prefix.
        self.assertEqual(step2[: len(step1) - 1], step1[:-1])
        self.assertTrue(step1[-1]["content"].startswith("[run status]"))
        self.assertIn("b.py", step2[-1]["content"])
        self.assertNotIn("files_read", step1[0]["content"])

    def test_window_slides_in_blocks(self) -> None:
        limit = agentic_loop.MAX_TRANSCRIPT_ACTIONS
        block = agentic_loop.TRANSCRIPT_WINDOW_BLOCK
        self.assertEqual(agentic_loop._transcript_window_start(limit), 0)
        first = agentic_loop._transcript_window_start(limit + 1)
        self.assertEqual(first, block)
        # Same start for a whole block of steps -> same prefix -> cache hits.
        for extra in range(1, block + 1):
            self.assertEqual(agentic_loop._transcript_window_start(limit + extra), first)
        self.assertEqual(agentic_loop._transcript_window_start(limit + block + 1), 2 * block)

    def test_tool_list_is_fixed_for_the_run(self) -> None:
        client = FakeClient(ToolCallResponse(tool_calls=(ToolCall(id="c1", name="read_file", arguments={"target_files": ["a.py"]}),)))
        state = _state(run_id="tools-1")
        agentic_loop.propose_next_action_with_tool_calling(_request(), state, _frame(), client_factory=lambda s: client, settings=_settings())
        narrower = TaskFrame(user_goal="explain", likely_needed_tools=[], likely_capabilities=[])
        agentic_loop.propose_next_action_with_tool_calling(_request(), _state(run_id="tools-1", files_read=["a.py"]), narrower, client_factory=lambda s: client, settings=_settings())
        names = [[t["name"] for t in call["tools"]] for call in client.calls]
        self.assertEqual(names[0], names[1][: len(names[0])])


class BatchedToolCallTests(unittest.TestCase):
    def setUp(self) -> None:
        agentic_loop._RUN_STABLE_PAYLOAD.clear()
        agentic_loop._RUN_TOOL_NAMES.clear()
        agentic_loop._RUN_ACTION_QUEUE.clear()

    def _three_reads(self):
        return ToolCallResponse(
            tool_calls=(
                ToolCall(id="c1", name="read_file", arguments={"target_files": ["a.py"]}),
                ToolCall(id="c2", name="read_file", arguments={"target_files": ["b.py"]}),
                ToolCall(id="c3", name="read_file", arguments={"target_files": ["c.py"]}),
            )
        )

    def _propose(self, client, state):
        return agentic_loop.propose_next_action_with_tool_calling(
            _request(), state, _frame(), client_factory=lambda s: client, settings=_settings()
        )

    def test_parallel_reads_cost_one_model_call(self) -> None:
        from repooperator_worker.agent_core.actions import ActionResult

        client = FakeClient(self._three_reads())
        taken, results = [], []
        seen = []
        for _ in range(3):
            action = self._propose(client, _state(run_id="batch-1", actions_taken=list(taken), action_results=list(results)))
            seen.append(action.target_files[0])
            taken.append(action)
            results.append(ActionResult(action_id=action.action_id, status="success", observation="ok", files_read=action.target_files))
        self.assertEqual(seen, ["a.py", "b.py", "c.py"])
        self.assertEqual(len(client.calls), 1)

    def test_queue_is_dropped_when_something_else_ran(self) -> None:
        from repooperator_worker.agent_core.actions import AgentAction, ActionResult

        client = FakeClient(self._three_reads())
        first = self._propose(client, _state(run_id="batch-2"))
        other = AgentAction(type="search_text", reason_summary="deterministic", payload={"query": "x"})
        state = _state(
            run_id="batch-2",
            actions_taken=[first, other],
            action_results=[
                ActionResult(action_id=first.action_id, status="success", observation="ok"),
                ActionResult(action_id=other.action_id, status="success", observation="ok"),
            ],
            files_read=["a.py"],
        )
        self._propose(client, state)
        self.assertEqual(len(client.calls), 2, "stale batch must not run after an unrelated step")

    def test_mutating_calls_are_never_batched(self) -> None:
        response = ToolCallResponse(
            tool_calls=(
                ToolCall(id="c1", name="read_file", arguments={"target_files": ["a.py"]}),
                ToolCall(id="c2", name="delete_file", arguments={"path": "a.py"}),
            )
        )
        client = FakeClient(response)
        first = self._propose(client, _state(run_id="batch-3"))
        self.assertEqual(first.type, "read_file")
        self.assertNotIn("batch-3", agentic_loop._RUN_ACTION_QUEUE)


class SequencedClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def generate_with_tools(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0) if self._responses else ToolCallResponse(text="done")


class GateFeedbackTests(unittest.TestCase):
    """Policy violations go back to the model as a failed call, not a silent veto."""

    def setUp(self) -> None:
        agentic_loop._RUN_STABLE_PAYLOAD.clear()
        agentic_loop._RUN_TOOL_NAMES.clear()
        agentic_loop._RUN_ACTION_QUEUE.clear()

    def test_model_corrects_itself_after_feedback(self) -> None:
        client = SequencedClient(
            [
                ToolCallResponse(text="This is a local-first coding agent."),  # answers with no evidence
                ToolCallResponse(tool_calls=(ToolCall(id="c2", name="inspect_repo_tree", arguments={}),)),
            ]
        )
        action = agentic_loop.propose_next_action_with_tool_calling(
            _request(), _state(), _frame(), client_factory=lambda s: client, settings=_settings()
        )
        self.assertEqual(action.type, "inspect_repo_tree")
        self.assertEqual(len(client.calls), 2)
        retry = client.calls[1]["messages"]
        feedback = retry[-1]["content"]
        self.assertIn('"ok": false', feedback)
        self.assertIn("no_evidence_yet", feedback)
        # The retry extends the first request, so it reuses its cached prefix.
        self.assertEqual(retry[: len(client.calls[0]["messages"])], client.calls[0]["messages"])

    def test_rejected_tool_call_gets_a_matching_tool_result(self) -> None:
        client = SequencedClient(
            [
                ToolCallResponse(tool_calls=(ToolCall(id="bad1", name="delete_file", arguments={"path": "a.py"}),)),
                ToolCallResponse(text="read_file returns the file contents."),
            ]
        )
        action = agentic_loop.propose_next_action_with_tool_calling(
            _request(), _state(files_read=["a.py"]), _frame(), client_factory=lambda s: client, settings=_settings()
        )
        self.assertEqual(action.type, "final_answer")
        retry = client.calls[1]["messages"]
        self.assertEqual(retry[-2]["tool_calls"][0]["id"], "bad1")
        self.assertEqual(retry[-1], {"role": "tool", "tool_call_id": "bad1", "content": retry[-1]["content"]})
        self.assertIn("read_only_request", retry[-1]["content"])

    def test_repeated_violation_falls_back_to_deterministic(self) -> None:
        client = SequencedClient([ToolCallResponse(text="answer"), ToolCallResponse(text="answer again")])
        action = agentic_loop.propose_next_action_with_tool_calling(
            _request(), _state(), _frame(), client_factory=lambda s: client, settings=_settings()
        )
        self.assertIsNone(action)
        self.assertEqual(len(client.calls), 1 + agentic_loop.MAX_GATE_FEEDBACK_RETRIES)


class PromptFileTests(unittest.TestCase):
    def test_prompt_is_read_from_file_and_reloaded(self) -> None:
        import os
        import tempfile
        from pathlib import Path
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "agent.md"
            path.write_text("v1", encoding="utf-8")
            with mock.patch.object(agentic_loop, "_prompt_path", return_value=path):
                self.assertEqual(agentic_loop.agent_system_prompt(), "v1")
                path.write_text("v2", encoding="utf-8")
                os.utime(path, (path.stat().st_atime, path.stat().st_mtime + 5))
                self.assertEqual(agentic_loop.agent_system_prompt(), "v2")
        agentic_loop._PROMPT_CACHE.update(path=None, mtime=None, text="")

    def test_bundled_prompt_exists_and_back_compat_constant(self) -> None:
        self.assertTrue(agentic_loop._BUNDLED_PROMPT_PATH.is_file())
        self.assertIn("RepoOperator", agentic_loop.AGENTIC_SYSTEM_PROMPT)
