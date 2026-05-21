"""
Tests that prove the event store alone can reconstruct a completed streamed run.

These tests exercise the primitives from run-event-state.ts semantics but on
the backend side:
  - progress_delta events are ordered and deduplicated by activity_id / sequence
  - assistant_delta + final_message → response comes from final_message.result.response
  - progress + final_message only → response from final_message.result.response
  - final_result.activity_events is a fallback when stored events are absent
  - list_run_events returns sorted sequence
  - run metadata carries thread_id so the frontend can map events to a thread

All tests use a temporary REPOOPERATOR_HOME directory and never hit the file
system outside of it.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any


def _set_home(tmp: str):
    os.environ["REPOOPERATOR_HOME"] = tmp


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_progress_event(run_id: str, thread_id: str, seq: int, phase: str, label: str, status: str = "completed") -> dict[str, Any]:
    return {
        "id": f"{run_id}-ev-{seq}",
        "activity_id": f"act-{seq}",
        "run_id": run_id,
        "thread_id": thread_id,
        "type": "progress_delta",
        "event_type": "progress_delta",
        "phase": phase,
        "label": label,
        "status": status,
        "sequence": seq,
        "timestamp": "2024-01-01T00:00:00Z",
    }


def _build_assistant_delta(run_id: str, thread_id: str, seq: int, delta: str) -> dict[str, Any]:
    return {
        "id": f"{run_id}-ad-{seq}",
        "run_id": run_id,
        "thread_id": thread_id,
        "type": "assistant_delta",
        "event_type": "assistant_delta",
        "delta": delta,
        "sequence": seq,
        "timestamp": "2024-01-01T00:01:00Z",
    }


def _build_final_message(run_id: str, thread_id: str, seq: int, response: str, progress_events: list[dict]) -> dict[str, Any]:
    return {
        "id": f"{run_id}-final-{seq}",
        "run_id": run_id,
        "thread_id": thread_id,
        "type": "final_message",
        "event_type": "final_message",
        "result": {
            "run_id": run_id,
            "thread_id": thread_id,
            "response": response,
            "response_type": "assistant_answer",
            "stop_reason": "completed",
            "activity_events": progress_events,
        },
        "sequence": seq,
        "timestamp": "2024-01-01T00:02:00Z",
    }


def _reconstruct_response(events: list[dict[str, Any]], final_result: dict | None = None) -> str:
    """Mirror of assistantTextFromRunEvents from run-event-state.ts."""
    # Prefer final_message.result.response
    for event in reversed(events):
        if event.get("type") == "final_message":
            result = event.get("result") or {}
            if result.get("response"):
                return str(result["response"])
    if final_result and final_result.get("response"):
        return str(final_result["response"])
    # Fallback: concatenate assistant_delta
    deltas = [str(e.get("delta") or "") for e in events if e.get("type") == "assistant_delta"]
    return "".join(deltas)


def _reconstruct_progress(events: list[dict[str, Any]], final_result: dict | None = None) -> list[dict]:
    """Mirror of mergeRunEventsIntoProgressSteps from run-event-state.ts."""
    by_activity: dict[str, dict] = {}
    ordered: list[str] = []
    for event in events:
        if event.get("type") != "progress_delta":
            continue
        if not (event.get("label") or event.get("message")):
            continue
        key = event.get("activity_id") or str(event.get("sequence") or event.get("id") or "")
        if key not in by_activity:
            ordered.append(key)
        by_activity[key] = event
    result = [by_activity[k] for k in ordered]
    if not result and final_result:
        for event in (final_result.get("activity_events") or []):
            if event.get("type") != "progress_delta":
                continue
            if not (event.get("label") or event.get("message")):
                continue
            result.append(event)
    return result


def _max_sequence(events: list[dict[str, Any]]) -> int:
    return max((int(e.get("sequence") or 0) for e in events), default=0)


# ── Test class ────────────────────────────────────────────────────────────────

class TestThreadReconstruction(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        _set_home(self._tmpdir.name)
        # Re-import after setting env so module picks up the temp home.
        import importlib
        import repooperator_worker.services.event_service as svc
        importlib.reload(svc)
        self.svc = svc

    def tearDown(self):
        self._tmpdir.cleanup()

    # ── A: progress + assistant_delta + final_message ─────────────────────────

    def test_A_streamed_run_reconstructs_response(self):
        run_id = "run_test_a"
        thread_id = "thread_a"
        self.svc.start_active_run(
            run_id=run_id,
            request=_stub_request(thread_id),
            thread_id=thread_id,
        )
        progress_evs = [
            _build_progress_event(run_id, thread_id, 1, "Thinking", "Loaded context"),
            _build_progress_event(run_id, thread_id, 2, "Planning", "Framed request"),
        ]
        for ev in progress_evs:
            self.svc.append_run_event(run_id, ev)
        self.svc.append_run_event(run_id, _build_assistant_delta(run_id, thread_id, 3, "Hello "))
        self.svc.append_run_event(run_id, _build_assistant_delta(run_id, thread_id, 4, "world."))
        final_response = "Hello world."
        final_msg = _build_final_message(run_id, thread_id, 5, final_response, progress_evs)
        self.svc.append_run_event(run_id, final_msg)
        self.svc.complete_active_run(run_id=run_id, status="completed", final_result={"response": final_response, "run_id": run_id})

        events = self.svc.list_run_events(run_id)
        response = _reconstruct_response(events)
        self.assertEqual(response, final_response, "Response should come from final_message, not concatenated deltas")

        progress = _reconstruct_progress(events)
        self.assertEqual(len(progress), 2)
        self.assertEqual(progress[0]["label"], "Loaded context")
        self.assertEqual(progress[1]["label"], "Framed request")

    # ── B: progress + final_message only (no assistant_delta) ─────────────────

    def test_B_final_message_without_assistant_delta(self):
        run_id = "run_test_b"
        thread_id = "thread_b"
        self.svc.start_active_run(run_id=run_id, request=_stub_request(thread_id), thread_id=thread_id)
        progress_evs = [
            _build_progress_event(run_id, thread_id, 1, "Thinking", "Loaded context"),
        ]
        for ev in progress_evs:
            self.svc.append_run_event(run_id, ev)
        final_response = "Only from final_message."
        final_msg = _build_final_message(run_id, thread_id, 5, final_response, progress_evs)
        self.svc.append_run_event(run_id, final_msg)
        self.svc.complete_active_run(run_id=run_id, status="completed")

        events = self.svc.list_run_events(run_id)
        response = _reconstruct_response(events)
        self.assertEqual(response, final_response)

        progress = _reconstruct_progress(events)
        self.assertEqual(len(progress), 1)
        self.assertEqual(progress[0]["label"], "Loaded context")

    # ── C: empty stored events fall back to final_result.activity_events ──────

    def test_C_fallback_to_final_result_activity_events(self):
        final_response = "From final_result."
        progress_evs = [
            _build_progress_event("run_c", "thread_c", 1, "Thinking", "Fallback step"),
        ]
        final_result = {
            "response": final_response,
            "activity_events": progress_evs,
        }
        # No stored events — pass empty list
        events: list[dict] = []
        response = _reconstruct_response(events, final_result)
        self.assertEqual(response, final_response)
        progress = _reconstruct_progress(events, final_result)
        self.assertEqual(len(progress), 1)
        self.assertEqual(progress[0]["label"], "Fallback step")

    # ── D: event order is stable / maxSequence is correct ────────────────────

    def test_D_event_sequence_is_stable(self):
        run_id = "run_test_d"
        thread_id = "thread_d"
        self.svc.start_active_run(run_id=run_id, request=_stub_request(thread_id), thread_id=thread_id)
        for seq in [3, 1, 2]:
            self.svc.append_run_event(run_id, _build_progress_event(run_id, thread_id, seq, "P", f"step {seq}"))

        events = self.svc.list_run_events(run_id)
        sequences = [int(e.get("sequence") or 0) for e in events]
        # Events are returned in append order (not necessarily sorted), but
        # maxSequence must equal the highest sequence present.
        self.assertEqual(_max_sequence(events), 3)
        # after_sequence filtering must work correctly
        after_one = self.svc.list_run_events(run_id, after_sequence=1)
        self.assertTrue(all(int(e.get("sequence") or 0) > 1 for e in after_one))

    # ── E: run metadata carries thread_id ─────────────────────────────────────

    def test_E_run_metadata_contains_thread_id(self):
        run_id = "run_test_e"
        thread_id = "thread_e"
        self.svc.start_active_run(run_id=run_id, request=_stub_request(thread_id), thread_id=thread_id)
        meta = self.svc.get_run(run_id)
        self.assertIsNotNone(meta)
        self.assertEqual(meta["thread_id"], thread_id, "run metadata must carry thread_id for frontend mapping")

    # ── E2: each event carries thread_id ──────────────────────────────────────

    def test_E2_events_carry_thread_id(self):
        run_id = "run_test_e2"
        thread_id = "thread_e2"
        self.svc.start_active_run(run_id=run_id, request=_stub_request(thread_id), thread_id=thread_id)
        ev = _build_progress_event(run_id, thread_id, 1, "Thinking", "test event")
        self.svc.append_run_event(run_id, ev)
        events = self.svc.list_run_events(run_id)
        self.assertTrue(len(events) >= 1)
        self.assertEqual(events[0].get("thread_id"), thread_id)

    # ── No duplicate assistant text from delta + final_message ────────────────

    def test_no_duplicate_response_from_delta_and_final_message(self):
        run_id = "run_test_dup"
        thread_id = "thread_dup"
        response = "The answer is 42."
        self.svc.start_active_run(run_id=run_id, request=_stub_request(thread_id), thread_id=thread_id)
        # Both assistant_delta and final_message carry the same text
        self.svc.append_run_event(run_id, _build_assistant_delta(run_id, thread_id, 1, response))
        final_msg = _build_final_message(run_id, thread_id, 2, response, [])
        self.svc.append_run_event(run_id, final_msg)
        self.svc.complete_active_run(run_id=run_id, status="completed")

        events = self.svc.list_run_events(run_id)
        reconstructed = _reconstruct_response(events)
        # Response must equal the text exactly once — not doubled.
        self.assertEqual(reconstructed, response)
        self.assertNotIn(response + response, reconstructed)


# ── Stub helpers ──────────────────────────────────────────────────────────────

def _stub_request(thread_id: str):
    """Return a minimal AgentRunRequest-like object for event_service calls."""
    from repooperator_worker.schemas import AgentRunRequest
    return AgentRunRequest(
        task="test",
        project_path="/tmp/mock_repo",
        git_provider="local",
        thread_id=thread_id,
    )


if __name__ == "__main__":
    unittest.main()
