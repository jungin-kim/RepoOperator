import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch


TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repooperator_worker.services.routine_service import RoutineStore  # noqa: E402


class RoutineServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = RoutineStore(Path(self.tmp.name))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_routine_persists(self) -> None:
        routine = self.store.create({"name": "Nightly", "repo_identity": "/repo", "task_prompt": "Summarize", "trigger": {"type": "manual"}})
        loaded = RoutineStore(Path(self.tmp.name)).get(routine.id)
        self.assertEqual(loaded.name, "Nightly")
        self.assertEqual(loaded.permission_profile.profile, "routine_safe")
        self.assertTrue(loaded.requires_approval_for_writes)

    def test_disabled_routine_does_not_run(self) -> None:
        routine = self.store.create({"name": "Every minute", "repo_identity": "/repo", "task_prompt": "Summarize", "enabled": False, "trigger": {"type": "interval", "interval_seconds": 1}})
        payloads = self.store._read_definitions()
        payloads[0]["next_run_at"] = "2000-01-01T00:00:00Z"
        self.store._write_definitions(payloads)
        self.assertEqual(self.store.enqueue_due(datetime.now(timezone.utc)), [])
        self.assertEqual(self.store.get(routine.id).enabled, False)

    def test_due_routine_enqueues_agent_run(self) -> None:
        routine = self.store.create({"name": "Due", "repo_identity": "/repo", "branch": "main", "thread_id": "thread-1", "task_prompt": "Summarize", "trigger": {"type": "interval", "interval_seconds": 60}})
        payloads = self.store._read_definitions()
        payloads[0]["next_run_at"] = "2000-01-01T00:00:00Z"
        self.store._write_definitions(payloads)
        with patch("repooperator_worker.services.routine_service.enqueue_message", return_value={"id": "queue-1"}) as enqueue:
            runs = self.store.enqueue_due(datetime.now(timezone.utc))
        self.assertEqual(runs[0].queued_message_id, "queue-1")
        enqueue.assert_called_once_with("thread-1", "/repo", "main", "Summarize")
        self.assertEqual(self.store.list_runs(routine.id)[0].routine_id, routine.id)

    def test_routine_run_cancel_uses_normal_cancellation(self) -> None:
        routine = self.store.create({"name": "Manual", "repo_identity": "/repo", "task_prompt": "Summarize"})
        self.store._append_run(type("Run", (), {"model_dump": lambda self: {"id": "rr1", "routine_id": routine.id, "status": "running", "run_id": "run-1"}})())
        with patch("repooperator_worker.services.routine_service.cancel_run", return_value={"id": "run-1"}) as cancel:
            cancelled = self.store.cancel_routine_run("rr1")
        self.assertEqual(cancelled.status, "cancelled")
        cancel.assert_called_once_with("run-1")


if __name__ == "__main__":
    unittest.main()
