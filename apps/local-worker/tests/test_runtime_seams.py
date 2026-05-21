import json
import sys
import unittest
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repooperator_worker.agent_core.memory import MemoryType, NoOpMemoryStore, new_memory_record  # noqa: E402
from repooperator_worker.agent_core.tasks import InMemoryTaskManager, TaskStatus, TaskType  # noqa: E402


class RuntimeSeamTests(unittest.TestCase):
    def test_task_manager_status_transitions_are_json_safe(self) -> None:
        manager = InMemoryTaskManager()
        task = manager.create_task(TaskType.LOCAL_AGENT_REVIEW, "Review repository", run_id="run-task", repo="demo")
        self.assertEqual(task.status, TaskStatus.PENDING)
        self.assertEqual(task.owner_run_id, "run-task")
        updated = manager.update_task(task.task_id, status=TaskStatus.RUNNING, progress_summary="Reading files")
        self.assertEqual(updated.status, TaskStatus.RUNNING)
        self.assertEqual(manager.get_task(task.task_id), updated)
        self.assertEqual(manager.list_tasks(), [updated])
        cancelled = manager.cancel_task(task.task_id)
        self.assertEqual(cancelled.status, TaskStatus.CANCELLED)
        self.assertIs(manager.cancel_task(task.task_id), cancelled)
        json.dumps(cancelled.model_dump(), ensure_ascii=False)

    def test_background_shell_task_creation_is_not_enabled(self) -> None:
        manager = InMemoryTaskManager()
        with self.assertRaises(ValueError):
            manager.create_task(TaskType.LOCAL_BASH, "Run shell", run_id="run-task")

    def test_noop_memory_store_is_json_safe_and_does_not_persist(self) -> None:
        store = NoOpMemoryStore()
        record = new_memory_record(
            id="mem_test",
            type=MemoryType.FEEDBACK,
            title="Test feedback",
            content="Prefer concise responses.",
            why="User preference",
            how_to_apply="Use short summaries.",
            source="unit-test",
        )
        self.assertEqual(store.load_project_memory("/repo"), [])
        self.assertEqual(store.search_memory("concise"), [])
        self.assertEqual(store.save_memory(record), record)
        packet = store.memory_context_packet("/repo", "concise")
        self.assertFalse(packet["enabled"])
        json.dumps(record.model_dump(), ensure_ascii=False)
        json.dumps(packet, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
