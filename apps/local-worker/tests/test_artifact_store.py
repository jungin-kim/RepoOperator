import json
import sys
import tempfile
import unittest
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repooperator_worker.agent_core.artifacts import ArtifactStore  # noqa: E402


class ArtifactStoreTests(unittest.TestCase):
    def test_large_payload_can_be_read_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ArtifactStore(base_dir=Path(tmp) / "artifacts")
            payload = {"content": "x" * 1000}
            record = store.write("run-artifact", "tool_result", payload)
            self.assertTrue(record.artifact_id)
            self.assertGreater(record.byte_size, 100)
            json.dumps(record.record_dump(), ensure_ascii=False)
            self.assertTrue(record.path)
            self.assertNotIn("path", record.record_dump())
            self.assertIn("path", record.internal_record_dump())
            self.assertEqual(store.read(record.artifact_id), payload)

    def test_secret_payload_is_redacted_before_storage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ArtifactStore(base_dir=Path(tmp) / "artifacts")
            token = "ghp_" + "A" * 36
            record = store.write("run-artifact", "tool_result", {"token": token})
            stored = store.read(record.artifact_id)
            self.assertTrue(record.redacted)
            self.assertNotIn(token, json.dumps(stored))
            self.assertIn("[REDACTED:github_token]", json.dumps(stored))


if __name__ == "__main__":
    unittest.main()
