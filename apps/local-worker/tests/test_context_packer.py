import json
import sys
import unittest
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repooperator_worker.agent_core.context_packer import pack_context  # noqa: E402
from repooperator_worker.agent_core.model_profile import ModelProfile  # noqa: E402
from repooperator_worker.schemas import AgentRunRequest  # noqa: E402


SMALL = ModelProfile(
    provider="test",
    model_name="tiny",
    context_window=16_000,
    max_output_tokens=2_000,
    supports_streaming=True,
    supports_tool_calls=False,
    supports_json_schema=False,
    supports_reasoning_signal=False,
    tokenizer_hint="unknown",
    compression_strategy="aggressive",
)
LARGE = ModelProfile(
    provider="test",
    model_name="large",
    context_window=300_000,
    max_output_tokens=8_000,
    supports_streaming=True,
    supports_tool_calls=False,
    supports_json_schema=False,
    supports_reasoning_signal=False,
    tokenizer_hint="unknown",
    compression_strategy="generous",
)


class ContextPackerTests(unittest.TestCase):
    def _request(self, task: str = "Fix app.py") -> AgentRunRequest:
        return AgentRunRequest(project_path="/tmp/repo", git_provider="local", branch="main", task=task)

    def test_small_model_compacts_more_than_large_model(self) -> None:
        content = "line\n" * 20_000
        state = {"evidence_store": {"contents": {"app.py": content, "noise.py": content}}}
        small = pack_context("summary_context", self._request(), state=state, profile=SMALL)
        large = pack_context("summary_context", self._request(), state=state, profile=LARGE)
        self.assertLess(small["compression"]["included_chars"], large["compression"]["included_chars"])
        self.assertTrue(small["compression"]["compacted"])

    def test_edit_and_repair_context_preserve_change_set_and_errors(self) -> None:
        state = {
            "files_read": ["app.py"],
            "evidence_store": {"contents": {"app.py": "def main():\n    return 1\n"}},
            "change_set_proposal": {
                "proposal_id": "p1",
                "status": "invalid",
                "changes": [{"path": "app.py", "operation": "modify", "summary": "Return two."}],
                "validation": {"status": "invalid", "errors": ["syntax error"]},
            },
            "validation_results": [{"kind": "change_set", "status": "invalid", "errors": ["syntax error"]}],
        }
        packet = pack_context("repair_context", self._request(), state=state, profile=SMALL)
        self.assertEqual(packet["current_user_request"], "Fix app.py")
        self.assertEqual(packet["active_change_set"]["proposal_id"], "p1")
        self.assertIn("syntax error", packet["validation_errors"])
        self.assertEqual(packet["previous_proposal"]["proposal_id"], "p1")
        json.dumps(packet, ensure_ascii=False)

    def test_summary_context_excludes_noisy_raw_files_but_keeps_summaries(self) -> None:
        state = {"evidence_store": {"contents": {"noise.log": "x" * 60_000}}}
        packet = pack_context("summary_context", self._request("Summarize project"), state=state, profile=SMALL)
        self.assertIn("noise.log", packet["file_evidence"]["summaries"])
        self.assertTrue(packet["file_evidence"]["omitted_files"])


if __name__ == "__main__":
    unittest.main()
