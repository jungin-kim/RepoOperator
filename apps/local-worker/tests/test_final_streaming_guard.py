import json
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

from repooperator_worker.agent_core.langgraph_runtime import stream_langgraph_controller  # noqa: E402
from repooperator_worker.schemas import AgentRunRequest  # noqa: E402
from repooperator_worker.services.event_service import list_run_events  # noqa: E402


class _BadStreamingFinalClient:
    @property
    def model_name(self) -> str:
        return "test-model"

    def generate_text(self, request):
        if "intent classifier" in request.system_prompt.lower():
            return "{}"
        if "bounded next-action planner" in request.system_prompt:
            return "{}"
        return "I cannot read files because the files object is empty."

    def stream_text(self, request):
        yield {"type": "reasoning_delta", "delta": "hidden"}
        yield {"type": "assistant_delta", "delta": "I cannot read files "}
        yield {"type": "assistant_delta", "delta": "because the files object is empty."}


class FinalStreamingGuardTests(unittest.TestCase):
    def test_streaming_emits_only_repaired_final_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            (repo / "README.md").write_text("# Demo\n\nA small documented project.\n", encoding="utf-8")
            request = AgentRunRequest(project_path=str(repo), git_provider="local", branch="main", task="README.md 설명해줘.")
            config = Path(tmp) / "config.json"
            config.write_text(json.dumps({"repooperatorHomeDir": str(Path(tmp) / ".repooperator")}), encoding="utf-8")
            with patch.dict(os.environ, {"REPOOPERATOR_CONFIG_PATH": str(config)}, clear=False), patch(
                "repooperator_worker.agent_core.graph.support.OpenAICompatibleModelClient",
                return_value=_BadStreamingFinalClient(),
            ), patch(
                "repooperator_worker.agent_core.graph.repository_support.get_active_repository",
                return_value=None,
            ):
                events = list(stream_langgraph_controller(request, run_id="run-final-guard"))
                stored_events = list_run_events("run-final-guard")
                stored_text = json.dumps(stored_events, ensure_ascii=False)
        assistant_text = "".join(str(event.get("delta") or "") for event in events if event.get("type") == "assistant_delta")
        final = next(event for event in events if event.get("type") == "final_message")
        self.assertNotIn("cannot read", assistant_text.lower())
        self.assertNotIn("files object is empty", assistant_text.lower())
        self.assertNotIn("hidden", assistant_text)
        self.assertNotIn("cannot read", stored_text.lower())
        self.assertTrue(any(event.get("type") == "assistant_delta" for event in stored_events))
        repair_events = [event for event in stored_events if event.get("activity_id") == "final-synthesis-repair"]
        self.assertTrue(repair_events)
        self.assertNotIn("files object is empty", json.dumps(repair_events, ensure_ascii=False).lower())
        self.assertIn("README.md", final["result"]["response"])
        self.assertEqual(assistant_text, final["result"]["response"])


if __name__ == "__main__":
    unittest.main()
