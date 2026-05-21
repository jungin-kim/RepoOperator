import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repooperator_worker.agent_core.context_budget import ContextBudget, compact_file_contents, estimate_chars  # noqa: E402
from repooperator_worker.agent_core.final_synthesis import _answer_with_model  # noqa: E402
from repooperator_worker.agent_core.state import AgentCoreState  # noqa: E402
from repooperator_worker.schemas import AgentRunRequest  # noqa: E402


class _PromptCapturingClient:
    def __init__(self) -> None:
        self.user_prompt = ""

    def stream_text(self, request):
        self.user_prompt = request.user_prompt
        yield {"type": "assistant_delta", "delta": "README.md and main.py provide the evidence."}

    def generate_text(self, request):
        self.user_prompt = request.user_prompt
        return "README.md and main.py provide the evidence."


class ContextBudgetTests(unittest.TestCase):
    def test_large_file_contents_are_compacted(self) -> None:
        compacted = compact_file_contents(
            {"README.md": "# Demo\n" + "a" * 200, "large.py": "def huge():\n" + "x" * 1000},
            ContextBudget(max_chars=300, reserved_final_answer_chars=50, max_file_chars=180),
        )
        self.assertTrue(compacted.compacted)
        self.assertIn("README.md", compacted.included_files)
        self.assertTrue(compacted.omitted_files)
        json.dumps(compacted.model_dump(), ensure_ascii=False)

    def test_explicit_files_are_preserved_before_non_explicit(self) -> None:
        compacted = compact_file_contents(
            {"noise.py": "n" * 400, "target.py": "t" * 120},
            ContextBudget(max_chars=220, reserved_final_answer_chars=20, max_file_chars=200),
            explicit_files=["target.py"],
        )
        self.assertIn("target.py", compacted.included_files)
        self.assertGreaterEqual(estimate_chars(compacted.included_files["target.py"]), 120)

    def test_final_synthesis_prompt_receives_compacted_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            request = AgentRunRequest(project_path=str(repo), git_provider="local", branch="main", task="Explain the project.")
            state = AgentCoreState(run_id="run-context-budget", thread_id="thread", repo=str(repo), branch="main", user_task=request.task)
            state.files_read = ["main.py", "README.md"]
            noise = "NOISE" * 40_000
            client = _PromptCapturingClient()
            with patch("repooperator_worker.agent_core.final_synthesis._compat_model_client", return_value=lambda: client):
                answer = _answer_with_model(
                    request,
                    {
                        "README.md": "# Demo\n" + "readme evidence\n" * 5000,
                        "main.py": "def main():\n    return 'entrypoint'\n",
                        "noise.log": noise,
                    },
                    state=state,
                )
            prompt_payload = json.loads(client.user_prompt)
            self.assertIn("context_compaction", prompt_payload)
            self.assertTrue(prompt_payload["context_compaction"]["compacted"])
            prompt_text = json.dumps(prompt_payload, ensure_ascii=False)
            self.assertIn("main.py", prompt_text)
            self.assertIn("README.md", prompt_text)
            self.assertNotIn(noise, prompt_text)
            self.assertIn("main.py", answer)


if __name__ == "__main__":
    unittest.main()
