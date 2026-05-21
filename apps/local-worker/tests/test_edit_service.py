import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repooperator_worker.schemas import AgentProposeFileRequest  # noqa: E402
from repooperator_worker.services.edit_service import propose_file_edit  # noqa: E402


class _RepairClient:
    calls = 0

    @property
    def model_name(self) -> str:
        return "test-model"

    def generate_text(self, _request):
        self.__class__.calls += 1
        if self.__class__.calls == 1:
            return json.dumps({"response": "not an edit"})
        return json.dumps(
            {
                "edits": [
                    {
                        "path": "app.py",
                        "replacement": "def main():\n    return 'ok'\n",
                        "summary": "Return a stable value.",
                    }
                ],
                "overall_summary": "Updated app behavior.",
                "tests": ["python app.py"],
            }
        )


class EditServiceTests(unittest.TestCase):
    def test_invalid_structured_output_retries_with_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp_home:
            repo = Path(temp_home) / "repo"
            repo.mkdir()
            subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
            (repo / "app.py").write_text("def main():\n    return None\n", encoding="utf-8")
            config_path = Path(temp_home) / ".repooperator" / "config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                json.dumps(
                    {
                        "permissions": {"mode": "auto_review"},
                        "model": {
                            "connectionMode": "local-runtime",
                            "provider": "vllm",
                            "baseUrl": "http://127.0.0.1:8001/v1",
                            "model": "test-model",
                        },
                    }
                ),
                encoding="utf-8",
            )
            _RepairClient.calls = 0
            with patch.dict(os.environ, {"REPOOPERATOR_CONFIG_PATH": str(config_path)}, clear=True), patch(
                "repooperator_worker.services.edit_service.OpenAICompatibleModelClient",
                return_value=_RepairClient(),
            ):
                proposal = propose_file_edit(
                    AgentProposeFileRequest(
                        project_path=str(repo),
                        relative_path="app.py",
                        instruction="Improve the return value.",
                    )
                )
        self.assertEqual(_RepairClient.calls, 2)
        self.assertIn("return 'ok'", proposal.proposed_content)


if __name__ == "__main__":
    unittest.main()
