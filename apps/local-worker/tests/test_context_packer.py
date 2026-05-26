import json
import sys
import unittest
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repooperator_worker.agent_core.context_packer import pack_context  # noqa: E402
from repooperator_worker.agent_core.actions import ActionResult  # noqa: E402
from repooperator_worker.agent_core.final_synthesis import validate_or_repair_final_answer  # noqa: E402
from repooperator_worker.agent_core.model_profile import ModelProfile  # noqa: E402
from repooperator_worker.agent_core.state import AgentCoreState  # noqa: E402
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

    def test_active_approval_and_change_set_are_never_dropped(self) -> None:
        huge_original = "old line\n" * 5_000
        huge_proposed = "new line\n" * 5_000
        proposal = {
            "proposal_id": "proposal-active",
            "status": "valid",
            "plan": {"summary": "Update app", "target_files": ["app.py"]},
            "changes": [
                {
                    "path": "app.py",
                    "operation": "modify",
                    "summary": "Change return value.",
                    "original_content": huge_original,
                    "proposed_content": huge_proposed,
                }
            ],
        }
        state = {
            "change_set_proposal": proposal,
            "pending_approval": {
                "kind": "change_set_apply",
                "proposal_id": "proposal-active",
                "change_set_proposal": proposal,
                "reason": "Apply validated change set.",
            },
        }
        packet = pack_context("edit", self._request(), state=state, profile=SMALL)
        self.assertEqual(packet["active_change_set"]["proposal_id"], "proposal-active")
        self.assertEqual(packet["active_approval"]["proposal_id"], "proposal-active")
        self.assertEqual(packet["context_pack_report"]["retained_proposal_id"], "proposal-active")
        self.assertNotIn(huge_original[:100], json.dumps(packet["active_change_set"]))

    def test_web_evidence_source_metadata_retained_without_raw_text(self) -> None:
        state = {
            "evidence_store": {
                "web_evidence": [
                    {
                        "title": "Docs",
                        "url": "https://docs.example.com/page",
                        "source": "docs.example.com",
                        "fetched_at": "2026-01-01T00:00:00Z",
                        "snippet": "Relevant summary.",
                        "text": "RAW_WEB_TEXT " * 10_000,
                    }
                ]
            }
        }
        packet = pack_context("web_research", self._request("Look up current docs"), state=state, profile=SMALL)
        sources = packet["context_pack_report"]["retained_web_sources"]
        self.assertEqual(sources[0]["url"], "https://docs.example.com/page")
        self.assertEqual(packet["web_evidence"][0]["source"], "docs.example.com")
        self.assertNotIn("RAW_WEB_TEXT", json.dumps(packet["web_evidence"]))

    def test_debug_context_report_is_json_safe_and_summary_only(self) -> None:
        state = {"evidence_store": {"contents": {"app.py": "print('hello')\n" * 1000}}}
        packet = pack_context("summary", self._request(), state=state, profile=SMALL)
        json.dumps(packet["context_pack_report"], ensure_ascii=False)
        json.dumps(packet["context_pack_summary"], ensure_ascii=False)
        self.assertNotIn("included_files", packet["context_pack_summary"])
        self.assertIn("included_sections", packet["context_pack_report"])

    def test_final_answer_repair_does_not_expose_raw_context_dump(self) -> None:
        request = self._request("Explain app.py")
        state = AgentCoreState(run_id="run-context-final", thread_id=None, repo="/tmp/repo", branch="main", user_task=request.task)
        state.files_read = ["app.py"]
        state.action_results = [
            ActionResult(
                action_id="read",
                status="success",
                files_read=["app.py"],
                payload={"contents": {"app.py": "def main():\n    return 1\n"}},
            )
        ]
        raw_dump = 'context_pack_report {"file_evidence": {"included_files": {"app.py": "def main(): return 1"}}}'
        repaired = validate_or_repair_final_answer(raw_dump, state, request)
        self.assertNotIn("context_pack_report", repaired)
        self.assertNotIn("included_files", repaired)

    def test_summary_context_excludes_noisy_raw_files_but_keeps_summaries(self) -> None:
        state = {"evidence_store": {"contents": {"noise.log": "x" * 60_000}}}
        packet = pack_context("summary_context", self._request("Summarize project"), state=state, profile=SMALL)
        self.assertIn("noise.log", packet["file_evidence"]["summaries"])
        self.assertTrue(packet["file_evidence"]["omitted_files"])

    def test_context_pack_report_includes_budget_and_target_carryover(self) -> None:
        state = {
            "files_read": ["main.py"],
            "evidence_store": {"contents": {"main.py": "def create_message(body):\n    return {'body': body}\n"}},
            "edit_target_candidates": [
                {
                    "path": "main.py",
                    "score": 91,
                    "role": "entrypoints",
                    "sources": ["read_file", "prior_target_candidate"],
                    "prior_reused": True,
                }
            ],
            "target_selection_diagnostics": {
                "selected_target_files": ["main.py"],
                "prior_evidence_reused": True,
                "candidates": [
                    {
                        "path": "main.py",
                        "score": 91,
                        "role": "entrypoints",
                        "sources": ["read_file", "prior_target_candidate"],
                        "prior_reused": True,
                    }
                ],
            },
        }
        packet = pack_context("edit", self._request("Prepare the proposal"), state=state, profile=SMALL)
        report = packet["context_pack_report"]
        self.assertIn("budget_usage", report)
        self.assertEqual(report["target_candidate_files"][0]["path"], "main.py")
        self.assertTrue(report["prior_evidence_reused"])
        self.assertEqual(packet["short_term_memory"]["target_candidate_summaries"][0]["path"], "main.py")
        self.assertTrue(packet["short_term_memory"]["carryover_summaries"])


if __name__ == "__main__":
    unittest.main()
