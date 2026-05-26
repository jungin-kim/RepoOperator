import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repooperator_worker.agent_core.change_set import ChangePlan, ChangeSetProposal, ProposedFileChange, validate_change_set  # noqa: E402
from repooperator_worker.services.worktree_sandbox_service import WorktreeSandbox, WorktreeSandboxService  # noqa: E402


class WorktreeSandboxServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name) / "repo"
        self.repo.mkdir()
        (self.repo / "app.js").write_text("function main() {\n  return 1;\n}\n", encoding="utf-8")
        self._git(["init"])
        self._git(["config", "user.email", "test@example.com"])
        self._git(["config", "user.name", "RepoOperator Test"])
        self._git(["add", "app.js"])
        self._git(["commit", "-m", "init"])
        self.service = WorktreeSandboxService(sandbox_root=Path(self.tmp.name) / "sandboxes")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _git(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(["git", *args], cwd=self.repo, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise AssertionError(result.stderr or result.stdout)
        return result

    def _proposal(self, proposed: str = "function main() {\n  return 2;\n}\n") -> ChangeSetProposal:
        original = (self.repo / "app.js").read_text(encoding="utf-8")
        proposal = ChangeSetProposal(
            plan=ChangePlan(summary="Modify app.js", target_files=["app.js"], operations=["modify"]),
            changes=[
                ProposedFileChange(
                    path="app.js",
                    operation="modify",
                    summary="Return a different value.",
                    original_content=original,
                    proposed_content=proposed,
                )
            ],
        )
        validation = validate_change_set(proposal, repo=str(self.repo))
        proposal.validation = validation
        proposal.status = validation.status
        proposal.validation_status = validation.status
        return proposal

    def test_temp_worktree_created_and_cleaned(self) -> None:
        sandbox = self.service.create_temp_worktree(str(self.repo))

        self.assertTrue(Path(sandbox.worktree_path).exists())
        self.assertTrue(Path(sandbox.worktree_path).resolve().is_relative_to(self.service.sandbox_root))

        cleanup = self.service.cleanup_worktree(sandbox)
        self.assertTrue(cleanup["removed"])
        self.assertFalse(Path(sandbox.worktree_path).exists())

    def test_applying_proposal_to_sandbox_does_not_modify_main_working_tree(self) -> None:
        proposal = self._proposal()
        original = (self.repo / "app.js").read_text(encoding="utf-8")
        sandbox = self.service.create_temp_worktree(str(self.repo))
        try:
            result = self.service.apply_change_set_to_worktree(sandbox, proposal)

            self.assertEqual(result.status, "valid")
            self.assertEqual((self.repo / "app.js").read_text(encoding="utf-8"), original)
            self.assertIn("return 2", (Path(sandbox.worktree_path) / "app.js").read_text(encoding="utf-8"))
        finally:
            self.service.cleanup_worktree(sandbox)

    def test_validation_result_attached_to_proposal(self) -> None:
        proposal = self._proposal()

        updated = self.service.validate_proposal_in_sandbox(
            project_path=str(self.repo),
            proposal=proposal,
            commands=[["node", "--check", "app.js"]],
        )

        self.assertEqual(updated["sandbox_validation"]["status"], "valid")
        self.assertIn("diff --git", updated["sandbox_validation"]["diff"])
        self.assertEqual(updated["sandbox_validation"]["commands"][0]["status"], "success")
        self.assertFalse(list(self.service.sandbox_root.glob("*")))

    def test_failed_sandbox_validation_marks_risk_without_invalidating_base_proposal(self) -> None:
        proposal = self._proposal("const value = ;\n")
        self.assertEqual(validate_change_set(proposal, repo=str(self.repo)).status, "valid")

        updated = self.service.validate_proposal_in_sandbox(
            project_path=str(self.repo),
            proposal=proposal,
            commands=[["node", "--check", "app.js"]],
        )

        self.assertEqual(updated["validation"]["status"], "valid")
        self.assertEqual(updated["sandbox_validation"]["status"], "failed")
        self.assertTrue(updated["changes"][0]["risk_notes"])

    def test_cleanup_rejects_paths_outside_sandbox_root(self) -> None:
        unsafe = WorktreeSandbox(
            project_path=str(self.repo),
            worktree_path=str(self.repo),
            sandbox_root=str(self.service.sandbox_root),
            base_ref="HEAD",
        )

        with self.assertRaises(ValueError):
            self.service.cleanup_worktree(unsafe)


if __name__ == "__main__":
    unittest.main()
