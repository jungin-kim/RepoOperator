import sys
import unittest
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repooperator_worker.agent_core.command_security import validate_argv_shape  # noqa: E402


class CommandSecurityTests(unittest.TestCase):
    def test_allows_common_argv_style_read_only_command(self) -> None:
        self.assertTrue(validate_argv_shape(["git", "log", "--oneline", "-n", "5"]).allowed)

    def test_rejects_shell_interpreter_command_string(self) -> None:
        result = validate_argv_shape(["bash", "-lc", "cat README.md"])
        self.assertFalse(result.allowed)
        self.assertIn("shell_interpreter_command_string", result.findings)

    def test_rejects_command_substitution(self) -> None:
        result = validate_argv_shape(["echo", "$(cat secret)"])
        self.assertFalse(result.allowed)
        self.assertIn("command_substitution", result.findings)

    def test_rejects_zsh_and_powershell_execution_primitives(self) -> None:
        self.assertFalse(validate_argv_shape(["zmodload", "zsh/system"]).allowed)
        self.assertFalse(validate_argv_shape(["powershell", "-Command", "Invoke-Expression whoami"]).allowed)

    def test_mutating_git_command_shape_is_allowed_for_policy_to_decide(self) -> None:
        self.assertTrue(validate_argv_shape(["git", "commit", "-m", "test"]).allowed)


if __name__ == "__main__":
    unittest.main()
