import sys
import unittest
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repooperator_worker.agent_core.secret_scanner import redact_secrets, scan_text_for_secrets  # noqa: E402


class SecretScannerTests(unittest.TestCase):
    def test_detects_and_redacts_high_confidence_tokens(self) -> None:
        text = "aws=AKIAABCDEFGHIJKLMNOP github=ghp_" + "A" * 36 + " slack=xoxb-1234567890-abcdef sk=sk-" + "B" * 32
        redacted, findings = redact_secrets(text)
        kinds = {finding.kind for finding in findings}
        self.assertTrue({"aws_access_key_id", "github_token", "slack_token", "openai_api_key"} <= kinds)
        self.assertNotIn("AKIAABCDEFGHIJKLMNOP", redacted)
        self.assertIn("[REDACTED:github_token]", redacted)

    def test_detects_private_key_and_jwt(self) -> None:
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signatureABC"
        key = "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----"
        kinds = {finding.kind for finding in scan_text_for_secrets(jwt + "\n" + key)}
        self.assertIn("jwt", kinds)
        self.assertIn("private_key", kinds)

    def test_ordinary_hash_is_not_flagged(self) -> None:
        findings = scan_text_for_secrets("sha256=" + "a" * 64)
        self.assertEqual(findings, [])


if __name__ == "__main__":
    unittest.main()
