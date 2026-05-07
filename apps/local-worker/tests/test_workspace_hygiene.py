import fnmatch
import sys
import unittest
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

ROOT_DIR = TESTS_DIR.parents[2]
STALE_PATTERNS = ("* 2.*", "* copy.*", "*.bak", "*.orig")
SKIP_DIRS = {
    ".git",
    ".next",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    "dist",
    "build",
    "coverage",
}


class WorkspaceHygieneTests(unittest.TestCase):
    def test_no_accidental_duplicate_source_files(self) -> None:
        stale: list[str] = []
        for root_name in ("apps", "packages"):
            root = ROOT_DIR / root_name
            for path in root.rglob("*"):
                if any(part in SKIP_DIRS for part in path.parts):
                    continue
                if path.is_file() and any(fnmatch.fnmatch(path.name, pattern) for pattern in STALE_PATTERNS):
                    stale.append(str(path.relative_to(ROOT_DIR)))
        self.assertEqual(stale, [])


if __name__ == "__main__":
    unittest.main()
