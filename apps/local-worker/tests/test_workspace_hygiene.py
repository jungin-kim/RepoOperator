import importlib.util
import sys
import unittest
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

ROOT_DIR = TESTS_DIR.parents[2]
HYGIENE_SCRIPT = ROOT_DIR / "scripts" / "check-workspace-hygiene.py"
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
    "runtime",
    "test-results",
}


def _load_hygiene_module():
    spec = importlib.util.spec_from_file_location("workspace_hygiene_script", HYGIENE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class WorkspaceHygieneTests(unittest.TestCase):
    def test_no_accidental_duplicate_source_files(self) -> None:
        stale: list[str] = []
        hygiene = _load_hygiene_module()
        for root_name in ("apps", "packages", "docs", "scripts"):
            root = ROOT_DIR / root_name
            for path in root.rglob("*"):
                if any(part in SKIP_DIRS for part in path.parts):
                    continue
                if path.is_file() and hygiene.is_conflict_artifact(path):
                    stale.append(str(path.relative_to(ROOT_DIR)))
        self.assertEqual(stale, [])

    def test_conflict_artifact_detection_covers_numeric_package_lock_copies(self) -> None:
        hygiene = _load_hygiene_module()
        bad_names = [
            "skills 2.py",
            "skills_service 10.py",
            "package-lock 6.json",
            "package-lock 11.json",
            "settings copy.toml",
            "module.bak",
            "module.orig",
        ]
        for name in bad_names:
            with self.subTest(name=name):
                self.assertTrue(hygiene.is_conflict_artifact(Path(name)))
        self.assertFalse(hygiene.is_conflict_artifact(Path("package-lock.json")))


if __name__ == "__main__":
    unittest.main()
