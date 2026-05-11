import json
import sys
import unittest
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repooperator_worker.agent_core.planner import PLANNER_ACTION_TYPES  # noqa: E402
from repooperator_worker.agent_core.tools.registry import ToolRegistry, get_default_tool_registry  # noqa: E402
from repooperator_worker.agent_core.tools.builtin import ReadFileTool  # noqa: E402


class ToolRegistryTests(unittest.TestCase):
    def test_default_registry_contains_stable_unique_tools(self) -> None:
        registry = get_default_tool_registry()
        self.assertEqual(
            registry.allowed_action_types(),
            [
                "inspect_repo_tree",
                "search_files",
                "search_text",
                "read_file",
                "analyze_repository",
                "preview_command",
                "inspect_git_state",
                "run_approved_command",
                "generate_edit",
                "ask_clarification",
                "final_answer",
            ],
        )
        self.assertEqual(len(registry.allowed_action_types()), len(set(registry.allowed_action_types())))

    def test_specs_are_json_safe_and_include_metadata(self) -> None:
        specs = get_default_tool_registry().specs_for_model()
        json.dumps(specs, ensure_ascii=False)
        by_name = {item["name"]: item for item in specs}
        self.assertTrue(by_name["read_file"]["read_only"])
        self.assertEqual(by_name["read_file"]["operation"], "read_file")
        self.assertTrue(by_name["search_files"]["concurrency_safe"])
        self.assertEqual(by_name["search_files"]["operation"], "search")
        self.assertTrue(by_name["search_text"]["read_only"])
        self.assertTrue(by_name["run_approved_command"]["requires_approval_by_default"])
        self.assertIn("input_schema", by_name["generate_edit"])
        self.assertTrue(all(item.get("operation") for item in specs))

    def test_planner_action_types_come_from_registry(self) -> None:
        self.assertEqual(PLANNER_ACTION_TYPES, set(get_default_tool_registry().allowed_action_types()))

    def test_duplicate_tool_names_are_rejected(self) -> None:
        registry = ToolRegistry([ReadFileTool()])
        with self.assertRaises(ValueError):
            registry.register(ReadFileTool())


if __name__ == "__main__":
    unittest.main()
