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
                "read_many_files",
                "analyze_repository",
                "preview_command",
                "inspect_git_state",
                "run_approved_command",
                "run_validation_command",
                "generate_change_set",
                "validate_change_set",
                "apply_change_set",
                "create_file",
                "modify_file",
                "delete_file",
                "rename_file",
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
        self.assertTrue(by_name["apply_change_set"]["requires_approval_by_default"])
        self.assertEqual(by_name["apply_change_set"]["operation"], "write")
        self.assertEqual(by_name["generate_change_set"]["operation"], "edit")
        self.assertEqual(by_name["validate_change_set"]["operation"], "validation")
        self.assertFalse(by_name["generate_change_set"]["permission_required"])
        for name in ("create_file", "modify_file", "delete_file", "rename_file"):
            self.assertTrue(by_name[name]["permission_required"])
            self.assertEqual(by_name[name]["side_effect_level"], "write")
        self.assertIn("input_schema", by_name["generate_edit"])
        required = {
            "name",
            "operation",
            "side_effect_level",
            "permission_required",
            "parallel_safe",
            "workspace_bound",
            "produces_artifact",
            "produces_evidence",
            "can_be_retried",
        }
        self.assertTrue(all(required.issubset(item) for item in specs))
        self.assertTrue(all(item.get("operation") for item in specs))

    def test_planner_action_types_come_from_registry(self) -> None:
        self.assertEqual(PLANNER_ACTION_TYPES, set(get_default_tool_registry().allowed_action_types()))

    def test_duplicate_tool_names_are_rejected(self) -> None:
        registry = ToolRegistry([ReadFileTool()])
        with self.assertRaises(ValueError):
            registry.register(ReadFileTool())


if __name__ == "__main__":
    unittest.main()
