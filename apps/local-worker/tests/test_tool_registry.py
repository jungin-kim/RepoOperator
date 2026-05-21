import json
import sys
import unittest
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repooperator_worker.agent_core.planner import PLANNER_ACTION_TYPES  # noqa: E402
from repooperator_worker.agent_core.tools import ToolSearch  # noqa: E402
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
                "search_web",
                "fetch_url",
                "summarize_web_evidence",
                "refresh_context_pack",
                "compact_thread_context",
                "generate_change_set",
                "validate_change_set",
                "apply_change_set",
                "git_status",
                "git_diff",
                "git_log",
                "git_branch_create",
                "git_commit",
                "git_push",
                "github_create_pr",
                "gitlab_create_mr",
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
        registry = get_default_tool_registry()
        specs = registry.internal_specs()
        json.dumps(specs, ensure_ascii=False)
        by_name = {item["name"]: item for item in specs}
        self.assertTrue(by_name["read_file"]["read_only"])
        self.assertEqual(by_name["read_file"]["operation"], "read_file")
        self.assertTrue(by_name["search_files"]["concurrency_safe"])
        self.assertEqual(by_name["search_files"]["operation"], "search")
        self.assertTrue(by_name["search_text"]["read_only"])
        self.assertTrue(by_name["run_approved_command"]["requires_approval_by_default"])
        self.assertTrue(by_name["search_web"]["network_access"])
        self.assertTrue(by_name["search_web"]["is_open_world"])
        self.assertIn("network", by_name["search_web"]["required_permissions"])
        self.assertEqual(by_name["search_web"]["operation"], "web_search")
        self.assertIn("web_research", by_name["fetch_url"]["capability_names"])
        self.assertIn("git_provider", by_name["git_commit"]["capability_names"])
        self.assertTrue(by_name["git_commit"]["requires_approval_by_default"])
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
            "is_destructive",
            "is_open_world",
            "interrupt_behavior",
            "idempotent",
            "should_defer",
            "always_load",
            "tool_search_keywords",
            "capability_names",
            "prompt_summary",
            "input_schema_summary",
            "output_schema_summary",
            "max_result_size_chars",
            "oversized_result_strategy",
            "permission_required",
            "parallel_safe",
            "workspace_bound",
            "produces_artifact",
            "produces_evidence",
            "evidence_kind",
            "progress_kind",
            "ui_renderer_kind",
            "grouped_display_key",
            "compact_label_template",
            "rejected_message_template",
            "error_message_template",
            "permission_matcher_kind",
            "required_permissions",
            "denial_recovery_hint",
            "can_be_retried",
        }
        self.assertTrue(all(required.issubset(item) for item in specs))
        self.assertTrue(all(item.get("operation") for item in specs))
        self.assertTrue(all(item.get("capability_names") for item in specs))
        self.assertTrue(all(isinstance(item.get("tool_search_keywords"), list) for item in specs))

    def test_specs_for_model_exposes_only_model_relevant_fields(self) -> None:
        specs = get_default_tool_registry().specs_for_model()
        json.dumps(specs, ensure_ascii=False)
        self.assertTrue(specs)
        internal_only = {
            "input_schema",
            "description",
            "permission_required",
            "parallel_safe",
            "permission_matcher_kind",
            "required_permissions",
            "denial_recovery_hint",
            "progress_kind",
            "ui_renderer_kind",
            "grouped_display_key",
            "compact_label_template",
            "rejected_message_template",
            "error_message_template",
            "oversized_result_strategy",
            "max_result_chars",
            "capabilities",
        }
        for item in specs:
            self.assertFalse(internal_only.intersection(item), item)
            self.assertIn("prompt_summary", item)
            self.assertIn("input_schema_summary", item)
            self.assertIn("output_schema_summary", item)

    def test_deferred_tools_are_not_loaded_by_default_model_specs(self) -> None:
        registry = get_default_tool_registry()
        default_names = {item["name"] for item in registry.specs_for_model()}
        for name in (
            "inspect_repo_tree",
            "search_files",
            "search_text",
            "read_file",
            "read_many_files",
            "generate_change_set",
            "validate_change_set",
            "final_answer",
        ):
            self.assertIn(name, default_names)
        self.assertNotIn("search_web", default_names)
        self.assertNotIn("fetch_url", default_names)
        self.assertNotIn("git_push", default_names)
        self.assertNotIn("github_create_pr", default_names)
        self.assertNotIn("apply_change_set", default_names)

        web_names = {item["name"] for item in registry.specs_for_model(capabilities=["web_research"])}
        git_names = {item["name"] for item in registry.specs_for_model(capabilities=["git_provider"])}
        self.assertIn("search_web", web_names)
        self.assertIn("fetch_url", web_names)
        self.assertIn("git_push", git_names)
        self.assertIn("github_create_pr", git_names)

    def test_tool_search_finds_relevant_deferred_tools(self) -> None:
        registry = get_default_tool_registry()
        web_names = [item["name"] for item in ToolSearch(registry).search(capability="web_research")]
        git_names = [item["name"] for item in registry.search_tools(capability="git_provider")]
        self.assertEqual(web_names, ["search_web", "fetch_url", "summarize_web_evidence"])
        self.assertIn("git_status", git_names)
        self.assertIn("git_push", git_names)
        self.assertIn("github_create_pr", git_names)

    def test_destructive_and_network_tools_have_permission_metadata(self) -> None:
        registry = get_default_tool_registry()
        specs = {item["name"]: item for item in registry.internal_specs()}
        destructive = [item for item in specs.values() if item["is_destructive"]]
        self.assertTrue(destructive)
        for item in destructive:
            self.assertTrue(item["requires_approval_by_default"], item["name"])
            self.assertTrue(item["permission_required"], item["name"])
            self.assertEqual(item["interrupt_behavior"], "approval", item["name"])
            self.assertTrue(item["required_permissions"], item["name"])
        network = [item for item in specs.values() if item["network_access"]]
        self.assertTrue(network)
        for item in network:
            self.assertTrue(item["is_open_world"], item["name"])
            self.assertIn(item["side_effect_level"], {"network", "remote_write"}, item["name"])
            self.assertTrue(item["required_permissions"], item["name"])

    def test_planner_action_types_come_from_registry(self) -> None:
        self.assertEqual(PLANNER_ACTION_TYPES, set(get_default_tool_registry().allowed_action_types()))

    def test_duplicate_tool_names_are_rejected(self) -> None:
        registry = ToolRegistry([ReadFileTool()])
        with self.assertRaises(ValueError):
            registry.register(ReadFileTool())


if __name__ == "__main__":
    unittest.main()
