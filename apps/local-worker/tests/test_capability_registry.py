import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repooperator_worker.agent_core.capabilities.builtin import get_default_capability_registry  # noqa: E402
from repooperator_worker.agent_core.capabilities.registry import CapabilityRegistry  # noqa: E402
from repooperator_worker.agent_core.tools.registry import get_default_tool_registry  # noqa: E402


class CapabilityRegistryTests(unittest.TestCase):
    def test_all_default_tools_map_to_capability(self) -> None:
        tools = get_default_tool_registry()
        missing = [name for name in tools.allowed_action_types() if not tools.capabilities_for_tool(name)]
        self.assertEqual(missing, [])

    def test_metadata_is_json_safe(self) -> None:
        registry = get_default_capability_registry()
        json.dumps(registry.specs_for_model(), ensure_ascii=False)
        by_name = {item["name"]: item for item in registry.specs_for_model()}
        self.assertTrue(by_name["repository_write"]["requires_approval"])
        self.assertTrue(by_name["web_research"]["network_access"])
        self.assertIn("network", by_name["web_research"]["required_permissions"])

    def test_unavailable_capabilities_are_not_selected(self) -> None:
        registry = get_default_capability_registry()
        web = replace(registry.get("web_research"), available=False)
        custom = CapabilityRegistry([*(spec for spec in registry.specs() if spec.name != "web_research"), web])
        self.assertNotIn("web_research", custom.selectable_names())
        self.assertEqual(custom.select_available(["web_research"]), [])

    def test_registry_does_not_drive_hard_workflow_routing(self) -> None:
        registry = get_default_capability_registry()
        self.assertFalse(any(name.endswith("_intent") for name in dir(registry)))
        self.assertFalse(hasattr(registry, "requested_workflow"))


if __name__ == "__main__":
    unittest.main()
