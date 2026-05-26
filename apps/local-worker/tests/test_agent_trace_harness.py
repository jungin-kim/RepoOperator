import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from repooperator_worker.agent_core.trace_harness import (  # noqa: E402
    FORBIDDEN_VISIBLE_MARKERS,
    TRACE_SCENARIO_NAMES,
    TRACE_UPDATE_ENV,
    compare_trace,
    run_agent_trace,
    snapshot_update_enabled,
    validate_trace_contract,
)


class AgentTraceHarnessTests(unittest.TestCase):
    def test_golden_trace_snapshots_pass(self) -> None:
        for scenario_name in TRACE_SCENARIO_NAMES:
            with self.subTest(scenario=scenario_name):
                snapshot = run_agent_trace("", scenario_name)
                self.assertEqual([], validate_trace_contract(snapshot))
                compare_trace(snapshot).assert_matches()

    def test_snapshot_mismatch_gives_readable_diff(self) -> None:
        snapshot = run_agent_trace("", "simple_project_summary")
        expected = snapshot.to_dict()
        expected["expected_graph_nodes"] = ["load_context", "wrong_node"]
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {TRACE_UPDATE_ENV: ""}, clear=False):
            path = Path(tmp) / "simple_project_summary.json"
            path.write_text(json.dumps(expected, indent=2, sort_keys=True), encoding="utf-8")
            comparison = compare_trace(snapshot, snapshot_dir=tmp)
        self.assertFalse(comparison.passed)
        self.assertIn("--- expected:simple_project_summary.json", comparison.diff)
        self.assertIn("+++ actual:simple_project_summary", comparison.diff)
        self.assertIn("wrong_node", comparison.diff)

    def test_update_mode_disabled_by_default(self) -> None:
        snapshot = run_agent_trace("", "simple_project_summary")
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {TRACE_UPDATE_ENV: ""}, clear=False):
            comparison = compare_trace(snapshot, snapshot_dir=tmp)
            self.assertFalse(comparison.passed)
            self.assertFalse((Path(tmp) / "simple_project_summary.json").exists())
            self.assertIn("Update mode is disabled", comparison.diff)
        self.assertFalse(snapshot_update_enabled({}))

    def test_trace_scenarios_are_deterministic(self) -> None:
        for scenario_name in TRACE_SCENARIO_NAMES:
            with self.subTest(scenario=scenario_name):
                first = run_agent_trace("", scenario_name).to_dict()
                second = run_agent_trace("", scenario_name).to_dict()
                self.assertEqual(first, second)

    def test_forbidden_visible_markers_cover_reasoning_leaks(self) -> None:
        expected = {
            "Work log",
            "Technical Log",
            "hidden reasoning",
            "<think>",
            "chain-of-thought",
            "chain of thought",
            "private_reasoning",
            "raw reasoning",
            "reasoning_delta",
        }
        self.assertTrue(expected.issubset(set(FORBIDDEN_VISIBLE_MARKERS)))

    def test_forbidden_visible_markers_are_contract_enforced_without_snapshot_churn(self) -> None:
        snapshot = run_agent_trace("", "simple_project_summary")
        payload = snapshot.to_dict()
        payload["expected_final_response_contract"]["text"] = "private_reasoning leaked"
        rebuilt = type(snapshot)(**payload)
        issues = validate_trace_contract(rebuilt)
        self.assertIn("visible trace contains forbidden marker: private_reasoning", issues)


if __name__ == "__main__":
    unittest.main()
