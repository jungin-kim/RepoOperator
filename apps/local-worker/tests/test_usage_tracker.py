"""Per-run token usage, including prompt-cache hits, reaches the response."""

from __future__ import annotations

import unittest
from unittest import mock

from repooperator_worker.services import model_client, usage_tracker


class UsageTrackerTests(unittest.TestCase):
    def test_calls_inside_a_run_are_attributed_to_it(self) -> None:
        with usage_tracker.tracking("run-u1"):
            usage_tracker.record({"input_tokens": 1000, "cached_input_tokens": 800, "output_tokens": 50})
            usage_tracker.record({"input_tokens": 1200, "cached_input_tokens": 1000, "output_tokens": 30})
        usage_tracker.record({"input_tokens": 999})  # outside any run: ignored
        snap = usage_tracker.snapshot("run-u1")
        self.assertEqual(snap["calls"], 2)
        self.assertEqual(snap["input_tokens"], 2200)
        self.assertEqual(snap["cached_input_tokens"], 1800)
        self.assertAlmostEqual(snap["cache_hit_ratio"], 0.818, places=3)

    def test_model_http_calls_record_usage(self) -> None:
        body = b'{"choices": [], "usage": {"prompt_tokens": 500, "completion_tokens": 7, "prompt_tokens_details": {"cached_tokens": 400}}}'

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return body

        with mock.patch.object(model_client.request, "urlopen", return_value=FakeResponse()):
            with usage_tracker.tracking("run-u2"):
                model_client._post_json(url="http://x", payload={}, headers={}, timeout=5, settings=mock.Mock())
        snap = usage_tracker.snapshot("run-u2")
        self.assertEqual((snap["input_tokens"], snap["cached_input_tokens"], snap["output_tokens"]), (500, 400, 7))

    def test_unknown_run_is_empty(self) -> None:
        self.assertEqual(usage_tracker.snapshot("never-ran"), {})


if __name__ == "__main__":
    unittest.main()
