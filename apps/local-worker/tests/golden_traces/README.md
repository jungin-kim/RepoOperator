# Golden Trace Snapshots

These snapshots are deterministic behavior contracts for the agent graph. They are not broad approval to refresh output whenever a test fails.

## Updating

1. Run the trace test once and read the diff.
2. Update only when the intended contract changed:

```bash
REPOOPERATOR_UPDATE_AGENT_TRACE_SNAPSHOTS=1 PYTHONPATH=apps/local-worker/src python3 -m unittest apps/local-worker/tests/test_agent_trace_harness.py
```

3. Rerun the trace test without update mode before committing.

## When Not To Update

Do not update snapshots to hide regressions in permission gates, write/no-write behavior, web source metadata, visible forbidden markers, raw context exposure, or final-response grounding.

A semantic contract failure means the harness found unsafe or inconsistent behavior even if the JSON looks close. A snapshot diff means the normalized trace changed; decide whether that change is intentional before refreshing.

## Future Trace Seam

TODO: add a fake-model real-LangGraph run that records the same contracts through production graph execution. Optional real-model evals can build on that later, but this deterministic harness should stay stable.
