"""Per-run model token usage, including prompt-cache hits.

Every non-streaming model call goes through ``model_client._post_json``, which
reports the provider's usage block here. The run that owns the call is taken
from a context variable the graph runtime binds for the duration of a run, so
planner, edit generation, subagents and final synthesis are all counted
without threading a run id through each call site.

Streaming final-answer calls do not report usage yet and are not counted.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

_current_run: ContextVar[str | None] = ContextVar("repooperator_usage_run", default=None)
_lock = threading.Lock()
_USAGE: dict[str, dict[str, int]] = {}
_LIMIT = 512

_COUNTED = ("input_tokens", "cached_input_tokens", "cache_write_tokens", "output_tokens")


@contextmanager
def tracking(run_id: str | None) -> Iterator[None]:
    """Attribute model calls made inside this block to ``run_id``."""

    token = _current_run.set(str(run_id) if run_id else None)
    try:
        yield
    finally:
        _current_run.reset(token)


def current_run() -> str | None:
    return _current_run.get()


def record(usage: dict[str, int] | None, *, run_id: str | None = None) -> None:
    """Add one model call's usage to the current (or given) run."""

    key = run_id or _current_run.get()
    if not key:
        return
    with _lock:
        totals = _USAGE.get(key)
        if totals is None:
            if len(_USAGE) >= _LIMIT:
                _USAGE.pop(next(iter(_USAGE)))
            totals = _USAGE[key] = {}
        for name in _COUNTED:
            value = int((usage or {}).get(name) or 0)
            if value:
                totals[name] = totals.get(name, 0) + value
        totals["calls"] = totals.get("calls", 0) + 1


def bump(name: str, *, run_id: str | None = None, amount: int = 1) -> None:
    """Count a non-token event for the run (e.g. gate feedback retries)."""

    key = run_id or _current_run.get()
    if not key:
        return
    with _lock:
        totals = _USAGE.setdefault(key, {})
        totals[name] = totals.get(name, 0) + amount


def snapshot(run_id: str | None) -> dict[str, Any]:
    """Usage totals for a run plus the prompt-cache hit ratio (0..1)."""

    if not run_id:
        return {}
    with _lock:
        totals = dict(_USAGE.get(str(run_id)) or {})
    if not totals:
        return {}
    prompt = totals.get("input_tokens", 0)
    totals["cache_hit_ratio"] = round(totals.get("cached_input_tokens", 0) / prompt, 3) if prompt else 0.0
    return totals
