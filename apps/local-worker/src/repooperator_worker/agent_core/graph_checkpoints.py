from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from langgraph.checkpoint.base import CheckpointTuple
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Interrupt

from repooperator_worker.services.event_service import append_run_event, list_run_events
from repooperator_worker.services.json_safe import json_safe


GRAPH_CHECKPOINT_EVENT = "langgraph_checkpoint"
GRAPH_CHECKPOINT_WRITE_EVENT = "langgraph_checkpoint_write"


class EventServiceLangGraphSaver(InMemorySaver):
    """LangGraph checkpointer with an event-service durability trail.

    LangGraph still receives a normal checkpointer, so graph step persistence,
    interrupt state, and resume semantics go through LangGraph's checkpoint API.
    Each checkpoint and pending write is mirrored into RepoOperator's run event
    storage as JSON-safe data. If this saver is re-created in a later process,
    ``get_tuple`` can reconstruct the latest checkpoint tuple from those events.
    """

    def put(self, config: dict[str, Any], checkpoint: dict[str, Any], metadata: dict[str, Any], new_versions: dict[str, Any]) -> dict[str, Any]:
        next_config = super().put(config, checkpoint, metadata, new_versions)
        self._persist_checkpoint(config, checkpoint, metadata, new_versions, next_config)
        return next_config

    def put_writes(
        self,
        config: dict[str, Any],
        writes: list[tuple[str, Any]] | tuple[tuple[str, Any], ...],
        task_id: str,
        task_path: str = "",
    ) -> None:
        super().put_writes(config, writes, task_id, task_path)
        self._persist_writes(config, writes, task_id, task_path)

    def get_tuple(self, config: dict[str, Any]) -> CheckpointTuple | None:
        existing = super().get_tuple(config)
        if existing is not None:
            return existing
        return self._load_tuple_from_events(config)

    def list(
        self,
        config: dict[str, Any] | None,
        *,
        filter: dict[str, Any] | None = None,
        before: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        yielded = 0
        for item in super().list(config, filter=filter, before=before, limit=limit):
            yielded += 1
            yield item
            if limit is not None and yielded >= limit:
                return
        if config is None:
            return
        restored = self._load_tuple_from_events(config)
        if restored is not None and yielded == 0:
            yield restored

    def _persist_checkpoint(
        self,
        config: dict[str, Any],
        checkpoint: dict[str, Any],
        metadata: dict[str, Any],
        new_versions: dict[str, Any],
        next_config: dict[str, Any],
    ) -> None:
        thread_key = _thread_key(config)
        run_id = _run_id_from_thread_key(thread_key)
        if not run_id:
            return
        configurable = dict(config.get("configurable") or {})
        next_configurable = dict(next_config.get("configurable") or {})
        try:
            append_run_event(
                run_id,
                {
                    "type": GRAPH_CHECKPOINT_EVENT,
                    "event_type": GRAPH_CHECKPOINT_EVENT,
                    "visibility": "debug",
                    "display": "secondary",
                    "thread_key": thread_key,
                    "checkpoint_ns": configurable.get("checkpoint_ns", ""),
                    "checkpoint_id": checkpoint.get("id") or next_configurable.get("checkpoint_id"),
                    "parent_checkpoint_id": configurable.get("checkpoint_id"),
                    "checkpoint": json_safe(checkpoint),
                    "metadata": json_safe(metadata),
                    "new_versions": json_safe(new_versions),
                    "langgraph_config": json_safe(next_config),
                },
            )
        except OSError:
            return

    def _persist_writes(
        self,
        config: dict[str, Any],
        writes: list[tuple[str, Any]] | tuple[tuple[str, Any], ...],
        task_id: str,
        task_path: str,
    ) -> None:
        thread_key = _thread_key(config)
        run_id = _run_id_from_thread_key(thread_key)
        checkpoint_id = (config.get("configurable") or {}).get("checkpoint_id")
        if not run_id or not checkpoint_id:
            return
        try:
            append_run_event(
                run_id,
                {
                    "type": GRAPH_CHECKPOINT_WRITE_EVENT,
                    "event_type": GRAPH_CHECKPOINT_WRITE_EVENT,
                    "visibility": "debug",
                    "display": "secondary",
                    "thread_key": thread_key,
                    "checkpoint_ns": (config.get("configurable") or {}).get("checkpoint_ns", ""),
                    "checkpoint_id": checkpoint_id,
                    "task_id": task_id,
                    "task_path": task_path,
                    "writes": [[channel, json_safe(value)] for channel, value in list(writes)],
                },
            )
        except OSError:
            return

    def _load_tuple_from_events(self, config: dict[str, Any]) -> CheckpointTuple | None:
        thread_key = _thread_key(config)
        run_id = _run_id_from_thread_key(thread_key)
        if not run_id:
            return None
        configurable = dict(config.get("configurable") or {})
        checkpoint_ns = str(configurable.get("checkpoint_ns") or "")
        requested_checkpoint_id = configurable.get("checkpoint_id")
        checkpoint_events = [
            event
            for event in list_run_events(run_id)
            if event.get("type") == GRAPH_CHECKPOINT_EVENT
            and event.get("thread_key") == thread_key
            and str(event.get("checkpoint_ns") or "") == checkpoint_ns
        ]
        if requested_checkpoint_id:
            checkpoint_events = [event for event in checkpoint_events if event.get("checkpoint_id") == requested_checkpoint_id]
        if not checkpoint_events:
            return None
        event = checkpoint_events[-1]
        checkpoint_id = str(event.get("checkpoint_id") or "")
        checkpoint = dict(event.get("checkpoint") or {})
        if checkpoint_id:
            checkpoint["id"] = checkpoint.get("id") or checkpoint_id
        pending_writes = self._load_writes_from_events(run_id, thread_key, checkpoint_ns, checkpoint_id)
        parent_id = event.get("parent_checkpoint_id")
        tuple_config = {
            "configurable": {
                "thread_id": thread_key,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }
        parent_config = (
            {
                "configurable": {
                    "thread_id": thread_key,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_id,
                }
            }
            if parent_id
            else None
        )
        return CheckpointTuple(
            config=tuple_config,
            checkpoint=checkpoint,
            metadata=dict(event.get("metadata") or {}),
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    def _load_writes_from_events(self, run_id: str, thread_key: str, checkpoint_ns: str, checkpoint_id: str) -> list[tuple[str, str, Any]]:
        writes: list[tuple[str, str, Any]] = []
        for event in list_run_events(run_id):
            if event.get("type") != GRAPH_CHECKPOINT_WRITE_EVENT:
                continue
            if event.get("thread_key") != thread_key:
                continue
            if str(event.get("checkpoint_ns") or "") != checkpoint_ns:
                continue
            if str(event.get("checkpoint_id") or "") != checkpoint_id:
                continue
            task_id = str(event.get("task_id") or "")
            for item in event.get("writes") or []:
                if not isinstance(item, list) or len(item) != 2:
                    continue
                channel = str(item[0])
                writes.append((task_id, channel, _restore_write_value(channel, item[1])))
        return writes


def _thread_key(config: dict[str, Any]) -> str:
    return str((config.get("configurable") or {}).get("thread_id") or "")


def _run_id_from_thread_key(thread_key: str) -> str:
    return thread_key.split("|", 1)[0] if thread_key else ""


def _restore_write_value(channel: str, value: Any) -> Any:
    if channel != "__interrupt__":
        return value
    if isinstance(value, list):
        return [_restore_interrupt(item) for item in value]
    return value


def _restore_interrupt(value: Any) -> Any:
    if isinstance(value, Interrupt):
        return value
    if not isinstance(value, dict):
        return value
    if "id" not in value or "value" not in value:
        return value
    return Interrupt(value=value.get("value"), id=str(value.get("id") or "placeholder-id"))
