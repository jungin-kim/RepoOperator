from repooperator_worker.config import get_settings
from repooperator_worker.agent_core.graph_checkpoints import GRAPH_CHECKPOINT_EVENT
from repooperator_worker.agent_core.model_profile import detect_model_profile
from repooperator_worker.agent_core.understanding_context import debug_context_payload
from repooperator_worker.services.active_repository import get_active_repository
from repooperator_worker.services.composio_service import get_composio_status
from repooperator_worker.services.event_service import get_active_runs, list_recent_runs, list_run_events
from repooperator_worker.services.memory_service import list_memory_items
from repooperator_worker.services.permissions_service import permission_profile
from repooperator_worker.services.skills_service import discover_skills
from repooperator_worker.services.thread_context_service import list_thread_context_items


def get_debug_runtime_status() -> dict:
    settings = get_settings()
    active = get_active_repository()
    profile = permission_profile(settings.permission_mode)
    model_profile = detect_model_profile(settings=settings)
    return {
        "worker": {
            "status": "ok",
            "service": "repooperator-local-worker",
        },
        "model": {
            "provider": settings.configured_model_provider,
            "connection_mode": settings.configured_model_connection_mode,
            "name": settings.configured_model_name,
            "base_url": settings.openai_base_url,
            "profile": model_profile.model_dump(),
        },
        "permissions": {
            "write_mode": settings.write_mode,
            "mode": profile["mode"],
            "sandbox": profile["sandbox"],
            "approval": profile["approval"],
            "tools": profile["tools"],
        },
        "repository": {
            "source": active.git_provider if active else None,
            "project_path": active.project_path if active else None,
            "branch": active.branch if active else None,
            "configured_default_source": settings.configured_git_provider,
            "configured_sources": settings.configured_repository_sources,
            "effective_sources": settings.configured_repository_sources,
        },
        "agent": {
            "orchestration_mode": "agent_core_controller",
        },
        "thread_context": list_thread_context_items(),
        "memory": list_memory_items(),
        "recent_runs": list_recent_runs(),
        "active_runs": get_active_runs(),
    }


def get_debug_context_status() -> dict:
    settings = get_settings()
    model_profile = detect_model_profile(settings=settings).model_dump()
    packs = _recent_context_pack_events()
    latest = packs[0] if packs else None
    state_payload = debug_context_payload(_latest_graph_state() or {})
    return {
        "model_profile": model_profile,
        "latest_pack": latest,
        "recent_packs": packs,
        "user_understanding_context": state_payload.get("user_understanding_context") or {},
        "evidence_basis": state_payload.get("evidence_basis") or {},
        "visible_rationale_log": state_payload.get("visible_rationale_log") or [],
    }


def integration_status() -> dict:
    try:
        status = get_composio_status()
    except RuntimeError as exc:
        status = {
            "provider": "Composio",
            "status": "error",
            "configured": True,
            "message": str(exc),
            "accounts": [],
            "toolkits": [],
            "tools_count": 0,
        }
    return {"integrations": [status]}


def _recent_context_pack_events(limit: int = 20) -> list[dict]:
    run_ids: list[str] = []
    for run in [*get_active_runs(), *list_recent_runs(limit=30)]:
        run_id = str(run.get("id") or "")
        if run_id and run_id not in run_ids:
            run_ids.append(run_id)
    packs: list[dict] = []
    for run_id in run_ids[:30]:
        for event in reversed(list_run_events(run_id)):
            summary = _context_pack_summary_from_event(event)
            if not summary:
                continue
            packs.append(
                {
                    "run_id": run_id,
                    "timestamp": event.get("timestamp"),
                    "thread_id": event.get("thread_id"),
                    "repo": event.get("repo"),
                    "branch": event.get("branch"),
                    "pack_kind": summary.get("kind") or summary.get("pack_kind"),
                    "trigger_node": summary.get("trigger_node"),
                    "compression_ratio": summary.get("compression_ratio"),
                    "estimated_input_tokens": summary.get("estimated_input_tokens"),
                    "estimated_output_reserve": summary.get("estimated_output_reserve"),
                    "included_sections": summary.get("included_sections") or [],
                    "excluded_sections": summary.get("excluded_sections") or [],
                    "retained_files": summary.get("retained_files") or [],
                    "omitted_files": summary.get("omitted_files") or [],
                    "retained_web_sources": summary.get("retained_web_sources") or [],
                    "warnings": summary.get("warnings") or [],
                }
            )
            if len(packs) >= limit:
                return packs
    return packs


def _context_pack_summary_from_event(event: dict) -> dict | None:
    if event.get("operation") != "context_pack" and event.get("event_type") != "context_pack":
        aggregate = event.get("aggregate") if isinstance(event.get("aggregate"), dict) else {}
        graph_summary = event.get("change_set_summary") if isinstance(event.get("change_set_summary"), dict) else {}
        if not aggregate.get("change_set_summary") and not graph_summary:
            return None
    aggregate = event.get("aggregate") if isinstance(event.get("aggregate"), dict) else {}
    if isinstance(aggregate.get("change_set_summary"), dict):
        return aggregate["change_set_summary"]
    if isinstance(event.get("change_set_summary"), dict):
        return event["change_set_summary"]
    if event.get("operation") == "context_pack":
        return aggregate
    return None


def _latest_graph_state() -> dict | None:
    run_ids: list[str] = []
    for run in [*get_active_runs(), *list_recent_runs(limit=30)]:
        run_id = str(run.get("id") or "")
        if run_id and run_id not in run_ids:
            run_ids.append(run_id)
    for run_id in run_ids[:30]:
        for event in reversed(list_run_events(run_id)):
            if event.get("type") != GRAPH_CHECKPOINT_EVENT:
                continue
            checkpoint = event.get("checkpoint") if isinstance(event.get("checkpoint"), dict) else {}
            values = checkpoint.get("channel_values") if isinstance(checkpoint.get("channel_values"), dict) else None
            if values is None:
                values = checkpoint.get("values") if isinstance(checkpoint.get("values"), dict) else None
            if isinstance(values, dict):
                state = dict(values)
                state.setdefault("run_id", run_id)
                state.setdefault("thread_id", event.get("thread_id"))
                state.setdefault("repo", event.get("repo"))
                state.setdefault("branch", event.get("branch"))
                return state
    return None
