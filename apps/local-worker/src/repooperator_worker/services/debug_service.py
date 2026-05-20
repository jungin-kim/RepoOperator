from repooperator_worker.config import get_settings
from repooperator_worker.agent_core.model_profile import detect_model_profile
from repooperator_worker.services.active_repository import get_active_repository
from repooperator_worker.services.composio_service import get_composio_status
from repooperator_worker.services.event_service import get_active_runs, list_recent_runs
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
