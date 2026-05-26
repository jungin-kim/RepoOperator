from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from repooperator_worker.config import get_settings
from repooperator_worker.schemas import (
    AgentProposeFileRequest,
    AgentProposeFileResponse,
    AgentRunRequest,
    AgentRunResponse,
    CommandRunRequest,
    CommandRunResponse,
    FileReadRequest,
    FileReadResponse,
    FileWriteRequest,
    FileWriteResponse,
    GitBranchCreateRequest,
    GitBranchCreateResponse,
    GitBranchListRequest,
    GitBranchListResponse,
    GitCheckoutRequest,
    GitCheckoutResponse,
    GitCommitRequest,
    GitCommitResponse,
    GitDiffRequest,
    GitDiffResponse,
    IDEContextUpdateRequest,
    GitMergeRequestCreateRequest,
    GitMergeRequestCreateResponse,
    PermissionModeRequest,
    PermissionModeResponse,
    GitPushRequest,
    GitPushResponse,
    HealthResponse,
    ProviderBranchesResponse,
    ProviderProjectsResponse,
    RecentProjectsResponse,
    RepoOpenPlanResponse,
    RepoOpenRequest,
    RepoOpenResponse,
    ThreadListResponse,
    ThreadSummary,
    ThreadUpsertRequest,
)
from repooperator_worker.services.edit_service import propose_file_edit
from repooperator_worker.services.agent_run_coordinator import (
    cancel_queued_message,
    cancel_run,
    enqueue_message,
    get_active_run,
    list_events,
    list_queue,
    resume_approval,
    start_run,
    steer_run,
    stream_run,
)
from repooperator_worker.services.apply_summary_service import generate_apply_summary
from repooperator_worker.services.command_service import (
    list_command_approvals,
    preview_command,
    revoke_command_approval,
    run_command_with_policy,
)
from repooperator_worker.services.file_service import read_text_file, write_text_file
from repooperator_worker.services.provider_service import (
    list_provider_branches,
    list_provider_projects,
    list_recent_projects,
    list_recent_project_paths,
)
from repooperator_worker.services.permissions_service import (
    get_permission_mode,
    permission_profile,
    update_permission_mode,
)
from repooperator_worker.services.git_service import (
    checkout_branch,
    commit_changes,
    create_branch,
    create_provider_merge_request,
    get_diff,
    list_local_branches,
    push_branch,
)
from repooperator_worker.services.ide_bridge_service import (
    clear_ide_context,
    get_ide_context,
    update_ide_context,
)
from repooperator_worker.services.repo_service import open_repository, plan_repository_open
from repooperator_worker.services.routine_service import get_default_routine_store
from repooperator_worker.services.thread_service import list_threads, upsert_thread
from repooperator_worker.services.tool_service import get_tools_status, preview_tool_run, run_tool
from repooperator_worker.services.debug_service import (
    get_debug_context_status,
    get_debug_runtime_status,
    integration_status,
)
from repooperator_worker.services.composio_service import (
    composio_connection_instructions,
    get_composio_status,
    list_composio_connected_accounts,
    list_composio_toolkits,
)
from repooperator_worker.services.event_service import (
    get_run,
    record_event,
)
from repooperator_worker.services.memory_service import (
    list_memory_items,
    record_applied_file_write,
)
from repooperator_worker.services.skills_service import discover_skills

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    profile = permission_profile(settings.permission_mode)
    return HealthResponse(
        status="ok",
        service="repooperator-local-worker",
        repo_base_dir=str(settings.repo_base_dir),
        configured_git_provider=settings.configured_git_provider,
        configured_repository_source=settings.configured_git_provider,
        configured_repository_sources=settings.configured_repository_sources,
        configured_model_connection_mode=settings.configured_model_connection_mode,
        configured_model_provider=settings.configured_model_provider,
        configured_model_name=settings.configured_model_name,
        configured_model_base_url=settings.openai_base_url,
        config_loaded_at=settings.config_loaded_at,
        config_source_path=str(settings.repooperator_config_path),
        config_hash=settings.config_hash,
        write_mode=settings.write_mode,
        permission_mode=profile["mode"],
        sandbox_scope=profile["sandbox"]["scope"],
        approval_policy=profile["approval"],
        tool_permissions=profile["tools"],
        recent_projects=list_recent_project_paths(),
    )


@router.get("/permissions", response_model=PermissionModeResponse)
def permissions_get() -> PermissionModeResponse:
    return get_permission_mode()


@router.post("/admin/reload-config")
def admin_reload_config() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "configured_model_connection_mode": settings.configured_model_connection_mode,
        "configured_model_provider": settings.configured_model_provider,
        "configured_model_name": settings.configured_model_name,
        "configured_model_base_url": settings.openai_base_url,
        "configured_git_provider": settings.configured_git_provider,
        "configured_repository_sources": settings.configured_repository_sources,
        "effective_repository_sources": settings.configured_repository_sources,
        "config_loaded_at": settings.config_loaded_at,
        "config_source_path": str(settings.repooperator_config_path),
        "config_hash": settings.config_hash,
        "api_key_configured": bool(settings.openai_api_key),
    }


@router.post("/permissions", response_model=PermissionModeResponse)
def permissions_post(request: PermissionModeRequest) -> PermissionModeResponse:
    try:
        return update_permission_mode(request.mode or request.write_mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/debug/runtime")
def debug_runtime() -> dict:
    return get_debug_runtime_status()


@router.get("/debug/memory")
def debug_memory() -> dict:
    return list_memory_items()


@router.get("/debug/context")
def debug_context() -> dict:
    return get_debug_context_status()


@router.get("/debug/skills")
def debug_skills() -> dict:
    return discover_skills()


@router.post("/ide/context")
def ide_context_update(request: IDEContextUpdateRequest) -> dict:
    try:
        return {"ide_context": update_ide_context(request.model_dump(exclude_none=True))}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/ide/context")
def ide_context_get(project_path: str, branch: str | None = None) -> dict:
    return {"ide_context": get_ide_context(project_path=project_path, branch=branch)}


@router.delete("/ide/context")
def ide_context_clear(project_path: str | None = None, branch: str | None = None) -> dict:
    return clear_ide_context(project_path=project_path, branch=branch)


@router.get("/debug/integrations")
def debug_integrations() -> dict:
    return integration_status()


@router.get("/routines")
def routines_list() -> dict:
    store = get_default_routine_store()
    return {"routines": [routine.model_dump() for routine in store.list()]}


@router.post("/routines")
def routines_create(payload: dict) -> dict:
    try:
        routine = get_default_routine_store().create(payload)
        return {"routine": routine.model_dump()}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/routines/{routine_id}/enable")
def routines_enable(routine_id: str, payload: dict) -> dict:
    try:
        routine = get_default_routine_store().update_enabled(routine_id, bool(payload.get("enabled", True)))
        return {"routine": routine.model_dump()}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/routines/{routine_id}/run-now")
def routines_run_now(routine_id: str) -> dict:
    try:
        run = get_default_routine_store().run_now(routine_id)
        return {"run": run.model_dump()}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/routines/{routine_id}/runs")
def routines_runs(routine_id: str) -> dict:
    return {"runs": [run.model_dump() for run in get_default_routine_store().list_runs(routine_id)]}


@router.get("/integrations/composio/status")
def composio_status() -> dict:
    try:
        return get_composio_status()
    except RuntimeError as exc:
        return {
            "provider": "Composio",
            "status": "error",
            "configured": True,
            "message": str(exc),
            "accounts": [],
            "toolkits": [],
            "tools_count": 0,
        }


@router.get("/integrations/composio/toolkits")
def composio_toolkits() -> dict:
    try:
        return list_composio_toolkits()
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/integrations/composio/accounts")
def composio_accounts() -> dict:
    try:
        return list_composio_connected_accounts()
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/integrations/composio/connect")
def composio_connect() -> dict:
    return composio_connection_instructions()


@router.get("/tools")
def tools_status() -> dict:
    return get_tools_status()


@router.post("/tools/run-preview")
def tools_run_preview(request: dict) -> dict:
    argv = request.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise HTTPException(status_code=400, detail="argv must be a list of strings.")
    preview = preview_tool_run(argv)
    record_event(
        event_type="tool_preview",
        repo=preview.get("cwd"),
        summary=f"Previewed local tool command: {' '.join(argv)}",
        tool=argv[0] if argv else None,
        command=argv,
    )
    return preview


@router.post("/tools/run")
def tools_run(request: dict) -> dict:
    argv = request.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise HTTPException(status_code=400, detail="argv must be a list of strings.")
    try:
        return run_tool(
            argv,
            confirmed=bool(request.get("confirmed") or request.get("remember_for_session")),
            approval_id=request.get("approval_id"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/commands/preview")
def commands_preview(request: dict) -> dict:
    argv = request.get("argv") or request.get("command")
    try:
        return preview_command(argv, reason=request.get("reason"), project_path=request.get("project_path"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/commands/run")
def commands_run(request: dict) -> dict:
    argv = request.get("argv") or request.get("command")
    try:
        decision = request.get("decision", "yes")
        if request.get("run_id"):
            run = get_run(str(request.get("run_id"))) or {}
            pending = run.get("pending_approval") if isinstance(run.get("pending_approval"), dict) else {}
            if pending.get("runtime") == "langgraph":
                return resume_approval(
                    str(request.get("run_id")),
                    {
                        "decision": decision,
                        "approval_id": request.get("approval_id"),
                        "command": argv if isinstance(argv, list) else [],
                        "remember_for_session": bool(request.get("remember_for_session")),
                    },
                )
        record_event(
            event_type="command_approval",
            summary=f"Command approval decision: {decision}",
            command=argv if isinstance(argv, list) else None,
            status="denied" if decision == "no_explain" else "ok",
        )
        if decision == "no_explain":
            return {
                "status": "denied",
                "command": argv,
                "stdout": "",
                "stderr": "",
                "exit_code": None,
                "message": "Command was not run. RepoOperator will explain another approach.",
            }
        return run_command_with_policy(
            argv,
            approval_id=request.get("approval_id"),
            remember_for_session=bool(request.get("remember_for_session")),
            reason=request.get("reason"),
            project_path=request.get("project_path"),
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/commands/approvals")
def commands_approvals() -> dict:
    return list_command_approvals()


@router.delete("/commands/approvals/{approval_id}")
def commands_approval_revoke(approval_id: str) -> dict:
    return revoke_command_approval(approval_id)


@router.get("/provider/projects", response_model=ProviderProjectsResponse)
def provider_projects(
    git_provider: str,
    search: str | None = None,
) -> ProviderProjectsResponse:
    try:
        return list_provider_projects(git_provider=git_provider, search=search)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/provider/recent-projects", response_model=RecentProjectsResponse)
def provider_recent_projects(limit: int = 20) -> RecentProjectsResponse:
    safe_limit = max(1, min(limit, 50))
    return RecentProjectsResponse(projects=list_recent_projects(limit=safe_limit))


@router.get("/provider/branches", response_model=ProviderBranchesResponse)
def provider_branches(
    git_provider: str,
    project_path: str,
) -> ProviderBranchesResponse:
    try:
        return list_provider_branches(
            git_provider=git_provider,
            project_path=project_path,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/threads", response_model=ThreadListResponse)
def threads_list() -> ThreadListResponse:
    return list_threads()


@router.post("/threads", response_model=ThreadSummary)
def threads_upsert(request: ThreadUpsertRequest) -> ThreadSummary:
    try:
        return upsert_thread(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/repo/open", response_model=RepoOpenResponse)
def repo_open(request: RepoOpenRequest) -> RepoOpenResponse:
    try:
        return open_repository(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/repo/open-plan", response_model=RepoOpenPlanResponse)
def repo_open_plan(request: RepoOpenRequest) -> RepoOpenPlanResponse:
    try:
        return plan_repository_open(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/fs/read", response_model=FileReadResponse)
def fs_read(request: FileReadRequest) -> FileReadResponse:
    try:
        return read_text_file(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/fs/write", response_model=FileWriteResponse)
def fs_write(request: FileWriteRequest) -> FileWriteResponse:
    try:
        response = write_text_file(request)
        record_applied_file_write(request)
        return response
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/cmd/run", response_model=CommandRunResponse)
def cmd_run(request: CommandRunRequest) -> CommandRunResponse:
    try:
        result = run_command_with_policy(
            request.command,
            approval_id=getattr(request, "approval_id", None),
            remember_for_session=bool(getattr(request, "remember_for_session", False)),
            project_path=request.project_path,
            reason="Compatibility command route. Commands still use RepoOperator command approval policy.",
        )
        return CommandRunResponse(
            project_path=request.project_path,
            command=result.get("display_command") or request.command,
            timeout_seconds=request.timeout_seconds or get_settings().default_command_timeout_seconds,
            exit_code=result.get("exit_code"),
            stdout=result.get("stdout", ""),
            stderr=result.get("stderr", ""),
            timed_out=False,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/git/diff", response_model=GitDiffResponse)
def git_diff(request: GitDiffRequest) -> GitDiffResponse:
    try:
        return get_diff(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/git/branches", response_model=GitBranchListResponse)
def git_branches(project_path: str) -> GitBranchListResponse:
    try:
        return list_local_branches(GitBranchListRequest(project_path=project_path))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/git/checkout", response_model=GitCheckoutResponse)
def git_checkout(request: GitCheckoutRequest) -> GitCheckoutResponse:
    try:
        return checkout_branch(request)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/git/branch", response_model=GitBranchCreateResponse)
def git_branch(request: GitBranchCreateRequest) -> GitBranchCreateResponse:
    try:
        return create_branch(request)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/git/commit", response_model=GitCommitResponse)
def git_commit(request: GitCommitRequest) -> GitCommitResponse:
    try:
        return commit_changes(request)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/git/push", response_model=GitPushResponse)
def git_push(request: GitPushRequest) -> GitPushResponse:
    try:
        return push_branch(request)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/git/merge-request", response_model=GitMergeRequestCreateResponse)
def git_merge_request(
    request: GitMergeRequestCreateRequest,
) -> GitMergeRequestCreateResponse:
    try:
        return create_provider_merge_request(request)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/agent/run", response_model=AgentRunResponse)
def agent_run(request: AgentRunRequest) -> AgentRunResponse:
    try:
        return start_run(request, stream=False)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/agent/run/stream")
def agent_run_stream(request: AgentRunRequest) -> StreamingResponse:
    _, event_stream = stream_run(request)

    return StreamingResponse(
        event_stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/agent/runs/active")
def agent_runs_active(thread_id: str | None = None) -> dict:
    return {"runs": get_active_run(thread_id=thread_id)}


@router.get("/agent/runs/{run_id}")
def agent_run_lookup(run_id: str) -> dict:
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    return run


@router.get("/agent/runs/{run_id}/events")
def agent_run_events(run_id: str, after_sequence: int = 0) -> dict:
    return {"events": list_events(run_id, after_sequence=after_sequence)}


@router.post("/agent/runs/{run_id}/steer")
def agent_run_steer(run_id: str, payload: dict) -> dict:
    try:
        return steer_run(
            run_id,
            content=payload.get("content"),
            queued_message_id=payload.get("queued_message_id"),
        )
    except ValueError as exc:
        status_code = 404 if "not found" in str(exc).lower() else 400
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc


@router.post("/agent/runs/{run_id}/cancel")
def agent_run_cancel(run_id: str) -> dict:
    try:
        return {"status": "cancelled", "run": cancel_run(run_id)}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/agent/runs/{run_id}/change-set/apply")
def agent_run_apply_change_set(run_id: str, payload: dict) -> dict:
    try:
        return resume_approval(
            run_id,
            {
                "decision": payload.get("decision") or "allow",
                "kind": "change_set_apply",
                "proposal_id": payload.get("proposal_id"),
            },
        )
    except ValueError as exc:
        status_code = 404 if "not found" in str(exc).lower() else 400
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc


@router.post("/agent/runs/{run_id}/change-set/reject")
def agent_run_reject_change_set(run_id: str, payload: dict) -> dict:
    try:
        return resume_approval(
            run_id,
            {
                "decision": payload.get("decision") or "deny",
                "kind": "change_set_apply",
                "proposal_id": payload.get("proposal_id"),
            },
        )
    except ValueError as exc:
        status_code = 404 if "not found" in str(exc).lower() else 400
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc


@router.post("/agent/queue")
def agent_queue_create(payload: dict) -> dict:
    try:
        item = enqueue_message(
            payload.get("thread_id"),
            str(payload.get("repo") or payload.get("project_path") or ""),
            payload.get("branch"),
            str(payload.get("content") or ""),
        )
        return {"item": item}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/agent/queue")
def agent_queue_list(thread_id: str | None = None, repo: str | None = None, branch: str | None = None) -> dict:
    return {"items": list_queue(thread_id=thread_id, repo=repo, branch=branch)}


@router.delete("/agent/queue/{queue_id}")
def agent_queue_cancel(queue_id: str) -> dict:
    try:
        return {"item": cancel_queued_message(queue_id)}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/agent/apply-summary")
def agent_apply_summary(payload: dict) -> dict:
    return generate_apply_summary(payload)


@router.post("/agent/propose-file", response_model=AgentProposeFileResponse)
def agent_propose_file(request: AgentProposeFileRequest) -> AgentProposeFileResponse:
    try:
        return propose_file_edit(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
