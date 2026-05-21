"""Run, resume, and stream entry points for the RepoOperator LangGraph runtime."""

from __future__ import annotations

from typing import Any, Iterator

from langgraph.types import Command

from repooperator_worker.agent_core.graph.adapters import _controller, _core_state_from_graph, _is_langgraph_checkpointer
from repooperator_worker.agent_core.graph.builder import build_repooperator_state_graph
from repooperator_worker.agent_core.graph.checkpoints import get_default_langgraph_checkpointer
from repooperator_worker.agent_core.graph.nodes.finalization import _response_with_change_set_payload
from repooperator_worker.agent_core.graph.state import graph_config_for_request, initial_graph_state
from repooperator_worker.agent_core.graph_state import request_to_snapshot, response_from_snapshot
from repooperator_worker.schemas import AgentRunRequest, AgentRunResponse
from repooperator_worker.services.event_service import append_run_event, list_run_events
from repooperator_worker.services.json_safe import safe_agent_response_payload, json_safe
from repooperator_worker.services.skills_service import enabled_skill_context

def build_compiled_repooperator_graph(*, checkpoint_adapter: Any | None = None) -> Any:
    checkpointer = checkpoint_adapter if _is_langgraph_checkpointer(checkpoint_adapter) else get_default_langgraph_checkpointer()
    return build_repooperator_state_graph().compile(checkpointer=checkpointer)

def run_langgraph_controller(
    request: AgentRunRequest,
    *,
    run_id: str | None = None,
    stream_final_answer: bool = False,
    checkpoint_adapter: Any | None = None,
) -> AgentRunResponse:
    run_id = run_id or "run_controller"
    _controller()._validate_active_repository(request)
    skills_context, skills_used = enabled_skill_context()
    initial_state = initial_graph_state(
        request,
        run_id=run_id,
        stream_final_answer=stream_final_answer,
        skills_context=skills_context,
        skills_used=skills_used,
    )
    compiled = build_compiled_repooperator_graph(checkpoint_adapter=checkpoint_adapter)
    config = graph_config_for_request(request, run_id)
    final_state = compiled.invoke(initial_state, config=config)
    if final_state.get("__interrupt__"):
        snapshot_state = dict(compiled.get_state(config).values or {})
        snapshot_state.setdefault("request_snapshot", request_to_snapshot(request))
        snapshot_state.setdefault("run_id", run_id)
        snapshot_state.setdefault("thread_id", request.thread_id)
        snapshot_state.setdefault("repo", request.project_path)
        snapshot_state.setdefault("branch", request.branch)
        return _response_from_interrupted_state(snapshot_state, request)
    response = response_from_snapshot(final_state.get("response_snapshot"))
    if isinstance(response, AgentRunResponse):
        return response
    core = _core_state_from_graph(final_state)
    if not core.final_response:
        core.final_response = final_state.get("final_response") or ""
    return _controller().build_final_response(core, request).model_copy(update={"agent_flow": "langgraph"})

def resume_langgraph_controller(
    request: AgentRunRequest,
    *,
    run_id: str,
    approval_decision: dict[str, Any],
    checkpoint_adapter: Any | None = None,
) -> AgentRunResponse:
    compiled = build_compiled_repooperator_graph(checkpoint_adapter=checkpoint_adapter)
    config = graph_config_for_request(request, run_id)
    final_state = compiled.invoke(Command(resume=json_safe(approval_decision)), config=config)
    if final_state.get("__interrupt__"):
        return _response_from_interrupted_state(dict(compiled.get_state(config).values or {}), request)
    response = response_from_snapshot(final_state.get("response_snapshot"))
    if isinstance(response, AgentRunResponse):
        return response
    core = _core_state_from_graph({**dict(final_state), "request_snapshot": request_to_snapshot(request), "run_id": run_id})
    return _controller().build_final_response(core, request).model_copy(update={"agent_flow": "langgraph"})

def stream_langgraph_controller(request: AgentRunRequest, *, run_id: str | None = None) -> Iterator[dict[str, Any]]:
    resolved_run_id = run_id or "run_controller"
    before_sequence = _latest_sequence(resolved_run_id)
    response = run_langgraph_controller(request, run_id=resolved_run_id, stream_final_answer=True)
    for event in list_run_events(resolved_run_id, after_sequence=before_sequence):
        if event.get("type") == "assistant_delta":
            before_sequence = int(event.get("sequence") or before_sequence)
            yield event
    if not any(event.get("type") == "assistant_delta" for event in list_run_events(resolved_run_id)):
        for chunk in _chunk_text(response.response):
            yield {"type": "assistant_delta", "delta": chunk, "streaming_mode": "post_hoc_chunking"}
    final = _controller()._response_json_safe(response.model_copy(update={"activity_events": []}), request)
    yield {"type": "final_message", "result": safe_agent_response_payload(final)}

def _response_from_interrupted_state(state: dict[str, Any], request: AgentRunRequest) -> AgentRunResponse:
    state = dict(state)
    state.setdefault("request_snapshot", request_to_snapshot(request))
    core = _core_state_from_graph(state)
    if not core.stop_reason:
        core.stop_reason = "waiting_approval"
    return _response_with_change_set_payload(
        _controller().build_final_response(core, request).model_copy(update={"agent_flow": "langgraph"}),
        state,
    )

def _stream_final_delta(run_id: str):
    def emit(delta: str) -> None:
        try:
            append_run_event(run_id, {"type": "assistant_delta", "delta": delta, "streaming_mode": "model_stream"})
        except OSError:
            return

    return emit

def _latest_sequence(run_id: str | None) -> int:
    if not run_id:
        return 0
    events = list_run_events(run_id)
    return max((int(event.get("sequence") or 0) for event in events), default=0)

def _chunk_text(text: str, *, size: int = 80) -> Iterator[str]:
    for index in range(0, len(text), size):
        yield text[index:index + size]
