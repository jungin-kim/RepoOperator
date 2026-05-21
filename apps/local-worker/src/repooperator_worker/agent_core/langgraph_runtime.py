"""Compatibility facade for the RepoOperator LangGraph runtime."""

from __future__ import annotations

from repooperator_worker.agent_core.tool_orchestrator import ToolOrchestrator
from repooperator_worker.agent_core.graph.adapters import (
    _controller,
    _core_state_from_graph,
    _execute_ad_hoc_action,
    _execute_if_action_type,
    _execute_pending_action,
    _is_langgraph_checkpointer,
    _latest_result,
    _pending_action,
    _request,
    _task_frame,
    _updates_from_core,
    _with_checkpoint_bump,
)
from repooperator_worker.agent_core.graph.builder import (
    build_analysis_graph,
    build_edit_graph,
    build_evidence_gathering_graph,
    build_finalization_graph,
    build_git_workflow_graph,
    build_repooperator_state_graph,
    build_supervisor_graph,
    build_validation_graph,
    build_web_research_graph,
)
from repooperator_worker.agent_core.graph.checkpoints import EventServiceLangGraphSaver, get_default_langgraph_checkpointer
from repooperator_worker.agent_core.graph.runtime import (
    build_compiled_repooperator_graph,
    resume_langgraph_controller,
    run_langgraph_controller,
    stream_langgraph_controller,
)
from repooperator_worker.agent_core.graph.routes import (
    route_after_apply,
    route_after_approval,
    route_after_change_plan,
    route_after_evidence,
    route_after_interrupt_resume,
    route_after_tool_result,
    route_after_understanding,
    route_after_validation,
    route_by_stage,
    route_next_node,
    route_to_final_or_continue,
    route_to_next_node,
)
from repooperator_worker.agent_core.graph.state import (
    APPEND_REDUCER_FIELDS,
    RepoOperatorGraphState,
    UNIQUE_APPEND_REDUCER_FIELDS,
    append_items,
    append_unique_items,
    graph_config_for_request,
    initial_graph_state,
)
from repooperator_worker.agent_core.graph.nodes.analysis import *  # noqa: F401,F403
from repooperator_worker.agent_core.graph.nodes.apply import *  # noqa: F401,F403
from repooperator_worker.agent_core.graph.nodes.context import *  # noqa: F401,F403
from repooperator_worker.agent_core.graph.nodes.edit import *  # noqa: F401,F403
from repooperator_worker.agent_core.graph.nodes.evidence import *  # noqa: F401,F403
from repooperator_worker.agent_core.graph.nodes.finalization import *  # noqa: F401,F403
from repooperator_worker.agent_core.graph.nodes.git import *  # noqa: F401,F403
from repooperator_worker.agent_core.graph.nodes.routine import *  # noqa: F401,F403
from repooperator_worker.agent_core.graph.nodes.supervisor import *  # noqa: F401,F403
from repooperator_worker.agent_core.graph.nodes.understanding import *  # noqa: F401,F403
from repooperator_worker.agent_core.graph.nodes.validation import *  # noqa: F401,F403
from repooperator_worker.agent_core.graph.nodes.web import *  # noqa: F401,F403
