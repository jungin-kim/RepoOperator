"""Compatibility facade for the RepoOperator LangGraph runtime."""

from __future__ import annotations

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
from repooperator_worker.agent_core.graph.nodes.finalization import final_emit_message_node
from repooperator_worker.agent_core.graph.nodes.supervisor import decompose_task_node, dispatch_work_units_node, reduce_work_reports_node, supervisor_node
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


__all__ = [
    "APPEND_REDUCER_FIELDS",
    "EventServiceLangGraphSaver",
    "RepoOperatorGraphState",
    "UNIQUE_APPEND_REDUCER_FIELDS",
    "append_items",
    "append_unique_items",
    "build_analysis_graph",
    "build_compiled_repooperator_graph",
    "build_edit_graph",
    "build_evidence_gathering_graph",
    "build_finalization_graph",
    "build_git_workflow_graph",
    "build_repooperator_state_graph",
    "build_supervisor_graph",
    "build_validation_graph",
    "build_web_research_graph",
    "decompose_task_node",
    "dispatch_work_units_node",
    "final_emit_message_node",
    "get_default_langgraph_checkpointer",
    "graph_config_for_request",
    "initial_graph_state",
    "reduce_work_reports_node",
    "resume_langgraph_controller",
    "route_after_apply",
    "route_after_approval",
    "route_after_change_plan",
    "route_after_evidence",
    "route_after_interrupt_resume",
    "route_after_tool_result",
    "route_after_understanding",
    "route_after_validation",
    "route_by_stage",
    "route_next_node",
    "route_to_final_or_continue",
    "route_to_next_node",
    "run_langgraph_controller",
    "stream_langgraph_controller",
    "supervisor_node",
]
