"""StateGraph topology builders for RepoOperator."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from repooperator_worker.agent_core.graph.state import RepoOperatorGraphState
from repooperator_worker.agent_core.graph.routes import route_after_change_plan, route_to_next_node
from repooperator_worker.agent_core.graph.nodes.analysis import (
    analysis_batch_files_node,
    analysis_file_role_node,
    analysis_graph_node,
    analysis_inventory_node,
    analysis_reduce_file_reports_node,
    analysis_route_batch_node,
    analysis_summarize_batch_node,
)
from repooperator_worker.agent_core.graph.nodes.apply import apply_change_set_node, await_approval_node, post_apply_validation_node
from repooperator_worker.agent_core.graph.nodes.context import capability_discovery_node, context_pack_node, load_context_node
from repooperator_worker.agent_core.graph.nodes.edit import (
    edit_generate_change_set_node,
    edit_locate_targets_node,
    edit_plan_change_set_node,
    edit_repair_change_set_node,
    edit_validate_change_set_node,
    generate_change_set_node,
    plan_change_set_node,
    repair_change_set_node,
    route_edit_after_validation,
    route_edit_next,
    route_edit_next_node,
    validate_change_set_node,
)
from repooperator_worker.agent_core.graph.nodes.evidence import (
    evidence_inspect_tree_node,
    evidence_rank_candidates_node,
    evidence_read_files_node,
    evidence_search_files_node,
    evidence_search_text_node,
    gather_evidence_node,
    route_evidence_next,
    route_evidence_next_node,
    update_evidence_store_node,
)
from repooperator_worker.agent_core.graph.nodes.finalization import (
    ask_clarification_node,
    final_build_response_node,
    final_emit_message_node,
    final_quality_guard_node,
    final_repair_answer_node,
    final_synthesis_node,
)
from repooperator_worker.agent_core.graph.nodes.git import (
    git_await_commit_approval_node,
    git_await_pr_approval_node,
    git_await_push_approval_node,
    git_commit_node,
    git_create_review_node,
    git_diff_node,
    git_propose_commit_summary_node,
    git_push_node,
    git_route_node,
    git_status_node,
    git_workflow_graph_node,
    route_git_workflow_next,
)
from repooperator_worker.agent_core.graph.nodes.routine import routine_enqueue_node
from repooperator_worker.agent_core.graph.nodes.supervisor import (
    decompose_task_node,
    dispatch_work_units_node,
    reduce_work_reports_node,
    supervisor_build_worker_tasks_node,
    supervisor_node,
    supervisor_reduce_worker_reports_node,
    supervisor_run_worker_tasks_node,
)
from repooperator_worker.agent_core.graph.nodes.understanding import build_task_plan_node, understand_request_node
from repooperator_worker.agent_core.graph.nodes.validation import (
    execute_tool_node,
    await_validation_approval_node,
    parse_validation_result_node,
    preview_selected_validation_command_node,
    run_selected_validation_command_node,
    select_validation_commands_node,
    validate_result_node,
    validation_approval_interrupt_node,
    validation_choose_node,
    validation_parse_errors_node,
    validation_preview_command_node,
    validation_route_next_node,
    validation_run_safe_node,
    validation_update_result_node,
)
from repooperator_worker.agent_core.graph.nodes.web import (
    route_web_research_next,
    web_decide_needed_node,
    web_fetch_sources_node,
    web_merge_evidence_node,
    web_research_graph_node,
    web_search_node,
    web_summarize_node,
)
from repooperator_worker.agent_core.graph.routes import route_next_node

def build_repooperator_state_graph() -> StateGraph:
    graph = StateGraph(RepoOperatorGraphState)
    graph.add_node("load_context", load_context_node)
    graph.add_node("capability_discovery", capability_discovery_node)
    graph.add_node("context_pack", context_pack_node)
    graph.add_node("understand_request", understand_request_node)
    graph.add_node("build_task_plan", build_task_plan_node)
    graph.add_node("route_next", route_next_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("gather_evidence", gather_evidence_node)
    graph.add_node("analysis_graph", analysis_graph_node)
    graph.add_node("execute_tool", execute_tool_node)
    graph.add_node("validate_result", validate_result_node)
    graph.add_node("plan_change_set", plan_change_set_node)
    graph.add_node("generate_change_set", generate_change_set_node)
    graph.add_node("validate_change_set", validate_change_set_node)
    graph.add_node("repair_change_set", repair_change_set_node)
    graph.add_node("ask_clarification", ask_clarification_node)
    graph.add_node("await_approval", await_approval_node)
    graph.add_node("await_change_approval", await_approval_node)
    graph.add_node("apply_change_set", apply_change_set_node)
    graph.add_node("post_apply_validation", post_apply_validation_node)
    graph.add_node("select_validation_commands", select_validation_commands_node)
    graph.add_node("preview_command", preview_selected_validation_command_node)
    graph.add_node("await_validation_approval", await_validation_approval_node)
    graph.add_node("run_validation_command", run_selected_validation_command_node)
    graph.add_node("parse_validation_result", parse_validation_result_node)
    graph.add_node("web_research_graph", web_research_graph_node)
    graph.add_node("git_workflow_graph", git_workflow_graph_node)
    graph.add_node("routine_enqueue_node", routine_enqueue_node)
    graph.add_node("decompose_task", decompose_task_node)
    graph.add_node("dispatch_work_units", dispatch_work_units_node)
    graph.add_node("reduce_work_reports", reduce_work_reports_node)
    graph.add_node("final_synthesis", final_synthesis_node)

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "capability_discovery")
    graph.add_edge("capability_discovery", "context_pack")
    graph.add_edge("context_pack", "understand_request")
    graph.add_edge("understand_request", "build_task_plan")
    graph.add_edge("build_task_plan", "route_next")
    graph.add_conditional_edges(
        "route_next",
        route_to_next_node,
        {
            "supervisor": "supervisor",
            "gather_evidence": "gather_evidence",
            "analysis_graph": "analysis_graph",
            "execute_tool": "execute_tool",
            "validate_result": "validate_result",
            "plan_change_set": "plan_change_set",
            "generate_change_set": "generate_change_set",
            "validate_change_set": "validate_change_set",
            "repair_change_set": "repair_change_set",
            "ask_clarification": "ask_clarification",
            "await_approval": "await_approval",
            "await_change_approval": "await_change_approval",
            "apply_change_set": "apply_change_set",
            "post_apply_validation": "post_apply_validation",
            "select_validation_commands": "select_validation_commands",
            "preview_command": "preview_command",
            "await_validation_approval": "await_validation_approval",
            "run_validation_command": "run_validation_command",
            "parse_validation_result": "parse_validation_result",
            "web_research_graph": "web_research_graph",
            "git_workflow_graph": "git_workflow_graph",
            "routine_enqueue_node": "routine_enqueue_node",
            "decompose_task": "decompose_task",
            "dispatch_work_units": "dispatch_work_units",
            "reduce_work_reports": "reduce_work_reports",
            "final_synthesis": "final_synthesis",
            END: END,
        },
    )
    graph.add_edge("supervisor", "route_next")
    graph.add_edge("gather_evidence", "validate_result")
    graph.add_edge("analysis_graph", "validate_result")
    graph.add_edge("execute_tool", "validate_result")
    graph.add_edge("validate_result", "route_next")
    graph.add_edge("plan_change_set", "generate_change_set")
    graph.add_edge("generate_change_set", "validate_change_set")
    graph.add_conditional_edges(
        "validate_change_set",
        route_after_change_plan,
        {
            "repair_change_set": "repair_change_set",
            "route_next": "route_next",
            "await_approval": "await_approval",
            "await_change_approval": "await_change_approval",
            "final_synthesis": "final_synthesis",
        },
    )
    graph.add_edge("repair_change_set", "route_next")
    graph.add_edge("ask_clarification", "final_synthesis")
    graph.add_edge("await_approval", "route_next")
    graph.add_edge("await_change_approval", "route_next")
    graph.add_edge("apply_change_set", "select_validation_commands")
    graph.add_edge("select_validation_commands", "preview_command")
    graph.add_edge("preview_command", "await_validation_approval")
    graph.add_edge("await_validation_approval", "run_validation_command")
    graph.add_edge("run_validation_command", "parse_validation_result")
    graph.add_edge("parse_validation_result", "route_next")
    graph.add_edge("post_apply_validation", "route_next")
    graph.add_edge("web_research_graph", "validate_result")
    graph.add_edge("git_workflow_graph", "route_next")
    graph.add_edge("routine_enqueue_node", "final_synthesis")
    graph.add_edge("decompose_task", "dispatch_work_units")
    graph.add_edge("dispatch_work_units", "reduce_work_reports")
    graph.add_edge("reduce_work_reports", "route_next")
    graph.add_edge("final_synthesis", END)
    return graph

def build_evidence_gathering_graph() -> StateGraph:
    graph = StateGraph(RepoOperatorGraphState)
    graph.add_node("route_evidence_next", route_evidence_next_node)
    graph.add_node("inspect_tree", evidence_inspect_tree_node)
    graph.add_node("rank_candidates", evidence_rank_candidates_node)
    graph.add_node("search_files", evidence_search_files_node)
    graph.add_node("search_text", evidence_search_text_node)
    graph.add_node("read_files", evidence_read_files_node)
    graph.add_node("update_evidence_store", update_evidence_store_node)
    graph.add_edge(START, "route_evidence_next")
    graph.add_conditional_edges(
        "route_evidence_next",
        route_evidence_next,
        {
            "inspect_tree": "inspect_tree",
            "rank_candidates": "rank_candidates",
            "search_files": "search_files",
            "search_text": "search_text",
            "read_files": "read_files",
            "update_evidence_store": "update_evidence_store",
            END: END,
        },
    )
    graph.add_edge("inspect_tree", "update_evidence_store")
    graph.add_edge("rank_candidates", "update_evidence_store")
    graph.add_edge("search_files", "update_evidence_store")
    graph.add_edge("search_text", "update_evidence_store")
    graph.add_edge("read_files", "update_evidence_store")
    graph.add_edge("update_evidence_store", END)
    return graph

def build_analysis_graph() -> StateGraph:
    graph = StateGraph(RepoOperatorGraphState)
    graph.add_node("inventory", analysis_inventory_node)
    graph.add_node("group_files", analysis_batch_files_node)
    graph.add_node("dispatch_file_role_workers", analysis_file_role_node)
    graph.add_node("reduce_file_reports", analysis_reduce_file_reports_node)
    graph.add_node("summarize_batch", analysis_summarize_batch_node)
    graph.add_node("route_batch_continue_or_end", analysis_route_batch_node)
    graph.add_edge(START, "inventory")
    graph.add_edge("inventory", "group_files")
    graph.add_edge("group_files", "dispatch_file_role_workers")
    graph.add_edge("dispatch_file_role_workers", "reduce_file_reports")
    graph.add_edge("reduce_file_reports", "summarize_batch")
    graph.add_edge("summarize_batch", "route_batch_continue_or_end")
    graph.add_edge("route_batch_continue_or_end", END)
    return graph

def build_edit_graph() -> StateGraph:
    graph = StateGraph(RepoOperatorGraphState)
    graph.add_node("route_edit_next", route_edit_next_node)
    graph.add_node("locate_targets", edit_locate_targets_node)
    graph.add_node("plan_change_set", edit_plan_change_set_node)
    graph.add_node("generate_change_set", edit_generate_change_set_node)
    graph.add_node("validate_change_set", edit_validate_change_set_node)
    graph.add_node("repair_change_set", edit_repair_change_set_node)
    graph.add_edge(START, "route_edit_next")
    graph.add_conditional_edges(
        "route_edit_next",
        route_edit_next,
        {
            "locate_targets": "locate_targets",
            "plan_change_set": "plan_change_set",
            "generate_change_set": "generate_change_set",
            "validate_change_set": "validate_change_set",
            "repair_change_set": "repair_change_set",
            END: END,
        },
    )
    graph.add_edge("locate_targets", "plan_change_set")
    graph.add_edge("plan_change_set", "generate_change_set")
    graph.add_edge("generate_change_set", "validate_change_set")
    graph.add_conditional_edges("validate_change_set", route_edit_after_validation, {"repair_change_set": "repair_change_set", END: END})
    graph.add_edge("repair_change_set", END)
    return graph

def build_validation_graph() -> StateGraph:
    graph = StateGraph(RepoOperatorGraphState)
    graph.add_node("choose_validation", validation_choose_node)
    graph.add_node("preview_command", validation_preview_command_node)
    graph.add_node("approval_interrupt_if_needed", validation_approval_interrupt_node)
    graph.add_node("run_safe_validation", validation_run_safe_node)
    graph.add_node("parse_errors", validation_parse_errors_node)
    graph.add_node("update_validation_result", validation_update_result_node)
    graph.add_node("route_validation_next", validation_route_next_node)
    graph.add_edge(START, "choose_validation")
    graph.add_edge("choose_validation", "preview_command")
    graph.add_edge("preview_command", "approval_interrupt_if_needed")
    graph.add_edge("approval_interrupt_if_needed", "run_safe_validation")
    graph.add_edge("run_safe_validation", "parse_errors")
    graph.add_edge("parse_errors", "update_validation_result")
    graph.add_edge("update_validation_result", "route_validation_next")
    graph.add_edge("route_validation_next", END)
    return graph

def build_web_research_graph() -> StateGraph:
    graph = StateGraph(RepoOperatorGraphState)
    graph.add_node("decide_web_needed", web_decide_needed_node)
    graph.add_node("search_web", web_search_node)
    graph.add_node("fetch_sources", web_fetch_sources_node)
    graph.add_node("summarize_web_evidence", web_summarize_node)
    graph.add_node("merge_web_evidence", web_merge_evidence_node)
    graph.add_edge(START, "decide_web_needed")
    graph.add_conditional_edges(
        "decide_web_needed",
        route_web_research_next,
        {
            "search_web": "search_web",
            "summarize_web_evidence": "summarize_web_evidence",
            END: END,
        },
    )
    graph.add_edge("search_web", "fetch_sources")
    graph.add_edge("fetch_sources", "summarize_web_evidence")
    graph.add_edge("summarize_web_evidence", "merge_web_evidence")
    graph.add_edge("merge_web_evidence", END)
    return graph

def build_git_workflow_graph() -> StateGraph:
    graph = StateGraph(RepoOperatorGraphState)
    graph.add_node("route_git_workflow", git_route_node)
    graph.add_node("git_status", git_status_node)
    graph.add_node("git_diff", git_diff_node)
    graph.add_node("propose_commit_summary", git_propose_commit_summary_node)
    graph.add_node("await_commit_approval", git_await_commit_approval_node)
    graph.add_node("git_commit", git_commit_node)
    graph.add_node("await_push_approval", git_await_push_approval_node)
    graph.add_node("git_push", git_push_node)
    graph.add_node("await_pr_approval", git_await_pr_approval_node)
    graph.add_node("create_pr_or_mr", git_create_review_node)
    graph.add_edge(START, "route_git_workflow")
    graph.add_conditional_edges(
        "route_git_workflow",
        route_git_workflow_next,
        {
            "git_status": "git_status",
            "git_diff": "git_diff",
            "propose_commit_summary": "propose_commit_summary",
            "git_commit": "git_commit",
            "git_push": "git_push",
            "create_pr_or_mr": "create_pr_or_mr",
            END: END,
        },
    )
    graph.add_edge("git_status", "route_git_workflow")
    graph.add_edge("git_diff", "route_git_workflow")
    graph.add_edge("propose_commit_summary", "await_commit_approval")
    graph.add_edge("await_commit_approval", END)
    graph.add_edge("git_commit", "await_push_approval")
    graph.add_edge("await_push_approval", END)
    graph.add_edge("git_push", "await_pr_approval")
    graph.add_edge("await_pr_approval", END)
    graph.add_edge("create_pr_or_mr", END)
    return graph

def build_finalization_graph() -> StateGraph:
    graph = StateGraph(RepoOperatorGraphState)
    graph.add_node("quality_guard", final_quality_guard_node)
    graph.add_node("repair_final_answer", final_repair_answer_node)
    graph.add_node("build_response", final_build_response_node)
    graph.add_node("emit_final_message", final_emit_message_node)
    graph.add_edge(START, "quality_guard")
    graph.add_edge("quality_guard", "repair_final_answer")
    graph.add_edge("repair_final_answer", "build_response")
    graph.add_edge("build_response", "emit_final_message")
    graph.add_edge("emit_final_message", END)
    return graph

def build_supervisor_graph() -> StateGraph:
    graph = StateGraph(RepoOperatorGraphState)
    graph.add_node("build_worker_tasks", supervisor_build_worker_tasks_node)
    graph.add_node("run_worker_task", supervisor_run_worker_tasks_node)
    graph.add_node("reduce_worker_reports", supervisor_reduce_worker_reports_node)
    graph.add_edge(START, "build_worker_tasks")
    graph.add_edge("build_worker_tasks", "run_worker_task")
    graph.add_edge("run_worker_task", "reduce_worker_reports")
    graph.add_edge("reduce_worker_reports", END)
    return graph
