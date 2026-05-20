from __future__ import annotations

from repooperator_worker.agent_core.capabilities.base import CapabilitySpec
from repooperator_worker.agent_core.capabilities.registry import CapabilityRegistry


REPOSITORY_READ_TOOLS = [
    "inspect_repo_tree",
    "search_files",
    "search_text",
    "read_file",
    "read_many_files",
    "analyze_repository",
]
REPOSITORY_WRITE_TOOLS = [
    "generate_change_set",
    "validate_change_set",
    "apply_change_set",
    "create_file",
    "modify_file",
    "delete_file",
    "rename_file",
    "generate_edit",
]
COMMAND_TOOLS = ["preview_command", "inspect_git_state", "run_approved_command", "run_validation_command"]
WEB_RESEARCH_TOOLS = ["search_web", "fetch_url", "summarize_web_evidence"]
GIT_PROVIDER_TOOLS = [
    "git_status",
    "git_diff",
    "git_log",
    "git_branch_create",
    "git_commit",
    "git_push",
    "github_create_pr",
    "gitlab_create_mr",
]
ROUTINE_TOOLS = ["routine_list", "routine_create", "routine_enable", "routine_run_now", "routine_runs"]
CONTEXT_MEMORY_TOOLS = ["ask_clarification", "final_answer"]
VALIDATION_TOOLS = ["validate_change_set", "run_validation_command"]
MULTI_AGENT_TOOLS: list[str] = []


def built_in_capabilities() -> list[CapabilitySpec]:
    return [
        CapabilitySpec(
            name="repository_read",
            category="repository_read",
            description="Read repository structure, text files, and local evidence without modifying files.",
            tools=REPOSITORY_READ_TOOLS,
            required_permissions=["repository_read"],
            side_effect_level="read",
            available=True,
        ),
        CapabilitySpec(
            name="repository_write",
            category="repository_write",
            description="Prepare, validate, and apply repository file changes through approved change sets.",
            tools=REPOSITORY_WRITE_TOOLS,
            required_permissions=["repository_write"],
            side_effect_level="write",
            requires_approval=True,
            available=True,
        ),
        CapabilitySpec(
            name="command_execution",
            category="command_execution",
            description="Preview and run local commands through command policy and approval checks.",
            tools=COMMAND_TOOLS,
            required_permissions=["command_execution"],
            side_effect_level="command",
            requires_approval=True,
            available=True,
        ),
        CapabilitySpec(
            name="web_research",
            category="web_research",
            description="Search and fetch external web evidence as untrusted source material.",
            tools=WEB_RESEARCH_TOOLS,
            required_permissions=["network"],
            side_effect_level="network",
            network_access=True,
            requires_approval=True,
            available=True,
        ),
        CapabilitySpec(
            name="git_provider",
            category="git_provider",
            description="Inspect local git state and perform approval-gated provider write workflows.",
            tools=GIT_PROVIDER_TOOLS,
            required_permissions=["git_read", "git_local_write", "git_remote_write"],
            side_effect_level="remote_write",
            network_access=True,
            requires_approval=True,
            available=True,
        ),
        CapabilitySpec(
            name="routine",
            category="routine",
            description="Persist and enqueue recurring agent runs without bypassing normal run permissions.",
            tools=ROUTINE_TOOLS,
            required_permissions=["routine_manage"],
            side_effect_level="none",
            requires_approval=False,
            available=True,
        ),
        CapabilitySpec(
            name="context_memory",
            category="context_memory",
            description="Pack model-aware short-term evidence memory for the current thread.",
            tools=CONTEXT_MEMORY_TOOLS,
            required_permissions=[],
            side_effect_level="none",
            available=True,
        ),
        CapabilitySpec(
            name="validation",
            category="validation",
            description="Validate proposals and safe commands before reporting or applying changes.",
            tools=VALIDATION_TOOLS,
            required_permissions=["validation"],
            side_effect_level="none",
            requires_approval=False,
            available=True,
        ),
        CapabilitySpec(
            name="multi_agent",
            category="multi_agent",
            description="Decompose complex work into bounded worker reports reduced by the supervisor graph.",
            tools=MULTI_AGENT_TOOLS,
            required_permissions=[],
            side_effect_level="none",
            available=True,
        ),
    ]


def get_default_capability_registry() -> CapabilityRegistry:
    return CapabilityRegistry(built_in_capabilities())


def built_in_tool_capability_map() -> dict[str, list[str]]:
    registry = get_default_capability_registry()
    return registry.tool_map()
