from __future__ import annotations

import sys
from typing import Any

from tools.base import BaseTool

from .common import *  # noqa: F403
from .config_tools import ConfigTool
from .interaction_tools import (
    AskUserQuestionTool,
    EnterPlanModeTool,
    ExitPlanModeTool,
    SendUserMessageTool,
    ToolSearchTool,
    _format_tool_search_item,
)
from .mcp_tools import ListMcpResourcesTool, ReadMcpResourceTool, _local_mcp_resources
from .process_runtime import *  # noqa: F403
from .project_tools import (
    AgentTool,
    SendMessageTool,
    TaskCreateTool,
    TaskGetTool,
    TaskListTool,
    TaskUpdateTool,
    TeamCreateTool,
    TeamDeleteTool,
    _agents_path,
    _load_records,
    _siyi_command,
    _task_records_path,
    _teams_path,
)
from .shell_tools import (
    BashTool,
    ExecCommandTool,
    PowerShellTool,
    StopCommandTool,
    WriteStdinTool,
)
from .skill_tools import (
    SkillTool,
    _dedupe_skill_roots,
    _discover_skills,
    _load_skill_paths,
    _skill_root_source,
    _skill_roots,
)
from .web_tools import WebFetchTool, WebSearchTool, _clean_duckduckgo_url
from .workflow_tools import WorkflowTool, _discover_workflows, _load_workflow_definition


def register_builtin_tools(registry: Any) -> None:
    tools: list[BaseTool] = [
        ExecCommandTool(),
        WriteStdinTool(),
        StopCommandTool(),
        WebSearchTool(),
        WebFetchTool(),
        AskUserQuestionTool(),
        EnterPlanModeTool(),
        ExitPlanModeTool(),
        SendUserMessageTool(),
        AgentTool(),
        SendMessageTool(),
        TeamCreateTool(),
        TeamDeleteTool(),
        TaskCreateTool(),
        TaskGetTool(),
        TaskListTool(),
        TaskUpdateTool(),
        SkillTool(),
        ListMcpResourcesTool(),
        ReadMcpResourceTool(),
        ConfigTool(),
    ]
    if _bash_exe() is not None:
        tools.insert(3, BashTool())
    if sys.platform.startswith("win"):
        tools.insert(4 if _bash_exe() is not None else 3, PowerShellTool())
    for tool in tools:
        registry.register(tool)

    registry.register(ToolSearchTool(registry.get_model_tools, registry.get_aliases))
    registry.register(WorkflowTool(lambda: registry))

    aliases = {
        "web_search": "WebSearch",
        "web_fetch": "WebFetch",
    }
    if registry.find_tool("PowerShell") is not None:
        aliases["shell"] = "PowerShell"
    elif registry.find_tool("Bash") is not None:
        aliases["shell"] = "Bash"
    for alias, target_name in aliases.items():
        target = registry.find_tool(target_name)
        if target is not None:
            registry.register_alias(alias, target_name)


__all__ = [
    "DEFAULT_NON_TTY_TIMEOUT_MS",
    "DEFAULT_OUTPUT_IDLE_TIMEOUT_MS",
    "DEFAULT_TTY_TIMEOUT_MS",
    "MAX_TOOL_WAIT_MS",
    "JsonObject",
    "PROCESSES",
    "AgentTool",
    "AskUserQuestionTool",
    "BashTool",
    "ConfigTool",
    "EnterPlanModeTool",
    "ExecCommandTool",
    "ExitPlanModeTool",
    "ListMcpResourcesTool",
    "PowerShellTool",
    "ReadMcpResourceTool",
    "SendMessageTool",
    "SendUserMessageTool",
    "SkillTool",
    "StopCommandTool",
    "TaskCreateTool",
    "TaskGetTool",
    "TaskListTool",
    "TaskUpdateTool",
    "TeamCreateTool",
    "TeamDeleteTool",
    "ToolSearchTool",
    "WebFetchTool",
    "WebSearchTool",
    "WorkflowTool",
    "WriteStdinTool",
    "ProcessSession",
    "ProcessSessionManager",
    "_StreamBuffer",
    "_agents_path",
    "_bash_exe",
    "_clean_duckduckgo_url",
    "_coerce_ms",
    "_coerce_source",
    "_combine_process_output",
    "_command_for_shell",
    "_project_root",
    "_dedupe_skill_roots",
    "_default_shell",
    "_discover_skills",
    "_discover_workflows",
    "_drain_reader_tasks",
    "_error",
    "_exec_error_result",
    "_exec_max_chars",
    "_exec_output_idle_timeout_ms",
    "_exec_result",
    "_exec_timeout_ms",
    "_format_process_result",
    "_format_tool_search_item",
    "_load_records",
    "_load_skill_paths",
    "_load_workflow_definition",
    "_local_mcp_resources",
    "_ok",
    "_powershell_exe",
    "_process_creation_kwargs",
    "_read_json_file",
    "_read_process_stream",
    "_resolve_path",
    "_run_process",
    "_schema",
    "_signal_process_tree",
    "_siyi_command",
    "_skill_root_source",
    "_skill_roots",
    "_state_dir",
    "_task_records_path",
    "_teams_path",
    "_terminate_process_tree",
    "_timeout_seconds",
    "_write_json_file",
    "register_builtin_tools",
]
