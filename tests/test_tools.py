from __future__ import annotations

import asyncio
import json
import shutil
import sys
from collections.abc import AsyncIterator
from pathlib import Path

from app.cli import parse_args
from app.main import build_runtime
from config.paths import get_global_permissions_path, get_project_state_dir, get_siyi_config_path
from engine.events import ErrorEvent, FinalAnswerEvent, ToolOutputDeltaEvent, ToolResultEvent
from engine.message_schema import ToolResultBlock, ToolUseBlock
from llm.base import LLMAdapter, LLMAssistantDone, LLMTextDelta, LLMToolUse
from runtime.usage_tracker import Usage
from tools.base import BaseTool, ToolContext, ToolResult, emit_tool_output_delta
import tools.builtin as builtin_module


def _context(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=str(tmp_path), trace_id="test", max_result_chars=20_000)


def test_default_registry_exposes_requested_tools(tmp_path: Path) -> None:
    runtime = build_runtime(parse_args(["--cwd", str(tmp_path)]))

    names = {tool.name for tool in runtime.tool_registry.get_tools()}
    model_tool_names = {
        schema["function"]["name"]
        for schema in runtime.tool_registry.to_model_tool_schemas()
    }
    alias_names = {
        "read_file",
        "edit_file",
        "write_file",
        "web_search",
        "web_fetch",
        "glob",
        "grep",
        "shell",
    }

    expected = {
        "Read",
        "Edit",
        "Write",
        "NotebookEdit",
        "exec_command",
        "write_stdin",
        "stop_command",
        "Glob",
        "Grep",
        "WebSearch",
        "WebFetch",
        "ToolSearch",
        "AskUserQuestion",
        "EnterPlanMode",
        "ExitPlanMode",
        "SendUserMessage",
        "Agent",
        "SendMessage",
        "TeamCreate",
        "TeamDelete",
        "TaskCreate",
        "TaskGet",
        "TaskList",
        "TaskUpdate",
        "Skill",
        "ListMcpResourcesTool",
        "ReadMcpResourceTool",
        "workflow",
        "Config",
    }
    if shutil.which("bash") is not None:
        expected.add("Bash")
    if sys.platform.startswith("win"):
        expected.add("PowerShell")
    assert expected <= names
    assert alias_names.isdisjoint(names)
    assert alias_names.isdisjoint(model_tool_names)
    assert expected <= model_tool_names


def test_registry_resolves_aliases_without_exposing_them_to_model(tmp_path: Path) -> None:
    runtime = build_runtime(parse_args(["--cwd", str(tmp_path)]))
    registry = runtime.tool_registry

    assert registry.get_aliases()["read_file"] == "Read"
    assert registry.find_tool("read_file") is registry.find_tool("Read")
    assert registry.find_tool("edit_file") is registry.find_tool("Edit")
    assert registry.find_tool("write_file") is registry.find_tool("Write")
    assert registry.find_tool("web_search") is registry.find_tool("WebSearch")
    assert registry.find_tool("web_fetch") is registry.find_tool("WebFetch")
    assert registry.find_tool("glob") is registry.find_tool("Glob")
    assert registry.find_tool("grep") is registry.find_tool("Grep")

    shell = registry.find_tool("shell")
    if sys.platform.startswith("win"):
        assert shell is registry.find_tool("PowerShell")
    elif shutil.which("bash") is not None:
        assert shell is registry.find_tool("Bash")


def test_tool_schemas_remove_count_limits_except_tool_search(tmp_path: Path) -> None:
    runtime = build_runtime(parse_args(["--cwd", str(tmp_path)]))

    read = runtime.tool_registry.find_tool("Read")
    glob = runtime.tool_registry.find_tool("Glob")
    grep = runtime.tool_registry.find_tool("Grep")
    web_search = runtime.tool_registry.find_tool("WebSearch")
    tool_search = runtime.tool_registry.find_tool("ToolSearch")
    agent = runtime.tool_registry.find_tool("Agent")
    task_create = runtime.tool_registry.find_tool("TaskCreate")
    assert read is not None and glob is not None and grep is not None
    assert web_search is not None and tool_search is not None
    assert agent is not None and task_create is not None

    assert "limit" not in read.input_schema["properties"]
    assert "max_results" not in glob.input_schema["properties"]
    assert "max_matches" not in grep.input_schema["properties"]
    assert "max_results" not in web_search.input_schema["properties"]
    assert tool_search.input_schema["properties"]["max_results"]["default"] == 50
    assert "run" not in agent.input_schema["properties"]
    assert "command" not in task_create.input_schema["properties"]
    assert "background" not in task_create.input_schema["properties"]
    assert "shell" not in task_create.input_schema["properties"]


def test_io_tools_expose_timeout_ms_and_default_to_ten_seconds(tmp_path: Path) -> None:
    runtime = build_runtime(parse_args(["--cwd", str(tmp_path)]))

    web_search = runtime.tool_registry.find_tool("WebSearch")
    web_fetch = runtime.tool_registry.find_tool("WebFetch")
    assert web_search is not None and web_fetch is not None

    for tool_name in ("PowerShell", "Bash", "WebSearch", "WebFetch"):
        tool = runtime.tool_registry.find_tool(tool_name)
        if tool is None:
            continue
        properties = tool.input_schema["properties"]
        assert properties["timeout_ms"]["default"] == 10000
        assert "timeout_seconds" not in properties
        assert "timeout" not in properties
        if tool_name in {"PowerShell", "Bash"}:
            assert "input" not in properties
            assert "run_in_background" not in properties

    assert builtin_module._timeout_seconds({}) == 10
    assert builtin_module._timeout_seconds({"timeout_ms": 30_000}) == 30
    assert builtin_module._timeout_seconds({"timeout": 120}) == 10
    assert builtin_module._timeout_seconds({"timeout_seconds": 45}) == 10

    context = _context(tmp_path)
    validation = web_search.validate_input({"query": "x", "timeout": 1}, context)
    assert not validation.ok
    validation = web_fetch.validate_input(
        {"url": "https://example.com", "timeout_seconds": 1},
        context,
    )
    assert not validation.ok


def test_file_tools_are_callable(tmp_path: Path) -> None:
    runtime = build_runtime(parse_args(["--cwd", str(tmp_path)]))
    context = _context(tmp_path)

    write = runtime.tool_registry.find_tool("Write")
    read = runtime.tool_registry.find_tool("Read")
    edit = runtime.tool_registry.find_tool("Edit")
    assert write is not None and read is not None and edit is not None

    write_result = asyncio.run(
        write.run({"file_path": "note.txt", "content": "alpha\nbeta\n"}, context)
    )
    assert write_result.success

    edit_result = asyncio.run(
        edit.run(
            {"file_path": "note.txt", "old_string": "beta", "new_string": "gamma"},
            context,
        )
    )
    assert edit_result.success

    read_result = asyncio.run(read.run({"file_path": "note.txt"}, context))
    assert read_result.success
    assert "gamma" in read_result.content


def test_notebook_edit_tool_is_callable(tmp_path: Path) -> None:
    runtime = build_runtime(parse_args(["--cwd", str(tmp_path)]))
    context = _context(tmp_path)
    path = tmp_path / "demo.ipynb"
    path.write_text(
        json.dumps(
            {
                "cells": [],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )

    tool = runtime.tool_registry.find_tool("NotebookEdit")
    assert tool is not None
    result = asyncio.run(
        tool.run(
            {
                "notebook_path": "demo.ipynb",
                "cell_index": 0,
                "action": "insert",
                "cell_type": "markdown",
                "source": "# Hello\n",
            },
            context,
        )
    )

    assert result.success
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["cells"][0]["cell_type"] == "markdown"


def test_local_execution_tools_are_callable(tmp_path: Path) -> None:
    runtime = build_runtime(parse_args(["--cwd", str(tmp_path)]))
    context = _context(tmp_path)
    tool = runtime.tool_registry.find_tool("PowerShell")
    if not sys.platform.startswith("win"):
        assert tool is None
        return
    assert tool is not None
    if shutil.which("pwsh") is None and shutil.which("powershell") is None and shutil.which("powershell.exe") is None:
        return

    result = asyncio.run(tool.run({"command": "Write-Output siyi-tool-ok"}, context))

    assert result.success
    assert "siyi-tool-ok" in result.content


def _available_shell(runtime) -> str | None:
    if runtime.tool_registry.find_tool("PowerShell") is not None:
        return "PowerShell"
    if runtime.tool_registry.find_tool("Bash") is not None:
        return "Bash"
    return None


def _script_command(script: Path, shell: str) -> str:
    if shell == "PowerShell":
        return f'& "{sys.executable}" "{script}"'
    return f'"{sys.executable}" "{script}"'


def test_shell_tool_closes_stdin_by_default(tmp_path: Path) -> None:
    runtime = build_runtime(parse_args(["--cwd", str(tmp_path)]))
    shell = _available_shell(runtime)
    if shell is None:
        return
    script = tmp_path / "stdin_check.py"
    script.write_text(
        "import sys\n"
        "data = sys.stdin.readline()\n"
        "print('stdin_closed' if data == '' else 'stdin_open', flush=True)\n",
        encoding="utf-8",
    )
    tool = runtime.tool_registry.find_tool(shell)
    assert tool is not None

    result = asyncio.run(
        tool.run({"command": _script_command(script, shell), "timeout_ms": 3000}, _context(tmp_path))
    )

    assert result.success
    assert "stdin_closed" in result.content


def test_shell_tool_timeout_returns_timed_out(tmp_path: Path) -> None:
    runtime = build_runtime(parse_args(["--cwd", str(tmp_path)]))
    shell = _available_shell(runtime)
    if shell is None:
        return
    script = tmp_path / "sleep.py"
    script.write_text("import time\ntime.sleep(5)\n", encoding="utf-8")
    tool = runtime.tool_registry.find_tool(shell)
    assert tool is not None

    result = asyncio.run(
        tool.run({"command": _script_command(script, shell), "timeout_ms": 100}, _context(tmp_path))
    )

    assert not result.success
    assert result.data is not None
    assert result.data["timed_out"] is True


def test_exec_command_supports_idle_stdin_poll_and_stop(tmp_path: Path) -> None:
    runtime = build_runtime(parse_args(["--cwd", str(tmp_path)]))
    shell = _available_shell(runtime)
    if shell is None:
        return
    script = tmp_path / "exec_interactive.py"
    script.write_text(
        "import sys, time\n"
        "print('ready', flush=True)\n"
        "line = sys.stdin.readline().strip()\n"
        "print('got:' + line, flush=True)\n"
        "time.sleep(0.3)\n"
        "print('later', flush=True)\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    context = _context(tmp_path)
    exec_tool = runtime.tool_registry.find_tool("exec_command")
    write_tool = runtime.tool_registry.find_tool("write_stdin")
    stop_tool = runtime.tool_registry.find_tool("stop_command")
    assert exec_tool is not None and write_tool is not None and stop_tool is not None

    async def run_session() -> tuple[dict, dict, dict, dict]:
        started_result = await exec_tool.run(
            {
                "shell": shell,
                "cmd": _script_command(script, shell),
                "tty": True,
                "output_idle_timeout_ms": 1000,
                "timeout_ms": 5000,
            },
            context,
        )
        started = json.loads(started_result.content)
        written_result = await write_tool.run(
            {
                "session_id": started["session_id"],
                "chars": "yes\n",
                "output_idle_timeout_ms": 100,
            },
            context,
        )
        written = json.loads(written_result.content)
        polled_result = await write_tool.run(
            {
                "session_id": started["session_id"],
                "chars": "",
                "output_idle_timeout_ms": 500,
            },
            context,
        )
        polled = json.loads(polled_result.content)
        stopped_result = await stop_tool.run({"session_id": started["session_id"]}, context)
        stopped = json.loads(stopped_result.content)
        return started, written, polled, stopped

    started, written, polled, stopped = asyncio.run(run_session())

    assert started["session_id"]
    assert started["status"] == "running"
    assert "ready" in started["output"]
    assert "got:yes" in written["output"]
    assert "later" in polled["output"]
    assert stopped["session_id"] is None
    assert stopped["status"] in {"stopped", "timed_out", "failed", "completed"}


def test_exec_command_non_tty_closes_stdin_and_timeout_kills(tmp_path: Path) -> None:
    runtime = build_runtime(parse_args(["--cwd", str(tmp_path)]))
    shell = _available_shell(runtime)
    if shell is None:
        return
    context = _context(tmp_path)
    exec_tool = runtime.tool_registry.find_tool("exec_command")
    write_tool = runtime.tool_registry.find_tool("write_stdin")
    stop_tool = runtime.tool_registry.find_tool("stop_command")
    assert exec_tool is not None and write_tool is not None and stop_tool is not None
    stdin_script = tmp_path / "exec_stdin_check.py"
    stdin_script.write_text(
        "import sys\n"
        "data = sys.stdin.readline()\n"
        "print('stdin_closed' if data == '' else 'stdin_open', flush=True)\n",
        encoding="utf-8",
    )
    sleep_script = tmp_path / "exec_sleep.py"
    sleep_script.write_text("import time\ntime.sleep(5)\n", encoding="utf-8")

    async def run_checks() -> tuple[dict, dict, dict]:
        closed_result = await exec_tool.run(
            {
                "shell": shell,
                "cmd": _script_command(stdin_script, shell),
                "tty": False,
                "output_idle_timeout_ms": 1000,
                "timeout_ms": 3000,
            },
            context,
        )
        non_tty_running_result = await exec_tool.run(
            {
                "shell": shell,
                "cmd": _script_command(sleep_script, shell),
                "tty": False,
                "output_idle_timeout_ms": 100,
                "timeout_ms": 5000,
            },
            context,
        )
        non_tty_running = json.loads(non_tty_running_result.content)
        non_tty_write_result = await write_tool.run(
            {
                "session_id": non_tty_running["session_id"],
                "chars": "x\n",
                "output_idle_timeout_ms": 100,
            },
            context,
        )
        await stop_tool.run({"session_id": non_tty_running["session_id"]}, context)
        timed_result = await exec_tool.run(
            {
                "shell": shell,
                "cmd": _script_command(sleep_script, shell),
                "tty": True,
                "output_idle_timeout_ms": 1000,
                "timeout_ms": 100,
            },
            context,
        )
        return (
            json.loads(closed_result.content),
            json.loads(non_tty_write_result.content),
            json.loads(timed_result.content),
        )

    closed, non_tty_write, timed = asyncio.run(run_checks())

    assert closed["session_id"] is None
    assert "stdin_closed" in closed["output"]
    assert non_tty_write["error"] == "stdin is not enabled for this session"
    assert timed["status"] == "timed_out"


def test_project_management_and_lookup_tools_are_callable(tmp_path: Path) -> None:
    runtime = build_runtime(parse_args(["--cwd", str(tmp_path)]))
    context = _context(tmp_path)
    state = get_project_state_dir(tmp_path)
    (state / "mcp").mkdir(parents=True)
    (state / "mcp" / "resources.json").write_text(
        json.dumps({"resources": [{"uri": "local://note", "name": "note", "text": "hello mcp"}]}),
        encoding="utf-8",
    )
    (state / "workflows").mkdir()
    (state / "workflows" / "empty.json").write_text(json.dumps({"steps": []}), encoding="utf-8")

    async def call(name: str, payload: dict) -> str:
        tool = runtime.tool_registry.find_tool(name)
        assert tool is not None, name
        result = await tool.run(payload, context)
        assert result.success, (name, result.content)
        return result.content

    async def run_smoke() -> None:
        await call("ToolSearch", {"query": "Read"})
        await call("Config", {"action": "set", "key": "alpha", "value": 1})
        await call("Config", {"action": "get", "key": "alpha"})
        await call("Skill", {"action": "list"})
        await call("ListMcpResourcesTool", {})
        await call("ReadMcpResourceTool", {"uri": "local://note"})
        await call("workflow", {"action": "list"})
        await call("workflow", {"action": "run", "name": "empty"})
        await call("Agent", {"action": "create", "name": "helper", "role": "test"})
        agents = json.loads(await call("Agent", {"action": "list"}))
        agent_id = agents[0]["agent_id"]
        await call("SendMessage", {"target_id": agent_id, "message": "hello"})
        await call("TeamCreate", {"name": "team", "agent_ids": [agent_id]})
        teams = json.loads((state / "teams.json").read_text(encoding="utf-8"))
        await call("TeamDelete", {"team_id": teams[0]["team_id"]})
        await call("TaskCreate", {"title": "task"})
        tasks = json.loads((state / "task_records.json").read_text(encoding="utf-8"))
        task_id = tasks[0]["task_id"]
        await call("TaskGet", {"task_id": task_id})
        await call("TaskList", {})
        await call("TaskUpdate", {"task_id": task_id, "status": "done"})

    asyncio.run(run_smoke())
    assert not (tmp_path / ".siyi").exists()


def test_skill_tool_uses_global_roots_and_configured_paths(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    cwd = tmp_path / "workspace"
    default_skill = home / ".siyi" / "skills" / "default-skill"
    project_skill = cwd / ".siyi" / "skills" / "project-skill"
    codex_skill = home / ".codex" / "skills" / "codex-skill"
    extra_root = tmp_path / "extra-skills"
    extra_skill = extra_root / "extra-skill"
    env_root = tmp_path / "env-skills"
    env_skill = env_root / "env-skill"
    for skill_dir in (default_skill, project_skill, codex_skill, extra_skill, env_skill):
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(f"# {skill_dir.name}", encoding="utf-8")
    (home / ".siyi").mkdir(exist_ok=True)
    (home / ".siyi" / "skill_paths.json").write_text(
        json.dumps({"paths": [str(extra_root), str(extra_root), str(tmp_path / "missing")]}),
        encoding="utf-8",
    )
    monkeypatch.setattr("config.paths.Path.home", lambda: home)
    monkeypatch.setenv("SIYI_SKILLS_DIR", str(env_root))
    runtime = build_runtime(parse_args(["--cwd", str(cwd)]))
    context = _context(cwd)
    skill = runtime.tool_registry.find_tool("Skill")
    assert skill is not None

    result = asyncio.run(skill.run({"action": "list"}, context))

    assert result.success
    assert result.data is not None
    skills = {item["name"]: item for item in result.data["skills"]}
    assert set(skills) == {"default-skill", "extra-skill", "env-skill"}
    assert skills["default-skill"]["source"] == "default"
    assert skills["extra-skill"]["source"] == "configured"
    assert skills["env-skill"]["source"] == "env"
    assert skills["extra-skill"]["root"] == str(extra_root.resolve())

    search = asyncio.run(skill.run({"action": "search", "query": extra_root.name}, context))
    assert search.success
    assert search.data is not None
    assert [item["name"] for item in search.data["skills"]] == ["extra-skill"]


def test_skill_tool_search_and_read_from_global_roots(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    skill_dir = home / ".siyi" / "skills" / "writer"
    skill_dir.mkdir(parents=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text("# Writer\nUse for writing.", encoding="utf-8")
    monkeypatch.setattr("config.paths.Path.home", lambda: home)
    monkeypatch.delenv("SIYI_SKILLS_DIR", raising=False)
    runtime = build_runtime(parse_args(["--cwd", str(tmp_path)]))
    context = _context(tmp_path)
    skill = runtime.tool_registry.find_tool("Skill")
    assert skill is not None

    search = asyncio.run(skill.run({"action": "search", "query": "writer"}, context))
    read = asyncio.run(skill.run({"action": "read", "name": "writer"}, context))

    assert search.success
    assert search.data is not None
    assert search.data["skills"][0]["path"] == str(skill_path)
    assert read.success
    assert read.content == "# Writer\nUse for writing."


def test_tool_search_finds_hidden_aliases(tmp_path: Path) -> None:
    runtime = build_runtime(parse_args(["--cwd", str(tmp_path)]))
    context = _context(tmp_path)
    tool_search = runtime.tool_registry.find_tool("ToolSearch")
    assert tool_search is not None

    result = asyncio.run(tool_search.run({"query": "read_file"}, context))

    assert result.success
    assert result.data is not None
    assert result.data["tools"][0]["name"] == "Read"
    assert "read_file" in result.data["tools"][0]["aliases"]
    assert "Read (aliases: read_file)" in result.content


def test_tool_search_does_not_find_removed_process_tools(tmp_path: Path) -> None:
    runtime = build_runtime(parse_args(["--cwd", str(tmp_path)]))
    context = _context(tmp_path)
    tool_search = runtime.tool_registry.find_tool("ToolSearch")
    assert tool_search is not None

    result = asyncio.run(tool_search.run({"query": "ProcessStart"}, context))

    assert result.success
    assert result.data is not None
    assert result.data["tools"] == []
    assert "ProcessStart" not in result.content


class ToolCallingLLM(LLMAdapter):
    def __init__(self) -> None:
        self.calls = 0

    async def stream_chat(
        self,
        messages,
        system_prompt: str,
        tools,
        temperature: float,
    ) -> AsyncIterator[LLMTextDelta | LLMToolUse | LLMAssistantDone]:
        del system_prompt, tools, temperature
        self.calls += 1
        if self.calls == 1:
            yield LLMToolUse(
                block=ToolUseBlock(
                    id="toolu_test",
                    name="Write",
                    input={"file_path": "tool-loop.txt", "content": "from tool"},
                )
            )
            yield LLMAssistantDone(usage=Usage(input_tokens=1, output_tokens=1))
            return

        assert any(
            message.has_tool_result()
            and any(
                isinstance(block, ToolResultBlock)
                and block.tool_use_id == "toolu_test"
                for block in message.content
            )
            for message in messages
        )
        yield LLMTextDelta(delta="tool loop complete")
        yield LLMAssistantDone(usage=Usage(input_tokens=1, output_tokens=1))


class MultiReadAliasLLM(LLMAdapter):
    def __init__(self) -> None:
        self.calls = 0

    async def stream_chat(
        self,
        messages,
        system_prompt: str,
        tools,
        temperature: float,
    ) -> AsyncIterator[LLMTextDelta | LLMToolUse | LLMAssistantDone]:
        del system_prompt, tools, temperature
        self.calls += 1
        if self.calls == 1:
            for index in range(4):
                yield LLMToolUse(
                    block=ToolUseBlock(
                        id=f"toolu_read_{index}",
                        name="read_file",
                        input={"file_path": f"file-{index}.txt"},
                    )
                )
            yield LLMAssistantDone(usage=Usage(input_tokens=1, output_tokens=1))
            return

        assert sum(1 for message in messages if message.has_tool_result()) >= 4
        yield LLMTextDelta(delta="read aliases complete")
        yield LLMAssistantDone(usage=Usage(input_tokens=1, output_tokens=1))


class ProgressTool(BaseTool):
    name = "ProgressTool"
    description = "Emit a progress delta for tests."
    read_only = True
    input_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    async def run(self, raw_input, context: ToolContext) -> ToolResult:
        del raw_input
        await emit_tool_output_delta(context, stream="stdout", delta="progress\n")
        return ToolResult(success=True, content="done")


class ProgressToolLLM(LLMAdapter):
    def __init__(self) -> None:
        self.calls = 0

    async def stream_chat(
        self,
        messages,
        system_prompt: str,
        tools,
        temperature: float,
    ) -> AsyncIterator[LLMTextDelta | LLMToolUse | LLMAssistantDone]:
        del messages, system_prompt, tools, temperature
        self.calls += 1
        if self.calls == 1:
            yield LLMToolUse(block=ToolUseBlock(id="toolu_progress", name="ProgressTool", input={}))
            yield LLMAssistantDone(usage=Usage(input_tokens=1, output_tokens=1))
            return
        yield LLMTextDelta(delta="done")
        yield LLMAssistantDone(usage=Usage(input_tokens=1, output_tokens=1))


def test_query_engine_executes_tool_use_and_continues(tmp_path: Path) -> None:
    config_path = get_siyi_config_path()
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps({"permission_mode": "custom"}), encoding="utf-8")
    permissions_path = get_global_permissions_path()
    permissions_path.write_text(
        json.dumps({"custom_permissions": {"allow": ["Write(*)"]}}),
        encoding="utf-8",
    )
    runtime = build_runtime(parse_args(["--cwd", str(tmp_path)]))
    runtime.query_engine.llm = ToolCallingLLM()

    async def collect():
        return [event async for event in runtime.query_engine.submit_user_input("write a file")]

    events = asyncio.run(collect())

    assert any(isinstance(event, ToolResultEvent) for event in events)
    assert isinstance(events[-1], FinalAnswerEvent)
    assert (tmp_path / "tool-loop.txt").read_text(encoding="utf-8") == "from tool"
    assert runtime.query_engine.llm.calls == 2


def test_concurrent_read_file_alias_calls_do_not_trigger_permission_prompt_crash(tmp_path: Path) -> None:
    for index in range(4):
        (tmp_path / f"file-{index}.txt").write_text(f"content {index}", encoding="utf-8")
    runtime = build_runtime(parse_args(["--cwd", str(tmp_path)]))
    runtime.query_engine.llm = MultiReadAliasLLM()

    async def collect():
        return [event async for event in runtime.query_engine.submit_user_input("read files")]

    events = asyncio.run(collect())

    assert not any(isinstance(event, ErrorEvent) for event in events)
    assert sum(1 for event in events if isinstance(event, ToolResultEvent)) == 4
    assert isinstance(events[-1], FinalAnswerEvent)
    assert runtime.query_engine.llm.calls == 2


def test_tool_output_delta_events_are_emitted_but_not_persisted(tmp_path: Path) -> None:
    config_path = get_siyi_config_path()
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps({"permission_mode": "custom"}), encoding="utf-8")
    permissions_path = get_global_permissions_path()
    permissions_path.write_text(
        json.dumps({"custom_permissions": {"allow": ["ProgressTool(*)"]}}),
        encoding="utf-8",
    )
    runtime = build_runtime(parse_args(["--cwd", str(tmp_path)]))
    runtime.tool_registry.register(ProgressTool())
    runtime.query_engine.llm = ProgressToolLLM()

    async def collect():
        return [event async for event in runtime.query_engine.submit_user_input("run progress tool")]

    events = asyncio.run(collect())

    assert any(isinstance(event, ToolOutputDeltaEvent) for event in events)
    assert isinstance(events[-1], FinalAnswerEvent)
    entries = [
        json.loads(line)
        for line in runtime.query_engine.session.path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert not any(
        entry.get("kind") == "event" and entry.get("event", {}).get("type") == "tool_output_delta"
        for entry in entries
    )
