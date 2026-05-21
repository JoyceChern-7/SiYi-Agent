from __future__ import annotations

import asyncio
import fnmatch
import html
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import httpx

from config.paths import get_global_skills_dir, get_project_state_dir, get_skill_paths_path
from runtime.ids import new_id
from tools.base import BaseTool, ToolContext, ToolResult, emit_tool_output_delta, truncate_text
from tools.governance import PermissionResult, ValidationResult
from tools.shell_analysis import analyze_bash, analyze_powershell


JsonObject = dict[str, Any]

DEFAULT_OUTPUT_IDLE_TIMEOUT_MS = 500 
# 交互式进程在没有输出时的空闲超时时间, 超过这个时间会直接将当前进程的输出结果返回给模型, 但不杀进程

DEFAULT_TTY_TIMEOUT_MS = 300_000
# 交互式进程的最大等待时间, 超过这个时间会强制杀死进程并返回结果.

DEFAULT_NON_TTY_TIMEOUT_MS = 10_000
# 非交互式进程的最大等待时间, 超过这个时间会强制杀死进程并返回结果.

MAX_TOOL_WAIT_MS = 30_000
# 工具的最大等待时间, 超过这个时间会强制返回结果, 但不杀进程.


def _schema(properties: JsonObject, required: list[str] | None = None) -> JsonObject:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def _ok(content: str, context: ToolContext, data: JsonObject | None = None) -> ToolResult:
    return ToolResult(
        success=True,
        content=truncate_text(content, context.max_result_chars),
        data=data,
    )


def _error(message: str, context: ToolContext, data: JsonObject | None = None) -> ToolResult:
    return ToolResult(
        success=False,
        content=truncate_text(message, context.max_result_chars),
        data=data,
        error=message,
    )


def _cwd(context: ToolContext) -> Path:
    return Path(context.cwd).expanduser().resolve()


def _resolve_path(context: ToolContext, value: str | None, *, default: str = ".") -> Path:
    raw = Path(value or default).expanduser()
    if not raw.is_absolute():
        raw = _cwd(context) / raw
    return raw.resolve()


def _state_dir(context: ToolContext) -> Path:
    path = get_project_state_dir(_cwd(context))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json_file(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _coerce_source(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return str(value).splitlines(keepends=True)


def _timeout_seconds(raw_input: JsonObject, *, default: float = 10.0) -> float:
    if raw_input.get("timeout_ms") is None:
        return default
    try:
        value = float(raw_input["timeout_ms"]) / 1000.0
    except (TypeError, ValueError):
        return default
    return max(0.001, value)


class ReadTool(BaseTool):
    name = "Read"
    description = "Read a text file from the local workspace with an optional line offset."
    read_only = True
    input_schema = _schema(
        {
            "file_path": {"type": "string", "description": "Path to read."},
            "path": {"type": "string", "description": "Alias for file_path."},
            "offset": {"type": "integer", "minimum": 1, "description": "1-based first line."},
        },
    )

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        path = _resolve_path(context, raw_input.get("file_path") or raw_input.get("path"))
        if not path.exists():
            return _error(f"File not found: {path}", context)
        if path.is_dir():
            return _error(f"Path is a directory: {path}", context)
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        offset = int(raw_input.get("offset") or 1)
        selected = lines[offset - 1 :]
        numbered = [
            f"{line_no:>6}\t{line}"
            for line_no, line in enumerate(selected, start=offset)
        ]
        content = "\n".join(numbered)
        return _ok(
            content,
            context,
            {
                "path": str(path),
                "total_lines": len(lines),
                "returned_lines": len(selected),
            },
        )


class WriteTool(BaseTool):
    name = "Write"
    description = "Write text to a local file, creating parent directories when needed."
    input_schema = _schema(
        {
            "file_path": {"type": "string", "description": "Path to write."},
            "path": {"type": "string", "description": "Alias for file_path."},
            "content": {"type": "string", "description": "File content."},
            "append": {"type": "boolean", "default": False},
        },
        required=["content"],
    )

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        path = _resolve_path(context, raw_input.get("file_path") or raw_input.get("path"))
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if raw_input.get("append") else "w"
        with path.open(mode, encoding="utf-8", newline="") as handle:
            handle.write(str(raw_input.get("content", "")))
        action = "appended" if raw_input.get("append") else "wrote"
        return _ok(f"{action} {path}", context, {"path": str(path)})


class EditTool(BaseTool):
    name = "Edit"
    description = "Replace text in an existing local file using exact string matching."
    input_schema = _schema(
        {
            "file_path": {"type": "string", "description": "Path to edit."},
            "path": {"type": "string", "description": "Alias for file_path."},
            "old_string": {"type": "string", "description": "Exact text to replace."},
            "new_string": {"type": "string", "description": "Replacement text."},
            "replace_all": {"type": "boolean", "default": False},
        },
        required=["old_string", "new_string"],
    )

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        path = _resolve_path(context, raw_input.get("file_path") or raw_input.get("path"))
        if not path.exists():
            return _error(f"File not found: {path}", context)
        old = str(raw_input["old_string"])
        new = str(raw_input["new_string"])
        if old == "":
            return _error("old_string must not be empty", context)
        text = path.read_text(encoding="utf-8", errors="replace")
        count = text.count(old)
        if count == 0:
            return _error("old_string was not found", context, {"path": str(path)})
        if count > 1 and not raw_input.get("replace_all"):
            return _error(
                f"old_string appears {count} times; set replace_all=true to replace all matches",
                context,
                {"path": str(path), "matches": count},
            )
        updated = text.replace(old, new) if raw_input.get("replace_all") else text.replace(old, new, 1)
        path.write_text(updated, encoding="utf-8", newline="")
        replaced = count if raw_input.get("replace_all") else 1
        return _ok(
            f"edited {path}; replacements={replaced}",
            context,
            {"path": str(path), "replacements": replaced},
        )


class NotebookEditTool(BaseTool):
    name = "NotebookEdit"
    description = "Edit a Jupyter notebook cell by replacing, inserting, or deleting cells."
    input_schema = _schema(
        {
            "notebook_path": {"type": "string", "description": "Path to .ipynb file."},
            "file_path": {"type": "string", "description": "Alias for notebook_path."},
            "cell_index": {"type": "integer", "minimum": 0},
            "action": {"type": "string", "enum": ["replace", "insert", "delete"], "default": "replace"},
            "cell_type": {"type": "string", "enum": ["code", "markdown", "raw"], "default": "code"},
            "source": {"description": "Cell source as a string or list of lines."},
        },
        required=["cell_index"],
    )

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        path = _resolve_path(
            context,
            raw_input.get("notebook_path") or raw_input.get("file_path"),
        )
        if not path.exists():
            return _error(f"Notebook not found: {path}", context)
        notebook = _read_json_file(path, {})
        cells = notebook.setdefault("cells", [])
        if not isinstance(cells, list):
            return _error("Invalid notebook: cells is not a list", context)
        index = int(raw_input["cell_index"])
        action = str(raw_input.get("action") or "replace")
        if action in {"replace", "delete"} and not 0 <= index < len(cells):
            return _error(f"cell_index out of range: {index}", context)
        if action == "insert" and not 0 <= index <= len(cells):
            return _error(f"cell_index out of range: {index}", context)

        if action == "delete":
            deleted = cells.pop(index)
            _write_json_file(path, notebook)
            return _ok(
                f"deleted cell {index} from {path}",
                context,
                {"path": str(path), "deleted_cell_type": deleted.get("cell_type")},
            )

        cell = {
            "cell_type": raw_input.get("cell_type") or "code",
            "metadata": {},
            "source": _coerce_source(raw_input.get("source")),
        }
        if cell["cell_type"] == "code":
            cell.update({"execution_count": None, "outputs": []})
        if action == "insert":
            cells.insert(index, cell)
        else:
            old_metadata = cells[index].get("metadata", {}) if isinstance(cells[index], dict) else {}
            cell["metadata"] = old_metadata
            cells[index] = cell
        _write_json_file(path, notebook)
        action_label = "inserted" if action == "insert" else "replaced"
        return _ok(
            f"{action_label} cell {index} in {path}",
            context,
            {"path": str(path), "cell_index": index, "action": action},
        )


async def _run_process(
    args: list[str],
    *,
    cwd: Path,
    timeout_seconds: float,
    context: ToolContext,
) -> tuple[int | None, str, str, bool]:
    started_at = time.perf_counter()
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    process = await asyncio.create_subprocess_exec(
        *args,
        cwd=str(cwd),
        stdin=subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **_process_creation_kwargs(),
    )

    stdout_task = asyncio.create_task(
        _read_process_stream(
            process.stdout,
            "stdout",
            stdout_parts,
            context,
            started_at=started_at,
        )
    )
    stderr_task = asyncio.create_task(
        _read_process_stream(
            process.stderr,
            "stderr",
            stderr_parts,
            context,
            started_at=started_at,
        )
    )
    try:
        returncode = await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
        timed_out = False
    except TimeoutError:
        timed_out = True
        await emit_tool_output_delta(
            context,
            stream="status",
            delta=f"command timed out after {timeout_seconds:g} seconds; terminating process tree\n",
            elapsed_ms=int((time.perf_counter() - started_at) * 1000),
        )
        await _terminate_process_tree(process)
        returncode = None

    await _drain_reader_tasks([stdout_task, stderr_task])
    stdout = "".join(stdout_parts)
    stderr = "".join(stderr_parts)
    if timed_out:
        timeout_message = f"command timed out after {timeout_seconds:g} seconds"
        stderr = f"{stderr}\n{timeout_message}".strip()
    return (returncode, stdout, stderr, timed_out)


def _process_creation_kwargs() -> JsonObject:
    if sys.platform.startswith("win"):
        flags = 0
        flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        flags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
        return {"creationflags": flags}
    return {"start_new_session": True}


async def _read_process_stream(
    stream: asyncio.StreamReader | None,
    stream_name: str,
    bucket: list[str],
    context: ToolContext,
    *,
    started_at: float,
    process_id: str | None = None,
) -> None:
    if stream is None:
        return
    while True:
        chunk = await stream.read(4096)
        if not chunk:
            return
        text = chunk.decode("utf-8", errors="replace")
        bucket.append(text)
        await emit_tool_output_delta(
            context,
            stream="stderr" if stream_name == "stderr" else "stdout",
            delta=text,
            process_id=process_id,
            elapsed_ms=int((time.perf_counter() - started_at) * 1000),
        )


async def _drain_reader_tasks(tasks: list[asyncio.Task[None]], *, timeout: float = 2.0) -> None:
    try:
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout)
    except TimeoutError:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _terminate_process_tree(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    await _signal_process_tree(process, force=False)
    try:
        await asyncio.wait_for(process.wait(), timeout=2)
        return
    except TimeoutError:
        pass
    await _signal_process_tree(process, force=True)
    with suppress(TimeoutError):
        await asyncio.wait_for(process.wait(), timeout=2)


async def _signal_process_tree(process: asyncio.subprocess.Process, *, force: bool) -> None:
    if process.returncode is not None:
        return
    if sys.platform.startswith("win"):
        args = ["taskkill", "/PID", str(process.pid), "/T"]
        if force:
            args.append("/F")
        with suppress(Exception):
            killer = await asyncio.create_subprocess_exec(
                *args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            await asyncio.wait_for(killer.wait(), timeout=5)
        return

    sig = signal.SIGKILL if force else signal.SIGTERM
    with suppress(ProcessLookupError, PermissionError):
        os.killpg(os.getpgid(process.pid), sig)
    with suppress(ProcessLookupError, PermissionError):
        process.send_signal(sig)


def _powershell_exe() -> str | None:
    return shutil.which("pwsh") or shutil.which("powershell") or shutil.which("powershell.exe")


def _bash_exe() -> str | None:
    return shutil.which("bash")


def _default_shell() -> str:
    if sys.platform.startswith("win"):
        return "PowerShell"
    if _bash_exe() is not None:
        return "Bash"
    return "PowerShell"


class PowerShellTool(BaseTool):
    name = "PowerShell"
    description = "Run a short non-interactive PowerShell command in the local working directory. Use exec_command with tty=true for long-running or interactive commands."
    input_schema = _schema(
        {
            "command": {"type": "string"},
            "description": {"type": "string"},
            "timeout_ms": {"type": "number", "minimum": 1, "default": 10000},
        },
        required=["command"],
    )

    def is_read_only(self, raw_input: JsonObject) -> bool:
        return analyze_powershell(str(raw_input.get("command") or "")).read_only

    def check_permissions(self, raw_input: JsonObject, context: ToolContext) -> PermissionResult:
        del context
        analysis = analyze_powershell(str(raw_input.get("command") or ""))
        return PermissionResult.allow(reason=analysis.reason, source=analysis.parser)

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        exe = _powershell_exe()
        if exe is None:
            return _error("PowerShell executable was not found", context)
        args = [exe, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", str(raw_input["command"])]
        timeout = _timeout_seconds(raw_input)
        return_code, stdout, stderr, timed_out = await _run_process(
            args,
            cwd=_cwd(context),
            timeout_seconds=timeout,
            context=context,
        )
        content = stdout if stdout else ""
        if stderr:
            content = f"{content}\n[stderr]\n{stderr}".strip()
        data = {"returncode": return_code, "timed_out": timed_out}
        if return_code == 0 and not timed_out:
            return _ok(content or "(command completed with no output)", context, data)
        return _error(content or "command failed", context, data)


class BashTool(BaseTool):
    name = "Bash"
    description = "Run a short non-interactive Bash command in the local working directory. Use exec_command with tty=true for long-running or interactive commands."
    input_schema = _schema(
        {
            "command": {"type": "string"},
            "description": {"type": "string"},
            "timeout_ms": {"type": "number", "minimum": 1, "default": 10000},
        },
        required=["command"],
    )

    def is_read_only(self, raw_input: JsonObject) -> bool:
        return analyze_bash(str(raw_input.get("command") or "")).read_only

    def check_permissions(self, raw_input: JsonObject, context: ToolContext) -> PermissionResult:
        del context
        analysis = analyze_bash(str(raw_input.get("command") or ""))
        return PermissionResult.allow(reason=analysis.reason, source=analysis.parser)

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        exe = _bash_exe()
        if exe is None:
            return _error("Bash executable was not found", context)
        args = [exe, "-lc", str(raw_input["command"])]
        timeout = _timeout_seconds(raw_input)
        return_code, stdout, stderr, timed_out = await _run_process(
            args,
            cwd=_cwd(context),
            timeout_seconds=timeout,
            context=context,
        )
        content = stdout if stdout else ""
        if stderr:
            content = f"{content}\n[stderr]\n{stderr}".strip()
        data = {"returncode": return_code, "timed_out": timed_out}
        if return_code == 0 and not timed_out:
            return _ok(content or "(command completed with no output)", context, data)
        return _error(content or "command failed", context, data)


def _command_for_shell(shell: str, command: str) -> list[str]:
    if shell.lower() in {"powershell", "pwsh"}:
        exe = _powershell_exe()
        if exe is None:
            raise RuntimeError("PowerShell executable was not found")
        return [exe, "-NoLogo", "-NoProfile", "-Command", command]
    if shell.lower() == "bash":
        exe = _bash_exe()
        if exe is None:
            raise RuntimeError("Bash executable was not found")
        return [exe, "-lc", command]
    raise RuntimeError(f"Unsupported shell: {shell}")


def _coerce_ms(raw_input: JsonObject, key: str, *, default: int, maximum: int | None = None) -> int:
    try:
        value = int(raw_input.get(key) if raw_input.get(key) is not None else default)
    except (TypeError, ValueError):
        value = default
    value = max(0, value)
    if maximum is not None:
        value = min(value, maximum)
    return value


class _StreamBuffer:
    def __init__(self, *, cap_chars: int = 1_048_576) -> None:
        self.cap_chars = cap_chars
        self.history = ""
        self.pending = ""
        self.history_omitted = 0
        self.pending_omitted = 0

    def append(self, text: str) -> None:
        self.history += text
        if len(self.history) > self.cap_chars:
            omitted = len(self.history) - self.cap_chars
            self.history = self.history[omitted:]
            self.history_omitted += omitted

        self.pending += text
        if len(self.pending) > self.cap_chars:
            omitted = len(self.pending) - self.cap_chars
            self.pending = self.pending[omitted:]
            self.pending_omitted += omitted

    def has_pending(self) -> bool:
        return bool(self.pending or self.pending_omitted)

    def consume(self) -> str:
        text = self.pending
        if self.pending_omitted:
            text = f"...[omitted {self.pending_omitted} chars]\n{text}"
        self.pending = ""
        self.pending_omitted = 0
        return text


@dataclass(slots=True)
class ProcessSession:
    process_id: str
    name: str
    shell: str
    command: str
    args: list[str]
    cwd: str
    process: asyncio.subprocess.Process
    tty: bool = True
    stdin_authorized: bool = True
    created_at: float = field(default_factory=time.time)
    status: str = "running"
    returncode: int | None = None
    stdin_closed: bool = False
    output_seq: int = 0
    stdout: _StreamBuffer = field(default_factory=_StreamBuffer)
    stderr: _StreamBuffer = field(default_factory=_StreamBuffer)
    output_event: asyncio.Event = field(default_factory=asyncio.Event)
    exit_event: asyncio.Event = field(default_factory=asyncio.Event)
    reader_tasks: list[asyncio.Task[None]] = field(default_factory=list)
    watcher_task: asyncio.Task[None] | None = None
    timeout_task: asyncio.Task[None] | None = None
    active_context: ToolContext | None = None
    active_started_at: float = 0.0

    def is_running(self) -> bool:
        return self.status == "running" and self.process.returncode is None

    def has_pending_output(self) -> bool:
        return self.stdout.has_pending() or self.stderr.has_pending()


class ProcessSessionManager:
    def __init__(self) -> None:
        self.sessions: dict[str, ProcessSession] = {}

    async def start(
        self,
        *,
        name: str,
        shell: str,
        command: str,
        cwd: Path,
        context: ToolContext,
        yield_time_ms: int,
        timeout_ms: int | None,
        tty: bool = True,
        output_idle_timeout_ms: int | None = None,
    ) -> tuple[ProcessSession, str, str]:
        args = _command_for_shell(shell, command)
        process = await asyncio.create_subprocess_exec(
            *args,
            cwd=str(cwd),
            stdin=asyncio.subprocess.PIPE if tty else subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **_process_creation_kwargs(),
        )
        session = ProcessSession(
            process_id=new_id("proc"),
            name=name,
            shell=shell,
            command=command,
            args=args,
            cwd=str(cwd),
            process=process,
            tty=tty,
            stdin_authorized=tty,
            stdin_closed=not tty,
            active_context=context,
            active_started_at=time.perf_counter(),
        )
        self.sessions[session.process_id] = session
        session.reader_tasks = [
            asyncio.create_task(self._read_stream(session, process.stdout, "stdout")),
            asyncio.create_task(self._read_stream(session, process.stderr, "stderr")),
        ]
        session.watcher_task = asyncio.create_task(self._watch_exit(session))
        if timeout_ms is not None:
            session.timeout_task = asyncio.create_task(self._enforce_timeout(session, timeout_ms))

        if output_idle_timeout_ms is None:
            await self._collect_for(session, context, yield_time_ms)
        else:
            await self._collect_until_idle(
                session,
                context,
                output_idle_timeout_ms=output_idle_timeout_ms,
            )
        return session, session.stdout.consume(), session.stderr.consume()

    def get(self, process_id: str) -> ProcessSession | None:
        return self.sessions.get(process_id)

    async def stop_all(self) -> None:
        sessions = list(self.sessions.values())
        for session in sessions:
            if session.is_running():
                session.status = "stopping"
                await _terminate_process_tree(session.process)
        for session in sessions:
            if session.is_running():
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(session.exit_event.wait(), timeout=2)
            await _drain_reader_tasks(session.reader_tasks)
            if session.status == "stopping":
                session.status = "stopped"
        self.sessions.clear()

    async def read(
        self,
        process_id: str,
        *,
        context: ToolContext,
        wait_ms: int,
    ) -> tuple[ProcessSession | None, str, str]:
        session = self.get(process_id)
        if session is None:
            return None, "", ""
        await self._wait_for_new_output(session, context, wait_ms)
        return session, session.stdout.consume(), session.stderr.consume()

    async def write(
        self,
        process_id: str,
        *,
        context: ToolContext,
        chars: str,
        close_stdin: bool,
        yield_time_ms: int,
        output_idle_timeout_ms: int | None = None,
    ) -> tuple[ProcessSession | None, str, str, str | None]:
        session = self.get(process_id)
        if session is None:
            return None, "", "", "process not found"
        error: str | None = None
        if chars:
            if not session.tty or not session.stdin_authorized:
                error = "stdin is not enabled for this session"
            elif session.stdin_closed or session.process.stdin is None:
                error = "stdin is closed"
            else:
                try:
                    session.process.stdin.write(chars.encode("utf-8"))
                    await session.process.stdin.drain()
                except (BrokenPipeError, ConnectionResetError):
                    session.stdin_closed = True
                    error = "stdin is closed"
        if close_stdin and not session.stdin_closed and session.process.stdin is not None:
            session.process.stdin.close()
            with suppress(Exception):
                await session.process.stdin.wait_closed()
            session.stdin_closed = True
        if output_idle_timeout_ms is None:
            await self._collect_for(session, context, yield_time_ms)
        else:
            await self._collect_until_idle(
                session,
                context,
                output_idle_timeout_ms=output_idle_timeout_ms,
            )
        return session, session.stdout.consume(), session.stderr.consume(), error

    async def stop(
        self,
        process_id: str,
        *,
        context: ToolContext,
    ) -> tuple[ProcessSession | None, str, str]:
        session = self.get(process_id)
        if session is None:
            return None, "", ""
        await self._activate(session, context)
        try:
            if session.is_running():
                session.status = "stopping"
                await emit_tool_output_delta(
                    context,
                    stream="status",
                    delta=f"stopping process {process_id}\n",
                    process_id=process_id,
                )
                await _terminate_process_tree(session.process)
                with suppress(TimeoutError):
                    await asyncio.wait_for(session.exit_event.wait(), timeout=2)
        finally:
            self._deactivate(session, context)
        await _drain_reader_tasks(session.reader_tasks)
        if session.status == "stopping":
            session.status = "stopped"
        return session, session.stdout.consume(), session.stderr.consume()

    async def _read_stream(
        self,
        session: ProcessSession,
        stream: asyncio.StreamReader | None,
        stream_name: str,
    ) -> None:
        if stream is None:
            return
        while True:
            chunk = await stream.read(4096)
            if not chunk:
                return
            text = chunk.decode("utf-8", errors="replace")
            if stream_name == "stderr":
                session.stderr.append(text)
            else:
                session.stdout.append(text)
            session.output_seq += 1
            session.output_event.set()
            context = session.active_context
            if context is not None:
                await emit_tool_output_delta(
                    context,
                    stream="stderr" if stream_name == "stderr" else "stdout",
                    delta=text,
                    process_id=session.process_id,
                    elapsed_ms=int((time.perf_counter() - session.active_started_at) * 1000),
                )

    async def _watch_exit(self, session: ProcessSession) -> None:
        session.returncode = await session.process.wait()
        await _drain_reader_tasks(session.reader_tasks)
        if session.status == "timed_out":
            pass
        elif session.status in {"stopping", "stopped"}:
            session.status = "stopped"
        else:
            session.status = "completed" if session.returncode == 0 else "failed"
        session.exit_event.set()
        session.output_event.set()

    async def _enforce_timeout(self, session: ProcessSession, timeout_ms: int) -> None:
        await asyncio.sleep(timeout_ms / 1000)
        if session.is_running():
            session.status = "timed_out"
            await _terminate_process_tree(session.process)

    async def _collect_for(
        self,
        session: ProcessSession,
        context: ToolContext,
        wait_ms: int,
    ) -> None:
        await self._activate(session, context)
        try:
            if wait_ms > 0 and session.is_running():
                with suppress(TimeoutError):
                    await asyncio.wait_for(session.exit_event.wait(), timeout=wait_ms / 1000)
        finally:
            self._deactivate(session, context)

    async def _collect_until_idle(
        self,
        session: ProcessSession,
        context: ToolContext,
        *,
        output_idle_timeout_ms: int,
    ) -> None:
        started_at = time.perf_counter()
        deadline = started_at + (MAX_TOOL_WAIT_MS / 1000)
        idle_seconds = max(0, output_idle_timeout_ms) / 1000
        last_seen_seq = session.output_seq
        last_output_at = started_at if session.has_pending_output() else None
        await self._activate(session, context)
        try:
            while True:
                if not session.is_running():
                    with suppress(TimeoutError):
                        await asyncio.wait_for(session.exit_event.wait(), timeout=0.05)
                    return

                now = time.perf_counter()
                if now >= deadline:
                    return
                if last_output_at is not None and now - last_output_at >= idle_seconds:
                    return

                timeout_seconds = deadline - now
                if last_output_at is not None:
                    timeout_seconds = min(timeout_seconds, idle_seconds - (now - last_output_at))
                else:
                    timeout_seconds = min(timeout_seconds, idle_seconds)
                timeout_seconds = max(0, timeout_seconds)

                session.output_event.clear()
                if session.output_seq != last_seen_seq:
                    last_seen_seq = session.output_seq
                    last_output_at = time.perf_counter()
                    continue
                if not session.is_running():
                    continue
                try:
                    await asyncio.wait_for(
                        self._wait_for_session_signal(session),
                        timeout=timeout_seconds,
                    )
                except TimeoutError:
                    return

                if session.output_seq != last_seen_seq:
                    last_seen_seq = session.output_seq
                    last_output_at = time.perf_counter()
        finally:
            self._deactivate(session, context)

    async def _wait_for_session_signal(self, session: ProcessSession) -> None:
        output_task = asyncio.create_task(session.output_event.wait())
        exit_task = asyncio.create_task(session.exit_event.wait())
        try:
            done, pending = await asyncio.wait(
                {output_task, exit_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                task.result()
        finally:
            for task in (output_task, exit_task):
                if not task.done():
                    task.cancel()

    async def _wait_for_new_output(
        self,
        session: ProcessSession,
        context: ToolContext,
        wait_ms: int,
    ) -> None:
        if session.has_pending_output() or not session.is_running() or wait_ms <= 0:
            return
        await self._activate(session, context)
        session.output_event.clear()
        try:
            with suppress(TimeoutError):
                await asyncio.wait_for(session.output_event.wait(), timeout=wait_ms / 1000)
        finally:
            self._deactivate(session, context)

    async def _activate(self, session: ProcessSession, context: ToolContext) -> None:
        session.active_context = context
        session.active_started_at = time.perf_counter()

    def _deactivate(self, session: ProcessSession, context: ToolContext) -> None:
        if session.active_context is context:
            session.active_context = None


PROCESSES = ProcessSessionManager()


def _format_process_result(
    session: ProcessSession,
    *,
    stdout: str,
    stderr: str,
    max_chars: int,
) -> str:
    content = (
        f"process_id={session.process_id} status={session.status} "
        f"returncode={session.returncode} stdin_closed={session.stdin_closed}"
    )
    if stdout:
        content += f"\n[stdout]\n{stdout}"
    if stderr:
        content += f"\n[stderr]\n{stderr}"
    return truncate_text(content.strip(), max_chars)


def _combine_process_output(stdout: str, stderr: str) -> str:
    content = stdout if stdout else ""
    if stderr:
        content = f"{content}\n[stderr]\n{stderr}".strip()
    return content


def _exec_timeout_ms(raw_input: JsonObject, *, tty: bool) -> int | None:
    if raw_input.get("timeout_ms") is None:
        return DEFAULT_TTY_TIMEOUT_MS if tty else DEFAULT_NON_TTY_TIMEOUT_MS
    try:
        return max(1, int(raw_input["timeout_ms"]))
    except (TypeError, ValueError):
        return DEFAULT_TTY_TIMEOUT_MS if tty else DEFAULT_NON_TTY_TIMEOUT_MS


def _exec_output_idle_timeout_ms(raw_input: JsonObject) -> int:
    return _coerce_ms(
        raw_input,
        "output_idle_timeout_ms",
        default=DEFAULT_OUTPUT_IDLE_TIMEOUT_MS,
        maximum=MAX_TOOL_WAIT_MS,
    )


def _exec_max_chars(raw_input: JsonObject, context: ToolContext) -> int:
    try:
        return max(1, int(raw_input.get("max_chars") or context.max_result_chars))
    except (TypeError, ValueError):
        return context.max_result_chars


def _exec_result(
    session: ProcessSession,
    *,
    stdout: str,
    stderr: str,
    context: ToolContext,
    started_at: float,
    max_chars: int,
    success: bool | None = None,
    error: str | None = None,
) -> ToolResult:
    output = truncate_text(_combine_process_output(stdout, stderr), max_chars)
    data: JsonObject = {
        "chunk_id": new_id("chunk"),
        "wall_time_seconds": round(time.perf_counter() - started_at, 4),
        "status": session.status,
        "exit_code": session.returncode,
        "session_id": session.process_id if session.is_running() else None,
        "stdin_closed": session.stdin_closed,
        "output": output,
    }
    if error:
        data["error"] = error
    if success is None:
        success = error is None and session.status not in {"failed", "timed_out"}
    content = json.dumps(data, ensure_ascii=False, indent=2)
    return ToolResult(
        success=success,
        content=truncate_text(content, context.max_result_chars),
        data=data,
        error=None,
    )


def _exec_error_result(
    message: str,
    context: ToolContext,
    *,
    started_at: float,
) -> ToolResult:
    data: JsonObject = {
        "chunk_id": new_id("chunk"),
        "wall_time_seconds": round(time.perf_counter() - started_at, 4),
        "status": "error",
        "exit_code": None,
        "session_id": None,
        "stdin_closed": True,
        "output": "",
        "error": message,
    }
    return ToolResult(
        success=False,
        content=json.dumps(data, ensure_ascii=False, indent=2),
        data=data,
        error=None,
    )


class ExecCommandTool(BaseTool):
    name = "exec_command"
    description = (
        "Runs a shell command and returns output. Set tty=true for long-running or "
        "interactive commands; if the process remains active, pass the returned "
        "session_id to write_stdin."
    )
    input_schema = _schema(
        {
            "cmd": {"type": "string", "description": "Shell command to execute."},
            "workdir": {
                "type": "string",
                "description": "Optional working directory; defaults to the current cwd.",
            },
            "shell": {
                "type": "string",
                "enum": ["PowerShell", "Bash"],
                "default": _default_shell(),
            },
            "tty": {
                "type": "boolean",
                "default": False,
                "description": "Keep stdin open for follow-up write_stdin calls.",
            },
            "timeout_ms": {
                "type": "integer",
                "minimum": 1,
                "description": "Maximum process lifetime. If reached, the process is terminated.",
            },
            "output_idle_timeout_ms": {
                "type": "integer",
                "minimum": 0,
                "maximum": MAX_TOOL_WAIT_MS,
                "default": DEFAULT_OUTPUT_IDLE_TIMEOUT_MS,
                "description": "Return partial output after stdout/stderr has been quiet for this long. Does not terminate the process.",
            },
            "max_chars": {"type": "integer", "minimum": 1, "default": 12000},
        },
        required=["cmd"],
    )

    def is_read_only(self, raw_input: JsonObject) -> bool:
        shell = str(raw_input.get("shell") or _default_shell())
        command = str(raw_input.get("cmd") or "")
        if shell == "Bash":
            return analyze_bash(command).read_only
        return analyze_powershell(command).read_only

    def check_permissions(self, raw_input: JsonObject, context: ToolContext) -> PermissionResult:
        del context
        shell = str(raw_input.get("shell") or _default_shell())
        command = str(raw_input.get("cmd") or "")
        if shell == "Bash":
            analysis = analyze_bash(command)
        else:
            analysis = analyze_powershell(command)
        return PermissionResult.allow(reason=analysis.reason, source=analysis.parser)

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        started_at = time.perf_counter()
        shell = str(raw_input.get("shell") or _default_shell())
        command = str(raw_input["cmd"])
        tty = bool(raw_input.get("tty"))
        try:
            session, stdout, stderr = await PROCESSES.start(
                name=command,
                shell=shell,
                command=command,
                cwd=_resolve_path(context, raw_input.get("workdir"), default="."),
                context=context,
                yield_time_ms=MAX_TOOL_WAIT_MS,
                timeout_ms=_exec_timeout_ms(raw_input, tty=tty),
                tty=tty,
                output_idle_timeout_ms=_exec_output_idle_timeout_ms(raw_input),
            )
        except RuntimeError as exc:
            return _exec_error_result(str(exc), context, started_at=started_at)
        return _exec_result(
            session,
            stdout=stdout,
            stderr=stderr,
            context=context,
            started_at=started_at,
            max_chars=_exec_max_chars(raw_input, context),
        )


class WriteStdinTool(BaseTool):
    name = "write_stdin"
    description = (
        "Writes characters to an active exec_command session and returns recent "
        "output. Use chars=\"\" to poll without writing."
    )
    input_schema = _schema(
        {
            "session_id": {"type": "string"},
            "chars": {"type": "string", "default": ""},
            "close_stdin": {"type": "boolean", "default": False},
            "output_idle_timeout_ms": {
                "type": "integer",
                "minimum": 0,
                "maximum": MAX_TOOL_WAIT_MS,
                "default": DEFAULT_OUTPUT_IDLE_TIMEOUT_MS,
                "description": "Return partial output after stdout/stderr has been quiet for this long. Does not terminate the process.",
            },
            "max_chars": {"type": "integer", "minimum": 1, "default": 12000},
        },
        required=["session_id"],
    )

    def is_read_only(self, raw_input: JsonObject) -> bool:
        return not raw_input.get("chars") and not bool(raw_input.get("close_stdin"))

    def check_permissions(self, raw_input: JsonObject, context: ToolContext) -> PermissionResult:
        del context
        if not raw_input.get("chars") and not bool(raw_input.get("close_stdin")):
            return PermissionResult.allow(reason="polling process output", source="tool")
        session = PROCESSES.get(str(raw_input.get("session_id") or ""))
        if session is None:
            return PermissionResult.deny(reason="process session not found", source="tool")
        if not session.tty or not session.stdin_authorized:
            return PermissionResult.deny(reason="stdin is not enabled for this session", source="tool")
        return PermissionResult.allow(reason="stdin authorized by exec_command session", source="tool")

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        started_at = time.perf_counter()
        session, stdout, stderr, error = await PROCESSES.write(
            str(raw_input["session_id"]),
            context=context,
            chars=str(raw_input.get("chars") or ""),
            close_stdin=bool(raw_input.get("close_stdin")),
            yield_time_ms=MAX_TOOL_WAIT_MS,
            output_idle_timeout_ms=_exec_output_idle_timeout_ms(raw_input),
        )
        if session is None:
            return _exec_error_result(f"Process not found: {raw_input['session_id']}", context, started_at=started_at)
        return _exec_result(
            session,
            stdout=stdout,
            stderr=stderr,
            context=context,
            started_at=started_at,
            max_chars=_exec_max_chars(raw_input, context),
            success=error is None and session.status not in {"failed", "timed_out"},
            error=error,
        )


class StopCommandTool(BaseTool):
    name = "stop_command"
    description = "Stops a running exec_command session."
    input_schema = _schema({"session_id": {"type": "string"}}, required=["session_id"])

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        started_at = time.perf_counter()
        session, stdout, stderr = await PROCESSES.stop(str(raw_input["session_id"]), context=context)
        if session is None:
            return _exec_error_result(f"Process not found: {raw_input['session_id']}", context, started_at=started_at)
        return _exec_result(
            session,
            stdout=stdout,
            stderr=stderr,
            context=context,
            started_at=started_at,
            max_chars=context.max_result_chars,
            success=True,
        )


class ProcessStartTool(BaseTool):
    name = "ProcessStart"
    model_visible = False
    description = "Start a long-running or interactive local shell process and return a process_id for later ProcessRead, ProcessWrite, or ProcessStop calls."
    input_schema = _schema(
        {
            "command": {"type": "string"},
            "shell": {"type": "string", "enum": ["PowerShell", "Bash"], "default": "PowerShell"},
            "name": {"type": "string"},
            "yield_time_ms": {"type": "integer", "minimum": 0, "maximum": 30000, "default": 1000},
            "timeout_ms": {"type": "integer", "minimum": 1},
        },
        required=["command"],
    )

    def check_permissions(self, raw_input: JsonObject, context: ToolContext) -> PermissionResult:
        del context
        shell = str(raw_input.get("shell") or "PowerShell")
        command = str(raw_input.get("command") or "")
        if shell == "Bash":
            analysis = analyze_bash(command)
        else:
            analysis = analyze_powershell(command)
        return PermissionResult.allow(reason=analysis.reason, source=analysis.parser)

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        shell = str(raw_input.get("shell") or "PowerShell")
        try:
            session, stdout, stderr = await PROCESSES.start(
                name=str(raw_input.get("name") or raw_input["command"]),
                shell=shell,
                command=str(raw_input["command"]),
                cwd=_cwd(context),
                context=context,
                yield_time_ms=_coerce_ms(raw_input, "yield_time_ms", default=1000, maximum=30000),
                timeout_ms=int(raw_input["timeout_ms"]) if raw_input.get("timeout_ms") is not None else None,
            )
        except RuntimeError as exc:
            return _error(str(exc), context)
        content = _format_process_result(
            session,
            stdout=stdout,
            stderr=stderr,
            max_chars=context.max_result_chars,
        )
        return _ok(
            content,
            context,
            {
                "process_id": session.process_id,
                "status": session.status,
                "returncode": session.returncode,
            },
        )


class ProcessReadTool(BaseTool):
    name = "ProcessRead"
    model_visible = False
    description = "Read new stdout and stderr from a process session started by ProcessStart."
    read_only = True
    input_schema = _schema(
        {
            "process_id": {"type": "string"},
            "wait_ms": {"type": "integer", "minimum": 0, "default": 1000},
            "max_chars": {"type": "integer", "minimum": 1, "default": 12000},
        },
        required=["process_id"],
    )

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        session, stdout, stderr = await PROCESSES.read(
            str(raw_input["process_id"]),
            context=context,
            wait_ms=_coerce_ms(raw_input, "wait_ms", default=1000),
        )
        if session is None:
            return _error(f"Process not found: {raw_input['process_id']}", context)
        max_chars = int(raw_input.get("max_chars") or 12000)
        content = _format_process_result(session, stdout=stdout, stderr=stderr, max_chars=max_chars)
        return _ok(
            content,
            context,
            {
                "process_id": session.process_id,
                "status": session.status,
                "returncode": session.returncode,
            },
        )


class ProcessWriteTool(BaseTool):
    name = "ProcessWrite"
    model_visible = False
    description = "Write stdin to a running process session, optionally close stdin, and read new output."
    input_schema = _schema(
        {
            "process_id": {"type": "string"},
            "chars": {"type": "string", "default": ""},
            "close_stdin": {"type": "boolean", "default": False},
            "yield_time_ms": {"type": "integer", "minimum": 0, "maximum": 30000, "default": 1000},
            "max_chars": {"type": "integer", "minimum": 1, "default": 12000},
        },
        required=["process_id"],
    )

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        session, stdout, stderr, error = await PROCESSES.write(
            str(raw_input["process_id"]),
            context=context,
            chars=str(raw_input.get("chars") or ""),
            close_stdin=bool(raw_input.get("close_stdin")),
            yield_time_ms=_coerce_ms(raw_input, "yield_time_ms", default=1000, maximum=30000),
        )
        if session is None:
            return _error(f"Process not found: {raw_input['process_id']}", context)
        max_chars = int(raw_input.get("max_chars") or 12000)
        content = _format_process_result(session, stdout=stdout, stderr=stderr, max_chars=max_chars)
        data = {
            "process_id": session.process_id,
            "status": session.status,
            "returncode": session.returncode,
            "stdin_closed": session.stdin_closed,
        }
        if error:
            return _error(content or error, context, data)
        return _ok(content, context, data)


class ProcessStopTool(BaseTool):
    name = "ProcessStop"
    model_visible = False
    description = "Stop a running process session and kill its process tree if needed."
    input_schema = _schema({"process_id": {"type": "string"}}, required=["process_id"])

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        session, stdout, stderr = await PROCESSES.stop(str(raw_input["process_id"]), context=context)
        if session is None:
            return _error(f"Process not found: {raw_input['process_id']}", context)
        content = _format_process_result(
            session,
            stdout=stdout,
            stderr=stderr,
            max_chars=context.max_result_chars,
        )
        return _ok(
            content,
            context,
            {
                "process_id": session.process_id,
                "status": session.status,
                "returncode": session.returncode,
            },
        )


class GlobTool(BaseTool):
    name = "Glob"
    description = "Find files by glob pattern under a path."
    read_only = True
    input_schema = _schema(
        {
            "pattern": {"type": "string"},
            "path": {"type": "string", "default": "."},
        },
        required=["pattern"],
    )

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        root = _resolve_path(context, raw_input.get("path"), default=".")
        matches = []
        for item in root.glob(str(raw_input["pattern"])):
            matches.append(str(item.resolve()))
        return _ok(
            "\n".join(matches) if matches else "(no matches)",
            context,
            {"matches": matches, "count": len(matches)},
        )


class GrepTool(BaseTool):
    name = "Grep"
    description = "Search local text files with a regular expression."
    read_only = True
    input_schema = _schema(
        {
            "pattern": {"type": "string"},
            "path": {"type": "string", "default": "."},
            "include": {"type": "string", "description": "Glob include filter such as *.py."},
            "exclude": {"type": "string", "description": "Glob exclude filter."},
            "case_sensitive": {"type": "boolean", "default": True},
        },
        required=["pattern"],
    )

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        root = _resolve_path(context, raw_input.get("path"), default=".")
        flags = 0 if raw_input.get("case_sensitive", True) else re.IGNORECASE
        try:
            regex = re.compile(str(raw_input["pattern"]), flags)
        except re.error as exc:
            return _error(f"Invalid regex: {exc}", context)
        include = raw_input.get("include")
        exclude = raw_input.get("exclude")
        results: list[JsonObject] = []
        files = [root] if root.is_file() else [path for path in root.rglob("*") if path.is_file()]
        for path in files:
            rel = str(path.relative_to(root)) if root.is_dir() else path.name
            if include and not fnmatch.fnmatch(path.name, str(include)) and not fnmatch.fnmatch(rel, str(include)):
                continue
            if exclude and (fnmatch.fnmatch(path.name, str(exclude)) or fnmatch.fnmatch(rel, str(exclude))):
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue
            for line_no, line in enumerate(lines, start=1):
                if regex.search(line):
                    results.append({"path": str(path), "line": line_no, "text": line})
        content = "\n".join(f"{item['path']}:{item['line']}: {item['text']}" for item in results)
        return _ok(content or "(no matches)", context, {"matches": results, "count": len(results)})


class _DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_link = False
        self.current_href: str | None = None
        self.current_text: list[str] = []
        self.results: list[JsonObject] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attrs_dict = dict(attrs)
        href = attrs_dict.get("href")
        css = attrs_dict.get("class") or ""
        if href and ("result__a" in css or "/l/?" in href):
            self.in_link = True
            self.current_href = href
            self.current_text = []

    def handle_data(self, data: str) -> None:
        if self.in_link:
            self.current_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or not self.in_link:
            return
        title = html.unescape(" ".join("".join(self.current_text).split()))
        url = _clean_duckduckgo_url(self.current_href or "")
        if title and url and not any(item["url"] == url for item in self.results):
            self.results.append({"title": title, "url": url})
        self.in_link = False
        self.current_href = None
        self.current_text = []


def _clean_duckduckgo_url(value: str) -> str:
    value = html.unescape(value)
    parsed = urlparse(value)
    query = parse_qs(parsed.query)
    if "uddg" in query and query["uddg"]:
        return unquote(query["uddg"][0])
    return value


class WebSearchTool(BaseTool):
    name = "WebSearch"
    description = "Search the web using DuckDuckGo HTML search and return result titles and URLs."
    read_only = True
    input_schema = _schema(
        {
            "query": {"type": "string"},
            "timeout_ms": {"type": "number", "minimum": 1, "default": 10000},
        },
        required=["query"],
    )

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        query = str(raw_input["query"])
        timeout = _timeout_seconds(raw_input)
        url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                response = await client.get(url, headers={"User-Agent": "SiYi/0.1"})
                response.raise_for_status()
        except httpx.HTTPError as exc:
            return _error(f"web search failed: {exc}", context)
        parser = _DuckDuckGoParser()
        parser.feed(response.text)
        results = parser.results
        content = "\n".join(f"{index + 1}. {item['title']}\n   {item['url']}" for index, item in enumerate(results))
        return _ok(content or "(no web results)", context, {"query": query, "results": results})


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in {"script", "style", "noscript"}:
            self.skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self.skip:
            self.skip -= 1

    def handle_data(self, data: str) -> None:
        if not self.skip:
            text = " ".join(data.split())
            if text:
                self.parts.append(text)


class WebFetchTool(BaseTool):
    name = "WebFetch"
    description = "Fetch a URL and return text content or raw response text."
    read_only = True
    input_schema = _schema(
        {
            "url": {"type": "string"},
            "raw": {"type": "boolean", "default": False},
            "max_chars": {"type": "integer", "minimum": 1, "default": 20000},
            "timeout_ms": {"type": "number", "minimum": 1, "default": 10000},
        },
        required=["url"],
    )

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        url = str(raw_input["url"])
        max_chars = int(raw_input.get("max_chars") or context.max_result_chars)
        timeout = _timeout_seconds(raw_input)
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                response = await client.get(url, headers={"User-Agent": "SiYi/0.1"})
                response.raise_for_status()
        except httpx.HTTPError as exc:
            return _error(f"web fetch failed: {exc}", context)
        content_type = response.headers.get("content-type", "")
        if raw_input.get("raw") or "html" not in content_type:
            text = response.text
        else:
            parser = _TextExtractor()
            parser.feed(response.text)
            text = "\n".join(parser.parts)
        return _ok(
            truncate_text(text, max_chars),
            context,
            {"url": str(response.url), "status_code": response.status_code, "content_type": content_type},
        )


class ToolSearchTool(BaseTool):
    name = "ToolSearch"
    description = "Search the currently available SiYi tools by name and description."
    read_only = True
    input_schema = _schema(
        {
            "query": {"type": "string", "default": ""},
            "max_results": {"type": "integer", "minimum": 1, "default": 50},
        },
    )

    def __init__(
        self,
        registry_provider: Callable[[], list[Any]],
        alias_provider: Callable[[], dict[str, str]] | None = None,
    ) -> None:
        self._registry_provider = registry_provider
        self._alias_provider = alias_provider or (lambda: {})

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        query = str(raw_input.get("query") or "").lower()
        max_results = int(raw_input.get("max_results") or 50)
        matches = []
        aliases_by_target: dict[str, list[str]] = {}
        for alias, target_name in self._alias_provider().items():
            aliases_by_target.setdefault(target_name, []).append(alias)
        for tool in self._registry_provider():
            aliases = sorted(aliases_by_target.get(tool.name, []))
            haystack = f"{tool.name} {' '.join(aliases)} {tool.description}".lower()
            if not query or query in haystack:
                matches.append(
                    {
                        "name": tool.name,
                        "aliases": aliases,
                        "description": tool.description,
                        "input_schema": tool.input_schema,
                    }
                )
            if len(matches) >= max_results:
                break
        content = "\n".join(
            _format_tool_search_item(item)
            for item in matches
        )
        return _ok(content or "(no matching tools)", context, {"tools": matches})


def _format_tool_search_item(item: JsonObject) -> str:
    aliases = item.get("aliases") or []
    alias_text = f" (aliases: {', '.join(aliases)})" if aliases else ""
    return f"{item['name']}{alias_text}: {item['description']}"


class AskUserQuestionTool(BaseTool):
    name = "AskUserQuestion"
    description = "Ask the human user a question and return a structured pending-question result."
    read_only = True
    input_schema = _schema(
        {
            "question": {"type": "string"},
            "choices": {"type": "array", "items": {"type": "string"}},
            "allow_freeform": {"type": "boolean", "default": True},
        },
        required=["question"],
    )

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        choices = raw_input.get("choices") or []
        lines = [f"Question for user: {raw_input['question']}"]
        if choices:
            lines.append("Choices:")
            lines.extend(f"- {choice}" for choice in choices)
        return _ok(
            "\n".join(lines),
            context,
            {
                "requires_user_input": True,
                "question": raw_input["question"],
                "choices": choices,
                "allow_freeform": bool(raw_input.get("allow_freeform", True)),
            },
        )


class EnterPlanModeTool(BaseTool):
    name = "EnterPlanMode"
    description = "Enter planning mode and persist the current plan state in the SiYi project state directory."
    input_schema = _schema(
        {
            "plan": {"type": "string", "description": "Optional plan text."},
            "reason": {"type": "string"},
        },
    )

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        state = {
            "active": True,
            "plan": raw_input.get("plan") or "",
            "reason": raw_input.get("reason") or "",
            "updated_at": time.time(),
        }
        _write_json_file(_state_dir(context) / "plan_mode.json", state)
        return _ok("entered plan mode", context, state)


class ExitPlanModeTool(BaseTool):
    name = "ExitPlanMode"
    description = "Exit planning mode and persist the current plan state in the SiYi project state directory."
    input_schema = _schema({"reason": {"type": "string"}})

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        state = {
            "active": False,
            "reason": raw_input.get("reason") or "",
            "updated_at": time.time(),
        }
        _write_json_file(_state_dir(context) / "plan_mode.json", state)
        return _ok("exited plan mode", context, state)


class SendUserMessageTool(BaseTool):
    name = "SendUserMessage"
    description = "Send a visible message to the user as a tool result."
    read_only = True
    input_schema = _schema({"message": {"type": "string"}}, required=["message"])

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        return _ok(str(raw_input["message"]), context, {"message": raw_input["message"]})


def _agents_path(context: ToolContext) -> Path:
    return _state_dir(context) / "agents.json"


def _teams_path(context: ToolContext) -> Path:
    return _state_dir(context) / "teams.json"


def _task_records_path(context: ToolContext) -> Path:
    return _state_dir(context) / "task_records.json"


def _load_records(path: Path) -> list[JsonObject]:
    data = _read_json_file(path, [])
    return data if isinstance(data, list) else []


class AgentTool(BaseTool):
    name = "Agent"
    description = "Create, list, get, or delete a named sub-agent record."
    input_schema = _schema(
        {
            "action": {"type": "string", "enum": ["create", "list", "get", "delete"], "default": "create"},
            "agent_id": {"type": "string"},
            "name": {"type": "string"},
            "role": {"type": "string"},
            "prompt": {"type": "string"},
        },
    )

    def is_read_only(self, raw_input: JsonObject) -> bool:
        return str(raw_input.get("action") or "create") in {"list", "get"}

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        path = _agents_path(context)
        agents = _load_records(path)
        action = str(raw_input.get("action") or "create")
        if action == "list":
            return _ok(json.dumps(agents, ensure_ascii=False, indent=2), context, {"agents": agents})
        if action == "get":
            agent = next((item for item in agents if item.get("agent_id") == raw_input.get("agent_id")), None)
            if agent is None:
                return _error(f"Agent not found: {raw_input.get('agent_id')}", context)
            return _ok(json.dumps(agent, ensure_ascii=False, indent=2), context, agent)
        if action == "delete":
            before = len(agents)
            agents = [item for item in agents if item.get("agent_id") != raw_input.get("agent_id")]
            _write_json_file(path, agents)
            return _ok(f"deleted {before - len(agents)} agent(s)", context, {"deleted": before - len(agents)})

        agent_id = new_id("agent")
        agent = {
            "agent_id": agent_id,
            "name": raw_input.get("name") or agent_id,
            "role": raw_input.get("role") or "general",
            "prompt": raw_input.get("prompt") or "",
            "messages": [],
            "created_at": time.time(),
        }
        agents.append(agent)
        _write_json_file(path, agents)
        return _ok(json.dumps(agent, ensure_ascii=False, indent=2), context, agent)


def _siyi_command(prompt: str, context: ToolContext) -> list[str]:
    exe = shutil.which("siyi")
    if exe:
        return [exe, "--cwd", context.cwd, "--internal-worker-prompt", prompt]
    env_pythonpath = os.environ.get("PYTHONPATH")
    src_path = str(Path(__file__).resolve().parents[1])
    os.environ["PYTHONPATH"] = f"{src_path}{os.pathsep}{env_pythonpath}" if env_pythonpath else src_path
    return [sys.executable, "-m", "app.cli", "--cwd", context.cwd, "--internal-worker-prompt", prompt]


class SendMessageTool(BaseTool):
    name = "SendMessage"
    description = "Append a message to an agent or team record."
    input_schema = _schema(
        {
            "target_id": {"type": "string"},
            "message": {"type": "string"},
            "target_type": {"type": "string", "enum": ["agent", "team"], "default": "agent"},
        },
        required=["target_id", "message"],
    )

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        target_type = str(raw_input.get("target_type") or "agent")
        path = _teams_path(context) if target_type == "team" else _agents_path(context)
        records = _load_records(path)
        id_key = "team_id" if target_type == "team" else "agent_id"
        target = next((item for item in records if item.get(id_key) == raw_input["target_id"]), None)
        if target is None:
            return _error(f"{target_type} not found: {raw_input['target_id']}", context)
        target.setdefault("messages", []).append(
            {"role": "user", "content": raw_input["message"], "created_at": time.time()}
        )
        _write_json_file(path, records)
        return _ok(f"sent message to {target_type} {raw_input['target_id']}", context, target)


class TeamCreateTool(BaseTool):
    name = "TeamCreate"
    description = "Create a team record containing agent ids."
    input_schema = _schema(
        {
            "name": {"type": "string"},
            "agent_ids": {"type": "array", "items": {"type": "string"}},
        },
        required=["name"],
    )

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        teams = _load_records(_teams_path(context))
        team = {
            "team_id": new_id("team"),
            "name": raw_input["name"],
            "agent_ids": raw_input.get("agent_ids") or [],
            "messages": [],
            "created_at": time.time(),
        }
        teams.append(team)
        _write_json_file(_teams_path(context), teams)
        return _ok(json.dumps(team, ensure_ascii=False, indent=2), context, team)


class TeamDeleteTool(BaseTool):
    name = "TeamDelete"
    description = "Delete a team record."
    input_schema = _schema({"team_id": {"type": "string"}}, required=["team_id"])

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        teams = _load_records(_teams_path(context))
        before = len(teams)
        teams = [item for item in teams if item.get("team_id") != raw_input["team_id"]]
        _write_json_file(_teams_path(context), teams)
        return _ok(f"deleted {before - len(teams)} team(s)", context, {"deleted": before - len(teams)})


class TaskCreateTool(BaseTool):
    name = "TaskCreate"
    description = "Create a task record."
    input_schema = _schema(
        {
            "title": {"type": "string"},
            "description": {"type": "string"},
            "status": {"type": "string", "default": "pending"},
        },
        required=["title"],
    )

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        records = _load_records(_task_records_path(context))
        task = {
            "task_id": new_id("taskrec"),
            "title": raw_input["title"],
            "description": raw_input.get("description") or "",
            "status": raw_input.get("status") or "pending",
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        records.append(task)
        _write_json_file(_task_records_path(context), records)
        return _ok(json.dumps(task, ensure_ascii=False, indent=2), context, task)


class TaskGetTool(BaseTool):
    name = "TaskGet"
    description = "Get a task record by id."
    read_only = True
    input_schema = _schema({"task_id": {"type": "string"}}, required=["task_id"])

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        records = _load_records(_task_records_path(context))
        task = next((item for item in records if item.get("task_id") == raw_input["task_id"]), None)
        if task is None:
            return _error(f"Task record not found: {raw_input['task_id']}", context)
        return _ok(json.dumps(task, ensure_ascii=False, indent=2), context, task)


class TaskListTool(BaseTool):
    name = "TaskList"
    description = "List task records."
    read_only = True
    input_schema = _schema({})

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        del raw_input
        records = _load_records(_task_records_path(context))
        data = {"records": records}
        return _ok(json.dumps(data, ensure_ascii=False, indent=2), context, data)


class TaskUpdateTool(BaseTool):
    name = "TaskUpdate"
    description = "Update fields on a task record."
    input_schema = _schema(
        {
            "task_id": {"type": "string"},
            "status": {"type": "string"},
            "title": {"type": "string"},
            "description": {"type": "string"},
            "metadata": {"type": "object"},
        },
        required=["task_id"],
    )

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        records = _load_records(_task_records_path(context))
        task = next((item for item in records if item.get("task_id") == raw_input["task_id"]), None)
        if task is None:
            return _error(f"Task record not found: {raw_input['task_id']}", context)
        for key in ("status", "title", "description", "metadata"):
            if key in raw_input and raw_input[key] is not None:
                task[key] = raw_input[key]
        task["updated_at"] = time.time()
        _write_json_file(_task_records_path(context), records)
        return _ok(json.dumps(task, ensure_ascii=False, indent=2), context, task)


class SkillTool(BaseTool):
    name = "Skill"
    description = "List, read, or search local skill files from global SiYi skill roots."
    read_only = True
    input_schema = _schema(
        {
            "action": {"type": "string", "enum": ["list", "read", "search"], "default": "list"},
            "name": {"type": "string"},
            "path": {"type": "string"},
            "query": {"type": "string"},
        },
    )

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        action = str(raw_input.get("action") or "list")
        roots = _skill_roots(context)
        skills = _discover_skills(roots)
        if action == "list":
            content = "\n".join(f"{item['name']}: {item['path']}" for item in skills)
            return _ok(content or "(no skills found)", context, {"skills": skills})
        if action == "search":
            query = str(raw_input.get("query") or "").lower()
            matches = [
                item
                for item in skills
                if query in item["name"].lower()
                or query in item["path"].lower()
                or query in item["root"].lower()
                or query in item["source"].lower()
            ]
            return _ok(
                "\n".join(f"{item['name']}: {item['path']}" for item in matches) or "(no matches)",
                context,
                {"skills": matches},
            )
        target = raw_input.get("path")
        if not target and raw_input.get("name"):
            target_item = next((item for item in skills if item["name"] == raw_input["name"]), None)
            target = target_item["path"] if target_item else None
        if not target:
            return _error("Skill read requires name or path", context)
        path = _resolve_path(context, str(target)) if not Path(str(target)).is_absolute() else Path(str(target))
        if not path.exists():
            return _error(f"Skill file not found: {path}", context)
        text = path.read_text(encoding="utf-8", errors="replace")
        return _ok(text, context, {"path": str(path)})


def _skill_roots(context: ToolContext) -> list[Path]:
    del context
    root_entries: list[tuple[Path, str]] = [(get_global_skills_dir(), "default")]
    root_entries.extend((path, "configured") for path in _load_skill_paths())
    env_root = os.environ.get("SIYI_SKILLS_DIR")
    if env_root:
        root_entries.append((Path(env_root).expanduser(), "env"))
    return [
        root.resolve()
        for root, _source in _dedupe_skill_roots(root_entries)
        if root.exists() and root.is_dir()
    ]


def _load_skill_paths() -> list[Path]:
    path = get_skill_paths_path()
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    raw_paths = payload.get("paths") if isinstance(payload, dict) else None
    if not isinstance(raw_paths, list):
        return []
    return [
        Path(str(raw_path)).expanduser()
        for raw_path in raw_paths
        if raw_path
    ]


def _dedupe_skill_roots(root_entries: list[tuple[Path, str]]) -> list[tuple[Path, str]]:
    seen: set[str] = set()
    result: list[tuple[Path, str]] = []
    for root, source in root_entries:
        resolved = root.expanduser().resolve()
        key = os.path.normcase(str(resolved))
        if key in seen:
            continue
        seen.add(key)
        result.append((resolved, source))
    return result


def _discover_skills(roots: list[Path]) -> list[JsonObject]:
    skills = []
    for root in roots:
        for path in root.rglob("SKILL.md"):
            skills.append(
                {
                    "name": path.parent.name,
                    "path": str(path),
                    "root": str(root),
                    "source": _skill_root_source(root),
                }
            )
    return sorted(skills, key=lambda item: item["name"])


def _skill_root_source(root: Path) -> str:
    root = root.resolve()
    for candidate, source in _dedupe_skill_roots(
        [(get_global_skills_dir(), "default")]
        + [(path, "configured") for path in _load_skill_paths()]
        + ([(Path(os.environ["SIYI_SKILLS_DIR"]).expanduser(), "env")] if os.environ.get("SIYI_SKILLS_DIR") else [])
    ):
        if candidate == root:
            return source
    return "unknown"


class ListMcpResourcesTool(BaseTool):
    name = "ListMcpResourcesTool"
    description = "List local MCP resource descriptors from the SiYi project state directory."
    read_only = True
    input_schema = _schema({"server": {"type": "string"}})

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        del raw_input
        resources = _local_mcp_resources(context)
        content = "\n".join(f"{item.get('uri')}: {item.get('name', '')}" for item in resources)
        return _ok(content or "(no local MCP resources)", context, {"resources": resources})


class ReadMcpResourceTool(BaseTool):
    name = "ReadMcpResourceTool"
    description = "Read a local MCP resource by uri from the SiYi project state directory."
    read_only = True
    input_schema = _schema({"uri": {"type": "string"}, "server": {"type": "string"}}, required=["uri"])

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        uri = str(raw_input["uri"])
        for item in _local_mcp_resources(context):
            if item.get("uri") == uri:
                if "text" in item:
                    return _ok(str(item["text"]), context, item)
                if "path" in item:
                    path = _resolve_path(context, str(item["path"]))
                    return _ok(path.read_text(encoding="utf-8", errors="replace"), context, item)
        parsed = urlparse(uri)
        if parsed.scheme == "file":
            path = Path(unquote(parsed.path)).resolve()
            return _ok(path.read_text(encoding="utf-8", errors="replace"), context, {"uri": uri})
        return _error(f"MCP resource not found: {uri}", context)


def _local_mcp_resources(context: ToolContext) -> list[JsonObject]:
    base = _state_dir(context) / "mcp"
    resources: list[JsonObject] = []
    descriptor = base / "resources.json"
    if descriptor.exists():
        data = _read_json_file(descriptor, [])
        if isinstance(data, list):
            resources.extend(item for item in data if isinstance(item, dict))
        elif isinstance(data, dict) and isinstance(data.get("resources"), list):
            resources.extend(item for item in data["resources"] if isinstance(item, dict))
    resource_dir = base / "resources"
    if resource_dir.exists():
        for path in resource_dir.rglob("*"):
            if path.is_file():
                resources.append({"uri": path.as_uri(), "name": path.name, "path": str(path)})
    return resources


class WorkflowTool(BaseTool):
    name = "workflow"
    description = "List, get, or run JSON workflow definitions from the SiYi project state directory."
    input_schema = _schema(
        {
            "action": {"type": "string", "enum": ["list", "get", "run"], "default": "list"},
            "name": {"type": "string"},
            "steps": {"type": "array", "items": {"type": "object"}},
        },
    )

    def __init__(self, registry_provider: Callable[[], Any]) -> None:
        self._registry_provider = registry_provider

    def is_read_only(self, raw_input: JsonObject) -> bool:
        return str(raw_input.get("action") or "list") in {"list", "get"}

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        action = str(raw_input.get("action") or "list")
        workflows = _discover_workflows(context)
        if action == "list":
            content = "\n".join(f"{item['name']}: {item['path']}" for item in workflows)
            return _ok(content or "(no workflows found)", context, {"workflows": workflows})
        definition = None
        if raw_input.get("name"):
            definition = _load_workflow_definition(context, str(raw_input["name"]))
            if definition is None:
                return _error(f"Workflow not found: {raw_input['name']}", context)
        if action == "get":
            return _ok(json.dumps(definition, ensure_ascii=False, indent=2), context, definition)
        steps = raw_input.get("steps") or (definition or {}).get("steps") or []
        if not isinstance(steps, list):
            return _error("workflow steps must be a list", context)
        outputs = []
        registry = self._registry_provider()
        for index, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                outputs.append({"step": index, "success": False, "content": "step is not an object"})
                continue
            tool_name = step.get("tool")
            tool_input = step.get("input") or {}
            tool = registry.find_tool(tool_name) if tool_name else None
            if tool is None:
                outputs.append({"step": index, "tool": tool_name, "success": False, "content": "tool not found"})
                continue
            if not isinstance(tool_input, dict):
                outputs.append({"step": index, "tool": tool_name, "success": False, "content": "tool input is not an object"})
                continue
            validation = tool.validate_input(tool_input, context)
            if not validation.ok:
                outputs.append({"step": index, "tool": tool_name, "success": False, "content": validation.reason})
                continue
            permission = await registry.permission_manager.authorize(tool, tool_input, context)
            if permission.decision != "allow":
                outputs.append({"step": index, "tool": tool_name, "success": False, "content": permission.reason})
                continue
            result = await tool.run(tool_input, context)
            outputs.append({"step": index, "tool": tool_name, **result.model_dump()})
        return _ok(json.dumps(outputs, ensure_ascii=False, indent=2), context, {"steps": outputs})


def _discover_workflows(context: ToolContext) -> list[JsonObject]:
    root = _state_dir(context) / "workflows"
    if not root.exists():
        return []
    return [{"name": path.stem, "path": str(path)} for path in sorted(root.glob("*.json"))]


def _load_workflow_definition(context: ToolContext, name: str) -> JsonObject | None:
    root = _state_dir(context) / "workflows"
    path = root / f"{name}.json"
    if not path.exists():
        return None
    data = _read_json_file(path, {})
    return data if isinstance(data, dict) else None


class ConfigTool(BaseTool):
    name = "Config"
    description = "Get, set, list, or delete local SiYi config values stored in the SiYi project state directory."
    input_schema = _schema(
        {
            "action": {"type": "string", "enum": ["get", "set", "list", "delete"], "default": "list"},
            "key": {"type": "string"},
            "value": {},
        },
    )

    def is_read_only(self, raw_input: JsonObject) -> bool:
        return str(raw_input.get("action") or "list") in {"get", "list"}

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        path = _state_dir(context) / "config.json"
        config = _read_json_file(path, {})
        if not isinstance(config, dict):
            config = {}
        action = str(raw_input.get("action") or "list")
        key = raw_input.get("key")
        if action == "list":
            return _ok(json.dumps(config, ensure_ascii=False, indent=2), context, {"config": config})
        if not key:
            return _error("Config action requires key", context)
        if action == "get":
            return _ok(json.dumps(config.get(key), ensure_ascii=False), context, {"key": key, "value": config.get(key)})
        if action == "delete":
            existed = key in config
            config.pop(key, None)
            _write_json_file(path, config)
            return _ok(f"deleted={existed}", context, {"key": key, "deleted": existed})
        config[str(key)] = raw_input.get("value")
        _write_json_file(path, config)
        return _ok(f"set {key}", context, {"key": key, "value": config[str(key)]})


def register_builtin_tools(registry: Any) -> None:
    tools: list[BaseTool] = [
        ReadTool(),
        EditTool(),
        WriteTool(),
        NotebookEditTool(),
        ExecCommandTool(),
        WriteStdinTool(),
        StopCommandTool(),
        ProcessStartTool(),
        ProcessReadTool(),
        ProcessWriteTool(),
        ProcessStopTool(),
        GlobTool(),
        GrepTool(),
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
        tools.insert(4, BashTool())
    if sys.platform.startswith("win"):
        tools.insert(5 if _bash_exe() is not None else 4, PowerShellTool())
    for tool in tools:
        registry.register(tool)

    registry.register(ToolSearchTool(registry.get_model_tools, registry.get_aliases))
    registry.register(WorkflowTool(lambda: registry))

    aliases = {
        "read_file": "Read",
        "edit_file": "Edit",
        "write_file": "Write",
        "web_search": "WebSearch",
        "web_fetch": "WebFetch",
        "glob": "Glob",
        "grep": "Grep",
    }
    if registry.find_tool("PowerShell") is not None:
        aliases["shell"] = "PowerShell"
    elif registry.find_tool("Bash") is not None:
        aliases["shell"] = "Bash"
    for alias, target_name in aliases.items():
        target = registry.find_tool(target_name)
        if target is not None:
            registry.register_alias(alias, target_name)
