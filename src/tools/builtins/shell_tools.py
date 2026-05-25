from __future__ import annotations

from .common import *
from .process_runtime import *

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
            cwd=_project_root(context),
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
            cwd=_project_root(context),
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
                "description": "Optional working directory; defaults to the project root.",
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
