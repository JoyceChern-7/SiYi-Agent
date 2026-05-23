from __future__ import annotations

from .common import *

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

__all__ = [
    "PROCESSES",
    "ProcessSession",
    "ProcessSessionManager",
    "_run_process",
    "_process_creation_kwargs",
    "_read_process_stream",
    "_drain_reader_tasks",
    "_terminate_process_tree",
    "_signal_process_tree",
    "_powershell_exe",
    "_bash_exe",
    "_default_shell",
    "_command_for_shell",
    "_coerce_ms",
    "_StreamBuffer",
    "_format_process_result",
    "_combine_process_output",
    "_exec_timeout_ms",
    "_exec_output_idle_timeout_ms",
    "_exec_max_chars",
    "_exec_result",
    "_exec_error_result",
]
