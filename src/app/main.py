from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from app.options import CLIOptions
from config.settings import AppSettings, load_settings
from engine.query_engine import QueryEngine
from llm.openai_adapter import OpenAIChatAdapter
from runtime.compaction import CompactionManager
from runtime.logging_utils import configure_logging
from runtime.permissions import (
    PermissionManager,
    ensure_permission_files,
    load_global_permission_mode,
)
from runtime.session_store import JsonlSessionStore
from runtime.token_budget import TokenBudget
from runtime.usage_tracker import UsageTracker
from tools.registry import ToolRegistry
from ui.renderer import ConsoleRenderer
from ui.repl import run_repl

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class AppRuntime:
    settings: AppSettings
    tool_registry: ToolRegistry
    llm_adapter: OpenAIChatAdapter
    session_store: JsonlSessionStore
    permission_manager: PermissionManager
    compaction_manager: CompactionManager
    token_budget: TokenBudget
    usage_tracker: UsageTracker
    query_engine: QueryEngine
    renderer: ConsoleRenderer


def _resolve_cwd(options: CLIOptions) -> Path:
    cwd = options.cwd or Path.cwd()
    if not cwd.exists() or not cwd.is_dir():
        raise ValueError(f"Working directory does not exist or is not a directory: {cwd}")
    return cwd.resolve()
# e.g:  if cwd = Path("../my-project"), then cwd.resolve() = Path("C:/Users/xxxx/Desktop/my-project")

def _initialize_process(options: CLIOptions) -> Path:
    cwd = _resolve_cwd(options)
    _activate_cwd(cwd)
    os.environ["SIYI_ENTRYPOINT"] = (
        "worker" if options.internal_worker else "cli"
    )
    return cwd


def _activate_cwd(cwd: Path) -> None:
    os.chdir(cwd)
    os.environ["SIYI_CWD"] = str(cwd)


def build_runtime(options: CLIOptions) -> AppRuntime:
    started_at = time.perf_counter()
    cwd = _initialize_process(options)
    settings = load_settings(options=options, cwd=cwd)

    configure_logging(debug=settings.runtime.debug)
    LOGGER.debug(
        "siyi.startup.begin",
        extra={
            "cwd": str(cwd),
            "non_interactive": settings.runtime.non_interactive,
            "resume": settings.runtime.resume,
        },
    )

    session_store = JsonlSessionStore(settings.runtime.session_dir)
    ensure_permission_files()
    permission_mode = load_global_permission_mode()
    session = session_store.open_session(
        requested_session=settings.runtime.resume,
        cwd=cwd,
        model=settings.model.model,
        permission_mode=permission_mode,
    )
    session_cwd = Path(session.metadata.cwd).expanduser().resolve()
    if not session_cwd.exists() or not session_cwd.is_dir():
        raise ValueError(f"Session working directory does not exist: {session_cwd}")
    if session_cwd != settings.runtime.cwd:
        _activate_cwd(session_cwd)
        settings.runtime.cwd = session_cwd
        cwd = session_cwd

    permission_manager = PermissionManager.from_settings(
        settings.tools,
        cwd=cwd,
        mode=session.metadata.permission_mode,
    )
    tool_registry = ToolRegistry.default(permission_manager=permission_manager)
    compaction_manager = CompactionManager(settings.runtime.compaction_enabled)
    token_budget = TokenBudget(settings.runtime)
    usage_tracker = UsageTracker(settings.model.pricing)
    llm_adapter = OpenAIChatAdapter.from_settings(settings.model)
    renderer = ConsoleRenderer(
        debug=settings.runtime.debug,
        print_thinking=settings.model.print_thinking,
    )
    query_engine = QueryEngine(
        session=session, 
        settings=settings, # AppSettings from settings.py, all settings are stored in this object, e.g: settings.model, settings.runtime, settings.tools
        llm=llm_adapter, 
        tool_registry=tool_registry,
        session_store=session_store, # 
        permission_manager=permission_manager,
        compaction_manager=compaction_manager,
        token_budget=token_budget,
        usage_tracker=usage_tracker,
    )

    runtime = AppRuntime(
        settings=settings,
        tool_registry=tool_registry,
        llm_adapter=llm_adapter,
        session_store=session_store,
        permission_manager=permission_manager,
        compaction_manager=compaction_manager,
        token_budget=token_budget,
        usage_tracker=usage_tracker,
        query_engine=query_engine,
        renderer=renderer,
    )
    LOGGER.debug(
        "siyi.startup.finished",
        extra={"startup_ms": round((time.perf_counter() - started_at) * 1000, 2)},
    )
    return runtime


async def _run_internal_worker(runtime: AppRuntime, prompt: str | None) -> int:
    if not prompt:
        runtime.renderer.render_error(
            "Internal worker mode requires a prompt."
        )
        return 2

    async for event in runtime.query_engine.submit_user_input(prompt):
        runtime.renderer.render_event(event)
    return 0 if not runtime.query_engine.last_error else 1


async def run(options: CLIOptions) -> int:
    try:
        runtime = build_runtime(options)
    except Exception as exc:  # noqa: BLE001 - startup must surface all failures cleanly
        configure_logging(debug=options.debug)
        if options.debug:
            LOGGER.exception("siyi.startup.failed")
        else:
            LOGGER.debug("siyi.startup.failed", exc_info=exc)
        print(f"siyi startup failed: {exc}", file=sys.stderr)
        return 1

    if runtime.settings.runtime.non_interactive:
        return await _run_internal_worker(runtime, runtime.settings.runtime.initial_prompt)

    await run_repl(runtime.query_engine, runtime.renderer)
    return 0
