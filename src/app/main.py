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
from runtime.project_store import ProjectStore
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
    project_store: ProjectStore
    session_store: JsonlSessionStore
    permission_manager: PermissionManager
    compaction_manager: CompactionManager
    token_budget: TokenBudget
    usage_tracker: UsageTracker
    query_engine: QueryEngine
    renderer: ConsoleRenderer


def _resolve_project_root(options: CLIOptions) -> Path:
    project_root = options.cwd or Path.cwd()
    if not project_root.exists() or not project_root.is_dir():
        raise ValueError(f"Project root does not exist or is not a directory: {project_root}")
    return project_root.resolve()

def _initialize_process(options: CLIOptions) -> Path:
    project_root = _resolve_project_root(options)
    os.environ["SIYI_ENTRYPOINT"] = (
        "worker" if options.internal_worker else "cli"
    )
    return project_root


def build_runtime(options: CLIOptions) -> AppRuntime:
    started_at = time.perf_counter()
    project_root = _initialize_process(options)
    settings = load_settings(options=options, project_root=project_root)

    configure_logging(debug=settings.runtime.debug)
    LOGGER.debug(
        "siyi.startup.begin",
        extra={
            "project_root": str(project_root),
            "non_interactive": settings.runtime.non_interactive,
            "resume": settings.runtime.resume,
        },
    )

    project_store = ProjectStore()
    session_store = JsonlSessionStore(settings.runtime.session_dir, project_store=project_store)
    ensure_permission_files()
    permission_mode = load_global_permission_mode()
    session = session_store.open_session(
        requested_session=settings.runtime.resume,
        project_root=project_root,
        permission_mode=permission_mode,
    )
    project = session_store.get_project_for_session(session.metadata)
    if project is None:
        raise ValueError(f"Session has no project metadata: {session.session_id}")
    settings.runtime.project_root = Path(project.project_root).expanduser().resolve()

    permission_manager = PermissionManager.from_settings(
        settings.tools,
        project_root=settings.runtime.project_root,
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
        project_store=project_store,
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
