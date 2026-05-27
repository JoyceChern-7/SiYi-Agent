from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from config.settings import AppSettings
from engine.events import AgentError
from engine.query_engine import QueryEngine
from llm.openai_adapter import OpenAIChatAdapter
from runtime.compaction import CompactionManager
from runtime.permissions import (
    PermissionApprovalBroker,
    PermissionManager,
    load_global_permission_mode,
)
from runtime.read_model import HistoryService, SessionSummary, TurnView
from runtime.project_store import ProjectStore
from runtime.session_runtime import SessionBusyError, SessionRuntime
from runtime.session_store import JsonlSessionStore, SessionHandle
from runtime.token_budget import TokenBudget
from runtime.turn_runtime import TurnRuntime
from runtime.usage_tracker import Usage, UsageTracker
from tools.registry import ToolRegistry


@dataclass(slots=True)
class SessionRuntimeManager:
    base_settings: AppSettings
    project_store: ProjectStore
    session_store: JsonlSessionStore
    approval_broker: PermissionApprovalBroker = field(default_factory=PermissionApprovalBroker)
    runtimes: dict[str, SessionRuntime] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def history_service(self) -> HistoryService:
        return HistoryService(self.session_store)

    async def create_session(self, project_root: Path | None = None) -> SessionSummary:
        root = (project_root or self.base_settings.runtime.project_root).expanduser().resolve()
        project = self.project_store.ensure_project(root)
        session = self.session_store.create_session(
            project=project,
            permission_mode=load_global_permission_mode(),
        )
        runtime = await self._build_runtime(session)
        async with self._lock:
            self.runtimes[session.session_id] = runtime
        return self._summary_for_session(session.session_id)

    async def get_or_create(self, session_id: str) -> SessionRuntime:
        runtime = self.runtimes.get(session_id)
        if runtime is not None:
            return runtime
        async with self._lock:
            runtime = self.runtimes.get(session_id)
            if runtime is not None:
                return runtime
            session = self.session_store.switch_session(session_id)
            runtime = await self._build_runtime(session)
            self.runtimes[session_id] = runtime
            return runtime

    def list_sessions(self) -> list[SessionSummary]:
        active_turns = {
            session_id: runtime.active_turn_id
            for session_id, runtime in self.runtimes.items()
            if runtime.active_turn_id is not None
        }
        last_errors = {
            session_id: runtime.query_engine.last_error
            for session_id, runtime in self.runtimes.items()
        }
        usage_by_session: dict[str, Usage | None] = {
            session_id: runtime.query_engine.usage_tracker.get_total_usage()
            for session_id, runtime in self.runtimes.items()
        }
        return self.history_service.list_session_summaries(
            active_turns=active_turns,
            last_errors=last_errors,
            usage_by_session=usage_by_session,
        )

    def get_turns(
        self,
        session_id: str,
        *,
        cursor: str | None = None,
        limit: int | None = None,
        items_view: str = "full",
    ) -> tuple[list[TurnView], str | None]:
        view = "summary" if items_view == "summary" else "full"
        return self.history_service.get_session_turns(
            session_id,
            cursor=cursor,
            limit=limit,
            items_view=view,
        )

    async def start_turn(self, session_id: str, text: str) -> TurnRuntime:
        runtime = await self.get_or_create(session_id)
        return await runtime.start_turn(text)

    async def subscribe_turn(
        self,
        session_id: str,
        turn_id: str,
        *,
        after_event_id: str | None = None,
    ):
        runtime = await self.get_or_create(session_id)
        async for event in runtime.subscribe_turn(turn_id, after_event_id=after_event_id):
            yield event

    async def interrupt_turn(self, session_id: str, turn_id: str) -> None:
        runtime = await self.get_or_create(session_id)
        await runtime.interrupt_turn(turn_id)

    def session_busy_error(self, session_id: str) -> AgentError:
        return SessionBusyError(session_id).error

    async def _build_runtime(self, session: SessionHandle) -> SessionRuntime:
        project = self.session_store.get_project_for_session(session.metadata)
        if project is None:
            raise ValueError(f"Session has no project metadata: {session.session_id}")
        settings = self.base_settings.model_copy(deep=True)
        settings.runtime.project_root = Path(project.project_root).expanduser().resolve()
        permission_manager = PermissionManager.from_settings(
            settings.tools,
            project_root=settings.runtime.project_root,
            mode=session.metadata.permission_mode,
        )
        permission_manager.set_approval_broker(self.approval_broker)
        tool_registry = ToolRegistry.default(permission_manager=permission_manager)
        compaction_manager = CompactionManager(settings.runtime.compaction_enabled)
        token_budget = TokenBudget(settings.runtime)
        usage_tracker = UsageTracker(settings.model.pricing)
        llm = OpenAIChatAdapter.from_settings(settings.model)
        engine = QueryEngine(
            session=session,
            settings=settings,
            llm=llm,
            tool_registry=tool_registry,
            session_store=self.session_store,
            permission_manager=permission_manager,
            compaction_manager=compaction_manager,
            token_budget=token_budget,
            usage_tracker=usage_tracker,
        )
        return SessionRuntime(engine)

    def _summary_for_session(self, session_id: str) -> SessionSummary:
        for summary in self.list_sessions():
            if summary.session_id == session_id:
                return summary
        raise ValueError(f"Session not found: {session_id}")
