from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import SecretStr

from config.user_settings import UserSettings, save_user_settings
from engine.query_engine import QueryEngine
from runtime.compaction import CompactionResult
from runtime.read_model import SessionSummary, TurnView


@dataclass(slots=True)
class SessionService:
    engine: QueryEngine

    def current(self):
        return self.engine.get_session_snapshot()

    def list(self) -> list[SessionSummary]:
        return self.engine.list_sessions()

    def turns(
        self,
        session_id: str | None = None,
        *,
        cursor: str | None = None,
        limit: int | None = None,
        items_view: str = "full",
    ) -> tuple[list[TurnView], str | None]:
        return self.engine.get_session_turns(
            session_id,
            cursor=cursor,
            limit=limit,
            items_view=items_view,
        )

    async def create(self):
        return await self.engine.new_session()

    async def switch(self, session_id: str):
        return await self.engine.switch_session(session_id)


@dataclass(slots=True)
class PermissionService:
    engine: QueryEngine

    def snapshot(self):
        return self.engine.get_permission_snapshot()

    def set_session_mode(self, mode: str):
        return self.engine.set_permission_mode(mode)

    def set_global_mode(self, mode: str):
        return self.engine.set_global_permission_mode(mode)


@dataclass(slots=True)
class CompactService:
    engine: QueryEngine

    async def compact(self, custom_instructions: str | None = None) -> CompactionResult:
        return await self.engine.compact(custom_instructions)


@dataclass(slots=True)
class AccountService:
    engine: QueryEngine

    def save_and_apply(self, settings: UserSettings, path: Path) -> Path:
        saved_path = save_user_settings(settings, path)
        self.apply(settings)
        return saved_path

    def apply(self, settings: UserSettings) -> None:
        selected_model = settings.models[settings.default_tier].model
        api_key = SecretStr(settings.api_key) if settings.api_key else None
        self.engine.settings.model = self.engine.settings.model.model_copy(
            update={
                "provider": settings.provider,
                "api_key": api_key,
                "base_url": settings.base_url,
                "model": selected_model,
            }
        )
        if hasattr(self.engine.llm, "api_key"):
            self.engine.llm.api_key = api_key
        if hasattr(self.engine.llm, "base_url"):
            self.engine.llm.base_url = settings.base_url
        if hasattr(self.engine.llm, "model"):
            self.engine.llm.model = selected_model


@dataclass(slots=True)
class AgentServices:
    session: SessionService
    permission: PermissionService
    compact: CompactService
    account: AccountService

    @classmethod
    def from_engine(cls, engine: QueryEngine) -> "AgentServices":
        return cls(
            session=SessionService(engine),
            permission=PermissionService(engine),
            compact=CompactService(engine),
            account=AccountService(engine),
        )
