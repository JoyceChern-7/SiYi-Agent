from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from engine.events import QueryEvent
from engine.query_engine import QueryEngine


class SessionBusyError(RuntimeError):
    def __init__(self, session_id: str) -> None:
        super().__init__(f"Session already has an active turn: {session_id}")
        self.session_id = session_id


@dataclass(slots=True)
class SessionRuntime:
    query_engine: QueryEngine
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    active_turn_id: str | None = None

    @property
    def session_id(self) -> str:
        return self.query_engine.session.session_id

    async def submit_user_input(self, text: str) -> AsyncIterator[QueryEvent]:
        if self.lock.locked():
            raise SessionBusyError(self.session_id)
        async with self.lock:
            async for event in self.query_engine.submit_user_input(text):
                self.active_turn_id = event.turn_id or self.active_turn_id
                yield event
            self.active_turn_id = None

    async def interrupt_turn(self, turn_id: str) -> None:
        await self.query_engine.interrupt_turn(turn_id)
