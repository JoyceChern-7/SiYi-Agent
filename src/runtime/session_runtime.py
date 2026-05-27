from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from engine.events import AgentError, QueryEvent
from engine.query_engine import QueryEngine
from runtime.ids import new_id
from runtime.turn_runtime import TurnRuntime


class SessionBusyError(RuntimeError):
    def __init__(self, session_id: str) -> None:
        super().__init__(f"Session already has an active turn: {session_id}")
        self.session_id = session_id
        self.error = AgentError(
            session_id=session_id,
            code="session_busy",
            category="concurrency",
            message="Session already has an active turn.",
            retryable=True,
            action="wait_or_interrupt",
        )


@dataclass(slots=True)
class SessionRuntime:
    query_engine: QueryEngine
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    active_turn_id: str | None = None
    active_turn: TurnRuntime | None = None
    turns: dict[str, TurnRuntime] = field(default_factory=dict)

    @property
    def session_id(self) -> str:
        return self.query_engine.session.session_id

    async def start_turn(self, text: str) -> TurnRuntime:
        if self.lock.locked():
            raise SessionBusyError(self.session_id)
        await self.lock.acquire()
        turn_runtime = TurnRuntime(turn_id=new_id("turn"))
        self.active_turn = turn_runtime
        self.active_turn_id = turn_runtime.turn_id
        self.turns[turn_runtime.turn_id] = turn_runtime
        task = asyncio.create_task(self._run_turn(text, turn_runtime))
        turn_runtime.active_task = task
        return turn_runtime

    async def submit_user_input(self, text: str) -> AsyncIterator[QueryEvent]:
        turn_runtime = await self.start_turn(text)
        async for event in turn_runtime.subscribe():
            yield event

    async def subscribe_turn(
        self,
        turn_id: str,
        *,
        after_event_id: str | None = None,
    ) -> AsyncIterator[QueryEvent]:
        turn_runtime = self.turns.get(turn_id)
        if turn_runtime is None:
            raise ValueError(f"Turn is not loaded: {turn_id}")
        async for event in turn_runtime.subscribe(after_event_id=after_event_id):
            yield event

    async def interrupt_turn(self, turn_id: str) -> None:
        if self.active_turn is None or self.active_turn_id != turn_id:
            raise ValueError(f"Turn is not active: {turn_id}")
        self.active_turn.cancel_event.set()
        await self.query_engine.interrupt_turn(turn_id)

    async def _run_turn(self, text: str, turn_runtime: TurnRuntime) -> None:
        try:
            async for event in self.query_engine.submit_user_input(
                text,
                turn_id=turn_runtime.turn_id,
                cancel_event=turn_runtime.cancel_event,
            ):
                await turn_runtime.publish(event)
                if event.type == "turn_completed":
                    turn_runtime.finish(getattr(event, "status", "completed"))
        except Exception as exc:  # noqa: BLE001 - background turns must publish failure
            error = AgentError(
                session_id=self.session_id,
                turn_id=turn_runtime.turn_id,
                code="runtime_error",
                category="runtime",
                message=str(exc) or type(exc).__name__,
                retryable=False,
                details={"exception_type": type(exc).__name__},
            )
            self.query_engine.last_error = error
            await turn_runtime.publish(error)
            turn_runtime.finish("failed")
        finally:
            if turn_runtime.finished_at is None:
                turn_runtime.finish(
                    "interrupted" if turn_runtime.cancel_event.is_set() else "completed"
                )
            if self.active_turn is turn_runtime:
                self.active_turn = None
                self.active_turn_id = None
            if self.lock.locked():
                self.lock.release()
