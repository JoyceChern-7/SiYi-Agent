from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Literal

from engine.events import QueryEvent

TurnStatus = Literal["running", "completed", "failed", "interrupted"]


class TurnInterruptedError(RuntimeError):
    def __init__(self, turn_id: str) -> None:
        super().__init__(f"Turn was interrupted: {turn_id}")
        self.turn_id = turn_id


class TurnNotActiveError(RuntimeError):
    pass


@dataclass(slots=True)
class TurnRuntime:
    turn_id: str | None = None
    status: TurnStatus = "running"
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    active_task: asyncio.Task[object] | None = None
    events: list[QueryEvent] = field(default_factory=list)
    subscribers: set[asyncio.Queue[QueryEvent | None]] = field(default_factory=set)
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: str | None = None

    async def publish(self, event: QueryEvent) -> None:
        self.events.append(event)
        stale: list[asyncio.Queue[QueryEvent | None]] = []
        for subscriber in self.subscribers:
            try:
                subscriber.put_nowait(event)
            except asyncio.QueueFull:
                stale.append(subscriber)
        for subscriber in stale:
            self.subscribers.discard(subscriber)

    async def subscribe(
        self,
        after_event_id: str | None = None,
    ) -> AsyncIterator[QueryEvent]:
        start_index = 0
        if after_event_id:
            for index, event in enumerate(self.events):
                if event.event_id == after_event_id:
                    start_index = index + 1
                    break
        for event in self.events[start_index:]:
            yield event
        if self.finished_at is not None:
            return

        queue: asyncio.Queue[QueryEvent | None] = asyncio.Queue()
        self.subscribers.add(queue)
        try:
            while True:
                event = await queue.get()
                if event is None:
                    return
                yield event
        finally:
            self.subscribers.discard(queue)

    def finish(self, status: TurnStatus) -> None:
        self.status = status
        self.finished_at = datetime.now(timezone.utc).isoformat()
        for subscriber in list(self.subscribers):
            subscriber.put_nowait(None)
