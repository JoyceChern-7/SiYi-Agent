from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi.testclient import TestClient

from app.api import create_app
from app.cli import parse_args
from app.main import build_runtime
from engine.events import TurnCompletedEvent
from llm.base import LLMAdapter, LLMAssistantDone, LLMTextDelta
from runtime.session_runtime import SessionBusyError
from runtime.session_runtime_manager import SessionRuntimeManager
from runtime.usage_tracker import Usage


class FakeLLM(LLMAdapter):
    def __init__(self, text: str = "answer") -> None:
        self.text = text

    async def stream_chat(
        self,
        messages,
        system_prompt: str,
        tools,
        temperature: float,
    ) -> AsyncIterator[LLMTextDelta | LLMAssistantDone]:
        del messages, system_prompt, tools, temperature
        yield LLMTextDelta(delta=self.text)
        yield LLMAssistantDone(usage=Usage(input_tokens=1, output_tokens=1))


class BlockingLLM(LLMAdapter):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def stream_chat(
        self,
        messages,
        system_prompt: str,
        tools,
        temperature: float,
    ) -> AsyncIterator[LLMTextDelta | LLMAssistantDone]:
        del messages, system_prompt, tools, temperature
        self.started.set()
        await self.release.wait()
        yield LLMTextDelta(delta="done")
        yield LLMAssistantDone(usage=Usage(input_tokens=1, output_tokens=1))


class ApiBlockingLLM(LLMAdapter):
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    async def stream_chat(
        self,
        messages,
        system_prompt: str,
        tools,
        temperature: float,
    ) -> AsyncIterator[LLMTextDelta | LLMAssistantDone]:
        del messages, system_prompt, tools, temperature
        self.started.set()
        while not self.release.is_set():
            await asyncio.sleep(0.01)
        yield LLMTextDelta(delta="done")
        yield LLMAssistantDone(usage=Usage(input_tokens=1, output_tokens=1))


def _manager(tmp_path: Path) -> SessionRuntimeManager:
    runtime = build_runtime(parse_args(["--cwd", str(tmp_path)]))
    return SessionRuntimeManager(
        base_settings=runtime.settings,
        project_store=runtime.project_store,
        session_store=runtime.session_store,
    )


async def _collect_turn_events(turn) -> list[object]:
    return [event async for event in turn.subscribe()]


def test_runtime_manager_reuses_and_lazy_loads_session_runtime(tmp_path: Path) -> None:
    initial = build_runtime(parse_args(["--cwd", str(tmp_path)]))
    initial.query_engine.llm = FakeLLM("persisted")
    asyncio.run(_collect_events(initial.query_engine, "hello"))
    session_id = initial.query_engine.session.session_id
    manager = SessionRuntimeManager(
        base_settings=initial.settings,
        project_store=initial.project_store,
        session_store=initial.session_store,
    )

    async def run():
        first = await manager.get_or_create(session_id)
        second = await manager.get_or_create(session_id)
        return first, second

    first, second = asyncio.run(run())

    assert first is second
    assert first.query_engine.turn_counter == 1
    assert first.query_engine.get_messages()[0].to_plain_text() == "hello"


def test_turn_runtime_replays_buffered_events_and_after_event_id(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    session_id = manager.session_store.latest_metadata().session_id  # type: ignore[union-attr]

    async def run() -> tuple[list[object], list[object]]:
        runtime = await manager.get_or_create(session_id)
        runtime.query_engine.llm = FakeLLM("buffered")
        turn = await manager.start_turn(session_id, "hello")
        assert turn.active_task is not None
        await turn.active_task
        all_events = [event async for event in turn.subscribe()]
        after_first = [event async for event in turn.subscribe(after_event_id=all_events[0].event_id)]
        return all_events, after_first

    all_events, after_first = asyncio.run(run())

    assert all_events[0].type == "turn_started"
    assert isinstance(all_events[-1], TurnCompletedEvent)
    assert all_events[-1].status == "completed"
    assert after_first[0].event_id != all_events[0].event_id


def test_runtime_manager_runs_different_sessions_in_parallel_and_rejects_same_session(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    first_session = manager.session_store.latest_metadata().session_id  # type: ignore[union-attr]
    second_root = tmp_path / "second"
    second_root.mkdir()

    async def run() -> tuple[str, str]:
        second_summary = await manager.create_session(second_root)
        first_runtime = await manager.get_or_create(first_session)
        second_runtime = await manager.get_or_create(second_summary.session_id)
        first_llm = BlockingLLM()
        second_llm = BlockingLLM()
        first_runtime.query_engine.llm = first_llm
        second_runtime.query_engine.llm = second_llm

        first_turn = await manager.start_turn(first_session, "first")
        second_turn = await manager.start_turn(second_summary.session_id, "second")
        await asyncio.wait_for(first_llm.started.wait(), timeout=2)
        await asyncio.wait_for(second_llm.started.wait(), timeout=2)
        try:
            await manager.start_turn(first_session, "busy")
        except SessionBusyError as exc:
            assert exc.error.code == "session_busy"
        else:
            raise AssertionError("expected session_busy")
        first_llm.release.set()
        second_llm.release.set()
        assert first_turn.active_task is not None and second_turn.active_task is not None
        await first_turn.active_task
        await second_turn.active_task
        return first_turn.status, second_turn.status

    first_status, second_status = asyncio.run(run())

    assert first_status == "completed"
    assert second_status == "completed"


def test_fastapi_starts_turn_and_streams_sse_events(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    session_id = manager.session_store.latest_metadata().session_id  # type: ignore[union-attr]

    async def prepare() -> None:
        runtime = await manager.get_or_create(session_id)
        runtime.query_engine.llm = FakeLLM("api answer")

    asyncio.run(prepare())
    app = create_app(manager)

    with TestClient(app) as client:
        response = client.post(f"/sessions/{session_id}/turns", json={"text": "hello api"})
        assert response.status_code == 200
        turn_id = response.json()["turn_id"]
        with client.stream("GET", f"/sessions/{session_id}/turns/{turn_id}/events") as stream:
            body = "".join(stream.iter_text())

    assert "event: turn_started" in body
    assert "event: assistant_delta" in body
    assert "event: turn_completed" in body
    assert '"type": "turn_completed"' in body


def test_fastapi_returns_409_for_busy_session(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    session_id = manager.session_store.latest_metadata().session_id  # type: ignore[union-attr]
    llm = ApiBlockingLLM()

    async def prepare() -> None:
        runtime = await manager.get_or_create(session_id)
        runtime.query_engine.llm = llm

    asyncio.run(prepare())
    app = create_app(manager)

    with TestClient(app) as client:
        first = client.post(f"/sessions/{session_id}/turns", json={"text": "blocking"})
        assert first.status_code == 200
        assert llm.started.wait(timeout=2)
        response = client.post(f"/sessions/{session_id}/turns", json={"text": "second"})
        llm.release.set()
        turn_id = first.json()["turn_id"]
        with client.stream("GET", f"/sessions/{session_id}/turns/{turn_id}/events") as stream:
            body = "".join(stream.iter_text())

    assert response.status_code == 409
    assert response.json()["code"] == "session_busy"
    assert "event: turn_completed" in body


def test_fastapi_interrupts_running_turn(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    session_id = manager.session_store.latest_metadata().session_id  # type: ignore[union-attr]
    llm = ApiBlockingLLM()

    async def prepare() -> None:
        runtime = await manager.get_or_create(session_id)
        runtime.query_engine.llm = llm

    asyncio.run(prepare())
    app = create_app(manager)

    with TestClient(app) as client:
        started = client.post(f"/sessions/{session_id}/turns", json={"text": "interrupt me"})
        assert started.status_code == 200
        assert llm.started.wait(timeout=2)
        turn_id = started.json()["turn_id"]
        interrupted = client.post(f"/sessions/{session_id}/turns/{turn_id}/interrupt")
        assert interrupted.status_code == 200
        with client.stream("GET", f"/sessions/{session_id}/turns/{turn_id}/events") as stream:
            body = "".join(stream.iter_text())

    assert "event: turn_completed" in body
    assert '"status": "interrupted"' in body


def test_fastapi_permission_endpoints_expose_pending_and_missing_resolution(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    app = create_app(manager)

    with TestClient(app) as client:
        pending = client.get("/permissions/pending")
        missing = client.post("/permissions/approval_missing/resolve", json={"approved": True})

    assert pending.status_code == 200
    assert pending.json() == []
    assert missing.status_code == 404


async def _collect_events(engine, prompt: str):
    return [event async for event in engine.submit_user_input(prompt)]
