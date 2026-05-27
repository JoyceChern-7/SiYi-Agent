from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

from app.cli import parse_args
from app.main import build_runtime
from config.paths import get_siyi_config_path
from engine.events import (
    AgentError,
    ToolOutputDeltaEvent,
    ToolResultEvent,
    TurnCompletedEvent,
    TurnStartedEvent,
)
from engine.message_schema import (
    ToolUseBlock,
    assistant_message_from_blocks,
    tool_result_message,
    user_message,
)
from engine.session_title import normalize_session_title
from llm.base import LLMAdapter, LLMAssistantDone, LLMTextDelta, LLMToolUse
from runtime.session_runtime import SessionBusyError, SessionRuntime
from runtime.session_store import SessionMetadata
from runtime.token_budget import AUTO_COMPACT_THRESHOLD_TOKENS
from runtime.compaction import CompactionManager, TIME_BASED_MC_CLEARED_MESSAGE
from runtime.usage_tracker import Usage
from tools.base import BaseTool, ToolContext, ToolResult, emit_tool_output_delta


class FakeLLMAdapter(LLMAdapter):
    def __init__(self, responses: list[str], *, emit_tool_use: bool = False) -> None:
        self.responses = responses
        self.emit_tool_use = emit_tool_use

    async def stream_chat(
        self,
        messages,
        system_prompt: str,
        tools,
        temperature: float,
    ) -> AsyncIterator[LLMTextDelta | LLMToolUse | LLMAssistantDone]:
        del messages, system_prompt, tools, temperature
        text = self.responses.pop(0)
        yield LLMTextDelta(delta=text)
        if self.emit_tool_use:
            yield LLMToolUse(
                block=ToolUseBlock(
                    name="web_search",
                    input={"query": "latest API release notes"},
                ),
            )
        yield LLMAssistantDone(usage=Usage(input_tokens=10, output_tokens=5))


class RaisingLLMAdapter(LLMAdapter):
    async def stream_chat(
        self,
        messages,
        system_prompt: str,
        tools,
        temperature: float,
    ) -> AsyncIterator[LLMTextDelta | LLMAssistantDone]:
        del messages, system_prompt, tools, temperature
        raise RuntimeError("boom")
        yield  # pragma: no cover


class BlockingLLMAdapter(LLMAdapter):
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


class BlockingTool(BaseTool):
    name = "BlockingTool"
    description = "Block until interrupted."
    input_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def run(self, raw_input, context: ToolContext) -> ToolResult:
        del raw_input, context
        self.started.set()
        await asyncio.Event().wait()
        return ToolResult(success=True, content="unreachable")


class FastReadOnlyTool(BaseTool):
    name = "FastReadOnlyTool"
    description = "Return immediately."
    read_only = True
    input_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    def __init__(self) -> None:
        self.completed = asyncio.Event()

    async def run(self, raw_input, context: ToolContext) -> ToolResult:
        del raw_input, context
        self.completed.set()
        return ToolResult(success=True, content="fast result")


class StreamingBlockingTool(BaseTool):
    name = "StreamingBlockingTool"
    description = "Emit output, then block."
    read_only = True
    input_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def run(self, raw_input, context: ToolContext) -> ToolResult:
        del raw_input
        await emit_tool_output_delta(context, stream="stdout", delta="partial output\n")
        self.started.set()
        await asyncio.Event().wait()
        return ToolResult(success=True, content="unreachable")


class BlockingToolLLM(LLMAdapter):
    async def stream_chat(
        self,
        messages,
        system_prompt: str,
        tools,
        temperature: float,
    ) -> AsyncIterator[LLMToolUse | LLMAssistantDone]:
        del messages, system_prompt, tools, temperature
        yield LLMToolUse(block=ToolUseBlock(id="toolu_block", name="BlockingTool", input={}))
        yield LLMAssistantDone(usage=Usage(input_tokens=1, output_tokens=1))


class ConcurrentToolsLLM(LLMAdapter):
    async def stream_chat(
        self,
        messages,
        system_prompt: str,
        tools,
        temperature: float,
    ) -> AsyncIterator[LLMToolUse | LLMAssistantDone]:
        del messages, system_prompt, tools, temperature
        yield LLMToolUse(block=ToolUseBlock(id="toolu_fast", name="FastReadOnlyTool", input={}))
        yield LLMToolUse(block=ToolUseBlock(id="toolu_blocking", name="StreamingBlockingTool", input={}))
        yield LLMAssistantDone(usage=Usage(input_tokens=1, output_tokens=1))


class OneToolThenAnswerLLM(LLMAdapter):
    def __init__(self) -> None:
        self.calls = 0

    async def stream_chat(
        self,
        messages,
        system_prompt: str,
        tools,
        temperature: float,
    ) -> AsyncIterator[LLMToolUse | LLMTextDelta | LLMAssistantDone]:
        del messages, system_prompt, tools, temperature
        self.calls += 1
        if self.calls == 1:
            yield LLMToolUse(block=ToolUseBlock(id="toolu_fast", name="FastReadOnlyTool", input={}))
            yield LLMAssistantDone(usage=Usage(input_tokens=1, output_tokens=1))
            return
        yield LLMTextDelta(delta="tool answer")
        yield LLMAssistantDone(usage=Usage(input_tokens=1, output_tokens=1))


class CapturingLLM(LLMAdapter):
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[list[str]] = []

    async def stream_chat(
        self,
        messages,
        system_prompt: str,
        tools,
        temperature: float,
    ) -> AsyncIterator[LLMTextDelta | LLMAssistantDone]:
        del system_prompt, tools, temperature
        self.calls.append([message.to_plain_text() for message in messages])
        yield LLMTextDelta(delta=self.responses.pop(0))
        yield LLMAssistantDone(usage=Usage(input_tokens=1, output_tokens=1))


class FakeTitleAgent:
    def __init__(self, title: str | None = None) -> None:
        self.title = title

    async def generate(self, first_user_input: str) -> str:
        if self.title is None:
            raise RuntimeError("title failed")
        return self.title


class PromptTooLongThenSummaryLLM(LLMAdapter):
    def __init__(self) -> None:
        self.calls = 0

    async def stream_chat(
        self,
        messages,
        system_prompt: str,
        tools,
        temperature: float,
    ) -> AsyncIterator[LLMTextDelta | LLMAssistantDone]:
        del messages, system_prompt, tools, temperature
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("Prompt is too long")
        yield LLMTextDelta(delta="<summary>compact summary</summary>")
        yield LLMAssistantDone(usage=Usage(input_tokens=1, output_tokens=1))


async def _collect_events(engine, prompt: str):
    return [event async for event in engine.submit_user_input(prompt)]


async def _collect_events_and_title(engine, prompt: str):
    events = [event async for event in engine.submit_user_input(prompt)]
    await engine.wait_for_title_task()
    return events


def test_query_engine_accumulates_multi_turn_history_and_resume(tmp_path: Path) -> None:
    runtime = build_runtime(parse_args(["--cwd", str(tmp_path)]))
    runtime.query_engine.llm = FakeLLMAdapter(["first answer", "second answer"])

    first_events = asyncio.run(_collect_events(runtime.query_engine, "first question"))
    second_events = asyncio.run(_collect_events(runtime.query_engine, "second question"))

    assert isinstance(first_events[0], TurnStartedEvent)
    assert isinstance(first_events[-1], TurnCompletedEvent)
    assert first_events[-1].status == "completed"
    assert not any(event.type in {"status", "final_answer"} for event in first_events)
    assert isinstance(second_events[-1], TurnCompletedEvent)
    assert second_events[-1].status == "completed"
    assert runtime.query_engine.current_turn is None
    assert runtime.query_engine.turn_counter == 2
    assert len(runtime.query_engine.get_messages()) == 4
    assert runtime.query_engine.get_messages()[0].to_plain_text() == "first question"
    assert runtime.query_engine.get_messages()[1].to_plain_text() == "first answer"
    assert runtime.query_engine.get_messages()[1].metadata["usage"] == {
        "input_tokens": 10,
        "output_tokens": 5,
        "cached_tokens": 0,
    }
    assert runtime.query_engine.get_messages()[2].to_plain_text() == "second question"
    assert runtime.query_engine.get_messages()[3].to_plain_text() == "second answer"

    resumed = build_runtime(
        parse_args(
            [
                "--cwd",
                str(tmp_path),
                "--resume",
                runtime.query_engine.session.session_id,
            ]
        )
    )
    assert resumed.query_engine.turn_counter == 2
    assert len(resumed.query_engine.get_messages()) == 4


def test_empty_prompt_emits_agent_error_without_error_event_wrapper(tmp_path: Path) -> None:
    runtime = build_runtime(parse_args(["--cwd", str(tmp_path)]))

    events = asyncio.run(_collect_events(runtime.query_engine, "   "))

    assert len(events) == 1
    assert isinstance(events[0], AgentError)
    assert events[0].type == "agent_error"
    assert events[0].code == "empty_prompt"
    assert events[0].category == "validation"
    assert runtime.query_engine.last_error == events[0]


def test_session_title_agent_updates_pending_session_name(tmp_path: Path) -> None:
    runtime = build_runtime(parse_args(["--cwd", str(tmp_path)]))
    runtime.query_engine.llm = FakeLLMAdapter(["answer"])
    runtime.query_engine.title_agent = FakeTitleAgent("实现 FastAPI")

    asyncio.run(_collect_events_and_title(runtime.query_engine, "帮我实现 FastAPI"))

    metadata = runtime.query_engine.session.metadata
    assert metadata.name == "实现 FastAPI"
    assert metadata.name_status == "ready"


def test_session_title_falls_back_to_truncated_prompt(tmp_path: Path) -> None:
    runtime = build_runtime(parse_args(["--cwd", str(tmp_path)]))
    runtime.query_engine.llm = FakeLLMAdapter(["answer"])
    runtime.query_engine.title_agent = FakeTitleAgent(None)

    asyncio.run(_collect_events_and_title(runtime.query_engine, "请分析这个项目并提出迁移建议"))

    metadata = runtime.query_engine.session.metadata
    assert metadata.name == "请分析这个项目并提出"
    assert metadata.name_status == "ready"


def test_session_title_limits_english_words() -> None:
    assert normalize_session_title(
        "one two three four five six seven eight nine ten eleven twelve"
    ) == "one two three four five six seven eight nine ten"


def test_session_runtime_rejects_same_session_concurrent_prompt(tmp_path: Path) -> None:
    runtime = build_runtime(parse_args(["--cwd", str(tmp_path)]))
    llm = BlockingLLMAdapter()
    runtime.query_engine.llm = llm
    runtime.query_engine.session.metadata.name_status = "ready"
    session_runtime = SessionRuntime(runtime.query_engine)

    async def run() -> SessionBusyError:
        first_task = asyncio.create_task(_collect_events(session_runtime, "first"))
        await llm.started.wait()
        try:
            await _collect_events(session_runtime, "second")
        except SessionBusyError as exc:
            error = exc
        else:
            raise AssertionError("expected SessionBusyError")
        finally:
            llm.release.set()
            await first_task
        return error

    error = asyncio.run(run())

    assert error.session_id == runtime.query_engine.session.session_id
    assert error.error.code == "session_busy"
    assert error.error.category == "concurrency"
    assert error.error.action == "wait_or_interrupt"


def test_session_runtime_interrupts_active_turn_and_releases_lock(tmp_path: Path) -> None:
    runtime = build_runtime(parse_args(["--cwd", str(tmp_path)]))
    llm = BlockingLLMAdapter()
    runtime.query_engine.llm = llm
    runtime.query_engine.session.metadata.name_status = "ready"
    session_runtime = SessionRuntime(runtime.query_engine)

    async def run() -> tuple[list[object], list[object]]:
        first_task = asyncio.create_task(_collect_events(session_runtime, "first"))
        await llm.started.wait()
        assert session_runtime.active_turn_id is not None
        await session_runtime.interrupt_turn(session_runtime.active_turn_id)
        interrupted_events = await first_task
        runtime.query_engine.llm = FakeLLMAdapter(["after interrupt"])
        next_events = await _collect_events(session_runtime, "second")
        return interrupted_events, next_events

    interrupted_events, next_events = asyncio.run(run())

    assert isinstance(interrupted_events[-1], TurnCompletedEvent)
    assert interrupted_events[-1].status == "interrupted"
    assert isinstance(next_events[-1], TurnCompletedEvent)
    assert next_events[-1].status == "completed"
    assert runtime.query_engine.current_turn is None
    assert len(runtime.query_engine.get_messages()) == 3
    assert runtime.query_engine.get_messages()[0].to_plain_text() == "first"
    assert runtime.query_engine.get_messages()[0].is_meta is True
    assert runtime.query_engine.get_messages()[0].metadata["hidden_reason"] == "interrupted_turn"


def test_session_runtime_interrupt_rejects_wrong_turn_id(tmp_path: Path) -> None:
    runtime = build_runtime(parse_args(["--cwd", str(tmp_path)]))
    llm = BlockingLLMAdapter()
    runtime.query_engine.llm = llm
    runtime.query_engine.session.metadata.name_status = "ready"
    session_runtime = SessionRuntime(runtime.query_engine)

    async def run() -> str:
        task = asyncio.create_task(_collect_events(session_runtime, "first"))
        await llm.started.wait()
        try:
            await session_runtime.interrupt_turn("turn_wrong")
        except ValueError as exc:
            message = str(exc)
        else:
            raise AssertionError("expected ValueError")
        finally:
            assert session_runtime.active_turn_id is not None
            await session_runtime.interrupt_turn(session_runtime.active_turn_id)
            await task
        return message

    message = asyncio.run(run())

    assert "not active" in message


def test_interrupt_adds_synthetic_tool_result_for_open_tool_call(tmp_path: Path) -> None:
    runtime = build_runtime(parse_args(["--cwd", str(tmp_path)]))
    tool = BlockingTool()
    runtime.tool_registry.register(tool)
    runtime.query_engine.llm = BlockingToolLLM()
    runtime.query_engine.session.metadata.name_status = "ready"
    runtime.query_engine.permission_manager.set_mode("full")
    session_runtime = SessionRuntime(runtime.query_engine)

    async def run() -> list[object]:
        task = asyncio.create_task(_collect_events(session_runtime, "run blocking tool"))
        await tool.started.wait()
        assert session_runtime.active_turn_id is not None
        await session_runtime.interrupt_turn(session_runtime.active_turn_id)
        return await task

    events = asyncio.run(run())
    synthetic_results = [
        event
        for event in events
        if isinstance(event, ToolResultEvent) and event.block.tool_use_id == "toolu_block"
    ]

    assert synthetic_results
    assert synthetic_results[0].block.is_error is True
    assert "interrupted" in synthetic_results[0].block.content
    assert isinstance(events[-1], TurnCompletedEvent)
    assert events[-1].status == "interrupted"
    assert any(message.has_tool_result() for message in runtime.query_engine.get_messages())
    assert all(
        message.is_meta
        for message in runtime.query_engine.get_messages()
        if message.metadata.get("hidden_reason") == "interrupted_turn"
    )


def test_interrupt_preserves_completed_concurrent_tool_results(tmp_path: Path) -> None:
    runtime = build_runtime(parse_args(["--cwd", str(tmp_path)]))
    fast_tool = FastReadOnlyTool()
    blocking_tool = StreamingBlockingTool()
    runtime.tool_registry.register(fast_tool)
    runtime.tool_registry.register(blocking_tool)
    runtime.query_engine.llm = ConcurrentToolsLLM()
    runtime.query_engine.session.metadata.name_status = "ready"
    runtime.query_engine.permission_manager.set_mode("full")
    session_runtime = SessionRuntime(runtime.query_engine)

    async def run() -> list[object]:
        task = asyncio.create_task(_collect_events(session_runtime, "run concurrent tools"))
        await fast_tool.completed.wait()
        await blocking_tool.started.wait()
        assert session_runtime.active_turn_id is not None
        await session_runtime.interrupt_turn(session_runtime.active_turn_id)
        return await task

    events = asyncio.run(run())
    results = [
        event
        for event in events
        if isinstance(event, ToolResultEvent)
    ]

    assert any(
        event.block.tool_use_id == "toolu_fast"
        and event.block.is_error is False
        and "fast result" in event.block.content
        for event in results
    )
    assert any(
        event.block.tool_use_id == "toolu_blocking"
        and event.block.is_error is True
        and "interrupted" in event.block.content
        for event in results
    )
    assert any(
        isinstance(event, ToolOutputDeltaEvent)
        and event.tool_use_id == "toolu_blocking"
        and "partial output" in event.delta
        for event in events
    )
    assert events[-1].status == "interrupted"


def test_interrupted_turn_is_hidden_from_next_prompt_and_resume(tmp_path: Path) -> None:
    runtime = build_runtime(parse_args(["--cwd", str(tmp_path)]))
    blocking_llm = BlockingLLMAdapter()
    runtime.query_engine.llm = blocking_llm
    runtime.query_engine.session.metadata.name_status = "ready"
    session_runtime = SessionRuntime(runtime.query_engine)

    async def interrupt_first_turn() -> None:
        task = asyncio.create_task(_collect_events(session_runtime, "typo prompt"))
        await blocking_llm.started.wait()
        assert session_runtime.active_turn_id is not None
        await session_runtime.interrupt_turn(session_runtime.active_turn_id)
        await task

    asyncio.run(interrupt_first_turn())

    capture = CapturingLLM(["fixed answer"])
    runtime.query_engine.llm = capture
    asyncio.run(_collect_events(session_runtime, "fixed prompt"))
    assert capture.calls
    assert all("typo prompt" not in message for message in capture.calls[0])

    resumed = build_runtime(
        parse_args(["--cwd", str(tmp_path), "--resume", runtime.query_engine.session.session_id])
    )
    resumed.query_engine.session.metadata.name_status = "ready"
    resumed_capture = CapturingLLM(["resumed answer"])
    resumed.query_engine.llm = resumed_capture
    asyncio.run(_collect_events(resumed.query_engine, "after resume"))

    assert resumed_capture.calls
    assert all("typo prompt" not in message for message in resumed_capture.calls[0])
    assert resumed.query_engine.get_messages()[0].is_meta is True
    assert resumed.query_engine.get_messages()[0].metadata["hidden_reason"] == "interrupted_turn"


def test_global_session_index_and_cross_project_switch(tmp_path: Path) -> None:
    first_project_root = tmp_path / "first"
    second_project_root = tmp_path / "second"
    first_project_root.mkdir()
    second_project_root.mkdir()

    first = build_runtime(parse_args(["--cwd", str(first_project_root)]))
    first.query_engine.llm = FakeLLMAdapter(["first answer"])
    asyncio.run(_collect_events(first.query_engine, "first question"))
    first_session_id = first.query_engine.session.session_id

    second = build_runtime(parse_args(["--cwd", str(second_project_root)]))
    second_session_id = second.query_engine.session.session_id

    sessions = second.query_engine.list_sessions()
    assert {session.session_id for session in sessions} >= {first_session_id, second_session_id}
    assert second.session_store.index_path.exists()

    snapshot = asyncio.run(second.query_engine.switch_session(first_session_id))

    assert snapshot.session_id == first_session_id
    assert Path(snapshot.project_root) == first_project_root.resolve()
    assert second.query_engine.get_messages()[0].to_plain_text() == "first question"
    assert second.query_engine.turn_counter == 1


def test_session_summary_reports_project_and_running_state(tmp_path: Path) -> None:
    runtime = build_runtime(parse_args(["--cwd", str(tmp_path)]))
    llm = BlockingLLMAdapter()
    runtime.query_engine.llm = llm
    runtime.query_engine.session.metadata.name_status = "ready"
    session_runtime = SessionRuntime(runtime.query_engine)

    async def run() -> tuple[str, list[object]]:
        task = asyncio.create_task(_collect_events(session_runtime, "running prompt"))
        await llm.started.wait()
        summaries = runtime.query_engine.list_sessions()
        active = next(
            item
            for item in summaries
            if item.session_id == runtime.query_engine.session.session_id
        )
        assert active.status == "running"
        assert active.active_turn_id == session_runtime.active_turn_id
        assert active.project_root == str(tmp_path.resolve())
        assert active.usage is not None
        llm.release.set()
        return active.active_turn_id or "", await task

    active_turn_id, events = asyncio.run(run())

    assert active_turn_id
    assert isinstance(events[-1], TurnCompletedEvent)
    assert runtime.query_engine.list_sessions()[0].status == "idle"


def test_session_turn_read_model_projects_full_and_summary_items(tmp_path: Path) -> None:
    runtime = build_runtime(parse_args(["--cwd", str(tmp_path)]))
    runtime.tool_registry.register(FastReadOnlyTool())
    runtime.query_engine.llm = OneToolThenAnswerLLM()
    runtime.query_engine.permission_manager.set_mode("full")

    asyncio.run(_collect_events(runtime.query_engine, "run one tool"))

    full_turns, next_cursor = runtime.query_engine.get_session_turns(items_view="full")
    assert next_cursor is None
    assert len(full_turns) == 1
    turn = full_turns[0]
    assert turn.status == "completed"
    assert [item.type for item in turn.items] == [
        "user_message",
        "tool_call",
        "assistant_message",
        "tool_result",
        "assistant_message",
    ]
    assert turn.items[0].text == "run one tool"
    assert turn.items[1].tool_name == "FastReadOnlyTool"
    assert turn.items[3].tool_use_id == "toolu_fast"
    assert turn.items[3].is_error is False

    summary_turns, _ = runtime.query_engine.get_session_turns(items_view="summary")
    assert [item.type for item in summary_turns[0].items] == [
        "user_message",
        "assistant_message",
    ]
    assert summary_turns[0].items[-1].text == "tool answer"


def test_session_permission_mode_inherits_global_default_and_switches(tmp_path: Path) -> None:
    first_cwd = tmp_path / "first"
    second_cwd = tmp_path / "second"
    first_cwd.mkdir()
    second_cwd.mkdir()
    config_path = get_siyi_config_path()
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps({"permission_mode": "full"}), encoding="utf-8")

    first = build_runtime(parse_args(["--cwd", str(first_cwd)]))
    assert first.query_engine.get_session_snapshot().permission_mode == "full"
    first_session_id = first.query_engine.session.session_id

    first.query_engine.set_permission_mode("custom")
    assert first.query_engine.get_session_snapshot().permission_mode == "custom"

    config_path.write_text(json.dumps({"permission_mode": "default"}), encoding="utf-8")
    second = build_runtime(parse_args(["--cwd", str(second_cwd)]))
    assert second.query_engine.get_session_snapshot().permission_mode == "default"

    snapshot = asyncio.run(second.query_engine.switch_session(first_session_id))

    assert snapshot.permission_mode == "custom"
    assert second.query_engine.permission_manager.mode == "custom"


def test_legacy_access_space_fields_in_config_and_metadata_are_ignored(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    legacy_mode_key = "work" + "space_mode"
    legacy_roots_key = "work" + "space_roots"
    config_path = get_siyi_config_path()
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "permission_mode": "default",
                legacy_mode_key: "custom",
                legacy_roots_key: [str(tmp_path / "extra")],
            }
        ),
        encoding="utf-8",
    )

    runtime = build_runtime(parse_args(["--cwd", str(project_root)]))
    snapshot = runtime.query_engine.get_session_snapshot()
    metadata_json = runtime.query_engine.session.metadata.to_json()
    assert not hasattr(snapshot, legacy_mode_key)
    assert legacy_mode_key not in metadata_json
    assert legacy_roots_key not in metadata_json

    legacy = SessionMetadata.from_json(
        {
            "session_id": "sess_legacy",
            "path": str(tmp_path / "sess_legacy.jsonl"),
            "project_id": runtime.query_engine.session.metadata.project_id,
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "name": "legacy",
            "name_status": "ready",
            "permission_mode": "default",
            legacy_mode_key: "custom",
            legacy_roots_key: [str(tmp_path / "extra")],
        }
    )
    assert legacy_mode_key not in legacy.to_json()
    assert legacy_roots_key not in legacy.to_json()


def test_session_switch_rejects_missing_project_root_without_changing_current_session(tmp_path: Path) -> None:
    first_project_root = tmp_path / "first"
    second_project_root = tmp_path / "second"
    first_project_root.mkdir()
    second_project_root.mkdir()

    first = build_runtime(parse_args(["--cwd", str(first_project_root)]))
    missing_session_id = first.query_engine.session.session_id
    second = build_runtime(parse_args(["--cwd", str(second_project_root)]))
    current_session_id = second.query_engine.session.session_id
    current_project_root = second.query_engine.settings.runtime.project_root
    first_project_root.rmdir()

    try:
        asyncio.run(second.query_engine.switch_session(missing_session_id))
    except ValueError as exc:
        assert "does not exist" in str(exc)
    else:
        raise AssertionError("switch should fail for a missing project root")

    assert second.query_engine.session.session_id == current_session_id
    assert second.query_engine.settings.runtime.project_root == current_project_root


def test_session_index_can_be_rebuilt_from_session_meta(tmp_path: Path) -> None:
    runtime = build_runtime(parse_args(["--cwd", str(tmp_path)]))
    session_id = runtime.query_engine.session.session_id
    runtime.session_store.index_path.unlink()

    sessions = runtime.session_store.list_sessions()

    assert [session.session_id for session in sessions] == [session_id]


def test_query_engine_does_not_block_when_budget_would_be_exceeded(tmp_path: Path) -> None:
    runtime = build_runtime(parse_args(["--cwd", str(tmp_path)]))
    runtime.query_engine.llm = FakeLLMAdapter(["still runs"])
    oversized_prompt = "x" * ((AUTO_COMPACT_THRESHOLD_TOKENS + 1) * 4)

    events = asyncio.run(
        _collect_events(
            runtime.query_engine,
            oversized_prompt,
        )
    )

    assert isinstance(events[-1], TurnCompletedEvent)
    assert events[-1].status == "completed"
    stored_messages = runtime.session_store.load_messages(runtime.query_engine.session.session_id)
    assert len(stored_messages) == 2
    assert stored_messages[0].to_plain_text() == oversized_prompt
    assert stored_messages[1].to_plain_text() == "still runs"


def test_query_engine_persists_user_message_before_model_failure(tmp_path: Path) -> None:
    runtime = build_runtime(parse_args(["--cwd", str(tmp_path)]))
    runtime.query_engine.llm = RaisingLLMAdapter()

    events = asyncio.run(_collect_events(runtime.query_engine, "hello"))

    assert isinstance(events[-1], TurnCompletedEvent)
    assert events[-1].status == "failed"
    assert events[-1].error is not None
    assert events[-1].error.code == "runtime_error"
    assert events[-1].error.message == "boom"
    stored_messages = runtime.session_store.load_messages(runtime.query_engine.session.session_id)
    assert len(stored_messages) == 1
    assert stored_messages[0].to_plain_text() == "hello"


def test_microcompact_projection_clears_old_tool_results() -> None:
    manager = CompactionManager(microcompact_gap_minutes=0, microcompact_keep_recent=1)
    first_tool = ToolUseBlock(id="toolu_old", name="WebSearch", input={"query": "old"})
    second_tool = ToolUseBlock(id="toolu_new", name="WebSearch", input={"query": "new"})
    messages = [
        assistant_message_from_blocks([first_tool]),
        tool_result_message(tool_use_id="toolu_old", content="old result"),
        assistant_message_from_blocks([second_tool]),
        tool_result_message(tool_use_id="toolu_new", content="new result"),
    ]

    result = manager.microcompact_projection(messages)

    assert result.compacted is True
    assert messages[1].to_plain_text() == "old result"
    assert result.messages[1].to_plain_text() == TIME_BASED_MC_CLEARED_MESSAGE
    assert result.messages[3].to_plain_text() == "new result"


def test_manual_compact_appends_segment_and_custom_instructions(tmp_path: Path) -> None:
    runtime = build_runtime(parse_args(["--cwd", str(tmp_path)]))
    runtime.query_engine.llm = FakeLLMAdapter(["<summary>summary about auth</summary>"])
    runtime.query_engine.mutable_messages.extend(
        [
            user_message("first question"),
            assistant_message_from_blocks([ToolUseBlock(id="toolu_search", name="WebSearch", input={"query": "a"})]),
            tool_result_message(tool_use_id="toolu_search", content="search content"),
            user_message("second question"),
        ]
    )
    runtime.query_engine.session.messages = list(runtime.query_engine.mutable_messages)

    result = asyncio.run(runtime.query_engine.compact("focus on auth"))

    assert result.compacted is True
    assert result.messages_to_append[0].metadata["subtype"] == "compact_boundary"
    assert result.messages_to_append[1].metadata["subtype"] == "compact_summary"
    assert result.messages_to_append[1].metadata["custom_instructions"] == "focus on auth"
    stored_messages = runtime.session_store.load_messages(runtime.query_engine.session.session_id)
    assert any(message.metadata.get("subtype") == "compact_summary" for message in stored_messages)


def test_compact_retries_prompt_too_long_by_truncating_head(tmp_path: Path) -> None:
    runtime = build_runtime(parse_args(["--cwd", str(tmp_path)]))
    llm = PromptTooLongThenSummaryLLM()
    runtime.query_engine.llm = llm
    runtime.query_engine.mutable_messages.extend(
        [
            user_message("first question"),
            assistant_message_from_blocks([ToolUseBlock(id="toolu_1", name="WebSearch", input={"query": "a"})]),
            tool_result_message(tool_use_id="toolu_1", content="file content"),
            user_message("second question"),
            assistant_message_from_blocks([ToolUseBlock(id="toolu_2", name="WebSearch", input={"query": "b"})]),
            tool_result_message(tool_use_id="toolu_2", content="file content"),
        ]
    )
    runtime.query_engine.session.messages = list(runtime.query_engine.mutable_messages)

    result = asyncio.run(runtime.query_engine.compact())

    assert result.compacted is True
    assert llm.calls == 2


def test_auto_compact_runs_once_over_threshold(tmp_path: Path) -> None:
    runtime = build_runtime(parse_args(["--cwd", str(tmp_path)]))
    runtime.query_engine.llm = FakeLLMAdapter(["<summary>auto summary</summary>", "final answer"])
    runtime.query_engine.mutable_messages.extend(
        [
            user_message("old question"),
            assistant_message_from_blocks([ToolUseBlock(id="toolu_old", name="WebSearch", input={"query": "a"})]),
            tool_result_message(tool_use_id="toolu_old", content="old result" * 130_000),
        ]
    )
    runtime.query_engine.session.messages = list(runtime.query_engine.mutable_messages)

    events = asyncio.run(_collect_events(runtime.query_engine, "new question"))

    assert isinstance(events[-1], TurnCompletedEvent)
    assert events[-1].status == "completed"
    assert any(
        message.metadata.get("subtype") == "compact_boundary"
        for message in runtime.query_engine.mutable_messages
    )
