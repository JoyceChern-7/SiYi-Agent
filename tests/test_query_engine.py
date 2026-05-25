from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

from app.cli import parse_args
from app.main import build_runtime
from config.paths import get_siyi_config_path
from engine.events import ErrorEvent, FinalAnswerEvent
from engine.message_schema import (
    ToolUseBlock,
    assistant_message_from_blocks,
    tool_result_message,
    user_message,
)
from engine.session_title import normalize_session_title
from llm.base import LLMAdapter, LLMAssistantDone, LLMTextDelta, LLMToolUse
from runtime.session_runtime import SessionBusyError, SessionRuntime
from runtime.token_budget import AUTO_COMPACT_THRESHOLD_TOKENS
from runtime.compaction import CompactionManager, TIME_BASED_MC_CLEARED_MESSAGE
from runtime.usage_tracker import Usage


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

    assert isinstance(first_events[-1], FinalAnswerEvent)
    assert isinstance(second_events[-1], FinalAnswerEvent)
    assert runtime.query_engine.current_turn is not None
    assert runtime.query_engine.current_turn.turn_index == 2
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

    assert isinstance(events[-1], FinalAnswerEvent)
    stored_messages = runtime.session_store.load_messages(runtime.query_engine.session.session_id)
    assert len(stored_messages) == 2
    assert stored_messages[0].to_plain_text() == oversized_prompt
    assert stored_messages[1].to_plain_text() == "still runs"


def test_query_engine_persists_user_message_before_model_failure(tmp_path: Path) -> None:
    runtime = build_runtime(parse_args(["--cwd", str(tmp_path)]))
    runtime.query_engine.llm = RaisingLLMAdapter()

    events = asyncio.run(_collect_events(runtime.query_engine, "hello"))

    assert isinstance(events[-1], ErrorEvent)
    assert events[-1].code == "RuntimeError"
    stored_messages = runtime.session_store.load_messages(runtime.query_engine.session.session_id)
    assert len(stored_messages) == 1
    assert stored_messages[0].to_plain_text() == "hello"


def test_microcompact_projection_clears_old_tool_results() -> None:
    manager = CompactionManager(microcompact_gap_minutes=0, microcompact_keep_recent=1)
    first_tool = ToolUseBlock(id="toolu_old", name="Read", input={"file_path": "old.txt"})
    second_tool = ToolUseBlock(id="toolu_new", name="Read", input={"file_path": "new.txt"})
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
            assistant_message_from_blocks([ToolUseBlock(id="toolu_read", name="Read", input={"file_path": "a.txt"})]),
            tool_result_message(tool_use_id="toolu_read", content="file content"),
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
            assistant_message_from_blocks([ToolUseBlock(id="toolu_1", name="Read", input={"file_path": "a.txt"})]),
            tool_result_message(tool_use_id="toolu_1", content="file content"),
            user_message("second question"),
            assistant_message_from_blocks([ToolUseBlock(id="toolu_2", name="Read", input={"file_path": "b.txt"})]),
            tool_result_message(tool_use_id="toolu_2", content="file content"),
        ]
    )
    runtime.query_engine.session.messages = list(runtime.query_engine.mutable_messages)

    result = asyncio.run(runtime.query_engine.compact())

    assert result.compacted is True
    assert llm.calls == 2


def test_read_restore_adds_recent_read_file(tmp_path: Path) -> None:
    read_file = tmp_path / "restore.txt"
    read_file.write_text("restored content", encoding="utf-8")
    runtime = build_runtime(parse_args(["--cwd", str(tmp_path)]))
    runtime.query_engine.llm = FakeLLMAdapter(["<summary>summary</summary>"])
    runtime.query_engine.mutable_messages.extend(
        [
            user_message("read it"),
            assistant_message_from_blocks(
                [ToolUseBlock(id="toolu_read", name="Read", input={"file_path": "restore.txt"})]
            ),
            tool_result_message(tool_use_id="toolu_read", content="old content"),
            user_message("x" * 60_000),
        ]
    )
    runtime.query_engine.session.messages = list(runtime.query_engine.mutable_messages)

    result = asyncio.run(runtime.query_engine.compact())

    restore_messages = [
        message
        for message in result.messages_to_append
        if message.metadata.get("subtype") == "post_compact_read_restore"
    ]
    assert restore_messages
    assert "restored content" in restore_messages[0].to_plain_text()


def test_read_restore_prefers_most_recent_duplicate_path(tmp_path: Path) -> None:
    recent_file = tmp_path / "recent.txt"
    older_file = tmp_path / "older.txt"
    recent_file.write_text("recent content", encoding="utf-8")
    older_file.write_text("older content", encoding="utf-8")
    runtime = build_runtime(parse_args(["--cwd", str(tmp_path)]))
    runtime.query_engine.llm = FakeLLMAdapter(["<summary>summary</summary>"])
    runtime.query_engine.mutable_messages.extend(
        [
            assistant_message_from_blocks(
                [ToolUseBlock(id="toolu_recent_1", name="Read", input={"file_path": "recent.txt"})]
            ),
            tool_result_message(tool_use_id="toolu_recent_1", content="old recent content"),
            assistant_message_from_blocks(
                [ToolUseBlock(id="toolu_older", name="Read", input={"file_path": "older.txt"})]
            ),
            tool_result_message(tool_use_id="toolu_older", content="older tool content"),
            assistant_message_from_blocks(
                [ToolUseBlock(id="toolu_recent_2", name="Read", input={"file_path": "recent.txt"})]
            ),
            tool_result_message(tool_use_id="toolu_recent_2", content="new recent content"),
            user_message("x" * 60_000),
        ]
    )
    runtime.query_engine.session.messages = list(runtime.query_engine.mutable_messages)

    result = asyncio.run(runtime.query_engine.compact())

    restore_messages = [
        message
        for message in result.messages_to_append
        if message.metadata.get("subtype") == "post_compact_read_restore"
    ]
    assert restore_messages
    assert "recent content" in restore_messages[0].to_plain_text()


def test_auto_compact_runs_once_over_threshold(tmp_path: Path) -> None:
    runtime = build_runtime(parse_args(["--cwd", str(tmp_path)]))
    runtime.query_engine.llm = FakeLLMAdapter(["<summary>auto summary</summary>", "final answer"])
    runtime.query_engine.mutable_messages.extend(
        [
            user_message("old question"),
            assistant_message_from_blocks([ToolUseBlock(id="toolu_old", name="Read", input={"file_path": "a.txt"})]),
            tool_result_message(tool_use_id="toolu_old", content="old result" * 130_000),
        ]
    )
    runtime.query_engine.session.messages = list(runtime.query_engine.mutable_messages)

    events = asyncio.run(_collect_events(runtime.query_engine, "new question"))

    assert isinstance(events[-1], FinalAnswerEvent)
    assert any(
        message.metadata.get("subtype") == "compact_boundary"
        for message in runtime.query_engine.mutable_messages
    )
