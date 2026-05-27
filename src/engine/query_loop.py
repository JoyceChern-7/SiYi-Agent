from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from contextlib import suppress
from typing import Protocol

from engine.events import (
    AgentError,
    AssistantDeltaEvent,
    AssistantMessageEvent,
    QueryEvent,
    ToolOutputDeltaEvent,
    ToolResultEvent,
    ToolCallEvent,
)
from engine.message_schema import (
    ContentBlock,
    ThinkingBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    assistant_message_from_blocks,
    normalize_messages_for_api,
    tool_result_message,
)
from engine.turn_state import QueryTurnState
from llm.base import LLMAdapter, LLMAssistantDone, LLMTextDelta, LLMThinkingDelta, LLMToolUse
from runtime.turn_runtime import TurnInterruptedError
from runtime.usage_tracker import Usage
from tools.base import ToolContext, ToolResult
from tools.registry import ToolRegistry


class QueryLoop(Protocol):
    async def run(
        self,
        turn: QueryTurnState,
        *,
        llm: LLMAdapter,
        system_prompt: str,
        tools: list[dict],
        temperature: float,
        tool_registry: ToolRegistry | None = None,
        tool_context: ToolContext | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncIterator[QueryEvent]:
        ...


class DefaultQueryLoop:
    async def run(
        self,
        turn: QueryTurnState,
        *,
        llm: LLMAdapter,
        system_prompt: str,
        tools: list[dict],
        temperature: float,
        tool_registry: ToolRegistry | None = None,
        tool_context: ToolContext | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncIterator[QueryEvent]:
        turn.stage = "running"

        while True:
            _raise_if_interrupted(turn, cancel_event)
            text_parts: list[str] = []
            thinking_parts: list[str] = []
            text_blocks: list[TextBlock] = []
            thinking_blocks: list[ThinkingBlock] = []
            tool_blocks: list[ToolUseBlock] = []
            response_usage = Usage()

            llm_stream = llm.stream_chat(
                messages=turn.messages_for_query,
                system_prompt=system_prompt,
                tools=tools,
                temperature=temperature,
            )
            while True:
                try:
                    llm_event = await _next_llm_event(llm_stream, turn, cancel_event)
                except StopAsyncIteration:
                    break
                _raise_if_interrupted(turn, cancel_event)
                if isinstance(llm_event, LLMTextDelta):
                    turn.append_stream_delta(llm_event.delta)
                    text_parts.append(llm_event.delta)
                    yield AssistantDeltaEvent(
                        session_id=turn.session_id,
                        turn_id=turn.turn_id,
                        delta=llm_event.delta,
                    )
                    continue

                if isinstance(llm_event, LLMThinkingDelta):
                    thinking_parts.append(llm_event.delta)
                    continue

                if isinstance(llm_event, LLMToolUse):
                    turn.add_tool_call(llm_event.block)
                    tool_blocks.append(llm_event.block)
                    yield ToolCallEvent(
                        session_id=turn.session_id,
                        turn_id=turn.turn_id,
                        block=llm_event.block,
                    )
                    continue

                if isinstance(llm_event, LLMAssistantDone):
                    turn.usage_delta.add(llm_event.usage)
                    response_usage.add(llm_event.usage)

            assistant_text = "".join(text_parts).strip()
            thinking_text = "".join(thinking_parts).strip()
            if thinking_text:
                thinking_blocks.append(ThinkingBlock(text=thinking_text))
            if assistant_text:
                text_blocks.append(TextBlock(text=assistant_text))
            if not text_blocks and not thinking_blocks and not tool_blocks:
                text_blocks.append(TextBlock(text="(empty assistant response)"))

            blocks: list[ContentBlock] = [*thinking_blocks, *text_blocks, *tool_blocks]
            metadata = {}
            if (
                response_usage.input_tokens
                or response_usage.output_tokens
                or response_usage.cached_tokens
            ):
                metadata["usage"] = response_usage.model_dump()
            metadata["turn_id"] = turn.turn_id
            metadata["turn_index"] = turn.turn_index
            assistant_message = assistant_message_from_blocks(blocks, metadata=metadata)
            turn.record_generated_message(assistant_message)

            if tool_blocks:
                turn.stop_reason = "tool_use"
            else:
                turn.stop_reason = "end_turn"

            _raise_if_interrupted(turn, cancel_event)
            yield AssistantMessageEvent(
                session_id=turn.session_id,
                turn_id=turn.turn_id,
                message=assistant_message,
            )

            if not tool_blocks:
                return

            for batch in _tool_batches(tool_blocks, tool_registry, tool_context):
                _raise_if_interrupted(turn, cancel_event)
                blocks_to_run = (
                    [batch.blocks]
                    if batch.concurrent
                    else [[block] for block in batch.blocks]
                )
                for blocks in blocks_to_run:
                    async for item in _execute_tool_batch(
                        blocks,
                        tool_registry,
                        tool_context,
                        cancel_event=cancel_event,
                        turn_id=turn.turn_id,
                    ):
                        if isinstance(item, _ToolCompleted):
                            error = (
                                item.error.model_copy(
                                    update={
                                        "session_id": turn.session_id,
                                        "turn_id": turn.turn_id,
                                    }
                                )
                                if item.error is not None
                                else None
                            )
                            result_message = tool_result_message(
                                tool_use_id=item.block.id,
                                content=item.result_block.content,
                                is_error=item.result_block.is_error,
                                metadata={
                                    "tool_name": item.block.name,
                                    "turn_id": turn.turn_id,
                                    "turn_index": turn.turn_index,
                                    "error": error.model_dump(mode="json") if error else None,
                                },
                            )
                            turn.record_generated_message(result_message)
                            yield ToolResultEvent(
                                session_id=turn.session_id,
                                turn_id=turn.turn_id,
                                block=item.result_block,
                                error=error,
                            )
                            continue
                        yield item

                    if cancel_event is not None and cancel_event.is_set():
                        _raise_if_interrupted(turn, cancel_event)

            turn.messages_for_query = normalize_messages_for_api(turn.messages)


class _ToolBatch:
    def __init__(self, blocks: list[ToolUseBlock], *, concurrent: bool) -> None:
        self.blocks = blocks
        self.concurrent = concurrent


@dataclass(slots=True)
class _ToolCompleted:
    block: ToolUseBlock
    result_block: ToolResultBlock
    error: AgentError | None = None


def _tool_batches(
    blocks: list[ToolUseBlock],
    tool_registry: ToolRegistry | None,
    tool_context: ToolContext | None,
) -> list[_ToolBatch]:
    batches: list[_ToolBatch] = []
    concurrent_blocks: list[ToolUseBlock] = []

    def flush_concurrent() -> None:
        nonlocal concurrent_blocks
        if concurrent_blocks:
            batches.append(_ToolBatch(concurrent_blocks, concurrent=True))
            concurrent_blocks = []

    for block in blocks:
        if _is_concurrency_safe(block, tool_registry, tool_context):
            concurrent_blocks.append(block)
            continue
        flush_concurrent()
        batches.append(_ToolBatch([block], concurrent=False))

    flush_concurrent()
    return batches


def _is_concurrency_safe(
    block: ToolUseBlock,
    tool_registry: ToolRegistry | None,
    tool_context: ToolContext | None,
) -> bool:
    if tool_registry is None or tool_context is None:
        return False
    tool = tool_registry.find_tool(block.name)
    if tool is None:
        return False
    try:
        validation = tool.validate_input(block.input, tool_context)
        return validation.ok and tool.is_concurrency_safe(block.input)
    except Exception:
        return False


async def _execute_tool(
    block: ToolUseBlock,
    tool_registry: ToolRegistry | None,
    tool_context: ToolContext | None,
    progress_queue: asyncio.Queue[ToolOutputDeltaEvent] | None = None,
    cancel_event: asyncio.Event | None = None,
) -> tuple[ToolResultBlock, AgentError | None]:
    if cancel_event is not None and cancel_event.is_set():
        raise TurnInterruptedError(tool_context.turn_id if tool_context else block.id)
    if tool_registry is None or tool_context is None:
        return _tool_error(
            block,
            code="tool_not_configured",
            category="tool",
            message="tool execution is not configured",
            retryable=False,
        )

    tool = tool_registry.find_tool(block.name)
    if tool is None:
        return _tool_error(
            block,
            code="tool_not_found",
            category="tool",
            message=f"tool not found: {block.name}",
            retryable=False,
        )

    execution_context = tool_context.model_copy(
        update={
            "tool_use_id": block.id,
            "tool_name": block.name,
            "progress_queue": progress_queue,
        }
    )

    validation = tool.validate_input(block.input, execution_context)
    if not validation.ok:
        return _tool_error(
            block,
            code="tool_validation_failed",
            category="tool",
            message=validation.reason or "tool input validation failed",
            retryable=False,
        )

    permission = await tool_registry.permission_manager.authorize(
        tool,
        block.input,
        execution_context,
    )
    if cancel_event is not None and cancel_event.is_set():
        raise TurnInterruptedError(tool_context.turn_id or block.id)
    if permission.decision != "allow":
        reason = permission.reason or f"permission decision: {permission.decision}"
        code = "approval_canceled" if permission.source == "approval_canceled" else "tool_denied"
        return _tool_error(
            block,
            code=code,
            category="permission",
            message=reason,
            retryable=code == "approval_canceled",
            action="retry_tool_call" if code == "approval_canceled" else "change_permission_or_retry",
        )

    try:
        result = await tool.run(block.input, execution_context)
    except Exception as exc:  # noqa: BLE001 - tool failures should flow back to the model
        result = ToolResult(
            success=False,
            content=f"{type(exc).__name__}: {exc}",
            error=str(exc),
        )
    result_block = tool.to_tool_result_block(block.id, result)
    return result_block, _agent_error_from_tool_result(block, result, result_block)


async def _execute_tool_batch(
    blocks: list[ToolUseBlock],
    tool_registry: ToolRegistry | None,
    tool_context: ToolContext | None,
    *,
    cancel_event: asyncio.Event | None = None,
    turn_id: str | None = None,
) -> AsyncIterator[QueryEvent | _ToolCompleted]:
    progress_queue: asyncio.Queue[QueryEvent] = asyncio.Queue()
    tasks = {
        asyncio.create_task(
            _execute_tool(
                block,
                tool_registry,
                tool_context,
                progress_queue,
                cancel_event,
            )
        ): block
        for block in blocks
    }
    cancel_task = asyncio.create_task(cancel_event.wait()) if cancel_event is not None else None

    try:
        while tasks:
            wait_for: set[asyncio.Task[object]] = set(tasks)
            if cancel_task is not None:
                wait_for.add(cancel_task)
            done, _pending = await asyncio.wait(
                wait_for,
                timeout=0.05,
                return_when=asyncio.FIRST_COMPLETED,
            )

            while not progress_queue.empty():
                yield progress_queue.get_nowait()

            interrupted = cancel_task is not None and cancel_task in done
            for task in [candidate for candidate in done if candidate in tasks]:
                block = tasks.pop(task)
                try:
                    result_block, error = task.result()
                except (asyncio.CancelledError, TurnInterruptedError):
                    interrupted = True
                    continue
                yield _ToolCompleted(block=block, result_block=result_block, error=error)

            if interrupted:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                while not progress_queue.empty():
                    yield progress_queue.get_nowait()
                raise TurnInterruptedError(turn_id or "")

        while not progress_queue.empty():
            yield progress_queue.get_nowait()
    finally:
        if cancel_task is not None and not cancel_task.done():
            cancel_task.cancel()
        for task in tasks:
            if not task.done():
                task.cancel()


def _tool_error(
    block: ToolUseBlock,
    *,
    code: str,
    category: str,
    message: str,
    retryable: bool,
    action: str | None = None,
) -> tuple[ToolResultBlock, AgentError]:
    error = AgentError(
        code=code,
        category=category,  # type: ignore[arg-type]
        message=message,
        retryable=retryable,
        action=action,
        tool_name=block.name,
        tool_use_id=block.id,
    )
    return ToolResultBlock(tool_use_id=block.id, content=message, is_error=True), error


def _agent_error_from_tool_result(
    block: ToolUseBlock,
    result: ToolResult,
    result_block: ToolResultBlock,
) -> AgentError | None:
    if not result_block.is_error:
        return None
    message = result.error or result_block.content or "tool failed"
    data = result.data or {}
    code = "tool_failed"
    category = "tool"
    retryable = False
    action = None
    process_id = None
    if bool(data.get("timed_out")):
        code = "shell_timeout"
        category = "process"
        retryable = True
        action = "retry_with_longer_timeout"
    elif "process belongs to another session" in message:
        code = "process_ownership_mismatch"
        category = "process"
    elif block.name in {"exec_command", "write_stdin", "stop_command", "PowerShell", "Bash"}:
        category = "process"
    if isinstance(data.get("session_id"), str):
        process_id = str(data["session_id"])
    return AgentError(
        code=code,
        category=category,  # type: ignore[arg-type]
        message=message,
        retryable=retryable,
        action=action,
        tool_name=block.name,
        tool_use_id=block.id,
        process_id=process_id,
        details=data,
    )


def _raise_if_interrupted(
    turn: QueryTurnState,
    cancel_event: asyncio.Event | None,
) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise TurnInterruptedError(turn.turn_id)


async def _next_llm_event(
    stream: AsyncIterator[object],
    turn: QueryTurnState,
    cancel_event: asyncio.Event | None,
) -> object:
    if cancel_event is None:
        return await stream.__anext__()
    next_task = asyncio.create_task(stream.__anext__())
    cancel_task = asyncio.create_task(cancel_event.wait())
    try:
        done, pending = await asyncio.wait(
            {next_task, cancel_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if cancel_task in done:
            next_task.cancel()
            with suppress(BaseException):
                await next_task
            raise TurnInterruptedError(turn.turn_id)
        cancel_task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        return next_task.result()
    finally:
        for task in (next_task, cancel_task):
            if not task.done():
                task.cancel()
