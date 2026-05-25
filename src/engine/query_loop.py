from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

from engine.events import (
    AssistantDeltaEvent,
    AssistantDoneEvent,
    QueryEvent,
    ToolOutputDeltaEvent,
    ToolResultEvent,
    ToolUseEvent,
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
    ) -> AsyncIterator[QueryEvent]:
        turn.stage = "running"

        while True:
            text_parts: list[str] = []
            thinking_parts: list[str] = []
            text_blocks: list[TextBlock] = []
            thinking_blocks: list[ThinkingBlock] = []
            tool_blocks: list[ToolUseBlock] = []
            response_usage = Usage()

            async for llm_event in llm.stream_chat(
                messages=turn.messages_for_query,
                system_prompt=system_prompt,
                tools=tools,
                temperature=temperature,
            ):
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
                    yield ToolUseEvent(
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
            assistant_message = assistant_message_from_blocks(blocks, metadata=metadata)
            turn.record_generated_message(assistant_message)

            if tool_blocks:
                turn.stop_reason = "tool_use"
            else:
                turn.stop_reason = "end_turn"

            yield AssistantDoneEvent(
                session_id=turn.session_id,
                turn_id=turn.turn_id,
                message=assistant_message,
            )

            if not tool_blocks:
                return

            for batch in _tool_batches(tool_blocks, tool_registry, tool_context):
                result_blocks: list[ToolResultBlock] = []
                if batch.concurrent:
                    async for item in _execute_tool_batch(
                        batch.blocks,
                        tool_registry,
                        tool_context,
                    ):
                        if isinstance(item, _ToolBatchExecutionResult):
                            result_blocks = item.result_blocks
                        else:
                            yield item
                else:
                    for block in batch.blocks:
                        async for item in _execute_tool_batch(
                            [block],
                            tool_registry,
                            tool_context,
                        ):
                            if isinstance(item, _ToolBatchExecutionResult):
                                result_blocks.extend(item.result_blocks)
                            else:
                                yield item

                for block, result_block in zip(batch.blocks, result_blocks, strict=True):
                    result_message = tool_result_message(
                        tool_use_id=block.id,
                        content=result_block.content,
                        is_error=result_block.is_error,
                        metadata={"tool_name": block.name},
                    )
                    turn.record_generated_message(result_message)
                    yield ToolResultEvent(
                        session_id=turn.session_id,
                        turn_id=turn.turn_id,
                        block=result_block,
                    )

            turn.messages_for_query = normalize_messages_for_api(turn.messages)


class _ToolBatch:
    def __init__(self, blocks: list[ToolUseBlock], *, concurrent: bool) -> None:
        self.blocks = blocks
        self.concurrent = concurrent


@dataclass(slots=True)
class _ToolBatchExecutionResult:
    result_blocks: list[ToolResultBlock]


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
) -> ToolResultBlock:
    if tool_registry is None or tool_context is None:
        return ToolResultBlock(
            tool_use_id=block.id,
            content="tool execution is not configured",
            is_error=True,
        )

    tool = tool_registry.find_tool(block.name)
    if tool is None:
        return ToolResultBlock(
            tool_use_id=block.id,
            content=f"tool not found: {block.name}",
            is_error=True,
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
        return ToolResultBlock(
            tool_use_id=block.id,
            content=validation.reason or "tool input validation failed",
            is_error=True,
        )

    permission = await tool_registry.permission_manager.authorize(
        tool,
        block.input,
        execution_context,
    )
    if permission.decision != "allow":
        reason = permission.reason or f"permission decision: {permission.decision}"
        return ToolResultBlock(
            tool_use_id=block.id,
            content=reason,
            is_error=True,
        )

    try:
        result = await tool.run(block.input, execution_context)
    except Exception as exc:  # noqa: BLE001 - tool failures should flow back to the model
        result = ToolResult(
            success=False,
            content=f"{type(exc).__name__}: {exc}",
            error=str(exc),
        )
    return tool.to_tool_result_block(block.id, result)


async def _execute_tool_batch(
    blocks: list[ToolUseBlock],
    tool_registry: ToolRegistry | None,
    tool_context: ToolContext | None,
) -> AsyncIterator[QueryEvent | _ToolBatchExecutionResult]:
    progress_queue: asyncio.Queue[QueryEvent] = asyncio.Queue()
    task = asyncio.gather(
        *[
            _execute_tool(
                block,
                tool_registry,
                tool_context,
                progress_queue,
            )
            for block in blocks
        ]
    )

    while not task.done():
        try:
            event = await asyncio.wait_for(progress_queue.get(), timeout=0.05)
        except TimeoutError:
            continue
        yield event

    while not progress_queue.empty():
        yield progress_queue.get_nowait()

    yield _ToolBatchExecutionResult(result_blocks=await task)
