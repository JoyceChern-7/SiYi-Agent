from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from engine.message_schema import (
    ContentBlock,
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    compact_boundary_message,
    compact_summary_message,
    get_messages_after_compact_boundary,
    make_message,
    user_message,
)
from llm.base import LLMAdapter, LLMAssistantDone, LLMTextDelta, LLMThinkingDelta, LLMToolUse
from runtime.ids import new_id
from runtime.token_budget import TokenBudget

TIME_BASED_MC_CLEARED_MESSAGE = "[Old tool result content cleared]"
COMPACTABLE_TOOLS = {
    "Read",
    "PowerShell",
    "Bash",
    "Grep",
    "Glob",
    "WebSearch",
    "WebFetch",
    "Edit",
    "Write",
}

DEFAULT_MICROCOMPACT_GAP_MINUTES = 60
DEFAULT_MICROCOMPACT_KEEP_RECENT = 5
POST_COMPACT_MAX_FILES_TO_RESTORE = 5
POST_COMPACT_TOKEN_BUDGET = 50_000
POST_COMPACT_MAX_TOKENS_PER_FILE = 5_000
COMPACT_MAX_PTL_RETRIES = 3

SESSION_MEMORY_MIN_TOKENS = 10_000
SESSION_MEMORY_MIN_TEXT_MESSAGES = 5
SESSION_MEMORY_MAX_TOKENS = 40_000

COMPACT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant tasked with summarizing conversations. "
    "Respond with text only. Do not call tools."
)

COMPACT_PROMPT = """CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.

Your task is to create a detailed summary of the conversation so far. The
summary will be used to continue development work after older context has been
compacted. Preserve the user's explicit requests, code decisions, file paths,
important errors, fixes, pending tasks, and the most recent state of the work.

Write the response in this structure:

<analysis>
Briefly reason about the important context to preserve.
</analysis>

<summary>
1. Primary Request and Intent:
2. Key Technical Concepts:
3. Files and Code Sections:
4. Errors and Fixes:
5. Problem Solving:
6. All User Messages:
7. Pending Tasks:
8. Current Work:
9. Optional Next Step:
</summary>
"""

PTL_RETRY_MARKER = "[earlier conversation truncated for compaction retry]"
PROMPT_TOO_LONG_MARKERS = (
    "prompt is too long",
    "context length",
    "context_length_exceeded",
    "maximum context",
    "too many tokens",
    "reduce the length",
    "request too large",
)


@dataclass(slots=True)
class CompactionResult:
    messages: list[Message]
    messages_to_append: list[Message] = field(default_factory=list)
    compacted: bool = False
    reason: str | None = None
    pre_tokens: int = 0
    post_tokens: int = 0
    tokens_saved: int = 0


class CompactionManager:
    def __init__(
        self,
        enabled: bool = True,
        *,
        microcompact_enabled: bool = True,
        microcompact_gap_minutes: int = DEFAULT_MICROCOMPACT_GAP_MINUTES,
        microcompact_keep_recent: int = DEFAULT_MICROCOMPACT_KEEP_RECENT,
    ) -> None:
        self.enabled = enabled
        self.microcompact_enabled = microcompact_enabled
        self.microcompact_gap_minutes = microcompact_gap_minutes
        self.microcompact_keep_recent = microcompact_keep_recent

    def maybe_compact(self, messages: list[Message]) -> CompactionResult:
        return self.microcompact_projection(messages)

    def microcompact_projection(
        self,
        messages: list[Message],
        *,
        now: datetime | None = None,
    ) -> CompactionResult:
        if not self.enabled or not self.microcompact_enabled:
            return CompactionResult(messages=messages, compacted=False)

        trigger = self._evaluate_time_based_microcompact(messages, now=now)
        if trigger is None:
            return CompactionResult(messages=messages, compacted=False)

        compactable_ids = _collect_compactable_tool_ids(messages)
        keep_recent = max(1, self.microcompact_keep_recent)
        keep_ids = set(compactable_ids[-keep_recent:])
        clear_ids = set(tool_id for tool_id in compactable_ids if tool_id not in keep_ids)
        if not clear_ids:
            return CompactionResult(messages=messages, compacted=False)

        tokens_saved = 0
        projected: list[Message] = []
        for message in messages:
            if message.role != "user" or not message.has_tool_result():
                projected.append(message)
                continue

            touched = False
            blocks: list[ContentBlock] = []
            for block in message.content:
                if (
                    isinstance(block, ToolResultBlock)
                    and block.tool_use_id in clear_ids
                    and block.content != TIME_BASED_MC_CLEARED_MESSAGE
                ):
                    tokens_saved += _estimate_text_tokens(block.content)
                    blocks.append(
                        block.model_copy(
                            update={"content": TIME_BASED_MC_CLEARED_MESSAGE}
                        )
                    )
                    touched = True
                    continue
                blocks.append(block)
            projected.append(
                message.model_copy(update={"content": blocks}, deep=True)
                if touched
                else message
            )

        if tokens_saved <= 0:
            return CompactionResult(messages=messages, compacted=False)

        return CompactionResult(
            messages=projected,
            compacted=True,
            reason=f"time_based_microcompact:{round(trigger)}m",
            tokens_saved=tokens_saved,
        )

    async def compact_conversation(
        self,
        messages: list[Message],
        *,
        llm: LLMAdapter,
        system_prompt: str,
        temperature: float,
        token_budget: TokenBudget,
        trigger: Literal["manual", "auto"] = "manual",
        custom_instructions: str | None = None,
        transcript_path: str | None = None,
        project_root: Path | None = None,
    ) -> CompactionResult:
        if not self.enabled:
            return CompactionResult(
                messages=get_messages_after_compact_boundary(messages),
                compacted=False,
                reason="compaction_disabled",
            )

        source_messages = [
            message
            for message in get_messages_after_compact_boundary(messages)
            if message.metadata.get("subtype") != "compact_boundary"
        ]
        if not source_messages:
            return CompactionResult(messages=[], compacted=False, reason="no_messages")

        pre_tokens = token_budget.estimate_request_tokens(
            source_messages,
            system_prompt,
            [],
        )

        keep_start = calculate_messages_to_keep_index(source_messages)
        if keep_start <= 0:
            messages_to_summarize = source_messages
            messages_to_keep: list[Message] = []
        else:
            messages_to_summarize = source_messages[:keep_start]
            messages_to_keep = source_messages[keep_start:]

        if not messages_to_summarize:
            return CompactionResult(
                messages=source_messages,
                compacted=False,
                reason="nothing_to_summarize",
                pre_tokens=pre_tokens,
                post_tokens=pre_tokens,
            )

        summary = await self._summarize_with_retries(
            messages_to_summarize,
            llm=llm,
            temperature=temperature,
            custom_instructions=custom_instructions,
        )
        summary_message = compact_summary_message(
            summary,
            custom_instructions=custom_instructions,
            transcript_path=transcript_path,
        )
        preserved_messages = [
            _copy_message_for_compact_segment(message) for message in messages_to_keep
        ]
        restore_messages = self._build_read_restore_messages(
            source_messages,
            messages_to_keep,
            project_root=project_root,
        )
        messages_after_boundary = [
            summary_message,
            *preserved_messages,
            *restore_messages,
        ]
        post_tokens = token_budget.estimate_request_tokens(
            messages_after_boundary,
            system_prompt,
            [],
        )
        tokens_saved = max(0, pre_tokens - post_tokens)
        boundary = compact_boundary_message(
            trigger=trigger,
            pre_tokens=pre_tokens,
            post_tokens=post_tokens,
            tokens_saved=tokens_saved,
            preserved_message_ids=[message.id for message in messages_to_keep],
        )
        return CompactionResult(
            messages=messages_after_boundary,
            messages_to_append=[boundary, *messages_after_boundary],
            compacted=True,
            reason=f"{trigger}_compact",
            pre_tokens=pre_tokens,
            post_tokens=post_tokens,
            tokens_saved=tokens_saved,
        )

    def mark_boundary(self, messages: list[Message], reason: str) -> list[Message]:
        return [*messages, compact_boundary_message(reason)]

    def recover_from_overflow(self, messages: list[Message]) -> CompactionResult:
        truncated = truncate_head_for_ptl_retry(get_messages_after_compact_boundary(messages))
        if truncated is None:
            return CompactionResult(messages=messages, compacted=False, reason="overflow_noop")
        return CompactionResult(messages=truncated, compacted=True, reason="overflow_truncated")

    def _evaluate_time_based_microcompact(
        self,
        messages: list[Message],
        *,
        now: datetime | None,
    ) -> float | None:
        last_assistant = next(
            (message for message in reversed(messages) if message.role == "assistant"),
            None,
        )
        if last_assistant is None:
            return None
        current = now or datetime.now(timezone.utc)
        timestamp = last_assistant.timestamp
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        gap_minutes = (current - timestamp).total_seconds() / 60
        if gap_minutes < self.microcompact_gap_minutes:
            return None
        return gap_minutes

    async def _summarize_with_retries(
        self,
        messages: list[Message],
        *,
        llm: LLMAdapter,
        temperature: float,
        custom_instructions: str | None,
    ) -> str:
        current = list(messages)
        for attempt in range(COMPACT_MAX_PTL_RETRIES + 1):
            try:
                summary = await _stream_compact_summary(
                    llm,
                    current,
                    temperature=temperature,
                    custom_instructions=custom_instructions,
                )
                if not _is_prompt_too_long(summary):
                    return summary
            except Exception as exc:  # noqa: BLE001 - provider errors are normalized here
                if not _is_prompt_too_long(str(exc)):
                    raise

            if attempt >= COMPACT_MAX_PTL_RETRIES:
                break
            truncated = truncate_head_for_ptl_retry(current)
            if truncated is None:
                break
            current = truncated
        raise RuntimeError("prompt too long while compacting conversation")

    def _build_read_restore_messages(
        self,
        messages: list[Message],
        preserved_messages: list[Message],
        *,
        project_root: Path | None,
    ) -> list[Message]:
        preserved_paths = _collect_read_paths(preserved_messages, project_root=project_root)
        candidates = [
            path
            for path in _collect_recent_read_paths(messages, project_root=project_root)
            if path not in preserved_paths
        ]
        restored: list[Message] = []
        used_tokens = 0
        for path in candidates[:POST_COMPACT_MAX_FILES_TO_RESTORE]:
            try:
                if not path.exists() or not path.is_file():
                    continue
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            content = _truncate_to_tokens(
                content,
                POST_COMPACT_MAX_TOKENS_PER_FILE,
                marker="\n\n[... file content truncated for post-compact restore]",
            )
            text = (
                "[post-compact restored Read file]\n"
                f"Path: {path}\n"
                "Content:\n"
                f"{content}"
            )
            message_tokens = _estimate_text_tokens(text)
            if used_tokens + message_tokens > POST_COMPACT_TOKEN_BUDGET:
                break
            used_tokens += message_tokens
            restored.append(
                make_message(
                    role="user",
                    message_type="user",
                    blocks=[TextBlock(text=text)],
                    is_meta=True,
                    metadata={
                        "subtype": "post_compact_read_restore",
                        "send_to_provider": True,
                        "path": str(path),
                    },
                )
            )
        return restored


async def _stream_compact_summary(
    llm: LLMAdapter,
    messages: list[Message],
    *,
    temperature: float,
    custom_instructions: str | None,
) -> str:
    prompt = COMPACT_PROMPT
    if custom_instructions and custom_instructions.strip():
        prompt += f"\n\nAdditional Instructions:\n{custom_instructions.strip()}"
    prompt += "\n\nREMINDER: Do NOT call tools. Respond with plain text only."

    text_parts: list[str] = []
    thinking_parts: list[str] = []
    request_messages = [*messages, user_message(prompt, is_meta=True)]
    request_messages[-1].metadata["send_to_provider"] = True

    async for event in llm.stream_chat(
        messages=request_messages,
        system_prompt=COMPACT_SYSTEM_PROMPT,
        tools=[],
        temperature=temperature,
    ):
        if isinstance(event, LLMTextDelta):
            text_parts.append(event.delta)
        elif isinstance(event, LLMThinkingDelta):
            thinking_parts.append(event.delta)
        elif isinstance(event, LLMToolUse):
            continue
        elif isinstance(event, LLMAssistantDone):
            continue

    summary = "".join(text_parts).strip() or "".join(thinking_parts).strip()
    if not summary:
        raise RuntimeError("compaction summary was empty")
    return _format_compact_summary(summary)


def calculate_messages_to_keep_index(messages: list[Message]) -> int:
    if not messages:
        return 0

    start_index = len(messages)
    total_tokens = 0
    text_message_count = 0
    floor = _last_compact_boundary_index(messages) + 1

    for index in range(len(messages) - 1, floor - 1, -1):
        message = messages[index]
        total_tokens += _estimate_message_tokens(message)
        if _has_text_blocks(message):
            text_message_count += 1
        start_index = index
        if total_tokens >= SESSION_MEMORY_MAX_TOKENS:
            break
        if (
            total_tokens >= SESSION_MEMORY_MIN_TOKENS
            and text_message_count >= SESSION_MEMORY_MIN_TEXT_MESSAGES
        ):
            break

    if start_index == floor and total_tokens < SESSION_MEMORY_MIN_TOKENS:
        return 0
    return adjust_index_to_preserve_api_invariants(messages, start_index)


def adjust_index_to_preserve_api_invariants(
    messages: list[Message],
    start_index: int,
) -> int:
    if start_index <= 0 or start_index >= len(messages):
        return start_index

    adjusted = start_index
    tool_result_ids: list[str] = []
    for message in messages[start_index:]:
        for block in message.content:
            if isinstance(block, ToolResultBlock):
                tool_result_ids.append(block.tool_use_id)

    kept_tool_use_ids = {
        block.id
        for message in messages[adjusted:]
        for block in message.content
        if isinstance(block, ToolUseBlock)
    }
    needed_ids = set(tool_id for tool_id in tool_result_ids if tool_id not in kept_tool_use_ids)
    for index in range(adjusted - 1, -1, -1):
        if not needed_ids:
            break
        message = messages[index]
        found = {
            block.id
            for block in message.content
            if isinstance(block, ToolUseBlock) and block.id in needed_ids
        }
        if found:
            adjusted = index
            needed_ids -= found

    assistant_ids = {
        message.id
        for message in messages[adjusted:]
        if message.role == "assistant"
    }
    for index in range(adjusted - 1, -1, -1):
        message = messages[index]
        if message.role == "assistant" and message.id in assistant_ids:
            adjusted = index

    return adjusted


def truncate_head_for_ptl_retry(messages: list[Message]) -> list[Message] | None:
    input_messages = (
        messages[1:]
        if messages
        and messages[0].role == "user"
        and messages[0].is_meta
        and messages[0].to_plain_text() == PTL_RETRY_MARKER
        else messages
    )
    groups = _group_messages_by_api_round(input_messages)
    if len(groups) < 2:
        return None
    sliced = [message for group in groups[1:] for message in group]
    if not sliced:
        return None
    if sliced[0].role == "assistant":
        marker = user_message(PTL_RETRY_MARKER, is_meta=True)
        marker.metadata["send_to_provider"] = True
        return [marker, *sliced]
    return sliced


def _group_messages_by_api_round(messages: list[Message]) -> list[list[Message]]:
    groups: list[list[Message]] = []
    current: list[Message] = []
    for message in messages:
        starts_user_round = (
            message.role == "user"
            and not message.is_meta
            and not message.has_tool_result()
        )
        if starts_user_round and current:
            groups.append(current)
            current = [message]
            continue
        current.append(message)
    if current:
        groups.append(current)
    return groups


def _copy_message_for_compact_segment(message: Message) -> Message:
    metadata = {
        **message.metadata,
        "send_to_provider": True,
        "preserved_after_compact": True,
        "original_message_id": message.id,
    }
    return message.model_copy(
        update={
            "id": new_id("msg"),
            "is_meta": True,
            "metadata": metadata,
        },
        deep=True,
    )


def _collect_compactable_tool_ids(messages: list[Message]) -> list[str]:
    ids: list[str] = []
    for message in messages:
        if message.role != "assistant":
            continue
        for block in message.content:
            if isinstance(block, ToolUseBlock) and block.name in COMPACTABLE_TOOLS:
                ids.append(block.id)
    return ids


def _collect_recent_read_paths(messages: list[Message], *, project_root: Path | None) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for message in reversed(messages):
        if message.role != "assistant":
            continue
        for block in reversed(message.content):
            if not isinstance(block, ToolUseBlock) or block.name != "Read":
                continue
            path = _resolve_read_path(block, project_root=project_root)
            if path is None or path in seen:
                continue
            seen.add(path)
            paths.append(path)
    return paths


def _collect_read_paths(messages: list[Message], *, project_root: Path | None) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for message in messages:
        if message.role != "assistant":
            continue
        for block in message.content:
            if not isinstance(block, ToolUseBlock) or block.name != "Read":
                continue
            path = _resolve_read_path(block, project_root=project_root)
            if path is None or path in seen:
                continue
            seen.add(path)
            paths.append(path)
    return paths


def _resolve_read_path(block: ToolUseBlock, *, project_root: Path | None) -> Path | None:
    raw_path = block.input.get("file_path") or block.input.get("path")
    if not raw_path:
        return None
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute() and project_root is not None:
        path = project_root / path
    return path.resolve()


def _format_compact_summary(summary: str) -> str:
    formatted = re.sub(r"<analysis>[\s\S]*?</analysis>", "", summary).strip()
    match = re.search(r"<summary>([\s\S]*?)</summary>", formatted)
    if match:
        formatted = match.group(1).strip()
    return re.sub(r"\n{3,}", "\n\n", formatted).strip()


def _is_prompt_too_long(value: str) -> bool:
    lower = value.lower()
    return any(marker in lower for marker in PROMPT_TOO_LONG_MARKERS)


def _last_compact_boundary_index(messages: list[Message]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].metadata.get("subtype") == "compact_boundary":
            return index
    return -1


def _has_text_blocks(message: Message) -> bool:
    return any(isinstance(block, TextBlock) and block.text for block in message.content)


def _estimate_message_tokens(message: Message) -> int:
    return _estimate_text_tokens(message.to_plain_text())


def _estimate_text_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _truncate_to_tokens(text: str, max_tokens: int, *, marker: str) -> str:
    if _estimate_text_tokens(text) <= max_tokens:
        return text
    char_budget = max(0, max_tokens * 4 - len(marker))
    return text[:char_budget] + marker
