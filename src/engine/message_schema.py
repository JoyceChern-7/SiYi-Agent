from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from runtime.ids import new_id

MessageRole = Literal["user", "assistant", "system"]
MessageType = Literal["user", "assistant", "system", "attachment", "progress"]


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ThinkingBlock(BaseModel):
    type: Literal["thinking"] = "thinking"
    text: str


class ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str = Field(default_factory=lambda: new_id("toolu"))
    name: str
    input: dict[str, Any]


class ToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool = False


ContentBlock = Annotated[
    TextBlock | ThinkingBlock | ToolUseBlock | ToolResultBlock,
    Field(discriminator="type"),
]


class Message(BaseModel):
    id: str = Field(default_factory=lambda: new_id("msg"))
    type: MessageType
    role: MessageRole
    content: list[ContentBlock]
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    is_meta: bool = False 
    # system message that is only for internal use, should be filtered out before sending to provider
    # 也就是"我们不希望 llm 看到的消息". 比如压缩前的消息.

    is_virtual: bool = False
    # 和 is_meta 不同, is_virtual 
    # 是指这个消息在任何场景下都不应该被视为真实的用户或助手消息.
 
    tool_use_result: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    # metadata 是一个开放的字段, 可以用来存储任意与消息相关的结构化信息. 
    # 例如 tool_use_message 可以在这里存 tool_use_id, 以便后续关联 tool_result_message

    def to_plain_text(self) -> str:
        parts: list[str] = []
        for block in self.content:
            if isinstance(block, TextBlock | ThinkingBlock):
                parts.append(block.text)
            elif isinstance(block, ToolUseBlock):
                parts.append(f"{block.name}:{block.input}")
            elif isinstance(block, ToolResultBlock):
                parts.append(block.content)
        return "\n".join(parts).strip()

    def has_tool_use(self) -> bool:
        return any(isinstance(block, ToolUseBlock) for block in self.content)

    def has_tool_result(self) -> bool:
        return any(isinstance(block, ToolResultBlock) for block in self.content)


def make_message(
    *,
    role: MessageRole,
    message_type: MessageType,
    blocks: list[ContentBlock],
    is_meta: bool = False,
    is_virtual: bool = False,
    tool_use_result: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Message:
    return Message(
        type=message_type,
        role=role,
        content=blocks,
        is_meta=is_meta,
        is_virtual=is_virtual,
        tool_use_result=tool_use_result,
        metadata=metadata or {},
    )


def user_message(text: str, *, is_meta: bool = False) -> Message:
    return make_message(
        role="user",
        message_type="user",
        blocks=[TextBlock(text=text)],
        is_meta=is_meta, 
    )


def system_message(text: str, *, is_meta: bool = False) -> Message:
    return make_message(
        role="system",
        message_type="system",
        blocks=[TextBlock(text=text)],
        is_meta=is_meta,
    )


def assistant_message(text: str, *, is_meta: bool = False) -> Message:
    return assistant_message_from_blocks(
        [TextBlock(text=text)],
        is_meta=is_meta,
    )


def assistant_message_from_blocks(
    blocks: list[ContentBlock],
    *,
    is_meta: bool = False,
    metadata: dict[str, Any] | None = None,
) -> Message:
    return make_message(
        role="assistant",
        message_type="assistant",
        blocks=blocks,
        is_meta=is_meta,
        metadata=metadata,
    )


def tool_result_message(
    *,
    tool_use_id: str,
    content: str,
    is_error: bool = False,
    metadata: dict[str, Any] | None = None,
) -> Message:
    block = ToolResultBlock(
        tool_use_id=tool_use_id,
        content=content,
        is_error=is_error,
    )
    return make_message(
        role="user",
        message_type="user",
        blocks=[block],
        tool_use_result={"tool_use_id": tool_use_id, "is_error": is_error},
        metadata=metadata,
    )


def compact_boundary_message(
    reason: str | None = None,
    *,
    trigger: str = "manual",
    pre_tokens: int = 0,
    post_tokens: int | None = None,
    tokens_saved: int | None = None,
    preserved_message_ids: list[str] | None = None,
) -> Message:
    details = reason or f"{trigger} compact"
    metadata: dict[str, Any] = {
        "subtype": "compact_boundary",
        "reason": details,
        "trigger": trigger,
        "pre_tokens": pre_tokens,
    }
    if post_tokens is not None:
        metadata["post_tokens"] = post_tokens
    if tokens_saved is not None:
        metadata["tokens_saved"] = tokens_saved
    if preserved_message_ids:
        metadata["preserved_message_ids"] = preserved_message_ids
    return make_message(
        role="system",
        message_type="system",
        blocks=[TextBlock(text=f"[compact_boundary] {details}")],
        is_meta=True,
        metadata=metadata,
    )


def compact_summary_message(
    summary: str,
    *,
    custom_instructions: str | None = None,
    transcript_path: str | None = None,
) -> Message:
    text = (
        "This session is being continued from a previous conversation that ran "
        "out of context. The summary below covers the earlier portion of the "
        "conversation.\n\n"
        f"{summary.strip()}"
    )
    if transcript_path:
        text += (
            "\n\nIf specific details from before compaction are needed, refer "
            f"to the full transcript at: {transcript_path}"
        )
    text += "\n\nRecent messages may be preserved verbatim after this summary."
    metadata: dict[str, Any] = {
        "subtype": "compact_summary",
        "send_to_provider": True,
    }
    if custom_instructions:
        metadata["custom_instructions"] = custom_instructions
    return make_message(
        role="user",
        message_type="user",
        blocks=[TextBlock(text=text)],
        is_meta=True,
        metadata=metadata,
    )


def get_messages_after_compact_boundary(messages: list[Message]) -> list[Message]:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.metadata.get("subtype") == "compact_boundary":
            return messages[index + 1 :]
    return messages


def normalize_messages_for_api(messages: list[Message]) -> list[Message]:
    normalized: list[Message] = []
    pending_user_text: list[str] = []

    def flush_pending_user() -> None:
        if not pending_user_text:
            return
        normalized.append(user_message("\n".join(pending_user_text)))
        pending_user_text.clear()

    for message in messages:
        if message.is_virtual:
            continue
        if message.metadata.get("subtype") == "compact_boundary":
            continue

        send_to_provider = bool(message.metadata.get("send_to_provider"))
        if message.is_meta and not send_to_provider:
            continue

        if (
            message.role == "user"
            and _is_plain_text_message(message)
        ):
            pending_user_text.append(message.to_plain_text())
            continue

        flush_pending_user()

        if message.role == "assistant":
            normalized.append(message)
            continue

        if message.role == "user":
            if message.has_tool_result():
                normalized.append(message)
                continue
            if _is_plain_text_message(message):
                normalized.append(message)
                continue

        if message.role == "system":
            normalized.append(message)

    flush_pending_user()
    return normalized


def _is_plain_text_message(message: Message) -> bool:
    return all(isinstance(block, TextBlock | ThinkingBlock) for block in message.content)
