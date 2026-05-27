from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from engine.message_schema import Message, ToolUseBlock
from runtime.ids import new_id
from runtime.usage_tracker import Usage

TurnStage = Literal[
    "created",
    "preflight",
    "running",
    "completed",
    "failed",
    "interrupted",
]


@dataclass(slots=True)
class QueryTurnState:
    session_id: str
    turn_index: int
    user_message: Message
    prompt_text: str
    turn_id: str = field(default_factory=lambda: new_id("turn"))
    started_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    finished_at: datetime | None = None
    stage: TurnStage = "created"
    messages: list[Message] = field(default_factory=list)
    messages_for_query: list[Message] = field(default_factory=list)
    generated_messages: list[Message] = field(default_factory=list)
    assistant_messages: list[Message] = field(default_factory=list)
    tool_result_messages: list[Message] = field(default_factory=list)
    tool_calls: list[ToolUseBlock] = field(default_factory=list)
    streamed_text_parts: list[str] = field(default_factory=list)
    usage_delta: Usage = field(default_factory=Usage)
    estimated_input_tokens: int = 0
    stop_reason: str | None = None
    error: str | None = None
    retryable_error: bool = False
    final_message: Message | None = None
    final_text: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    def append_stream_delta(self, delta: str) -> None:
        if delta:
            self.streamed_text_parts.append(delta)

    def add_tool_call(self, block: ToolUseBlock) -> None:
        self.tool_calls.append(block)

    def record_generated_message(self, message: Message) -> None:
        self.generated_messages.append(message)
        self.messages.append(message)
        if message.role == "assistant":
            self.assistant_messages.append(message)
        elif message.role == "user" and message.has_tool_result():
            self.tool_result_messages.append(message)
        if message.role == "assistant":
            self.final_message = message
            self.final_text = message.to_plain_text()

    def mark_failed(self, error: str, *, retryable: bool = False) -> None:
        self.stage = "failed"
        self.error = error
        self.retryable_error = retryable
        self.finished_at = datetime.now(timezone.utc)

    def mark_interrupted(self, error: str = "interrupted") -> None:
        self.stage = "interrupted"
        self.error = error
        self.retryable_error = True
        self.finished_at = datetime.now(timezone.utc)

    def mark_completed(self, stop_reason: str | None = None) -> None:
        self.stage = "completed"
        self.stop_reason = stop_reason
        self.finished_at = datetime.now(timezone.utc)

    @property
    def duration_ms(self) -> float | None:
        if self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds() * 1000
