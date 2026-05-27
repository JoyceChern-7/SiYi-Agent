from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from engine.message_schema import Message, ToolResultBlock, ToolUseBlock
from runtime.ids import new_id
from runtime.usage_tracker import Usage


class QueryEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: new_id("evt"))
    type: str
    session_id: str | None = None
    turn_id: str | None = None


AgentErrorCategory = Literal[
    "validation",
    "concurrency",
    "provider",
    "permission",
    "tool",
    "process",
    "runtime",
    "config",
]


class AgentError(QueryEvent):
    type: Literal["agent_error"] = "agent_error"
    code: str
    category: AgentErrorCategory
    message: str
    retryable: bool = False
    action: str | None = None
    tool_name: str | None = None
    tool_use_id: str | None = None
    process_id: str | None = None
    approval_id: str | None = None
    trace_id: str = Field(default_factory=lambda: new_id("trace"))
    details: dict[str, Any] = Field(default_factory=dict)


class AssistantDeltaEvent(QueryEvent):
    type: Literal["assistant_delta"] = "assistant_delta"
    delta: str


class TurnStartedEvent(QueryEvent):
    type: Literal["turn_started"] = "turn_started"
    turn_index: int


class AssistantMessageEvent(QueryEvent):
    type: Literal["assistant_message"] = "assistant_message"
    message: Message


class ToolCallEvent(QueryEvent):
    type: Literal["tool_call"] = "tool_call"
    block: ToolUseBlock


class ToolResultEvent(QueryEvent):
    type: Literal["tool_result"] = "tool_result"
    block: ToolResultBlock
    error: AgentError | None = None


class ToolOutputDeltaEvent(QueryEvent):
    type: Literal["tool_output_delta"] = "tool_output_delta"
    tool_use_id: str
    tool_name: str
    stream: Literal["stdout", "stderr", "status"]
    delta: str
    process_id: str | None = None
    elapsed_ms: int = 0


class PermissionRequestEvent(QueryEvent):
    type: Literal["permission_request"] = "permission_request"
    approval_id: str
    tool_name: str
    tool_input: dict[str, Any]
    project_root: str


class ApprovalResolvedEvent(QueryEvent):
    type: Literal["approval_resolved"] = "approval_resolved"
    approval_id: str
    approved: bool
    tool_name: str


class TurnCompletedEvent(QueryEvent):
    type: Literal["turn_completed"] = "turn_completed"
    status: Literal["completed", "failed", "interrupted"]
    usage: Usage | None = None
    estimated_cost: float | None = None
    stop_reason: str | None = None
    error: AgentError | None = None


class SessionUpdatedEvent(QueryEvent):
    type: Literal["session_updated"] = "session_updated"
    name: str | None = None
    name_status: str | None = None
    permission_mode: str | None = None

