from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from engine.events import AgentError
from engine.message_schema import (
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from runtime.session_store import JsonlSessionStore, _apply_message_patch
from runtime.usage_tracker import Usage

SessionStatus = Literal["idle", "running", "waiting_on_permission", "failed"]
TurnStatus = Literal["running", "completed", "failed", "interrupted"]
ItemsView = Literal["summary", "full"]
ItemType = Literal[
    "user_message",
    "assistant_message",
    "tool_call",
    "tool_result",
    "permission_request",
    "approval_resolved",
    "agent_error",
    "compact_boundary",
    "compact_summary",
]


class SessionSummary(BaseModel):
    session_id: str
    name: str
    name_status: str
    project_id: str
    project_root: str
    created_at: str
    updated_at: str
    permission_mode: str
    message_count: int
    status: SessionStatus = "idle"
    active_turn_id: str | None = None
    last_error: AgentError | None = None
    usage: Usage | None = None


class ItemView(BaseModel):
    type: ItemType
    id: str
    timestamp: str | None = None
    text: str | None = None
    role: str | None = None
    tool_name: str | None = None
    tool_use_id: str | None = None
    tool_input: dict[str, Any] | None = None
    content: str | None = None
    is_error: bool | None = None
    approval_id: str | None = None
    approved: bool | None = None
    status: str | None = None
    error: AgentError | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TurnView(BaseModel):
    turn_id: str
    turn_index: int | None = None
    status: TurnStatus = "running"
    items_view: ItemsView = "full"
    started_at: str | None = None
    completed_at: str | None = None
    duration_ms: int | None = None
    error: AgentError | None = None
    items: list[ItemView] = Field(default_factory=list)


@dataclass
class _TurnAccumulator:
    turn_id: str
    turn_index: int | None = None
    status: TurnStatus = "running"
    started_at: str | None = None
    completed_at: str | None = None
    duration_ms: int | None = None
    error: AgentError | None = None
    items: list[ItemView] = field(default_factory=list)

    def to_view(self, items_view: ItemsView) -> TurnView:
        items = list(self.items)
        if items_view == "summary":
            user_item = next((item for item in items if item.type == "user_message"), None)
            assistant_item = next(
                (item for item in reversed(items) if item.type == "assistant_message"),
                None,
            )
            items = [item for item in (user_item, assistant_item) if item is not None]
        return TurnView(
            turn_id=self.turn_id,
            turn_index=self.turn_index,
            status=self.status,
            items_view=items_view,
            started_at=self.started_at,
            completed_at=self.completed_at,
            duration_ms=self.duration_ms,
            error=self.error,
            items=items,
        )


class HistoryService:
    def __init__(self, session_store: JsonlSessionStore) -> None:
        self.session_store = session_store

    def list_session_summaries(
        self,
        *,
        active_session_id: str | None = None,
        active_turn_id: str | None = None,
        last_error: AgentError | None = None,
        usage: Usage | None = None,
        active_turns: dict[str, str] | None = None,
        last_errors: dict[str, AgentError | None] | None = None,
        usage_by_session: dict[str, Usage | None] | None = None,
    ) -> list[SessionSummary]:
        summaries: list[SessionSummary] = []
        active_turn_map = dict(active_turns or {})
        if active_session_id and active_turn_id:
            active_turn_map[active_session_id] = active_turn_id
        error_map = dict(last_errors or {})
        if active_session_id:
            error_map[active_session_id] = last_error
        usage_map = dict(usage_by_session or {})
        if active_session_id:
            usage_map[active_session_id] = usage
        for metadata in self.session_store.list_sessions():
            project = self.session_store.get_project_for_session(metadata)
            active_turn = active_turn_map.get(metadata.session_id)
            summaries.append(
                SessionSummary(
                    session_id=metadata.session_id,
                    name=metadata.name,
                    name_status=metadata.name_status,
                    project_id=metadata.project_id,
                    project_root=project.project_root if project is not None else "",
                    created_at=metadata.created_at,
                    updated_at=metadata.updated_at,
                    permission_mode=metadata.permission_mode,
                    message_count=metadata.message_count,
                    status="running" if active_turn else "idle",
                    active_turn_id=active_turn,
                    last_error=error_map.get(metadata.session_id),
                    usage=usage_map.get(metadata.session_id),
                )
            )
        return summaries

    def get_session_turns(
        self,
        session_id: str,
        *,
        cursor: str | None = None,
        limit: int | None = None,
        items_view: ItemsView = "full",
    ) -> tuple[list[TurnView], str | None]:
        turns = self._load_turns(session_id, items_view=items_view)
        start = _cursor_to_index(cursor)
        end = len(turns) if limit is None else min(len(turns), start + max(limit, 0))
        page = turns[start:end]
        next_cursor = str(end) if end < len(turns) else None
        return page, next_cursor

    def _load_turns(self, session_id: str, *, items_view: ItemsView) -> list[TurnView]:
        metadata = self.session_store.get_metadata(session_id)
        path = metadata.path if metadata is not None else self.session_store.root / f"{session_id}.jsonl"
        if not path.exists():
            return []

        turns: list[_TurnAccumulator] = []
        by_id: dict[str, _TurnAccumulator] = {}
        messages_by_id: dict[str, Message] = {}
        items_by_message_id: dict[str, ItemView] = {}
        pending_items: list[ItemView] = []
        current_turn: _TurnAccumulator | None = None

        for entry in _read_jsonl(path):
            timestamp = str(entry.get("timestamp") or "") or None
            kind = entry.get("kind")
            if kind == "message":
                message = Message.model_validate(entry["message"])
                messages_by_id[message.id] = message
                item = _item_from_message(message, timestamp)
                if item is None:
                    continue
                items_by_message_id[message.id] = item
                target = _turn_for_message(message, by_id, current_turn)
                if target is None:
                    pending_items.append(item)
                else:
                    target.items.append(item)
                continue

            if kind == "message_patch":
                message = messages_by_id.get(str(entry.get("message_id") or ""))
                patch = entry.get("patch")
                if message is not None and isinstance(patch, dict):
                    _apply_message_patch(message, patch)
                    item = items_by_message_id.get(message.id)
                    if item is not None:
                        item.metadata = dict(message.metadata)
                continue

            if kind != "event" or not isinstance(entry.get("event"), dict):
                continue
            event = entry["event"]
            event_type = event.get("type")

            if event_type == "turn_started":
                turn_id = str(event.get("turn_id") or "")
                if not turn_id:
                    continue
                current_turn = by_id.get(turn_id)
                if current_turn is None:
                    current_turn = _TurnAccumulator(
                        turn_id=turn_id,
                        turn_index=_optional_int(event.get("turn_index")),
                        started_at=timestamp,
                    )
                    by_id[turn_id] = current_turn
                    turns.append(current_turn)
                if pending_items:
                    current_turn.items.extend(pending_items)
                    pending_items = []
                continue

            turn_id = str(event.get("turn_id") or "")
            target = by_id.get(turn_id) if turn_id else current_turn
            if target is None and turn_id:
                target = _TurnAccumulator(turn_id=turn_id, started_at=timestamp)
                by_id[turn_id] = target
                turns.append(target)
            if target is None:
                continue

            item = _item_from_event(event, timestamp)
            if item is not None:
                target.items.append(item)

            if event_type == "turn_completed":
                status = event.get("status")
                if status in {"completed", "failed", "interrupted"}:
                    target.status = status
                target.completed_at = timestamp
                error = event.get("error")
                if isinstance(error, dict):
                    target.error = AgentError.model_validate(error)
                if target.started_at and target.completed_at:
                    target.duration_ms = _duration_ms(target.started_at, target.completed_at)
                current_turn = None

        for item in pending_items:
            if not turns:
                turn = _TurnAccumulator(turn_id="turn_unassigned")
                turns.append(turn)
            turns[-1].items.append(item)
        return [turn.to_view(items_view) for turn in turns]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entries.append(json.loads(line))
    return entries


def _item_from_message(message: Message, timestamp: str | None) -> ItemView | None:
    subtype = message.metadata.get("subtype")
    if subtype == "compact_boundary":
        return _text_item("compact_boundary", message, timestamp)
    if subtype == "compact_summary":
        return _text_item("compact_summary", message, timestamp)
    if message.role == "assistant":
        return _text_item("assistant_message", message, timestamp)
    if message.role == "user" and message.has_tool_result():
        for block in message.content:
            if isinstance(block, ToolResultBlock):
                return ItemView(
                    type="tool_result",
                    id=message.id,
                    timestamp=timestamp,
                    tool_name=_optional_str(message.metadata.get("tool_name")),
                    tool_use_id=block.tool_use_id,
                    content=block.content,
                    is_error=block.is_error,
                    error=_agent_error_from_metadata(message.metadata),
                    metadata=dict(message.metadata),
                )
    if message.role == "user" and _is_text_message(message):
        return _text_item("user_message", message, timestamp)
    return None


def _item_from_event(event: dict[str, Any], timestamp: str | None) -> ItemView | None:
    event_type = event.get("type")
    if event_type == "tool_call" and isinstance(event.get("block"), dict):
        block = event["block"]
        return ItemView(
            type="tool_call",
            id=str(block.get("id") or event.get("event_id") or ""),
            timestamp=timestamp,
            tool_name=str(block.get("name") or ""),
            tool_use_id=str(block.get("id") or ""),
            tool_input=block.get("input") if isinstance(block.get("input"), dict) else {},
            status="started",
        )
    if event_type == "permission_request":
        return ItemView(
            type="permission_request",
            id=str(event.get("approval_id") or event.get("event_id") or ""),
            timestamp=timestamp,
            approval_id=_optional_str(event.get("approval_id")),
            tool_name=_optional_str(event.get("tool_name")),
            tool_input=event.get("tool_input") if isinstance(event.get("tool_input"), dict) else {},
            status="pending",
        )
    if event_type == "approval_resolved":
        return ItemView(
            type="approval_resolved",
            id=str(event.get("approval_id") or event.get("event_id") or ""),
            timestamp=timestamp,
            approval_id=_optional_str(event.get("approval_id")),
            tool_name=_optional_str(event.get("tool_name")),
            approved=bool(event.get("approved")),
            status="approved" if event.get("approved") else "denied",
        )
    if event_type == "agent_error":
        error = AgentError.model_validate(event)
        return ItemView(
            type="agent_error",
            id=error.event_id,
            timestamp=timestamp,
            text=error.message,
            error=error,
        )
    return None


def _text_item(item_type: ItemType, message: Message, timestamp: str | None) -> ItemView:
    return ItemView(
        type=item_type,
        id=message.id,
        timestamp=timestamp,
        text=message.to_plain_text(),
        role=message.role,
        metadata=dict(message.metadata),
    )


def _turn_for_message(
    message: Message,
    by_id: dict[str, _TurnAccumulator],
    current_turn: _TurnAccumulator | None,
) -> _TurnAccumulator | None:
    turn_id = message.metadata.get("turn_id")
    if isinstance(turn_id, str) and turn_id in by_id:
        return by_id[turn_id]
    return current_turn


def _is_text_message(message: Message) -> bool:
    return all(isinstance(block, TextBlock | ThinkingBlock) for block in message.content)


def _cursor_to_index(cursor: str | None) -> int:
    if cursor is None:
        return 0
    try:
        return max(int(cursor), 0)
    except ValueError:
        return 0


def _optional_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _optional_str(value: object) -> str | None:
    return str(value) if value is not None else None


def _agent_error_from_metadata(metadata: dict[str, Any]) -> AgentError | None:
    error = metadata.get("error")
    if isinstance(error, dict):
        return AgentError.model_validate(error)
    return None


def _duration_ms(started_at: str, completed_at: str) -> int | None:
    from datetime import datetime

    try:
        start = datetime.fromisoformat(started_at)
        end = datetime.fromisoformat(completed_at)
    except ValueError:
        return None
    return int((end - start).total_seconds() * 1000)
