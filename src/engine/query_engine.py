from __future__ import annotations

import logging
import asyncio
from dataclasses import dataclass
from collections.abc import AsyncIterator
from pathlib import Path

from config.paths import get_global_permissions_path, get_siyi_config_path
from config.settings import AppSettings
from engine.events import (
    AgentError,
    QueryEvent,
    SessionUpdatedEvent,
    ToolResultEvent,
    TurnCompletedEvent,
    TurnStartedEvent,
)
from engine.message_schema import (
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    get_messages_after_compact_boundary,
    normalize_messages_for_api,
    tool_result_message,
    user_message,
)
from engine.query_loop import DefaultQueryLoop, QueryLoop
from engine.session_title import SessionTitleAgent, fallback_session_title, normalize_session_title
from engine.turn_state import QueryTurnState
from llm.base import LLMAdapter
from runtime.compaction import CompactionManager, CompactionResult
from runtime.permissions import (
    PermissionManager,
    load_global_permission_mode,
    save_global_permission_mode,
)
from runtime.read_model import HistoryService, SessionSummary, TurnView
from runtime.ids import new_id
from runtime.session_store import JsonlSessionStore, SessionHandle
from runtime.token_budget import BudgetSnapshot, TokenBudget
from runtime.turn_runtime import TurnInterruptedError
from runtime.usage_tracker import UsageTracker
from tools.base import ToolContext
from tools.registry import ToolRegistry

LOGGER = logging.getLogger(__name__)


def _usage_has_tokens(usage: object) -> bool:
    return bool(
        getattr(usage, "input_tokens", 0)
        or getattr(usage, "output_tokens", 0)
        or getattr(usage, "cached_tokens", 0)
    )


def _agent_error_from_exception(
    exc: Exception,
    *,
    session_id: str | None = None,
    turn_id: str | None = None,
) -> AgentError:
    message = str(exc) or type(exc).__name__
    lowered = message.lower()
    code = "runtime_error"
    category = "runtime"
    retryable = False
    action = None
    if "api key" in lowered or "authentication" in lowered or "unauthorized" in lowered or "401" in lowered:
        code = "provider_auth_invalid"
        category = "provider"
        action = "open_provider_settings"
    elif "rate limit" in lowered or "429" in lowered:
        code = "provider_rate_limited"
        category = "provider"
        retryable = True
        action = "retry_after"
    return AgentError(
        session_id=session_id,
        turn_id=turn_id,
        code=code,
        category=category,  # type: ignore[arg-type]
        message=message,
        retryable=retryable,
        action=action,
        details={"exception_type": type(exc).__name__},
    )


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    session_id: str
    name: str
    name_status: str
    project_root: str
    project_id: str
    project_state_dir: str
    session_path: str
    turn_count: int
    message_count: int
    completed_turns: int
    last_error: AgentError | None
    total_usage: dict[str, int]
    estimated_total_cost: float
    permission_mode: str


@dataclass(frozen=True, slots=True)
class PermissionSnapshot:
    session_mode: str
    global_mode: str
    config_path: str
    permissions_path: str


class QueryEngine:
    def __init__(
        self,
        session: SessionHandle,
        settings: AppSettings,
        llm: LLMAdapter,
        tool_registry: ToolRegistry,
        session_store: JsonlSessionStore, 
        # why JsonlSessionStore? 
        # Because we want to persist the session data in a jsonl file, so that we can load it later and continue the session. 
        # This is also useful for debugging and auditing purposes.
        permission_manager: PermissionManager,
        compaction_manager: CompactionManager,
        token_budget: TokenBudget,
        usage_tracker: UsageTracker,
        query_loop: QueryLoop | None = None,
    ) -> None:
        self.session = session
        self.settings = settings
        self.llm = llm
        self.tool_registry = tool_registry
        self.session_store = session_store
        self.permission_manager = permission_manager
        self.compaction_manager = compaction_manager
        self.token_budget = token_budget
        self.usage_tracker = usage_tracker
        self.query_loop = query_loop or DefaultQueryLoop()
        self.history_service = HistoryService(session_store)
        self.title_agent = SessionTitleAgent(llm)
        self._title_task: asyncio.Task[None] | None = None
        self.mutable_messages: list[Message] = list(session.messages)
        self.last_error: AgentError | None = None
        self.current_turn: QueryTurnState | None = None
        self._active_cancel_event: asyncio.Event | None = None
        self.turn_counter = self._derive_turn_counter()
        self.auto_compact_failures = 0
        self.usage_tracker.rebuild_from_messages(self.mutable_messages)

    async def submit_user_input(
        self,
        text: str,
        *,
        turn_id: str | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncIterator[QueryEvent]:
        prompt = text.strip() 
        if not prompt:
            error = AgentError(
                session_id=self.session.session_id,
                turn_id=turn_id,
                code="empty_prompt",
                category="validation",
                message="Prompt is empty.",
                retryable=False,
            )
            self.last_error = error
            self.session_store.append_event(self.session, error)
            yield error
            return

        self.last_error = None
        if self.session.metadata.name_status == "pending" and self.turn_counter == 0:
            self._start_title_task(prompt)
        user_msg = self._build_user_message(prompt)
        turn = self._create_turn_state(prompt, user_msg, turn_id=turn_id)
        self.current_turn = turn
        self._active_cancel_event = cancel_event

        self._append_message(user_msg)
        turn_started = TurnStartedEvent(
            session_id=turn.session_id,
            turn_id=turn.turn_id,
            turn_index=turn.turn_index,
        )
        self.session_store.append_event(self.session, turn_started)
        yield turn_started

        persisted_generated_count = 0
        try:
            await self._prepare_turn(turn)
            async for event in self.query_loop.run(
                turn,
                llm=self.llm,
                system_prompt=self.settings.model.system_prompt,
                tools=self.tool_registry.to_model_tool_schemas(),
                temperature=self.settings.model.temperature,
                tool_registry=self.tool_registry,
                tool_context=ToolContext(
                    project_root=self.project_root,
                    trace_id=turn.turn_id,
                    session_id=turn.session_id,
                    turn_id=turn.turn_id,
                    max_result_chars=self.settings.tools.max_tool_result_chars,
                ),
                cancel_event=cancel_event,
            ):
                persisted_generated_count = self._drain_generated_messages(
                    turn,
                    persisted_generated_count,
                )
                # _attach_turn_metadata 负责补全事件中的 session_id 和 turn_id
                # 最后还是返回 Event.
                emitted = self._attach_turn_metadata(event, turn)
                self._persist_event(emitted)
                # 最后将事件 yield 出去，供 UI 层消费并渲染
                yield emitted 
        except TurnInterruptedError:
            await self._cancel_turn_side_effects(turn.turn_id)
            synthetic_events = self._complete_pending_tool_results(turn)
            persisted_generated_count = self._drain_generated_messages(
                turn,
                persisted_generated_count,
            )
            self._hide_interrupted_turn_from_model(turn)
            turn.mark_interrupted()
            completed = self._build_turn_completed_event(turn, status="interrupted")
            self.session_store.append_event(self.session, completed)
            for event in synthetic_events:
                yield event
            yield completed
            return
        except Exception as exc:  # noqa: BLE001
            if self.settings.runtime.debug:
                LOGGER.exception("query_engine.turn_failed")
            else:
                LOGGER.debug("query_engine.turn_failed", exc_info=exc)
            error = _agent_error_from_exception(exc, session_id=turn.session_id, turn_id=turn.turn_id)
            turn.mark_failed(error.code, retryable=error.retryable)
            self.last_error = error
            completed = self._build_turn_completed_event(
                turn,
                status="failed",
                error=error,
            )
            self.session_store.append_event(self.session, completed)
            yield completed
            return
        finally:
            if self.current_turn is turn:
                self.current_turn = None
            if self._active_cancel_event is cancel_event:
                self._active_cancel_event = None

        persisted_generated_count = self._drain_generated_messages(
            turn,
            persisted_generated_count,
        )
        turn.mark_completed(turn.stop_reason)

        completed = self._build_turn_completed_event(turn, status="completed")
        self.session_store.append_event(self.session, completed)
        yield completed

    def _build_user_message(self, text: str) -> Message:
        return user_message(text)

    def _create_turn_state(
        self,
        prompt: str,
        user_msg: Message,
        *,
        turn_id: str | None = None,
    ) -> QueryTurnState:
        self.turn_counter += 1
        prior_messages = list(self.mutable_messages)
        turn = QueryTurnState(
            session_id=self.session.session_id,
            turn_index=self.turn_counter,
            user_message=user_msg,
            prompt_text=prompt,
            turn_id=turn_id or new_id("turn"),
            messages=[*prior_messages, user_msg],
        )
        user_msg.metadata["turn_id"] = turn.turn_id
        user_msg.metadata["turn_index"] = turn.turn_index
        return turn

    async def _prepare_turn(self, turn: QueryTurnState) -> BudgetSnapshot:
        turn.stage = "preflight"
        turn.messages_for_query = self._build_messages_for_query(turn.messages)
        budget = self._evaluate_budget(turn.messages_for_query)

        if self._should_try_auto_compact(turn.messages, budget):
            try:
                compaction_result = await self._compact_messages(
                    turn.messages,
                    trigger="auto",
                )
            except Exception as exc:  # noqa: BLE001 - auto compact should not kill the user turn
                self.auto_compact_failures += 1
                LOGGER.debug("query_engine.auto_compact_failed", exc_info=exc)
            else:
                if compaction_result.compacted:
                    self._append_compaction_messages(compaction_result.messages_to_append)
                    turn.messages = list(self.mutable_messages)
                    turn.messages_for_query = normalize_messages_for_api(
                        compaction_result.messages
                    )
                    budget = self._evaluate_budget(turn.messages_for_query)
                    self.auto_compact_failures = 0
                else:
                    self.auto_compact_failures += 1

        turn.estimated_input_tokens = budget.estimated_tokens
        return budget

    async def compact(self, custom_instructions: str | None = None) -> CompactionResult:
        result = await self._compact_messages(
            self.mutable_messages,
            trigger="manual",
            custom_instructions=custom_instructions,
        )
        if result.compacted:
            self._append_compaction_messages(result.messages_to_append)
            self.auto_compact_failures = 0
        return result

    async def new_session(self) -> SessionSnapshot:
        project = self.session_store.project_store.ensure_project(self.settings.runtime.project_root)
        session = self.session_store.create_session(
            project=project,
            permission_mode=load_global_permission_mode(),
        )
        self._activate_session(session)
        return self.get_session_snapshot()

    async def switch_session(self, session_id: str) -> SessionSnapshot:
        metadata = self.session_store.get_metadata(session_id)
        if metadata is None:
            raise ValueError(f"Session not found: {session_id}")
        project = self.session_store.get_project_for_session(metadata)
        if project is None:
            raise ValueError(f"Session has no project metadata: {session_id}")
        project_root = Path(project.project_root).expanduser().resolve()
        if not project_root.exists() or not project_root.is_dir():
            raise ValueError(f"Session project root does not exist: {project_root}")

        session = self.session_store.switch_session(session_id)
        self.settings.runtime.project_root = project_root
        self.permission_manager.reload_for_project_root(project_root, mode=metadata.permission_mode)
        self._activate_session(session)
        return self.get_session_snapshot()

    def list_sessions(self) -> list[SessionSummary]:
        return self.history_service.list_session_summaries(
            active_session_id=self.session.session_id,
            active_turn_id=self.current_turn.turn_id if self.current_turn is not None else None,
            last_error=self.last_error,
            usage=self.usage_tracker.get_total_usage(),
        )

    def get_session_turns(
        self,
        session_id: str | None = None,
        *,
        cursor: str | None = None,
        limit: int | None = None,
        items_view: str = "full",
    ) -> tuple[list[TurnView], str | None]:
        target_session_id = session_id or self.session.session_id
        view = "summary" if items_view == "summary" else "full"
        return self.history_service.get_session_turns(
            target_session_id,
            cursor=cursor,
            limit=limit,
            items_view=view,
        )

    def get_permission_snapshot(self) -> PermissionSnapshot:
        return PermissionSnapshot(
            session_mode=self.permission_manager.mode,
            global_mode=load_global_permission_mode(),
            config_path=str(get_siyi_config_path()),
            permissions_path=str(get_global_permissions_path()),
        )

    def set_permission_mode(self, mode: str) -> PermissionSnapshot:
        permission_mode = self.permission_manager.set_mode(mode)
        self.session.metadata.permission_mode = permission_mode
        self.session_store.update_metadata(self.session)
        self._append_session_updated_event(permission_mode=permission_mode)
        return self.get_permission_snapshot()

    def set_global_permission_mode(self, mode: str) -> PermissionSnapshot:
        save_global_permission_mode(mode)
        return self.get_permission_snapshot()

    async def _compact_messages(
        self,
        messages: list[Message],
        *,
        trigger: str,
        custom_instructions: str | None = None,
    ) -> CompactionResult:
        return await self.compaction_manager.compact_conversation(
            list(messages),
            llm=self.llm,
            system_prompt=self.settings.model.system_prompt,
            temperature=self.settings.model.temperature,
            token_budget=self.token_budget,
            trigger="auto" if trigger == "auto" else "manual",
            custom_instructions=custom_instructions,
            transcript_path=str(self.session.path),
            project_root=Path(self.project_root),
        )

    def _build_messages_for_query(self, messages: list[Message]) -> list[Message]:
        visible_messages = get_messages_after_compact_boundary(messages)
        compaction_result = self.compaction_manager.microcompact_projection(visible_messages)
        return normalize_messages_for_api(compaction_result.messages)

    def _evaluate_budget(self, messages: list[Message]) -> BudgetSnapshot:
        return self.token_budget.evaluate(
            messages=messages,
            system_prompt=self.settings.model.system_prompt,
            tools=self.tool_registry.to_model_tool_schemas(),
        )

    def _should_try_auto_compact(
        self,
        messages: list[Message],
        budget: BudgetSnapshot,
    ) -> bool:
        if not self.settings.runtime.compaction_enabled:
            return False
        if not self.settings.runtime.auto_compact_enabled:
            return False
        if self.auto_compact_failures >= 3:
            return False
        if not budget.should_autocompact:
            return False
        visible_messages = get_messages_after_compact_boundary(messages)
        return any(message.role == "assistant" for message in visible_messages)

    def _append_message(self, message: Message) -> None:
        self.mutable_messages.append(message)
        self.session_store.append_message(self.session, message)

    def _append_compaction_messages(self, messages: list[Message]) -> None:
        for message in messages:
            self.mutable_messages.append(message)
            self.session_store.append_message(self.session, message)

    # store generated messages in the turn state until the turn is completed,
    # then persist them all at once.
    def _drain_generated_messages(
        self,
        turn: QueryTurnState,
        already_persisted: int,
    ) -> int:
        pending = turn.generated_messages[already_persisted:]
        if not pending:
            return already_persisted
        for message in pending:
            self.mutable_messages.append(message)
            self.session_store.append_message(self.session, message)
        return len(turn.generated_messages)

    def _attach_turn_metadata(
        self,
        event: QueryEvent,
        turn: QueryTurnState,
    ) -> QueryEvent:
        update: dict[str, str] = {}
        if event.session_id is None:
            update["session_id"] = turn.session_id
        if event.turn_id is None:
            update["turn_id"] = turn.turn_id
        return event.model_copy(update=update) if update else event

    def _persist_event(self, event: QueryEvent) -> None:
        if event.type in {"assistant_delta", "tool_output_delta"}:
            return
        self.session_store.append_event(self.session, event)

    def _build_turn_completed_event(
        self,
        turn: QueryTurnState,
        *,
        status: str,
        error: AgentError | None = None,
    ) -> TurnCompletedEvent:
        usage_snapshot = None
        if status == "completed" or _usage_has_tokens(turn.usage_delta):
            usage_snapshot = self.usage_tracker.record_turn(turn.turn_id, turn.usage_delta)
        if error is None and status == "interrupted":
            error = AgentError(
                session_id=turn.session_id,
                turn_id=turn.turn_id,
                code="turn_interrupted",
                category="runtime",
                message="turn interrupted",
                retryable=False,
            )
        elif error is None and status == "failed":
            error = AgentError(
                session_id=turn.session_id,
                turn_id=turn.turn_id,
                code=turn.error or "runtime_error",
                category="runtime",
                message=turn.error or "turn failed",
                retryable=turn.retryable_error,
            )
        if error is not None:
            self.last_error = error
        return TurnCompletedEvent(
            session_id=turn.session_id,
            turn_id=turn.turn_id,
            status=status,  # type: ignore[arg-type]
            usage=self.usage_tracker.get_total_usage(),
            estimated_cost=usage_snapshot.estimated_cost if usage_snapshot is not None else None,
            stop_reason=turn.stop_reason,
            error=error,
        )

    def _complete_pending_tool_results(self, turn: QueryTurnState) -> list[ToolResultEvent]:
        assistant_tool_ids: set[str] = set()
        for message in turn.assistant_messages:
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    assistant_tool_ids.add(block.id)
        completed_ids: set[str] = set()
        for message in turn.tool_result_messages:
            for block in message.content:
                if isinstance(block, ToolResultBlock):
                    completed_ids.add(block.tool_use_id)
        events: list[ToolResultEvent] = []
        for block in turn.tool_calls:
            if block.id not in assistant_tool_ids or block.id in completed_ids:
                continue
            result_block = ToolResultBlock(
                tool_use_id=block.id,
                content="tool call was interrupted before it returned a result",
                is_error=True,
            )
            error = AgentError(
                session_id=turn.session_id,
                turn_id=turn.turn_id,
                code="turn_interrupted",
                category="runtime",
                message=result_block.content,
                retryable=False,
                tool_name=block.name,
                tool_use_id=block.id,
            )
            result_message = tool_result_message(
                tool_use_id=block.id,
                content=result_block.content,
                is_error=True,
                metadata={
                    "tool_name": block.name,
                    "interrupted": True,
                    "turn_id": turn.turn_id,
                    "turn_index": turn.turn_index,
                    "error": error.model_dump(mode="json"),
                },
            )
            turn.record_generated_message(result_message)
            event = ToolResultEvent(
                session_id=turn.session_id,
                turn_id=turn.turn_id,
                block=result_block,
                error=error,
            )
            self.session_store.append_event(self.session, event)
            events.append(event)
        return events

    def _hide_interrupted_turn_from_model(self, turn: QueryTurnState) -> None:
        patch = {
            "is_meta": True,
            "metadata": {
                "hidden_from_model": True,
                "hidden_reason": "interrupted_turn",
                "turn_id": turn.turn_id,
            },
        }
        seen: set[str] = set()
        for message in [turn.user_message, *turn.generated_messages]:
            if message.id in seen:
                continue
            seen.add(message.id)
            self.session_store.patch_message(self.session, message.id, patch)

    async def _cancel_turn_side_effects(self, turn_id: str) -> None:
        if self.permission_manager.approval_broker is not None:
            await self.permission_manager.approval_broker.cancel_turn(turn_id)
        from tools.builtin import PROCESSES

        await PROCESSES.stop_for_turn(turn_id)

    def _append_session_updated_event(
        self,
        *,
        name: str | None = None,
        name_status: str | None = None,
        permission_mode: str | None = None,
    ) -> None:
        event = SessionUpdatedEvent(
            session_id=self.session.session_id,
            name=name,
            name_status=name_status,
            permission_mode=permission_mode,
        )
        self.session_store.append_event(self.session, event)

    def _derive_turn_counter(self) -> int:
        return sum(
            1
            for message in self.mutable_messages
            if message.role == "user"
            and not message.is_meta
            and not message.metadata.get("preserved_after_compact")
            and not message.has_tool_result()
        )

    def _activate_session(self, session: SessionHandle) -> None:
        self.session = session
        self.mutable_messages = list(session.messages)
        self.permission_manager.set_mode(session.metadata.permission_mode)
        self.current_turn = None
        self.last_error = None
        self.turn_counter = self._derive_turn_counter()
        self.auto_compact_failures = 0
        self.usage_tracker.rebuild_from_messages(self.mutable_messages)
        project = self.session_store.get_project_for_session(session.metadata)
        if project is not None:
            self.settings.runtime.project_root = Path(project.project_root).expanduser().resolve()

    @property
    def project_root(self) -> str:
        project = self.session_store.get_project_for_session(self.session.metadata)
        if project is None:
            return str(self.settings.runtime.project_root)
        return project.project_root

    @property
    def project_state_dir(self) -> str:
        project = self.session_store.get_project_for_session(self.session.metadata)
        return project.project_state_dir if project is not None else ""

    def _start_title_task(self, first_user_input: str) -> None:
        if self._title_task is not None and not self._title_task.done():
            return
        self._title_task = asyncio.create_task(self._name_session(self.session, first_user_input))

    async def _name_session(self, session: SessionHandle, first_user_input: str) -> None:
        try:
            title = normalize_session_title(await self.title_agent.generate(first_user_input))
            if not title:
                raise ValueError("empty generated title")
        except Exception as exc:  # noqa: BLE001 - title generation must not fail the user turn
            LOGGER.debug("query_engine.session_title_failed", exc_info=exc)
            title = fallback_session_title(first_user_input)
        try:
            self.session_store.set_generated_name(session, title)
            self.session_store.append_event(
                session,
                SessionUpdatedEvent(
                    session_id=session.session_id,
                    name=title,
                    name_status="ready",
                ),
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("query_engine.session_title_persist_failed", exc_info=exc)

    async def wait_for_title_task(self) -> None:
        if self._title_task is not None:
            await self._title_task

    async def interrupt_turn(self, turn_id: str) -> None:
        turn = self.current_turn
        if turn is None or turn.turn_id != turn_id:
            raise ValueError(f"Turn is not active: {turn_id}")
        if self._active_cancel_event is None:
            raise ValueError(f"Turn cannot be interrupted yet: {turn_id}")
        self._active_cancel_event.set()
        await self._cancel_turn_side_effects(turn_id)

    def transcript_preview(self) -> str:
        return "\n".join(
            f"{message.role}: {message.to_plain_text()}"
            for message in self.mutable_messages
            if any(isinstance(block, TextBlock) for block in message.content)
        )

    def get_messages(self) -> list[Message]:
        return list(self.mutable_messages)

    def get_recent_messages(self, limit: int = 10, *, include_meta: bool = False) -> list[Message]:
        visible_messages = [
            message
            for message in self.mutable_messages
            if include_meta or not message.is_meta
        ]
        if limit <= 0:
            return []
        return visible_messages[-limit:]

    def get_last_user_prompt(self) -> str | None:
        for message in reversed(self.mutable_messages):
            if message.role != "user":
                continue
            if message.is_meta or message.has_tool_result():
                continue
            return message.to_plain_text() or None
        return None

    def get_session_snapshot(self) -> SessionSnapshot:
        total_usage = self.usage_tracker.get_total_usage()
        return SessionSnapshot(
            session_id=self.session.session_id,
            name=self.session.metadata.name,
            name_status=self.session.metadata.name_status,
            project_root=self.project_root,
            project_id=self.session.metadata.project_id,
            project_state_dir=self.project_state_dir,
            session_path=str(self.session.path),
            turn_count=self.turn_counter,
            message_count=len(self.mutable_messages),
            completed_turns=len(self.usage_tracker.get_turn_history()),
            last_error=self.last_error,
            total_usage=total_usage.model_dump(),
            estimated_total_cost=self.usage_tracker.estimate_cost(total_usage),
            permission_mode=self.permission_manager.mode,
        )
