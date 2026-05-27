from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.cli import parse_args
from app.options import CLIOptions
from config.settings import load_settings
from engine.events import AgentError, QueryEvent
from runtime.logging_utils import configure_logging
from runtime.permissions import ensure_permission_files
from runtime.project_store import ProjectStore
from runtime.session_runtime import SessionBusyError
from runtime.session_runtime_manager import SessionRuntimeManager
from runtime.session_store import JsonlSessionStore


class CreateSessionRequest(BaseModel):
    project_root: str | None = None


class StartTurnRequest(BaseModel):
    text: str


class StartTurnResponse(BaseModel):
    turn_id: str
    status: str


class ResolveApprovalRequest(BaseModel):
    approved: bool


class ResolveApprovalResponse(BaseModel):
    approval_id: str
    resolved: bool


def build_api_manager(options: CLIOptions | None = None) -> SessionRuntimeManager:
    options = options or parse_args([])
    project_root = (options.cwd or Path.cwd()).expanduser().resolve()
    if not project_root.exists() or not project_root.is_dir():
        raise ValueError(f"Project root does not exist or is not a directory: {project_root}")
    settings = load_settings(options=options, project_root=project_root)
    configure_logging(debug=settings.runtime.debug)
    ensure_permission_files()
    project_store = ProjectStore()
    session_store = JsonlSessionStore(settings.runtime.session_dir, project_store=project_store)
    return SessionRuntimeManager(
        base_settings=settings,
        project_store=project_store,
        session_store=session_store,
    )


def create_app(manager: SessionRuntimeManager | None = None) -> FastAPI:
    app = FastAPI(title="SiYi API")
    app.state.runtime_manager = manager

    @app.get("/sessions")
    async def list_sessions():
        runtime_manager = _manager(request_app=app)
        return [summary.model_dump(mode="json") for summary in runtime_manager.list_sessions()]

    @app.post("/sessions")
    async def create_session(payload: CreateSessionRequest):
        runtime_manager = _manager(request_app=app)
        project_root = Path(payload.project_root).expanduser().resolve() if payload.project_root else None
        try:
            summary = await runtime_manager.create_session(project_root)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return summary.model_dump(mode="json")

    @app.get("/sessions/{session_id}/turns")
    async def list_turns(
        session_id: str,
        cursor: str | None = None,
        limit: int | None = None,
        items_view: str = "full",
    ):
        runtime_manager = _manager(request_app=app)
        turns, next_cursor = runtime_manager.get_turns(
            session_id,
            cursor=cursor,
            limit=limit,
            items_view=items_view,
        )
        return {
            "data": [turn.model_dump(mode="json") for turn in turns],
            "next_cursor": next_cursor,
        }

    @app.post("/sessions/{session_id}/turns")
    async def start_turn(session_id: str, payload: StartTurnRequest):
        runtime_manager = _manager(request_app=app)
        try:
            turn = await runtime_manager.start_turn(session_id, payload.text)
        except SessionBusyError as exc:
            return JSONResponse(status_code=409, content=exc.error.model_dump(mode="json"))
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return StartTurnResponse(turn_id=str(turn.turn_id), status=turn.status).model_dump(mode="json")

    @app.get("/sessions/{session_id}/turns/{turn_id}/events")
    async def stream_turn_events(
        session_id: str,
        turn_id: str,
        after_event_id: str | None = None,
        last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
    ):
        runtime_manager = _manager(request_app=app)
        after = after_event_id or last_event_id

        async def stream():
            try:
                async for event in runtime_manager.subscribe_turn(
                    session_id,
                    turn_id,
                    after_event_id=after,
                ):
                    yield _sse_event(event)
            except ValueError:
                error = AgentError(
                    session_id=session_id,
                    turn_id=turn_id,
                    code="turn_not_loaded",
                    category="runtime",
                    message=f"Turn is not loaded: {turn_id}",
                    retryable=False,
                )
                yield _sse_event(error)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/sessions/{session_id}/turns/{turn_id}/interrupt")
    async def interrupt_turn(session_id: str, turn_id: str):
        runtime_manager = _manager(request_app=app)
        try:
            await runtime_manager.interrupt_turn(session_id, turn_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"turn_id": turn_id, "status": "interrupt_requested"}

    @app.get("/permissions/pending")
    async def pending_permissions():
        runtime_manager = _manager(request_app=app)
        pending = await runtime_manager.approval_broker.pending()
        return [item.model_dump(mode="json") for item in pending]

    @app.post("/permissions/{approval_id}/resolve")
    async def resolve_permission(approval_id: str, payload: ResolveApprovalRequest):
        runtime_manager = _manager(request_app=app)
        resolved = await runtime_manager.approval_broker.resolve(approval_id, payload.approved)
        if not resolved:
            raise HTTPException(status_code=404, detail=f"Approval not found: {approval_id}")
        return ResolveApprovalResponse(approval_id=approval_id, resolved=True).model_dump(mode="json")

    return app


def _manager(*, request_app: FastAPI) -> SessionRuntimeManager:
    runtime_manager = getattr(request_app.state, "runtime_manager", None)
    if runtime_manager is None:
        runtime_manager = build_api_manager()
        request_app.state.runtime_manager = runtime_manager
    return runtime_manager


def _sse_event(event: QueryEvent) -> str:
    payload = event.model_dump(mode="json")
    return (
        f"id: {event.event_id}\n"
        f"event: {event.type}\n"
        f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    )


app = create_app()
