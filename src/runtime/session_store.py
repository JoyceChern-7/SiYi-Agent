from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.paths import get_project_id, get_project_state_dir
from engine.message_schema import Message
from runtime.ids import new_id

LOGGER = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class SessionMetadata:
    session_id: str
    path: Path
    cwd: str
    project_id: str
    project_state_dir: str
    created_at: str
    updated_at: str
    model: str
    permission_mode: str = "default"
    message_count: int = 0
    legacy: bool = False

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "SessionMetadata":
        return cls(
            session_id=str(payload["session_id"]),
            path=Path(str(payload["path"])).expanduser().resolve(),
            cwd=str(payload.get("cwd") or ""),
            project_id=str(payload.get("project_id") or ""),
            project_state_dir=str(payload.get("project_state_dir") or ""),
            created_at=str(payload.get("created_at") or _now()),
            updated_at=str(payload.get("updated_at") or payload.get("created_at") or _now()),
            model=str(payload.get("model") or ""),
            permission_mode=str(payload.get("permission_mode") or "default"),
            message_count=int(payload.get("message_count") or 0),
            legacy=bool(payload.get("legacy", False)),
        )

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        root: Path,
        cwd: Path,
        model: str,
        permission_mode: str = "default",
    ) -> "SessionMetadata":
        timestamp = _now()
        project_state_dir = get_project_state_dir(cwd)
        return cls(
            session_id=session_id,
            path=(root / f"{session_id}.jsonl").resolve(),
            cwd=str(cwd.expanduser().resolve()),
            project_id=get_project_id(cwd),
            project_state_dir=str(project_state_dir),
            created_at=timestamp,
            updated_at=timestamp,
            model=model,
            permission_mode=permission_mode,
            message_count=0,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "path": str(self.path),
            "cwd": self.cwd,
            "project_id": self.project_id,
            "project_state_dir": self.project_state_dir,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "model": self.model,
            "permission_mode": self.permission_mode,
            "message_count": self.message_count,
            "legacy": self.legacy,
        }


@dataclass(slots=True)
class SessionHandle:
    session_id: str
    path: Path
    messages: list[Message]
    metadata: SessionMetadata


class JsonlSessionStore:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"

    def open_session(
        self,
        requested_session: str | bool | None,
        cwd: Path,
        model: str = "",
        permission_mode: str = "default",
    ) -> SessionHandle:
        if isinstance(requested_session, str):
            metadata = self.get_metadata(requested_session)
            if metadata is None:
                raise ValueError(f"Session not found: {requested_session}")
            if metadata.legacy or not metadata.cwd:
                raise ValueError(f"Session has no cwd metadata: {requested_session}")
            return self._open_metadata(metadata, cwd_for_log=Path(metadata.cwd))

        if requested_session is True:
            metadata = self.latest_metadata()
            if metadata is not None:
                return self._open_metadata(metadata, cwd_for_log=Path(metadata.cwd))

        return self.create_session(cwd=cwd, model=model, permission_mode=permission_mode)

    def create_session(
        self,
        *,
        cwd: Path,
        model: str = "",
        permission_mode: str = "default",
    ) -> SessionHandle:
        metadata = SessionMetadata.create(
            session_id=new_id("sess"),
            root=self.root,
            cwd=cwd,
            model=model,
            permission_mode=permission_mode,
        )
        metadata.path.parent.mkdir(parents=True, exist_ok=True)
        self._append_entry(
            metadata.path,
            {
                "kind": "session_meta",
                "session_id": metadata.session_id,
                "timestamp": metadata.created_at,
                "meta": metadata.to_json(),
            },
        )
        self._save_metadata(metadata)
        LOGGER.info(
            "session.created",
            extra={
                "session_id": metadata.session_id,
                "cwd": metadata.cwd,
                "project_id": metadata.project_id,
            },
        )
        return SessionHandle(
            session_id=metadata.session_id,
            path=metadata.path,
            messages=[],
            metadata=metadata,
        )

    def switch_session(self, session_id: str) -> SessionHandle:
        metadata = self.get_metadata(session_id)
        if metadata is None:
            raise ValueError(f"Session not found: {session_id}")
        if metadata.legacy or not metadata.cwd:
            raise ValueError(f"Session has no cwd metadata: {session_id}")
        return self._open_metadata(metadata, cwd_for_log=Path(metadata.cwd))

    def append_message(self, session: SessionHandle, message: Message) -> None:
        session.messages.append(message)
        self._append_entry(
            session.path,
            {
                "kind": "message",
                "session_id": session.session_id,
                "timestamp": _now(),
                "message": message.model_dump(mode="json"),
            },
        )
        session.metadata.message_count = len(session.messages)
        self._touch_metadata(session.metadata)

    def append_messages(self, session: SessionHandle, messages: list[Message]) -> None:
        for message in messages:
            self.append_message(session, message)

    def append_event(self, session: SessionHandle, event: Any) -> None:
        payload = event.model_dump(mode="json") if hasattr(event, "model_dump") else event
        self._append_entry(
            session.path,
            {
                "kind": "event",
                "session_id": session.session_id,
                "timestamp": _now(),
                "event": payload,
            },
        )
        self._touch_metadata(session.metadata)

    def update_metadata(self, session: SessionHandle) -> None:
        session.metadata.updated_at = _now()
        self._append_entry(
            session.path,
            {
                "kind": "session_meta",
                "session_id": session.session_id,
                "timestamp": session.metadata.updated_at,
                "meta": session.metadata.to_json(),
            },
        )
        self._save_metadata(session.metadata)

    def load_messages(self, session_id: str) -> list[Message]:
        metadata = self.get_metadata(session_id)
        path = metadata.path if metadata is not None else self.root / f"{session_id}.jsonl"
        if not path.exists():
            return []

        messages: list[Message] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry.get("kind") == "message":
                    messages.append(Message.model_validate(entry["message"]))
        return messages

    def list_sessions(self) -> list[SessionMetadata]:
        return sorted(
            self._read_index().values(),
            key=lambda metadata: metadata.updated_at,
            reverse=True,
        )

    def list_session_ids(self) -> list[str]:
        return [metadata.session_id for metadata in self.list_sessions()]

    def latest_metadata(self) -> SessionMetadata | None:
        for metadata in self.list_sessions():
            if not metadata.legacy and metadata.cwd and Path(metadata.cwd).exists():
                return metadata
        return None

    def get_metadata(self, session_id: str) -> SessionMetadata | None:
        index = self._read_index()
        metadata = index.get(session_id)
        if metadata is not None:
            return metadata

        path = self.root / f"{session_id}.jsonl"
        if not path.exists():
            return None
        metadata = self._read_metadata_from_file(path)
        self._save_metadata(metadata)
        return metadata

    def _open_metadata(self, metadata: SessionMetadata, *, cwd_for_log: Path) -> SessionHandle:
        messages = self.load_messages(metadata.session_id)
        metadata.message_count = len(messages)
        self._save_metadata(metadata)
        LOGGER.info(
            "session.opened",
            extra={
                "session_id": metadata.session_id,
                "cwd": str(cwd_for_log),
                "project_id": metadata.project_id,
            },
        )
        return SessionHandle(
            session_id=metadata.session_id,
            path=metadata.path,
            messages=messages,
            metadata=metadata,
        )

    def _touch_metadata(self, metadata: SessionMetadata) -> None:
        metadata.updated_at = _now()
        self._save_metadata(metadata)

    def _save_metadata(self, metadata: SessionMetadata) -> None:
        index = self._read_index()
        index[metadata.session_id] = metadata
        self._write_index(index)

    def _read_index(self) -> dict[str, SessionMetadata]:
        if not self.index_path.exists():
            return self._rebuild_index()
        with self.index_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        raw_sessions = payload.get("sessions") if isinstance(payload, dict) else None
        if not isinstance(raw_sessions, dict):
            return self._rebuild_index()
        sessions: dict[str, SessionMetadata] = {}
        for session_id, raw_metadata in raw_sessions.items():
            if not isinstance(raw_metadata, dict):
                continue
            try:
                metadata = SessionMetadata.from_json(raw_metadata)
            except (KeyError, TypeError, ValueError):
                continue
            sessions[str(session_id)] = metadata
        return sessions

    def _write_index(self, sessions: dict[str, SessionMetadata]) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "sessions": {
                session_id: metadata.to_json()
                for session_id, metadata in sorted(sessions.items())
            },
        }
        with self.index_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    def _rebuild_index(self) -> dict[str, SessionMetadata]:
        sessions: dict[str, SessionMetadata] = {}
        for path in sorted(self.root.glob("sess_*.jsonl")):
            metadata = self._read_metadata_from_file(path)
            sessions[metadata.session_id] = metadata
        self._write_index(sessions)
        return sessions

    def _read_metadata_from_file(self, path: Path) -> SessionMetadata:
        messages = 0
        created_at = _now()
        updated_at = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
        found_metadata: SessionMetadata | None = None
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    if entry.get("kind") == "session_meta" and isinstance(entry.get("meta"), dict):
                        metadata = SessionMetadata.from_json(entry["meta"])
                        metadata.path = path.resolve()
                        found_metadata = metadata
                    if entry.get("kind") == "message":
                        messages += 1
                    created_at = str(entry.get("timestamp") or created_at)
        if found_metadata is not None:
            found_metadata.updated_at = updated_at
            found_metadata.message_count = messages
            return found_metadata
        return SessionMetadata(
            session_id=path.stem,
            path=path.resolve(),
            cwd="",
            project_id="",
            project_state_dir="",
            created_at=created_at,
            updated_at=updated_at,
            model="",
            permission_mode="default",
            message_count=messages,
            legacy=True,
        )

    def _count_messages(self, path: Path) -> int:
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry.get("kind") == "message":
                    count += 1
        return count

    def _append_entry(self, path: Path, entry: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False))
            handle.write("\n")
