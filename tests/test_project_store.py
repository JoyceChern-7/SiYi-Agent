from __future__ import annotations

import json
import threading
from pathlib import Path

from runtime.project_store import ProjectStore
from runtime.session_store import JsonlSessionStore


def test_project_store_reuses_project_for_same_root(tmp_path: Path) -> None:
    project_root = tmp_path / "workspace"
    other_root = tmp_path / "other"
    project_root.mkdir()
    other_root.mkdir()
    store = ProjectStore(tmp_path / "projects")

    first = store.ensure_project(project_root)
    second = store.ensure_project(project_root)
    other = store.ensure_project(other_root)

    assert first.project_id == second.project_id
    assert first.project_id != other.project_id
    assert Path(first.project_state_dir).exists()


def test_session_metadata_uses_project_and_omits_cwd_model_and_project_state(tmp_path: Path) -> None:
    project_root = tmp_path / "workspace"
    project_root.mkdir()
    project_store = ProjectStore(tmp_path / "projects")
    session_store = JsonlSessionStore(tmp_path / "sessions", project_store=project_store)
    project = project_store.ensure_project(project_root)

    session = session_store.create_session(project=project)
    payload = json.loads(session_store.index_path.read_text(encoding="utf-8"))
    metadata = payload["sessions"][session.session_id]

    assert metadata["project_id"] == project.project_id
    assert metadata["name"] == "新会话"
    assert metadata["name_status"] == "pending"
    assert "cwd" not in metadata
    assert "model" not in metadata
    assert "project_state_dir" not in metadata


def test_legacy_session_cwd_is_migrated_to_project(tmp_path: Path) -> None:
    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    project_store = ProjectStore(tmp_path / "projects")
    session_store = JsonlSessionStore(tmp_path / "sessions", project_store=project_store)
    session_path = session_store.root / "sess_legacy.jsonl"
    session_path.write_text(
        json.dumps(
            {
                "kind": "session_meta",
                "session_id": "sess_legacy",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "meta": {
                    "session_id": "sess_legacy",
                    "path": str(session_path),
                    "cwd": str(legacy_root),
                    "project_state_dir": str(tmp_path / "old-state"),
                    "model": "old-model",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    metadata = session_store.get_metadata("sess_legacy")
    assert metadata is not None
    project = session_store.get_project_for_session(metadata)

    assert project is not None
    assert Path(project.project_root) == legacy_root.resolve()
    assert metadata.name == "新会话"
    assert metadata.name_status == "ready"


def test_session_index_writes_are_locked(tmp_path: Path) -> None:
    project_root = tmp_path / "workspace"
    project_root.mkdir()
    project_store = ProjectStore(tmp_path / "projects")
    session_store = JsonlSessionStore(tmp_path / "sessions", project_store=project_store)
    project = project_store.ensure_project(project_root)
    sessions = [session_store.create_session(project=project) for _ in range(8)]

    def update_name(index: int) -> None:
        session_store.set_generated_name(sessions[index], f"session {index}")

    threads = [threading.Thread(target=update_name, args=(index,)) for index in range(len(sessions))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    names = {metadata.name for metadata in session_store.list_sessions()}
    assert {f"session {index}" for index in range(len(sessions))} <= names
