from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.paths import get_project_id, get_project_state_dir, get_projects_dir


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class ProjectMetadata:
    project_id: str
    project_root: str
    project_state_dir: str
    created_at: str
    updated_at: str

    @classmethod
    def create(cls, project_root: Path) -> "ProjectMetadata":
        root = project_root.expanduser().resolve()
        timestamp = _now()
        return cls(
            project_id=get_project_id(root),
            project_root=str(root),
            project_state_dir=str(get_project_state_dir(root)),
            created_at=timestamp,
            updated_at=timestamp,
        )

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "ProjectMetadata":
        project_root = Path(str(payload["project_root"])).expanduser().resolve()
        project_id = str(payload.get("project_id") or get_project_id(project_root))
        project_state_dir = str(
            payload.get("project_state_dir") or get_project_state_dir(project_root)
        )
        return cls(
            project_id=project_id,
            project_root=str(project_root),
            project_state_dir=project_state_dir,
            created_at=str(payload.get("created_at") or _now()),
            updated_at=str(payload.get("updated_at") or payload.get("created_at") or _now()),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "project_root": self.project_root,
            "project_state_dir": self.project_state_dir,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class ProjectStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or get_projects_dir()).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        self._lock = threading.RLock()

    def ensure_project(self, project_root: Path) -> ProjectMetadata:
        root = project_root.expanduser().resolve()
        if not root.exists() or not root.is_dir():
            raise ValueError(f"Project root does not exist or is not a directory: {root}")
        project_id = get_project_id(root)
        with self._lock:
            projects = self._read_index_unlocked()
            metadata = projects.get(project_id)
            if metadata is None:
                metadata = ProjectMetadata.create(root)
            else:
                metadata.project_root = str(root)
                metadata.project_state_dir = str(get_project_state_dir(root))
                metadata.updated_at = _now()
            Path(metadata.project_state_dir).mkdir(parents=True, exist_ok=True)
            projects[project_id] = metadata
            self._write_index_unlocked(projects)
            return metadata

    def get_project(self, project_id: str) -> ProjectMetadata | None:
        with self._lock:
            return self._read_index_unlocked().get(project_id)

    def list_projects(self) -> list[ProjectMetadata]:
        with self._lock:
            return sorted(
                self._read_index_unlocked().values(),
                key=lambda metadata: metadata.updated_at,
                reverse=True,
            )

    def _read_index_unlocked(self) -> dict[str, ProjectMetadata]:
        if not self.index_path.exists():
            return {}
        with self.index_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        raw_projects = raw.get("projects", raw) if isinstance(raw, dict) else {}
        projects: dict[str, ProjectMetadata] = {}
        if not isinstance(raw_projects, dict):
            return projects
        for project_id, raw_metadata in raw_projects.items():
            if not isinstance(raw_metadata, dict):
                continue
            try:
                metadata = ProjectMetadata.from_json(raw_metadata)
            except Exception:
                continue
            projects[str(project_id)] = metadata
        return projects

    def _write_index_unlocked(self, projects: dict[str, ProjectMetadata]) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "projects": {
                project_id: metadata.to_json()
                for project_id, metadata in sorted(projects.items())
            }
        }
        tmp_path = self.index_path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(tmp_path, self.index_path)
