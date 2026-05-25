from __future__ import annotations

import os
import re
from hashlib import sha256
from pathlib import Path


def get_siyi_config_home() -> Path:
    raw_path = os.environ.get("SIYI_CONFIG_DIR")
    if raw_path:
        return Path(raw_path).expanduser().resolve()
    return (Path.home() / ".SiYi").resolve()


def get_siyi_home() -> Path:
    return (Path.home() / ".siyi").resolve()


def get_siyi_config_path() -> Path:
    return (get_siyi_home() / "config.json").resolve()


def get_global_permissions_path() -> Path:
    return (get_siyi_home() / "permissions.json").resolve()


def get_projects_dir() -> Path:
    return (get_siyi_home() / "projects").resolve()


def get_project_id(project_root: Path | str) -> str:
    resolved = Path(project_root).expanduser().resolve()
    safe_name = _safe_project_name(resolved.name or "workspace")
    digest = sha256(os.path.normcase(str(resolved)).encode("utf-8")).hexdigest()[:10]
    return f"{safe_name}-{digest}"


def get_project_state_dir(project_root: Path | str) -> Path:
    return (get_projects_dir() / get_project_id(project_root)).resolve()


def get_session_dir() -> Path:
    raw_path = os.environ.get("SIYI_SESSION_DIR")
    if raw_path:
        return Path(raw_path).expanduser().resolve()
    return (get_siyi_home() / "sessions").resolve()


def get_user_settings_path() -> Path:
    return get_siyi_config_home() / "settings.json"


def get_global_skills_dir() -> Path:
    return get_siyi_home() / "skills"


def get_skill_paths_path() -> Path:
    return get_siyi_home() / "skill_paths.json"


def _safe_project_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-._")
    if not safe:
        return "workspace"
    return safe[:64].rstrip("-._") or "workspace"
