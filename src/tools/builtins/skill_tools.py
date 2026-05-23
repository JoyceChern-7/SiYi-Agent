from __future__ import annotations

from .common import *

class SkillTool(BaseTool):
    name = "Skill"
    description = "List, read, or search local skill files from global SiYi skill roots."
    read_only = True
    input_schema = _schema(
        {
            "action": {"type": "string", "enum": ["list", "read", "search"], "default": "list"},
            "name": {"type": "string"},
            "path": {"type": "string"},
            "query": {"type": "string"},
        },
    )

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        action = str(raw_input.get("action") or "list")
        roots = _skill_roots(context)
        skills = _discover_skills(roots)
        if action == "list":
            content = "\n".join(f"{item['name']}: {item['path']}" for item in skills)
            return _ok(content or "(no skills found)", context, {"skills": skills})
        if action == "search":
            query = str(raw_input.get("query") or "").lower()
            matches = [
                item
                for item in skills
                if query in item["name"].lower()
                or query in item["path"].lower()
                or query in item["root"].lower()
                or query in item["source"].lower()
            ]
            return _ok(
                "\n".join(f"{item['name']}: {item['path']}" for item in matches) or "(no matches)",
                context,
                {"skills": matches},
            )
        target = raw_input.get("path")
        if not target and raw_input.get("name"):
            target_item = next((item for item in skills if item["name"] == raw_input["name"]), None)
            target = target_item["path"] if target_item else None
        if not target:
            return _error("Skill read requires name or path", context)
        path = _resolve_path(context, str(target)) if not Path(str(target)).is_absolute() else Path(str(target))
        if not path.exists():
            return _error(f"Skill file not found: {path}", context)
        text = path.read_text(encoding="utf-8", errors="replace")
        return _ok(text, context, {"path": str(path)})

def _skill_roots(context: ToolContext) -> list[Path]:
    del context
    root_entries: list[tuple[Path, str]] = [(get_global_skills_dir(), "default")]
    root_entries.extend((path, "configured") for path in _load_skill_paths())
    env_root = os.environ.get("SIYI_SKILLS_DIR")
    if env_root:
        root_entries.append((Path(env_root).expanduser(), "env"))
    return [
        root.resolve()
        for root, _source in _dedupe_skill_roots(root_entries)
        if root.exists() and root.is_dir()
    ]

def _load_skill_paths() -> list[Path]:
    path = get_skill_paths_path()
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    raw_paths = payload.get("paths") if isinstance(payload, dict) else None
    if not isinstance(raw_paths, list):
        return []
    return [
        Path(str(raw_path)).expanduser()
        for raw_path in raw_paths
        if raw_path
    ]

def _dedupe_skill_roots(root_entries: list[tuple[Path, str]]) -> list[tuple[Path, str]]:
    seen: set[str] = set()
    result: list[tuple[Path, str]] = []
    for root, source in root_entries:
        resolved = root.expanduser().resolve()
        key = os.path.normcase(str(resolved))
        if key in seen:
            continue
        seen.add(key)
        result.append((resolved, source))
    return result

def _discover_skills(roots: list[Path]) -> list[JsonObject]:
    skills = []
    for root in roots:
        for path in root.rglob("SKILL.md"):
            skills.append(
                {
                    "name": path.parent.name,
                    "path": str(path),
                    "root": str(root),
                    "source": _skill_root_source(root),
                }
            )
    return sorted(skills, key=lambda item: item["name"])

def _skill_root_source(root: Path) -> str:
    root = root.resolve()
    for candidate, source in _dedupe_skill_roots(
        [(get_global_skills_dir(), "default")]
        + [(path, "configured") for path in _load_skill_paths()]
        + ([(Path(os.environ["SIYI_SKILLS_DIR"]).expanduser(), "env")] if os.environ.get("SIYI_SKILLS_DIR") else [])
    ):
        if candidate == root:
            return source
    return "unknown"
