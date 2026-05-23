from __future__ import annotations

from .common import *

class ReadTool(BaseTool):
    name = "Read"
    description = "Read a text file from the local workspace with an optional line offset."
    read_only = True
    input_schema = _schema(
        {
            "file_path": {"type": "string", "description": "Path to read."},
            "path": {"type": "string", "description": "Alias for file_path."},
            "offset": {"type": "integer", "minimum": 1, "description": "1-based first line."},
        },
    )

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        path = _resolve_path(context, raw_input.get("file_path") or raw_input.get("path"))
        if not path.exists():
            return _error(f"File not found: {path}", context)
        if path.is_dir():
            return _error(f"Path is a directory: {path}", context)
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        offset = int(raw_input.get("offset") or 1)
        selected = lines[offset - 1 :]
        numbered = [
            f"{line_no:>6}\t{line}"
            for line_no, line in enumerate(selected, start=offset)
        ]
        content = "\n".join(numbered)
        return _ok(
            content,
            context,
            {
                "path": str(path),
                "total_lines": len(lines),
                "returned_lines": len(selected),
            },
        )

class WriteTool(BaseTool):
    name = "Write"
    description = "Write text to a local file, creating parent directories when needed."
    input_schema = _schema(
        {
            "file_path": {"type": "string", "description": "Path to write."},
            "path": {"type": "string", "description": "Alias for file_path."},
            "content": {"type": "string", "description": "File content."},
            "append": {"type": "boolean", "default": False},
        },
        required=["content"],
    )

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        path = _resolve_path(context, raw_input.get("file_path") or raw_input.get("path"))
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if raw_input.get("append") else "w"
        with path.open(mode, encoding="utf-8", newline="") as handle:
            handle.write(str(raw_input.get("content", "")))
        action = "appended" if raw_input.get("append") else "wrote"
        return _ok(f"{action} {path}", context, {"path": str(path)})

class EditTool(BaseTool):
    name = "Edit"
    description = "Replace text in an existing local file using exact string matching."
    input_schema = _schema(
        {
            "file_path": {"type": "string", "description": "Path to edit."},
            "path": {"type": "string", "description": "Alias for file_path."},
            "old_string": {"type": "string", "description": "Exact text to replace."},
            "new_string": {"type": "string", "description": "Replacement text."},
            "replace_all": {"type": "boolean", "default": False},
        },
        required=["old_string", "new_string"],
    )

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        path = _resolve_path(context, raw_input.get("file_path") or raw_input.get("path"))
        if not path.exists():
            return _error(f"File not found: {path}", context)
        old = str(raw_input["old_string"])
        new = str(raw_input["new_string"])
        if old == "":
            return _error("old_string must not be empty", context)
        text = path.read_text(encoding="utf-8", errors="replace")
        count = text.count(old)
        if count == 0:
            return _error("old_string was not found", context, {"path": str(path)})
        if count > 1 and not raw_input.get("replace_all"):
            return _error(
                f"old_string appears {count} times; set replace_all=true to replace all matches",
                context,
                {"path": str(path), "matches": count},
            )
        updated = text.replace(old, new) if raw_input.get("replace_all") else text.replace(old, new, 1)
        path.write_text(updated, encoding="utf-8", newline="")
        replaced = count if raw_input.get("replace_all") else 1
        return _ok(
            f"edited {path}; replacements={replaced}",
            context,
            {"path": str(path), "replacements": replaced},
        )

class NotebookEditTool(BaseTool):
    name = "NotebookEdit"
    description = "Edit a Jupyter notebook cell by replacing, inserting, or deleting cells."
    input_schema = _schema(
        {
            "notebook_path": {"type": "string", "description": "Path to .ipynb file."},
            "file_path": {"type": "string", "description": "Alias for notebook_path."},
            "cell_index": {"type": "integer", "minimum": 0},
            "action": {"type": "string", "enum": ["replace", "insert", "delete"], "default": "replace"},
            "cell_type": {"type": "string", "enum": ["code", "markdown", "raw"], "default": "code"},
            "source": {"description": "Cell source as a string or list of lines."},
        },
        required=["cell_index"],
    )

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        path = _resolve_path(
            context,
            raw_input.get("notebook_path") or raw_input.get("file_path"),
        )
        if not path.exists():
            return _error(f"Notebook not found: {path}", context)
        notebook = _read_json_file(path, {})
        cells = notebook.setdefault("cells", [])
        if not isinstance(cells, list):
            return _error("Invalid notebook: cells is not a list", context)
        index = int(raw_input["cell_index"])
        action = str(raw_input.get("action") or "replace")
        if action in {"replace", "delete"} and not 0 <= index < len(cells):
            return _error(f"cell_index out of range: {index}", context)
        if action == "insert" and not 0 <= index <= len(cells):
            return _error(f"cell_index out of range: {index}", context)

        if action == "delete":
            deleted = cells.pop(index)
            _write_json_file(path, notebook)
            return _ok(
                f"deleted cell {index} from {path}",
                context,
                {"path": str(path), "deleted_cell_type": deleted.get("cell_type")},
            )

        cell = {
            "cell_type": raw_input.get("cell_type") or "code",
            "metadata": {},
            "source": _coerce_source(raw_input.get("source")),
        }
        if cell["cell_type"] == "code":
            cell.update({"execution_count": None, "outputs": []})
        if action == "insert":
            cells.insert(index, cell)
        else:
            old_metadata = cells[index].get("metadata", {}) if isinstance(cells[index], dict) else {}
            cell["metadata"] = old_metadata
            cells[index] = cell
        _write_json_file(path, notebook)
        action_label = "inserted" if action == "insert" else "replaced"
        return _ok(
            f"{action_label} cell {index} in {path}",
            context,
            {"path": str(path), "cell_index": index, "action": action},
        )

class GlobTool(BaseTool):
    name = "Glob"
    description = "Find files by glob pattern under a path."
    read_only = True
    input_schema = _schema(
        {
            "pattern": {"type": "string"},
            "path": {"type": "string", "default": "."},
        },
        required=["pattern"],
    )

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        root = _resolve_path(context, raw_input.get("path"), default=".")
        matches = []
        for item in root.glob(str(raw_input["pattern"])):
            matches.append(str(item.resolve()))
        return _ok(
            "\n".join(matches) if matches else "(no matches)",
            context,
            {"matches": matches, "count": len(matches)},
        )

class GrepTool(BaseTool):
    name = "Grep"
    description = "Search local text files with a regular expression."
    read_only = True
    input_schema = _schema(
        {
            "pattern": {"type": "string"},
            "path": {"type": "string", "default": "."},
            "include": {"type": "string", "description": "Glob include filter such as *.py."},
            "exclude": {"type": "string", "description": "Glob exclude filter."},
            "case_sensitive": {"type": "boolean", "default": True},
        },
        required=["pattern"],
    )

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        root = _resolve_path(context, raw_input.get("path"), default=".")
        flags = 0 if raw_input.get("case_sensitive", True) else re.IGNORECASE
        try:
            regex = re.compile(str(raw_input["pattern"]), flags)
        except re.error as exc:
            return _error(f"Invalid regex: {exc}", context)
        include = raw_input.get("include")
        exclude = raw_input.get("exclude")
        results: list[JsonObject] = []
        files = [root] if root.is_file() else [path for path in root.rglob("*") if path.is_file()]
        for path in files:
            rel = str(path.relative_to(root)) if root.is_dir() else path.name
            if include and not fnmatch.fnmatch(path.name, str(include)) and not fnmatch.fnmatch(rel, str(include)):
                continue
            if exclude and (fnmatch.fnmatch(path.name, str(exclude)) or fnmatch.fnmatch(rel, str(exclude))):
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue
            for line_no, line in enumerate(lines, start=1):
                if regex.search(line):
                    results.append({"path": str(path), "line": line_no, "text": line})
        content = "\n".join(f"{item['path']}:{item['line']}: {item['text']}" for item in results)
        return _ok(content or "(no matches)", context, {"matches": results, "count": len(results)})
