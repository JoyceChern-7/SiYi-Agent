from __future__ import annotations

from .common import *

class WorkflowTool(BaseTool):
    name = "workflow"
    description = "List, get, or run JSON workflow definitions from the SiYi project state directory."
    input_schema = _schema(
        {
            "action": {"type": "string", "enum": ["list", "get", "run"], "default": "list"},
            "name": {"type": "string"},
            "steps": {"type": "array", "items": {"type": "object"}},
        },
    )

    def __init__(self, registry_provider: Callable[[], Any]) -> None:
        self._registry_provider = registry_provider

    def is_read_only(self, raw_input: JsonObject) -> bool:
        return str(raw_input.get("action") or "list") in {"list", "get"}

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        action = str(raw_input.get("action") or "list")
        workflows = _discover_workflows(context)
        if action == "list":
            content = "\n".join(f"{item['name']}: {item['path']}" for item in workflows)
            return _ok(content or "(no workflows found)", context, {"workflows": workflows})
        definition = None
        if raw_input.get("name"):
            definition = _load_workflow_definition(context, str(raw_input["name"]))
            if definition is None:
                return _error(f"Workflow not found: {raw_input['name']}", context)
        if action == "get":
            return _ok(json.dumps(definition, ensure_ascii=False, indent=2), context, definition)
        steps = raw_input.get("steps") or (definition or {}).get("steps") or []
        if not isinstance(steps, list):
            return _error("workflow steps must be a list", context)
        outputs = []
        registry = self._registry_provider()
        for index, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                outputs.append({"step": index, "success": False, "content": "step is not an object"})
                continue
            tool_name = step.get("tool")
            tool_input = step.get("input") or {}
            tool = registry.find_tool(tool_name) if tool_name else None
            if tool is None:
                outputs.append({"step": index, "tool": tool_name, "success": False, "content": "tool not found"})
                continue
            if not isinstance(tool_input, dict):
                outputs.append({"step": index, "tool": tool_name, "success": False, "content": "tool input is not an object"})
                continue
            validation = tool.validate_input(tool_input, context)
            if not validation.ok:
                outputs.append({"step": index, "tool": tool_name, "success": False, "content": validation.reason})
                continue
            permission = await registry.permission_manager.authorize(tool, tool_input, context)
            if permission.decision != "allow":
                outputs.append({"step": index, "tool": tool_name, "success": False, "content": permission.reason})
                continue
            result = await tool.run(tool_input, context)
            outputs.append({"step": index, "tool": tool_name, **result.model_dump()})
        return _ok(json.dumps(outputs, ensure_ascii=False, indent=2), context, {"steps": outputs})

def _discover_workflows(context: ToolContext) -> list[JsonObject]:
    root = _state_dir(context) / "workflows"
    if not root.exists():
        return []
    return [{"name": path.stem, "path": str(path)} for path in sorted(root.glob("*.json"))]

def _load_workflow_definition(context: ToolContext, name: str) -> JsonObject | None:
    root = _state_dir(context) / "workflows"
    path = root / f"{name}.json"
    if not path.exists():
        return None
    data = _read_json_file(path, {})
    return data if isinstance(data, dict) else None
