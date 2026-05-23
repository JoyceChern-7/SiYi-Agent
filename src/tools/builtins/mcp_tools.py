from __future__ import annotations

from .common import *

class ListMcpResourcesTool(BaseTool):
    name = "ListMcpResourcesTool"
    description = "List local MCP resource descriptors from the SiYi project state directory."
    read_only = True
    input_schema = _schema({"server": {"type": "string"}})

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        del raw_input
        resources = _local_mcp_resources(context)
        content = "\n".join(f"{item.get('uri')}: {item.get('name', '')}" for item in resources)
        return _ok(content or "(no local MCP resources)", context, {"resources": resources})

class ReadMcpResourceTool(BaseTool):
    name = "ReadMcpResourceTool"
    description = "Read a local MCP resource by uri from the SiYi project state directory."
    read_only = True
    input_schema = _schema({"uri": {"type": "string"}, "server": {"type": "string"}}, required=["uri"])

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        uri = str(raw_input["uri"])
        for item in _local_mcp_resources(context):
            if item.get("uri") == uri:
                if "text" in item:
                    return _ok(str(item["text"]), context, item)
                if "path" in item:
                    path = _resolve_path(context, str(item["path"]))
                    return _ok(path.read_text(encoding="utf-8", errors="replace"), context, item)
        parsed = urlparse(uri)
        if parsed.scheme == "file":
            path = Path(unquote(parsed.path)).resolve()
            return _ok(path.read_text(encoding="utf-8", errors="replace"), context, {"uri": uri})
        return _error(f"MCP resource not found: {uri}", context)

def _local_mcp_resources(context: ToolContext) -> list[JsonObject]:
    base = _state_dir(context) / "mcp"
    resources: list[JsonObject] = []
    descriptor = base / "resources.json"
    if descriptor.exists():
        data = _read_json_file(descriptor, [])
        if isinstance(data, list):
            resources.extend(item for item in data if isinstance(item, dict))
        elif isinstance(data, dict) and isinstance(data.get("resources"), list):
            resources.extend(item for item in data["resources"] if isinstance(item, dict))
    resource_dir = base / "resources"
    if resource_dir.exists():
        for path in resource_dir.rglob("*"):
            if path.is_file():
                resources.append({"uri": path.as_uri(), "name": path.name, "path": str(path)})
    return resources
