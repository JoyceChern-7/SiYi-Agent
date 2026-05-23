from __future__ import annotations

from .common import *

class ConfigTool(BaseTool):
    name = "Config"
    description = "Get, set, list, or delete local SiYi config values stored in the SiYi project state directory."
    input_schema = _schema(
        {
            "action": {"type": "string", "enum": ["get", "set", "list", "delete"], "default": "list"},
            "key": {"type": "string"},
            "value": {},
        },
    )

    def is_read_only(self, raw_input: JsonObject) -> bool:
        return str(raw_input.get("action") or "list") in {"get", "list"}

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        path = _state_dir(context) / "config.json"
        config = _read_json_file(path, {})
        if not isinstance(config, dict):
            config = {}
        action = str(raw_input.get("action") or "list")
        key = raw_input.get("key")
        if action == "list":
            return _ok(json.dumps(config, ensure_ascii=False, indent=2), context, {"config": config})
        if not key:
            return _error("Config action requires key", context)
        if action == "get":
            return _ok(json.dumps(config.get(key), ensure_ascii=False), context, {"key": key, "value": config.get(key)})
        if action == "delete":
            existed = key in config
            config.pop(key, None)
            _write_json_file(path, config)
            return _ok(f"deleted={existed}", context, {"key": key, "deleted": existed})
        config[str(key)] = raw_input.get("value")
        _write_json_file(path, config)
        return _ok(f"set {key}", context, {"key": key, "value": config[str(key)]})
