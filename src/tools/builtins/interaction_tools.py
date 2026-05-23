from __future__ import annotations

from .common import *

class ToolSearchTool(BaseTool):
    name = "ToolSearch"
    description = "Search the currently available SiYi tools by name and description."
    read_only = True
    input_schema = _schema(
        {
            "query": {"type": "string", "default": ""},
            "max_results": {"type": "integer", "minimum": 1, "default": 50},
        },
    )

    def __init__(
        self,
        registry_provider: Callable[[], list[Any]],
        alias_provider: Callable[[], dict[str, str]] | None = None,
    ) -> None:
        self._registry_provider = registry_provider
        self._alias_provider = alias_provider or (lambda: {})

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        query = str(raw_input.get("query") or "").lower()
        max_results = int(raw_input.get("max_results") or 50)
        matches = []
        aliases_by_target: dict[str, list[str]] = {}
        for alias, target_name in self._alias_provider().items():
            aliases_by_target.setdefault(target_name, []).append(alias)
        for tool in self._registry_provider():
            aliases = sorted(aliases_by_target.get(tool.name, []))
            haystack = f"{tool.name} {' '.join(aliases)} {tool.description}".lower()
            if not query or query in haystack:
                matches.append(
                    {
                        "name": tool.name,
                        "aliases": aliases,
                        "description": tool.description,
                        "input_schema": tool.input_schema,
                    }
                )
            if len(matches) >= max_results:
                break
        content = "\n".join(
            _format_tool_search_item(item)
            for item in matches
        )
        return _ok(content or "(no matching tools)", context, {"tools": matches})

def _format_tool_search_item(item: JsonObject) -> str:
    aliases = item.get("aliases") or []
    alias_text = f" (aliases: {', '.join(aliases)})" if aliases else ""
    return f"{item['name']}{alias_text}: {item['description']}"

class AskUserQuestionTool(BaseTool):
    name = "AskUserQuestion"
    description = "Ask the human user a question and return a structured pending-question result."
    read_only = True
    input_schema = _schema(
        {
            "question": {"type": "string"},
            "choices": {"type": "array", "items": {"type": "string"}},
            "allow_freeform": {"type": "boolean", "default": True},
        },
        required=["question"],
    )

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        choices = raw_input.get("choices") or []
        lines = [f"Question for user: {raw_input['question']}"]
        if choices:
            lines.append("Choices:")
            lines.extend(f"- {choice}" for choice in choices)
        return _ok(
            "\n".join(lines),
            context,
            {
                "requires_user_input": True,
                "question": raw_input["question"],
                "choices": choices,
                "allow_freeform": bool(raw_input.get("allow_freeform", True)),
            },
        )

class EnterPlanModeTool(BaseTool):
    name = "EnterPlanMode"
    description = "Enter planning mode and persist the current plan state in the SiYi project state directory."
    input_schema = _schema(
        {
            "plan": {"type": "string", "description": "Optional plan text."},
            "reason": {"type": "string"},
        },
    )

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        state = {
            "active": True,
            "plan": raw_input.get("plan") or "",
            "reason": raw_input.get("reason") or "",
            "updated_at": time.time(),
        }
        _write_json_file(_state_dir(context) / "plan_mode.json", state)
        return _ok("entered plan mode", context, state)

class ExitPlanModeTool(BaseTool):
    name = "ExitPlanMode"
    description = "Exit planning mode and persist the current plan state in the SiYi project state directory."
    input_schema = _schema({"reason": {"type": "string"}})

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        state = {
            "active": False,
            "reason": raw_input.get("reason") or "",
            "updated_at": time.time(),
        }
        _write_json_file(_state_dir(context) / "plan_mode.json", state)
        return _ok("exited plan mode", context, state)

class SendUserMessageTool(BaseTool):
    name = "SendUserMessage"
    description = "Send a visible message to the user as a tool result."
    read_only = True
    input_schema = _schema({"message": {"type": "string"}}, required=["message"])

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        return _ok(str(raw_input["message"]), context, {"message": raw_input["message"]})
