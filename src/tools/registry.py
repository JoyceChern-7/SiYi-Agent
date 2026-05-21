from __future__ import annotations

from dataclasses import dataclass, field

from runtime.permissions import PermissionManager
from tools.base import Tool


@dataclass(slots=True)
class ToolRegistry:
    permission_manager: PermissionManager
    tools: dict[str, Tool] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)

    @classmethod
    def default(cls, permission_manager: PermissionManager) -> "ToolRegistry":
        registry = cls(permission_manager=permission_manager)
        from tools.builtin import register_builtin_tools

        register_builtin_tools(registry)
        return registry

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def register_alias(self, alias: str, target_name: str) -> None:
        self.aliases[alias] = target_name

    def get_tools(self) -> list[Tool]:
        return list(self.tools.values())

    def get_model_tools(self) -> list[Tool]:
        return [
            tool
            for tool in self.get_tools()
            if getattr(tool, "model_visible", True)
        ]

    def get_aliases(self) -> dict[str, str]:
        return dict(self.aliases)

    def aliases_for(self, target_name: str) -> list[str]:
        return [
            alias
            for alias, candidate_target_name in self.aliases.items()
            if candidate_target_name == target_name
        ]

    def canonical_name_for(self, name: str) -> str:
        return self.aliases.get(name, name)

    def find_tool(self, name: str) -> Tool | None:
        tool = self.tools.get(name)
        if tool is not None:
            return tool
        target_name = self.aliases.get(name)
        if target_name is None:
            return None
        return self.tools.get(target_name)

    def to_model_tool_schemas(self) -> list[dict]:
        schemas: list[dict] = []
        for tool in self.get_model_tools():
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                }
            )
        return schemas
