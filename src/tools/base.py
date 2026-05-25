from __future__ import annotations

from typing import Any, Literal, Protocol

from jsonschema import Draft7Validator

from pydantic import BaseModel

from engine.message_schema import ToolResultBlock
from tools.governance import PermissionResult, ValidationResult


class ToolContext(BaseModel):
    project_root: str
    trace_id: str
    session_id: str | None = None
    turn_id: str | None = None
    max_result_chars: int = 20_000
    tool_use_id: str | None = None
    tool_name: str | None = None
    progress_queue: Any | None = None


class ToolResult(BaseModel):
    success: bool
    content: str
    data: dict[str, Any] | None = None
    error: str | None = None


class Tool(Protocol):
    name: str
    description: str
    input_schema: dict[str, Any]
    model_visible: bool

    async def run(self, raw_input: dict[str, Any], context: ToolContext) -> ToolResult:
        ...

    def validate_input(self, raw_input: dict[str, Any], context: ToolContext) -> ValidationResult:
        ...

    def check_permissions(self, raw_input: dict[str, Any], context: ToolContext) -> PermissionResult:
        ...

    def is_read_only(self, raw_input: dict[str, Any]) -> bool:
        ...

    def is_concurrency_safe(self, raw_input: dict[str, Any]) -> bool:
        ...

    def to_tool_result_block(self, tool_use_id: str, result: ToolResult) -> ToolResultBlock:
        ...


class BaseTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    model_visible: bool = True
    read_only: bool = False

    def validate_input(
        self,
        raw_input: dict[str, Any],
        context: ToolContext,
    ) -> ValidationResult:
        del context
        errors = sorted(
            Draft7Validator(self.input_schema).iter_errors(raw_input),
            key=lambda error: list(error.path),
        )
        if not errors:
            return ValidationResult.allow()
        error = errors[0]
        path = ".".join(str(part) for part in error.path)
        prefix = f"{path}: " if path else ""
        return ValidationResult.reject(f"{prefix}{error.message}")

    def check_permissions(
        self,
        raw_input: dict[str, Any],
        context: ToolContext,
    ) -> PermissionResult:
        del raw_input, context
        return PermissionResult.allow(source="tool")

    def is_read_only(self, raw_input: dict[str, Any]) -> bool:
        del raw_input
        return self.read_only

    def is_concurrency_safe(self, raw_input: dict[str, Any]) -> bool:
        return self.is_read_only(raw_input)

    def to_tool_result_block(
        self,
        tool_use_id: str,
        result: ToolResult,
    ) -> ToolResultBlock:
        content = result.content
        if result.data is not None:
            content = content or str(result.data)
        if result.error:
            content = f"{content}\n{result.error}".strip()
        return ToolResultBlock(
            tool_use_id=tool_use_id,
            content=content,
            is_error=not result.success,
        )


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}\n...[truncated {omitted} chars]"


async def emit_tool_output_delta(
    context: ToolContext,
    *,
    stream: Literal["stdout", "stderr", "status"],
    delta: str,
    process_id: str | None = None,
    elapsed_ms: int = 0,
) -> None:
    if not delta or context.progress_queue is None:
        return
    if context.tool_use_id is None or context.tool_name is None:
        return

    from engine.events import ToolOutputDeltaEvent

    await context.progress_queue.put(
        ToolOutputDeltaEvent(
            session_id=context.session_id,
            turn_id=context.turn_id,
            tool_use_id=context.tool_use_id,
            tool_name=context.tool_name,
            stream=stream,
            delta=delta,
            process_id=process_id,
            elapsed_ms=elapsed_ms,
        )
    )
