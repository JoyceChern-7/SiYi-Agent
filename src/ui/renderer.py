from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich.text import Text

from engine.query_engine import SessionSnapshot
from runtime.read_model import SessionSummary
from engine.events import (
    AgentError,
    AssistantDeltaEvent,
    AssistantMessageEvent,
    ApprovalResolvedEvent,
    QueryEvent,
    PermissionRequestEvent,
    ToolOutputDeltaEvent,
    ToolResultEvent,
    ToolCallEvent,
    TurnCompletedEvent,
    TurnStartedEvent,
)
from engine.message_schema import Message


class ConsoleRenderer:
    def __init__(self, debug: bool = False, print_thinking: bool = False) -> None:
        self.console = Console()
        self.debug = debug
        self.print_thinking = print_thinking
        self._assistant_open = False
        self._tool_output_open = False

    def render_event(self, event: QueryEvent) -> None:
        if isinstance(event, TurnStartedEvent):
            self.render_turn_started(event)
        elif isinstance(event, AssistantDeltaEvent):
            self.render_assistant_delta(event.delta)
        elif isinstance(event, AssistantMessageEvent):
            self.render_assistant_done()
        elif isinstance(event, ToolCallEvent):
            self.render_tool_call(event)
        elif isinstance(event, ToolOutputDeltaEvent):
            self.render_tool_output_delta(event)
        elif isinstance(event, PermissionRequestEvent):
            self.render_permission_request(event)
        elif isinstance(event, ApprovalResolvedEvent):
            self.render_approval_resolved(event)
        elif isinstance(event, ToolResultEvent):
            self.render_tool_result(event)
        elif isinstance(event, TurnCompletedEvent):
            self.render_turn_completed(event)
        elif isinstance(event, AgentError):
            self.render_error(event.message)

    def render_welcome(self, *, session_id: str, project_root: str, model: str) -> None:
        self.console.print("[bold]SiYi[/bold] interactive mode")
        self.console.print(
            f"[dim]session={session_id} model={model} project_root={project_root}[/dim]"
        )
        self.console.print("[dim]Type /help for available commands.[/dim]")

    def render_user_message(self, text: str) -> None:
        self.console.print(Text(f"> {text}", style="dim"))

    def render_assistant_delta(self, delta: str) -> None:
        if not self._assistant_open:
            self.console.print(Text("● ", style="bold green"), end="")
            self._assistant_open = True
        self.console.print(delta, end="")

    def render_assistant_done(self) -> None:
        if self._assistant_open:
            self.console.print()
            self._assistant_open = False

    def render_tool_call(self, event: ToolCallEvent) -> None:
        self.render_assistant_done()
        self.console.print(f"[dim yellow]tool call:[/dim yellow] {event.block.name} {event.block.input}")

    def render_tool_result(self, event: ToolResultEvent) -> None:
        self.render_assistant_done()
        if self._tool_output_open:
            self.console.print()
            self._tool_output_open = False
        style = "dim red" if event.block.is_error else "dim green"
        self.console.print(f"[{style}]tool result:[/{style}] {event.block.content}")

    def render_tool_output_delta(self, event: ToolOutputDeltaEvent) -> None:
        self.render_assistant_done()
        style = "dim red" if event.stream == "stderr" else "dim"
        self.console.print(Text(event.delta, style=style), end="")
        self._tool_output_open = True

    def render_status(self, message: str) -> None:
        if self.debug:
            self.console.print(f"[dim]{message}[/dim]")

    def render_turn_started(self, event: TurnStartedEvent) -> None:
        self.render_status(f"turn={event.turn_index}")

    def render_permission_request(self, event: PermissionRequestEvent) -> None:
        self.render_assistant_done()
        self.console.print(
            f"[yellow]permission requested:[/yellow] {event.tool_name} {event.tool_input}"
        )

    def render_approval_resolved(self, event: ApprovalResolvedEvent) -> None:
        self.render_status(
            f"approval {event.approval_id}: {'approved' if event.approved else 'denied'}"
        )

    def render_turn_completed(self, event: TurnCompletedEvent) -> None:
        if event.status == "failed" and event.error is not None:
            self.render_error(event.error.message)
            return
        if event.status == "interrupted":
            self.render_note("turn interrupted")
            return
        if self.debug and event.usage:
            self.console.print(f"[dim]usage={event.usage.model_dump()}[/dim]")

    def render_error(self, message: str) -> None:
        self.render_assistant_done()
        self.console.print(f"[bold red]error:[/bold red] {message}")

    def render_note(self, message: str) -> None:
        self.render_assistant_done()
        self.console.print(f"[dim]{message}[/dim]")

    def render_help(self, commands: list[object]) -> None:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Command", style="cyan", no_wrap=True)
        table.add_column("Description")
        for command in commands:
            table.add_row(command.usage, command.description)
        self.console.print(table)

    def render_session(self, snapshot: SessionSnapshot) -> None:
        table = Table(show_header=False, box=None)
        table.add_column("Field", style="cyan", no_wrap=True)
        table.add_column("Value")
        table.add_row("session", snapshot.session_id)
        table.add_row("name", snapshot.name)
        table.add_row("name_status", snapshot.name_status)
        table.add_row("project_root", snapshot.project_root)
        table.add_row("project", snapshot.project_id)
        table.add_row("project_state", snapshot.project_state_dir)
        table.add_row("session_path", snapshot.session_path)
        table.add_row("permission_mode", snapshot.permission_mode)
        table.add_row("turns", str(snapshot.turn_count))
        table.add_row("completed_turns", str(snapshot.completed_turns))
        table.add_row("messages", str(snapshot.message_count))
        table.add_row("last_error", snapshot.last_error.message if snapshot.last_error else "-")
        table.add_row("usage", str(snapshot.total_usage))
        table.add_row("estimated_cost", f"{snapshot.estimated_total_cost:.6f}")
        self.console.print(table)

    def render_sessions(
        self,
        sessions: list[SessionSummary],
        *,
        current_session_id: str,
    ) -> None:
        table = Table(title="Sessions")
        table.add_column("")
        table.add_column("session")
        table.add_column("name")
        table.add_column("status")
        table.add_column("permission")
        table.add_column("messages", justify="right")
        table.add_column("updated")
        for metadata in sessions:
            marker = "*" if metadata.session_id == current_session_id else ""
            table.add_row(
                marker,
                metadata.session_id,
                metadata.name,
                metadata.name_status,
                metadata.permission_mode,
                str(metadata.message_count),
                metadata.updated_at,
            )
        self.console.print(table)

    def render_history(self, messages: list[Message], *, limit: int) -> None:
        self.render_assistant_done()
        if not messages:
            self.console.print("[dim](no transcript messages available)[/dim]")
            return
        self.console.print(f"[bold]Recent Messages[/bold] [dim](last {limit})[/dim]")
        for message in messages:
            style = {
                "user": "cyan",
                "assistant": "green",
                "system": "magenta",
            }.get(message.role, "white")
            self.console.print(
                Text(f"{message.role}: {message.to_plain_text()}", style=style)
            )

    def render_debug_transcript(self, transcript: str) -> None:
        self.render_assistant_done()
        self.console.print(transcript or "(no messages)")
