from __future__ import annotations

from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style

from engine.query_engine import QueryEngine
from runtime.session_runtime import SessionRuntime
from ui.controller import REPLController
from ui.input_parser import parse_input
from ui.prompt_io import PromptIO
from ui.renderer import ConsoleRenderer


async def run_repl(
    engine: QueryEngine,
    renderer: ConsoleRenderer,
    *,
    session_runtime: SessionRuntime | None = None,
) -> None:
    style = Style.from_dict({"prompt": "ansibrightblack"})
    session: PromptSession[str] = PromptSession(style=style)
    controller = REPLController(
        engine,
        renderer,
        prompt_io=PromptIO(session),
        session_runtime=session_runtime,
    )
    controller.show_welcome()

    while True:
        try:
            raw = await session.prompt_async([("class:prompt", "> ")])
        except (EOFError, KeyboardInterrupt):
            renderer.console.print()
            return

        parsed = parse_input(raw)
        should_continue = await controller.handle(parsed)
        if not should_continue:
            return
