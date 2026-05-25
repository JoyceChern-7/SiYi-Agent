from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from pydantic import SecretStr

from config.paths import get_global_skills_dir, get_skill_paths_path, get_user_settings_path
from config.user_settings import (
    DEFAULT_MODEL_TIERS,
    MODEL_TIER_LABELS,
    UserModelTier,
    UserSettings,
    default_user_settings,
    load_user_settings,
    save_user_settings,
)
from engine.query_engine import QueryEngine
from runtime.permissions import PermissionRequest
from ui.input_parser import ParsedInput
from ui.prompt_io import PromptIO
from ui.renderer import ConsoleRenderer


@dataclass(slots=True)
class CommandSpec:
    name: str
    usage: str
    description: str


class REPLController:
    def __init__(
        self,
        engine: QueryEngine,
        renderer: ConsoleRenderer,
        prompt_io: PromptIO | None = None,
    ) -> None:
        self.engine = engine
        self.renderer = renderer
        self.prompt_io = prompt_io
        self._permission_prompt_lock = asyncio.Lock()
        permission_manager = getattr(self.engine, "permission_manager", None)
        if self.prompt_io is not None and permission_manager is not None:
            permission_manager.set_requester(self._ask_permission)
        self.command_specs = [
            CommandSpec("help", "/help", "Show the available REPL commands."),
            CommandSpec("login", "/login", "Configure LLM API settings."),
            CommandSpec("compact", "/compact [instructions]", "Compact the current session context."),
            CommandSpec("session", "/session", "Show the current session summary."),
            CommandSpec("sessions", "/sessions", "List global sessions."),
            CommandSpec("session_new", "/session_new", "Start a new global session for the current project."),
            CommandSpec("session_switch", "/session_switch <session_id>", "Switch to a global session and follow its project."),
            CommandSpec("permission", "/permission [--global] <default|full|custom>", "Show or change the permission mode."),
            CommandSpec("history", "/history [count]", "Show recent transcript messages."),
            CommandSpec("retry", "/retry", "Resubmit the most recent user prompt."),
            CommandSpec("add_skill_path", "/add_skill_path <path>", "Add a global skill search path."),
            CommandSpec("add_skills", "/add_skills <github_address>", "Clone a GitHub skills repository into ~/.siyi/skills."),
            CommandSpec("debug", "/debug", "Print a plain-text transcript preview."),
            CommandSpec("clear", "/clear", "Clear the terminal output."),
            CommandSpec("exit", "/exit", "Exit the REPL."),
            CommandSpec("quit", "/quit", "Exit the REPL."),
        ]

    def show_welcome(self) -> None:
        snapshot = self.engine.get_session_snapshot()
        self.renderer.render_welcome(
            session_id=snapshot.session_id,
            project_root=snapshot.project_root,
            model=self.engine.settings.model.model,
        )

    async def handle(self, parsed: ParsedInput) -> bool:
        if parsed.kind == "empty":
            return True
        if parsed.kind == "command":
            return await self._handle_command(parsed)
        await self._submit_prompt(parsed.text)
        return True

    async def _handle_command(self, parsed: ParsedInput) -> bool:
        assert parsed.command is not None
        command = parsed.command.lower()

        if command in {"exit", "quit"}:
            return False
        if command == "clear":
            self.renderer.console.clear()
            return True
        if command == "help":
            self.renderer.render_help(self.command_specs)
            return True
        if command == "login":
            await self._handle_login()
            return True
        if command == "compact":
            await self._handle_compact(parsed.args)
            return True
        if command == "session":
            self.renderer.render_session(self.engine.get_session_snapshot())
            return True
        if command == "sessions":
            self.renderer.render_sessions(
                self.engine.list_sessions(),
                current_session_id=self.engine.get_session_snapshot().session_id,
            )
            return True
        if command == "session_new":
            if parsed.args:
                self.renderer.render_error("usage: /session_new")
                return True
            try:
                snapshot = await self.engine.new_session()
            except Exception as exc:  # noqa: BLE001 - keep REPL alive
                self.renderer.render_error(f"failed to create session: {exc}")
                return True
            self.renderer.render_note(f"started new session: {snapshot.session_id}")
            return True
        if command == "session_switch":
            if len(parsed.args) != 1:
                self.renderer.render_error("usage: /session_switch <session_id>")
                return True
            try:
                snapshot = await self.engine.switch_session(parsed.args[0])
            except Exception as exc:  # noqa: BLE001 - keep REPL alive
                self.renderer.render_error(f"failed to switch session: {exc}")
                return True
            self.renderer.render_note(
                f"switched to session: {snapshot.session_id} project_root={snapshot.project_root}"
            )
            return True
        if command == "permission":
            self._handle_permission(parsed.args)
            return True
        if command == "history":
            return self._handle_history(parsed.args)
        if command == "debug":
            self.renderer.render_debug_transcript(self.engine.transcript_preview())
            return True
        if command == "retry":
            prompt = self.engine.get_last_user_prompt()
            if prompt is None:
                self.renderer.render_error("no previous user prompt is available to retry")
                return True
            self.renderer.render_note("retrying the most recent user prompt")
            await self._submit_prompt(prompt)
            return True
        if command == "add_skill_path":
            self._handle_add_skill_path(parsed.args)
            return True
        if command == "add_skills":
            await self._handle_add_skills(parsed.args)
            return True

        self.renderer.render_error(f"unknown command: /{parsed.command}")
        return True

    def _handle_permission(self, args: list[str]) -> None:
        valid_modes = {"default", "full", "custom"}
        if not args:
            snapshot = self.engine.get_permission_snapshot()
            self.renderer.render_note(
                "permission mode: "
                f"session={snapshot.session_mode} global={snapshot.global_mode}\n"
                f"config={snapshot.config_path}\n"
                f"permissions={snapshot.permissions_path}"
            )
            return

        if args[0] == "--global":
            if len(args) != 2 or args[1] not in valid_modes:
                self.renderer.render_error("usage: /permission --global <default|full|custom>")
                return
            snapshot = self.engine.set_global_permission_mode(args[1])
            self.renderer.render_note(
                f"global permission mode set to {snapshot.global_mode}; "
                f"current session remains {snapshot.session_mode}"
            )
            return

        if len(args) != 1 or args[0] not in valid_modes:
            self.renderer.render_error("usage: /permission [--global] <default|full|custom>")
            return

        snapshot = self.engine.set_permission_mode(args[0])
        self.renderer.render_note(f"session permission mode set to {snapshot.session_mode}")

    def _handle_history(self, args: list[str]) -> bool:
        if len(args) > 1:
            self.renderer.render_error("usage: /history [count]")
            return True

        limit = 10
        if args:
            try:
                limit = int(args[0])
            except ValueError:
                self.renderer.render_error("history count must be an integer")
                return True
            if limit <= 0:
                self.renderer.render_error("history count must be greater than zero")
                return True

        self.renderer.render_history(
            self.engine.get_recent_messages(limit=limit),
            limit=limit,
        )
        return True

    def _handle_add_skill_path(self, args: list[str]) -> None:
        raw_path = " ".join(args).strip()
        if not raw_path:
            self.renderer.render_error("usage: /add_skill_path <path>")
            return

        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            self.renderer.render_error(f"skill path does not exist: {path}")
            return
        if not path.is_dir():
            self.renderer.render_error(f"skill path is not a directory: {path}")
            return

        config_path = get_skill_paths_path()
        try:
            paths = _load_configured_skill_paths(config_path)
            saved = _add_configured_skill_path(paths, path)
            _save_configured_skill_paths(config_path, saved)
        except ValueError as exc:
            self.renderer.render_error(str(exc))
            return
        self.renderer.render_note(f"added skill path: {path}")

    async def _handle_add_skills(self, args: list[str]) -> None:
        address = " ".join(args).strip()
        if not address:
            self.renderer.render_error("usage: /add_skills <github_address>")
            return

        try:
            clone_url, repo_name = _github_clone_target(address)
        except ValueError as exc:
            self.renderer.render_error(str(exc))
            return

        skills_dir = get_global_skills_dir()
        target = skills_dir / repo_name
        if target.exists():
            self.renderer.render_error(f"skills repository already exists: {target}")
            return

        skills_dir.mkdir(parents=True, exist_ok=True)
        self.renderer.render_note(f"cloning {clone_url} into {target}")
        try:
            process = await asyncio.create_subprocess_exec(
                "git",
                "clone",
                clone_url,
                str(target),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            self.renderer.render_error("git executable was not found")
            return

        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            detail = (stderr or stdout).decode("utf-8", errors="replace").strip()
            self.renderer.render_error(f"git clone failed: {detail or f'exit code {process.returncode}'}")
            return
        self.renderer.render_note(f"installed skills repository: {target}")

    async def _submit_prompt(self, prompt: str) -> None:
        async for event in self.engine.submit_user_input(prompt):
            self.renderer.render_event(event)

    async def _handle_compact(self, args: list[str]) -> None:
        instructions = " ".join(args).strip() or None
        try:
            result = await self.engine.compact(instructions)
        except Exception as exc:  # noqa: BLE001 - surface compact failures in the REPL
            self.renderer.render_error(f"compact failed: {exc}")
            return
        if not result.compacted:
            self.renderer.render_note(f"compact skipped: {result.reason or 'not needed'}")
            return
        self.renderer.render_note(
            "compacted context: "
            f"{result.pre_tokens} -> {result.post_tokens} tokens, "
            f"saved ~{result.tokens_saved}"
        )

    async def _ask_permission(self, request: PermissionRequest) -> bool:
        if self.prompt_io is None:
            return False
        async with self._permission_prompt_lock:
            tool_input = json.dumps(request.tool_input, ensure_ascii=False, indent=2)
            self.renderer.render_note(
                f"permission request: {request.tool_name}\n"
                f"tool_input: {tool_input}\n"
                f"project_root: {request.project_root}"
            )
            choice = await self.prompt_io.choice(
                "Allow this tool call",
                choices=("yes", "no"),
                default="no",
            )
            return choice == "yes"

    async def _handle_login(self) -> None:
        if self.prompt_io is None:
            self.renderer.render_error("/login is only available in interactive mode")
            return

        existing = load_user_settings()
        defaults = default_user_settings()
        merged_models = {
            **defaults.models,
            **existing.models,
        }
        self.renderer.render_note("SiYi API setup: LLM")
        provider = await self.prompt_io.text("LLM provider", default=existing.provider)
        api_key = await self.prompt_io.secret("LLM API Key", default=existing.api_key)
        base_url = await self.prompt_io.text("LLM Base URL", default=existing.base_url)

        configured_models: dict[str, UserModelTier] = {}
        for tier, fallback_model in DEFAULT_MODEL_TIERS.items():
            current = merged_models.get(tier)
            label = MODEL_TIER_LABELS[tier]
            model = await self.prompt_io.text(
                f"{label} model",
                default=current.model if current else fallback_model,
            )
            configured_models[tier] = UserModelTier(label=label, model=model)

        default_tier = await self.prompt_io.choice(
            "Default tier",
            choices=("swift", "balanced", "depth"),
            default=existing.default_tier if existing.default_tier in DEFAULT_MODEL_TIERS else "balanced",
        )

        settings = UserSettings(
            provider=provider or "openai-compatible",
            api_key=api_key,
            base_url=base_url or None,
            default_tier=default_tier,
            models=configured_models,
        )
        path = save_user_settings(settings, get_user_settings_path())
        self._apply_login_settings(settings)
        self.renderer.render_note(f"saved API settings to {path}")

    def _apply_login_settings(self, settings: UserSettings) -> None:
        selected_model = settings.models[settings.default_tier].model
        api_key = SecretStr(settings.api_key) if settings.api_key else None
        self.engine.settings.model = self.engine.settings.model.model_copy(
            update={
                "provider": settings.provider,
                "api_key": api_key,
                "base_url": settings.base_url,
                "model": selected_model,
            }
        )
        if hasattr(self.engine.llm, "api_key"):
            self.engine.llm.api_key = api_key
        if hasattr(self.engine.llm, "base_url"):
            self.engine.llm.base_url = settings.base_url
        if hasattr(self.engine.llm, "model"):
            self.engine.llm.model = selected_model


def _load_configured_skill_paths(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read skill paths config: {path}") from exc
    raw_paths = payload.get("paths") if isinstance(payload, dict) else None
    if raw_paths is None:
        return []
    if not isinstance(raw_paths, list) or not all(isinstance(item, str) for item in raw_paths):
        raise ValueError(f"invalid skill paths config: {path}")
    return raw_paths


def _add_configured_skill_path(paths: list[str], path: Path) -> list[str]:
    normalized = str(path)
    seen = {os.path.normcase(str(Path(item).expanduser().resolve())) for item in paths}
    if os.path.normcase(normalized) in seen:
        return paths
    return [*paths, normalized]


def _save_configured_skill_paths(path: Path, paths: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"paths": paths}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _github_clone_target(address: str) -> tuple[str, str]:
    address = address.strip()
    shorthand_match = re.fullmatch(r"([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?", address)
    if shorthand_match:
        owner, repo = shorthand_match.groups()
        return f"https://github.com/{owner}/{repo}.git", repo

    ssh_match = re.fullmatch(r"git@github\.com:([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?", address)
    if ssh_match:
        _owner, repo = ssh_match.groups()
        return address if address.endswith(".git") else f"{address}.git", repo

    parsed = urlparse(address)
    if parsed.scheme in {"http", "https"} and parsed.netloc.lower() == "github.com":
        parts = [part for part in parsed.path.strip("/").split("/") if part]
        if len(parts) == 2:
            owner, repo = parts
            repo = repo.removesuffix(".git")
            return f"https://github.com/{owner}/{repo}.git", repo

    raise ValueError("expected a GitHub repository such as owner/repo, https://github.com/owner/repo, or git@github.com:owner/repo.git")
