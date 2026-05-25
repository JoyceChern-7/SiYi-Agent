from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from types import SimpleNamespace

from config.settings import ModelSettings
from runtime.permissions import PermissionRequest
from ui.controller import REPLController
from ui.input_parser import parse_input


@dataclass(slots=True)
class _Snapshot:
    session_id: str = "sess_test"
    name: str = "新会话"
    name_status: str = "ready"
    project_root: str = "C:/workspace"
    project_id: str = "workspace-123"
    project_state_dir: str = "C:/home/.siyi/projects/workspace-123"
    session_path: str = "C:/home/.siyi/sessions/sess_test.jsonl"
    turn_count: int = 1
    message_count: int = 2
    completed_turns: int = 1
    last_error: str | None = None
    total_usage: dict[str, int] = None  # type: ignore[assignment]
    estimated_total_cost: float = 0.0
    permission_mode: str = "default"

    def __post_init__(self) -> None:
        if self.total_usage is None:
            self.total_usage = {"input_tokens": 10, "output_tokens": 5, "cached_tokens": 0}


class _FakeRenderer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def render_help(self, commands) -> None:
        self.calls.append(("help", [command.name for command in commands]))

    def render_session(self, snapshot) -> None:
        self.calls.append(("session", snapshot.session_id))

    def render_sessions(self, sessions, *, current_session_id: str) -> None:
        self.calls.append(("sessions", [item.session_id for item in sessions], current_session_id))

    def render_history(self, messages, *, limit: int) -> None:
        self.calls.append(("history", limit, list(messages)))

    def render_debug_transcript(self, transcript: str) -> None:
        self.calls.append(("debug", transcript))

    def render_error(self, message: str) -> None:
        self.calls.append(("error", message))

    def render_note(self, message: str) -> None:
        self.calls.append(("note", message))

    def render_user_message(self, text: str) -> None:
        self.calls.append(("user", text))

    def render_event(self, event) -> None:
        self.calls.append(("event", event))


class _FakeEngine:
    def __init__(self) -> None:
        self.snapshot = _Snapshot()
        self.recent_messages = ["user:hello", "assistant:world"]
        self.last_user_prompt = "hello again"
        self.submitted_prompts: list[str] = []
        self.compact_instructions: str | None = None
        self.sessions = [
            SimpleNamespace(session_id="sess_test", permission_mode="default"),
            SimpleNamespace(session_id="sess_other", permission_mode="full"),
        ]
        self.settings = SimpleNamespace(
            model=ModelSettings(),
        )
        self.llm = SimpleNamespace(api_key=None, base_url=None, model=None)

    def get_session_snapshot(self):
        return self.snapshot

    def list_sessions(self):
        return self.sessions

    async def new_session(self):
        self.snapshot = _Snapshot(session_id="sess_new")
        return self.snapshot

    async def switch_session(self, session_id: str):
        self.snapshot = _Snapshot(session_id=session_id)
        return self.snapshot

    def get_permission_snapshot(self):
        return SimpleNamespace(
            session_mode=self.snapshot.permission_mode,
            global_mode="default",
            config_path="C:/home/.siyi/config.json",
            permissions_path="C:/home/.siyi/permissions.json",
        )

    def set_permission_mode(self, mode: str):
        self.snapshot = _Snapshot(
            session_id=self.snapshot.session_id,
            permission_mode=mode,
        )
        return self.get_permission_snapshot()

    def set_global_permission_mode(self, mode: str):
        return SimpleNamespace(
            session_mode=self.snapshot.permission_mode,
            global_mode=mode,
            config_path="C:/home/.siyi/config.json",
            permissions_path="C:/home/.siyi/permissions.json",
        )

    def get_recent_messages(self, limit: int = 10, *, include_meta: bool = False):
        del include_meta
        return self.recent_messages[-limit:]

    def transcript_preview(self) -> str:
        return "user: hello\nassistant: world"

    def get_last_user_prompt(self) -> str | None:
        return self.last_user_prompt

    async def submit_user_input(self, prompt: str):
        self.submitted_prompts.append(prompt)
        yield {"prompt": prompt}

    async def compact(self, custom_instructions: str | None = None):
        self.compact_instructions = custom_instructions
        return SimpleNamespace(
            compacted=True,
            pre_tokens=100,
            post_tokens=20,
            tokens_saved=80,
            reason="manual_compact",
            custom_instructions=custom_instructions,
        )


def test_parse_input_splits_command_args() -> None:
    parsed = parse_input("/history 5")

    assert parsed.kind == "command"
    assert parsed.command == "history"
    assert parsed.args == ["5"]


def test_controller_renders_help_and_session() -> None:
    controller = REPLController(_FakeEngine(), _FakeRenderer())
    renderer = controller.renderer

    asyncio.run(controller.handle(parse_input("/help")))
    asyncio.run(controller.handle(parse_input("/session")))

    assert renderer.calls[0] == (
        "help",
        [
            "help",
            "login",
            "compact",
            "session",
            "sessions",
            "session_new",
            "session_switch",
            "permission",
            "history",
            "retry",
            "add_skill_path",
            "add_skills",
            "debug",
            "clear",
            "exit",
            "quit",
        ],
    )
    assert renderer.calls[1] == ("session", "sess_test")


def test_controller_session_commands() -> None:
    engine = _FakeEngine()
    renderer = _FakeRenderer()
    controller = REPLController(engine, renderer)

    asyncio.run(controller.handle(parse_input("/sessions")))
    asyncio.run(controller.handle(parse_input("/session_new")))
    asyncio.run(controller.handle(parse_input("/session_switch sess_other")))

    assert renderer.calls[0] == ("sessions", ["sess_test", "sess_other"], "sess_test")
    assert renderer.calls[1] == ("note", "started new session: sess_new")
    assert renderer.calls[2] == ("note", "switched to session: sess_other project_root=C:/workspace")


def test_controller_permission_commands() -> None:
    engine = _FakeEngine()
    renderer = _FakeRenderer()
    controller = REPLController(engine, renderer)

    asyncio.run(controller.handle(parse_input("/permission")))
    asyncio.run(controller.handle(parse_input("/permission full")))
    asyncio.run(controller.handle(parse_input("/permission --global custom")))
    asyncio.run(controller.handle(parse_input("/permission nope")))

    assert renderer.calls[0][0] == "note"
    assert "session=default global=default" in renderer.calls[0][1]
    assert renderer.calls[1] == ("note", "session permission mode set to full")
    assert renderer.calls[2] == (
        "note",
        "global permission mode set to custom; current session remains full",
    )
    assert renderer.calls[3][0] == "error"


def test_controller_history_validates_count() -> None:
    controller = REPLController(_FakeEngine(), _FakeRenderer())
    renderer = controller.renderer

    asyncio.run(controller.handle(parse_input("/history nope")))
    asyncio.run(controller.handle(parse_input("/history 2")))

    assert renderer.calls[0] == ("error", "history count must be an integer")
    assert renderer.calls[1][0] == "history"
    assert renderer.calls[1][1] == 2


def test_controller_retry_resubmits_last_prompt() -> None:
    engine = _FakeEngine()
    renderer = _FakeRenderer()
    controller = REPLController(engine, renderer)

    asyncio.run(controller.handle(parse_input("/retry")))

    assert renderer.calls[0] == ("note", "retrying the most recent user prompt")
    assert renderer.calls[1] == ("event", {"prompt": "hello again"})
    assert engine.submitted_prompts == ["hello again"]


def test_controller_compact_passes_custom_instructions() -> None:
    engine = _FakeEngine()
    renderer = _FakeRenderer()
    controller = REPLController(engine, renderer)

    asyncio.run(controller.handle(parse_input("/compact focus on auth")))

    assert renderer.calls[0][0] == "note"
    assert "100 -> 20 tokens" in renderer.calls[0][1]
    assert engine.compact_instructions == "focus on auth"


def test_controller_does_not_echo_user_prompt() -> None:
    engine = _FakeEngine()
    renderer = _FakeRenderer()
    controller = REPLController(engine, renderer)

    asyncio.run(controller.handle(parse_input("hello")))

    assert ("user", "hello") not in renderer.calls
    assert renderer.calls == [("event", {"prompt": "hello"})]


class _FakePromptIO:
    async def secret(self, label: str, *, default: str | None = None) -> str:
        del label, default
        return "login-key"

    async def text(self, label: str, *, default: str | None = None) -> str:
        del default
        values = {
            "LLM provider": "openai-compatible",
            "LLM Base URL": "https://login.example.test/v1",
            "Swift model": "swift-model",
            "Balanced model": "balanced-model",
            "Depth model": "depth-model",
        }
        return values[label]

    async def choice(self, label: str, *, choices: tuple[str, ...], default: str) -> str:
        del label, choices, default
        return "depth"


class _TrackingPromptIO:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    async def choice(self, label: str, *, choices: tuple[str, ...], default: str) -> str:
        del label, choices, default
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return "yes"


def test_controller_login_saves_and_applies_user_settings(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SIYI_CONFIG_DIR", str(tmp_path))
    engine = _FakeEngine()
    renderer = _FakeRenderer()
    controller = REPLController(engine, renderer, prompt_io=_FakePromptIO())

    asyncio.run(controller.handle(parse_input("/login")))

    saved_payload = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
    assert saved_payload["provider"] == "openai-compatible"
    assert "system_prompt" not in saved_payload
    assert "model" not in saved_payload
    assert engine.settings.model.provider == "openai-compatible"
    assert engine.settings.model.api_key.get_secret_value() == "login-key"
    assert engine.settings.model.base_url == "https://login.example.test/v1"
    assert engine.settings.model.model == "depth-model"
    assert engine.llm.model == "depth-model"
    assert engine.submitted_prompts == []


def test_controller_add_skill_path_validates_and_persists(tmp_path, monkeypatch) -> None:
    home = tmp_path / "home"
    extra = tmp_path / "extra-skills"
    extra.mkdir()
    monkeypatch.setattr("config.paths.Path.home", lambda: home)
    engine = _FakeEngine()
    renderer = _FakeRenderer()
    controller = REPLController(engine, renderer)

    asyncio.run(controller.handle(parse_input("/add_skill_path")))
    asyncio.run(controller.handle(parse_input(f"/add_skill_path {extra}")))
    asyncio.run(controller.handle(parse_input(f"/add_skill_path {extra}")))

    config_path = home / ".siyi" / "skill_paths.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload == {"paths": [str(extra.resolve())]}
    assert renderer.calls[0] == ("error", "usage: /add_skill_path <path>")
    assert renderer.calls[1] == ("note", f"added skill path: {extra.resolve()}")
    assert renderer.calls[2] == ("note", f"added skill path: {extra.resolve()}")


def test_controller_add_skill_path_rejects_missing_and_file_paths(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("config.paths.Path.home", lambda: tmp_path / "home")
    file_path = tmp_path / "not-a-dir"
    file_path.write_text("x", encoding="utf-8")
    controller = REPLController(_FakeEngine(), _FakeRenderer())
    renderer = controller.renderer

    asyncio.run(controller.handle(parse_input(f"/add_skill_path {tmp_path / 'missing'}")))
    asyncio.run(controller.handle(parse_input(f"/add_skill_path {file_path}")))

    assert renderer.calls[0][0] == "error"
    assert "does not exist" in renderer.calls[0][1]
    assert renderer.calls[1][0] == "error"
    assert "not a directory" in renderer.calls[1][1]


def test_github_clone_target_accepts_supported_forms() -> None:
    from ui.controller import _github_clone_target

    assert _github_clone_target("owner/repo") == ("https://github.com/owner/repo.git", "repo")
    assert _github_clone_target("https://github.com/owner/repo") == (
        "https://github.com/owner/repo.git",
        "repo",
    )
    assert _github_clone_target("https://github.com/owner/repo.git") == (
        "https://github.com/owner/repo.git",
        "repo",
    )
    assert _github_clone_target("git@github.com:owner/repo.git") == (
        "git@github.com:owner/repo.git",
        "repo",
    )


def test_controller_add_skills_rejects_existing_target(tmp_path, monkeypatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr("config.paths.Path.home", lambda: home)
    target = home / ".siyi" / "skills" / "repo"
    target.mkdir(parents=True)
    controller = REPLController(_FakeEngine(), _FakeRenderer())

    asyncio.run(controller.handle(parse_input("/add_skills owner/repo")))

    assert controller.renderer.calls == [("error", f"skills repository already exists: {target}")]


def test_controller_add_skills_clones_github_repo(tmp_path, monkeypatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr("config.paths.Path.home", lambda: home)
    calls = []

    class _FakeProcess:
        returncode = 0

        async def communicate(self):
            return b"ok", b""

    async def fake_create_subprocess_exec(*args, stdout=None, stderr=None):
        calls.append((args, stdout, stderr))
        return _FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    controller = REPLController(_FakeEngine(), _FakeRenderer())

    asyncio.run(controller.handle(parse_input("/add_skills owner/repo")))

    target = home / ".siyi" / "skills" / "repo"
    assert calls[0][0] == ("git", "clone", "https://github.com/owner/repo.git", str(target))
    assert controller.renderer.calls[-1] == ("note", f"installed skills repository: {target}")


def test_controller_add_skills_surfaces_git_failure(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("config.paths.Path.home", lambda: tmp_path / "home")

    class _FakeProcess:
        returncode = 1

        async def communicate(self):
            return b"", b"fatal clone failed"

    async def fake_create_subprocess_exec(*args, stdout=None, stderr=None):
        del args, stdout, stderr
        return _FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    controller = REPLController(_FakeEngine(), _FakeRenderer())

    asyncio.run(controller.handle(parse_input("/add_skills owner/repo")))

    assert controller.renderer.calls[-1] == ("error", "git clone failed: fatal clone failed")


def test_permission_prompts_are_serialized() -> None:
    prompt_io = _TrackingPromptIO()
    controller = REPLController(_FakeEngine(), _FakeRenderer(), prompt_io=prompt_io)
    request = PermissionRequest(
        tool_name="Write",
        tool_input={"file_path": "a.txt"},
        project_root="C:/workspace",
    )

    async def run_two_prompts() -> list[bool]:
        return await asyncio.gather(
            controller._ask_permission(request),
            controller._ask_permission(request),
        )

    assert asyncio.run(run_two_prompts()) == [True, True]
    assert prompt_io.max_active == 1
