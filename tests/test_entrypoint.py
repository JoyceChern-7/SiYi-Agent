from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.cli import parse_args
from app.main import build_runtime, run


def test_parse_args_entry_layer_flags(tmp_path: Path) -> None:
    options = parse_args(
        [
            "--model",
            "gpt-test",
            "--system-prompt",
            "test prompt",
            "--resume",
            "session-1",
            "--debug",
            "--print-thinking",
            "--cwd",
            str(tmp_path),
        ]
    )

    assert options.model == "gpt-test"
    assert options.system_prompt == "test prompt"
    assert options.resume == "session-1"
    assert options.debug is True
    assert options.print_thinking is True
    assert options.cwd == tmp_path.resolve()
    assert options.non_interactive is False
    assert options.prompt is None


def test_build_runtime_wires_core_services_for_internal_worker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    monkeypatch.delenv("SIYI_SESSION_DIR", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    options = parse_args(["--cwd", str(tmp_path), "--internal-worker-prompt", "hello"])
    runtime = build_runtime(options)

    assert runtime.settings.runtime.project_root == tmp_path.resolve()
    assert runtime.settings.runtime.session_dir == (
        home.resolve() / ".siyi" / "sessions"
    )
    assert runtime.settings.runtime.initial_prompt == "hello"
    assert runtime.settings.runtime.non_interactive is True
    assert runtime.query_engine.session.session_id.startswith("sess_")
    assert runtime.session_store.root.exists()


def test_public_non_interactive_flag_is_removed(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--cwd", str(tmp_path), "--non-interactive"])


def test_internal_worker_without_prompt_fails_after_runtime(tmp_path: Path) -> None:
    options = parse_args(["--cwd", str(tmp_path), "--internal-worker-prompt", ""])

    import asyncio

    exit_code = asyncio.run(run(options))

    assert exit_code == 2
    assert Path(os.environ["SIYI_SESSION_DIR"]).exists()
