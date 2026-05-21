from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from config.settings import ToolSettings
from config.paths import get_global_permissions_path, get_siyi_config_path
from engine.message_schema import ToolUseBlock
from engine.query_loop import _tool_batches
from runtime.permissions import PermissionManager, ensure_permission_files
from tools.base import BaseTool, ToolContext, ToolResult
from tools.shell_analysis import analyze_bash, analyze_powershell
from tools.registry import ToolRegistry


class _DummyWriteTool(BaseTool):
    name = "Write"
    description = "dummy mutating tool"
    input_schema = {"type": "object", "properties": {}, "additionalProperties": True}

    async def run(self, raw_input, context: ToolContext) -> ToolResult:
        del raw_input
        return ToolResult(success=True, content=context.cwd)


def _context(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=str(tmp_path), trace_id="test")


def test_ensure_permission_files_creates_first_run_defaults() -> None:
    ensure_permission_files()

    config_payload = json.loads(get_siyi_config_path().read_text(encoding="utf-8"))
    permissions_payload = json.loads(get_global_permissions_path().read_text(encoding="utf-8"))

    assert config_payload == {"permission_mode": "default"}
    assert "version" not in permissions_payload
    assert permissions_payload["custom_permissions"] == {"allow": [], "ask": [], "deny": []}
    assert permissions_payload["shell_exec_rules"]["unmatched"] == "ask"
    assert ["git", "status"] in permissions_payload["shell_exec_rules"]["allow_prefix"]
    assert permissions_payload["sandbox"] == {
        "enabled": False,
        "fail_if_unavailable": False,
        "allow_unsandboxed_commands": True,
    }


def test_ensure_permission_files_does_not_overwrite_existing_files() -> None:
    config_path = get_siyi_config_path()
    permissions_path = get_global_permissions_path()
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps({"permission_mode": "full", "extra": "keep"}),
        encoding="utf-8",
    )
    permissions_path.write_text(
        json.dumps({"custom_permissions": {"allow": ["Write(*)"]}, "extra": "keep"}),
        encoding="utf-8",
    )

    ensure_permission_files()

    assert json.loads(config_path.read_text(encoding="utf-8")) == {
        "permission_mode": "full",
        "extra": "keep",
    }
    assert json.loads(permissions_path.read_text(encoding="utf-8")) == {
        "custom_permissions": {"allow": ["Write(*)"]},
        "extra": "keep",
    }


def test_default_shell_permission_uses_exec_rules(tmp_path: Path) -> None:
    manager = PermissionManager.from_settings(ToolSettings(), cwd=tmp_path)

    result = manager.check("PowerShell", {"command": "Get-ChildItem"})

    assert result.decision == "allow"
    assert result.source == "shell_exec_rules"


def test_default_process_session_permissions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = PermissionManager.from_settings(ToolSettings(), cwd=tmp_path)

    assert manager.check("exec_command", {"cmd": "Get-ChildItem"}).decision == "allow"
    assert manager.check("ProcessStart", {"command": "npm run dev"}).decision == "ask"
    assert manager.check("ProcessRead", {"process_id": "proc_test"}).decision == "allow"
    assert manager.check("ProcessWrite", {"process_id": "proc_test", "chars": "y\n"}).decision == "ask"
    assert manager.check("ProcessStop", {"process_id": "proc_test"}).decision == "ask"
    assert manager.check("write_stdin", {"session_id": "proc_test", "chars": ""}).decision == "allow"
    assert manager.check("stop_command", {"session_id": "proc_test"}).decision == "allow"


def test_alias_tools_inherit_target_permissions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = PermissionManager.from_settings(ToolSettings(), cwd=tmp_path)
    registry = ToolRegistry.default(manager)
    context = _context(tmp_path)
    requester_calls = 0

    async def requester(_request) -> bool:
        nonlocal requester_calls
        requester_calls += 1
        return True

    manager.set_requester(requester)

    async def authorize_alias(name: str, payload: dict) -> str:
        tool = registry.find_tool(name)
        assert tool is not None
        result = await manager.authorize(tool, payload, context)
        return result.decision

    assert asyncio.run(authorize_alias("read_file", {"file_path": "README.md"})) == "allow"
    assert asyncio.run(authorize_alias("glob", {"pattern": "*.py"})) == "allow"
    assert asyncio.run(authorize_alias("grep", {"pattern": "x"})) == "allow"
    assert asyncio.run(authorize_alias("web_search", {"query": "docs"})) == "allow"
    assert requester_calls == 0


def test_shell_alias_inherits_target_shell_permission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = PermissionManager.from_settings(ToolSettings(), cwd=tmp_path)
    registry = ToolRegistry.default(manager)
    context = _context(tmp_path)
    shell = registry.find_tool("shell")
    if shell is None:
        return

    result = asyncio.run(manager.authorize(shell, {"command": "npx --yes playwright"}, context))

    assert result.decision == "deny"
    assert "permission required" in (result.reason or "")


def test_custom_permissions_use_global_siyi_permissions_file(tmp_path: Path) -> None:
    permissions_path = get_global_permissions_path()
    permissions_path.parent.mkdir(parents=True)
    permissions_path.write_text(
        json.dumps({"custom_permissions": {"deny": ["Write(*)"]}}),
        encoding="utf-8",
    )

    manager = PermissionManager.from_settings(ToolSettings(), cwd=tmp_path, mode="custom")
    result = manager.check("Write", {"file_path": "a.txt"})

    assert result.decision == "deny"
    assert result.source == str(permissions_path)


def test_full_permission_mode_allows_non_shell_but_keeps_shell_rules(tmp_path: Path) -> None:
    manager = PermissionManager.from_settings(ToolSettings(), cwd=tmp_path, mode="full")

    assert manager.check("Write", {"file_path": "a.txt"}).decision == "allow"
    assert manager.check("PowerShell", {"command": "npx --yes playwright"}).decision == "ask"
    assert manager.check("PowerShell", {"command": "python -c \"print(1)\""}).decision == "allow"


def test_custom_shell_unmatched_uses_configured_fallback(tmp_path: Path) -> None:
    permissions_path = get_global_permissions_path()
    permissions_path.parent.mkdir(parents=True)
    permissions_path.write_text(
        json.dumps({"shell_exec_rules": {"unmatched": "deny", "allow_prefix": []}}),
        encoding="utf-8",
    )
    manager = PermissionManager.from_settings(ToolSettings(), cwd=tmp_path, mode="custom")

    result = manager.check("PowerShell", {"command": "python -c \"print(1)\""})

    assert result.decision == "deny"
    assert "did not match" in (result.reason or "")


def test_shell_exec_rules_precedence_and_legacy_permission_format(tmp_path: Path) -> None:
    permissions_path = get_global_permissions_path()
    permissions_path.parent.mkdir(parents=True)
    permissions_path.write_text(
        json.dumps(
            {
                "permissions": {"allow": ["Write(*)"]},
                "shell_exec_rules": {
                    "allow_prefix": [["rm"]],
                    "ask_prefix": [["rm"]],
                    "deny_prefix": [["rm"]],
                    "ask_glob": ["*Invoke-Expression*"],
                },
            }
        ),
        encoding="utf-8",
    )
    manager = PermissionManager.from_settings(ToolSettings(), cwd=tmp_path, mode="custom")

    assert manager.check("Write", {"file_path": "a.txt"}).decision == "allow"
    assert manager.check("Bash", {"command": "rm file.txt"}).decision == "deny"
    assert manager.check("PowerShell", {"command": "Invoke-Expression whoami"}).decision == "ask"


def test_cwd_siyi_permission_config_is_ignored(tmp_path: Path) -> None:
    legacy_dir = tmp_path / ".siyi"
    legacy_dir.mkdir()
    (legacy_dir / "permissions.json").write_text(
        json.dumps({"permissions": {"deny": ["Bash(*)"]}}),
        encoding="utf-8",
    )

    manager = PermissionManager.from_settings(ToolSettings(), cwd=tmp_path)
    result = manager.check("Bash", {"command": "git status"})

    assert result.decision == "allow"
    assert result.source == "shell_exec_rules"


def test_interactive_approval_is_current_call_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = PermissionManager.from_settings(ToolSettings(), cwd=tmp_path)
    approvals = 0

    async def approve_once(_request) -> bool:
        nonlocal approvals
        approvals += 1
        return True

    manager.set_requester(approve_once)

    import asyncio

    first = asyncio.run(manager.authorize(_DummyWriteTool(), {}, _context(tmp_path)))
    second = manager.check("Write", {})

    assert first.decision == "allow"
    assert second.decision == "ask"
    assert approvals == 1


def test_sandbox_required_on_native_windows_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    permissions_path = get_global_permissions_path()
    permissions_path.parent.mkdir(parents=True)
    permissions_path.write_text(
        json.dumps(
            {
                "sandbox": {
                    "enabled": True,
                    "allow_unsandboxed_commands": False,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("runtime.permissions.sys.platform", "win32")
    manager = PermissionManager.from_settings(ToolSettings(), cwd=tmp_path)

    result = manager.check("PowerShell", {"command": "Get-ChildItem"})

    assert result.decision == "deny"
    assert "native Windows" in (result.reason or "")


def test_shell_read_only_analysis_is_conservative() -> None:
    assert analyze_powershell("Get-ChildItem | Select-Object Name").read_only
    assert not analyze_powershell("New-Item test.txt").read_only
    assert analyze_bash("git status | head -n 5").read_only
    assert not analyze_bash("git status > out.txt").read_only


def test_tool_batches_group_consecutive_concurrency_safe_tools(tmp_path: Path) -> None:
    registry = ToolRegistry.default(PermissionManager.from_settings(ToolSettings(), cwd=tmp_path))
    context = _context(tmp_path)

    batches = _tool_batches(
        [
            ToolUseBlock(name="Read", input={"file_path": "a.txt"}),
            ToolUseBlock(name="Glob", input={"pattern": "*.txt"}),
            ToolUseBlock(name="Write", input={"file_path": "a.txt", "content": "x"}),
            ToolUseBlock(name="Grep", input={"pattern": "x"}),
        ],
        registry,
        context,
    )

    assert [len(batch.blocks) for batch in batches] == [2, 1, 1]
    assert [batch.concurrent for batch in batches] == [True, False, True]
