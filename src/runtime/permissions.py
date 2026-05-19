from __future__ import annotations

import fnmatch
import json
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from config.paths import get_global_permissions_path, get_siyi_config_home, get_siyi_config_path
from config.settings import ToolSettings
from tools.base import ToolContext
from tools.governance import PermissionDecision, PermissionResult
from tools.shell_analysis import analyze_bash, analyze_powershell


PermissionRequester = Callable[["PermissionRequest"], Awaitable[bool]]
PermissionMode = Literal["default", "full", "custom"]


class SandboxPolicy(BaseModel):
    enabled: bool = False
    fail_if_unavailable: bool = False
    allow_unsandboxed_commands: bool = True


class PermissionRules(BaseModel):
    allow: list[str] = Field(default_factory=list)
    ask: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)


class ShellExecRules(BaseModel):
    allow_prefix: list[list[str] | str] = Field(default_factory=lambda: [
        ["git", "status"],
        ["git", "diff"],
        ["git", "log"],
        ["rg"],
        ["ls"],
        ["Get-ChildItem"],
    ])
    ask_prefix: list[list[str] | str] = Field(default_factory=lambda: [
        ["git", "push"],
        ["npm", "install"],
        ["npx"],
        ["rm"],
        ["Remove-Item"],
        ["del"],
    ])
    deny_prefix: list[list[str] | str] = Field(default_factory=list)
    ask_glob: list[str] = Field(default_factory=lambda: [
        "*rm -rf*",
        "*Remove-Item* -Recurse*",
        "*curl*|*powershell*",
        "*Invoke-Expression*",
    ])
    deny_glob: list[str] = Field(default_factory=list)
    unmatched: str = "ask"


class PermissionConfig(BaseModel):
    custom_permissions: PermissionRules = Field(default_factory=PermissionRules)
    shell_exec_rules: ShellExecRules = Field(default_factory=ShellExecRules)
    sandbox: SandboxPolicy = Field(default_factory=SandboxPolicy)
    source_path: Path | None = None


class PermissionRequest(BaseModel):
    tool_name: str
    tool_input: dict[str, Any]
    reason: str
    summary: str
    cwd: str


def ensure_permission_files() -> None:
    config_path = get_siyi_config_path()
    if not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps({"permission_mode": "default"}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    permissions_path = get_global_permissions_path()
    if not permissions_path.exists():
        permissions_path.parent.mkdir(parents=True, exist_ok=True)
        permissions_path.write_text(
            json.dumps(_default_permissions_payload(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _default_permissions_payload() -> dict[str, Any]:
    return {
        "custom_permissions": PermissionRules().model_dump(mode="json"),
        "shell_exec_rules": ShellExecRules().model_dump(mode="json"),
        "sandbox": SandboxPolicy().model_dump(mode="json"),
    }


def load_global_permission_mode() -> PermissionMode:
    path = get_siyi_config_path()
    if not path.exists():
        return "default"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "default"
    if not isinstance(payload, dict):
        return "default"
    return _coerce_permission_mode(payload.get("permission_mode"))


def save_global_permission_mode(mode: str) -> Path:
    permission_mode = _coerce_permission_mode(mode)
    path = get_siyi_config_path()
    payload: dict[str, Any] = {}
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = {}
        if isinstance(raw, dict):
            payload.update(raw)
    payload["permission_mode"] = permission_mode
    payload.pop("version", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _coerce_permission_mode(value: object) -> PermissionMode:
    if value in {"default", "full", "custom"}:
        return value  # type: ignore[return-value]
    return "default"


@dataclass(slots=True)
class PermissionManager:
    settings: ToolSettings
    config: PermissionConfig = field(default_factory=PermissionConfig)
    mode: PermissionMode = "default"
    requester: PermissionRequester | None = None

    @classmethod
    def from_settings(
        cls,
        settings: ToolSettings,
        *,
        cwd: Path | None = None,
        mode: str | None = None,
    ) -> "PermissionManager":
        config = load_permission_config(cwd)
        if config.source_path is None:
            config.sandbox = SandboxPolicy(
                enabled=settings.sandbox_enabled,
                fail_if_unavailable=settings.sandbox_fail_if_unavailable,
                allow_unsandboxed_commands=settings.allow_unsandboxed_commands,
            )
        return cls(
            settings=settings,
            config=config,
            mode=_coerce_permission_mode(mode or load_global_permission_mode()),
        )

    def reload_for_cwd(self, cwd: Path, *, mode: str | None = None) -> None:
        self.config = load_permission_config(cwd)
        if self.config.source_path is None:
            self.config.sandbox = SandboxPolicy(
                enabled=self.settings.sandbox_enabled,
                fail_if_unavailable=self.settings.sandbox_fail_if_unavailable,
                allow_unsandboxed_commands=self.settings.allow_unsandboxed_commands,
            )
        if mode is not None:
            self.set_mode(mode)

    def set_mode(self, mode: str) -> PermissionMode:
        self.mode = _coerce_permission_mode(mode)
        return self.mode

    def set_requester(self, requester: PermissionRequester | None) -> None:
        self.requester = requester

    def check(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> PermissionResult:
        del context
        if self._is_shell_tool(tool_name, tool_input) and not self.settings.shell_enabled:
            return PermissionResult.deny(reason="shell tool is disabled", source="settings")
        if tool_name == "Read" and not self.settings.read_file_enabled:
            return PermissionResult.deny(reason="file reads are disabled", source="settings")
        if tool_name in {"WebSearch", "WebFetch", "web_search", "web_fetch"} and not self.settings.web_search_enabled:
            return PermissionResult.deny(reason="web tools are disabled", source="settings")

        sandbox_result = self._check_sandbox_policy(tool_name, tool_input)
        if sandbox_result.decision != "allow":
            return sandbox_result

        if self._is_shell_tool(tool_name, tool_input):
            return self._check_shell_tool(tool_name, tool_input)

        rule_result = self._check_rule_lists(tool_name, tool_input)
        if rule_result is not None:
            return rule_result

        return self._default_decision(tool_name, tool_input)

    async def authorize(
        self,
        tool: object,
        raw_input: dict[str, Any],
        context: ToolContext,
    ) -> PermissionResult:
        permission_tool_name = str(getattr(tool, "permission_name", getattr(tool, "name", "")))
        display_tool_name = context.tool_name or str(getattr(tool, "name", ""))
        tool_decision = getattr(tool, "check_permissions")(raw_input, context)
        global_decision = self.check(permission_tool_name, raw_input, {"cwd": context.cwd})
        decision = _combine_permissions(tool_decision, global_decision)
        if decision.decision != "ask":
            return decision

        if self.requester is None:
            return PermissionResult.deny(
                reason=f"permission required but no interactive approver is available: {decision.reason}",
                source=decision.source or "permission",
            )

        approved = await self.requester(
            PermissionRequest(
                tool_name=display_tool_name,
                tool_input=raw_input,
                reason=decision.reason or "permission required",
                summary=_tool_input_summary(display_tool_name, raw_input),
                cwd=context.cwd,
            )
        )
        if approved:
            return PermissionResult.allow(reason="approved for this tool call", source="interactive")
        return PermissionResult.deny(reason="user denied this tool call", source="interactive")

    def _check_rule_lists(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> PermissionResult | None:
        rules, source = self._active_tool_rules()
        if rules is None:
            return None
        payload = _permission_payload(tool_name, tool_input)
        candidate = f"{tool_name}({payload})"
        for decision in ("deny", "ask", "allow"):
            decision_rules = getattr(rules, decision)
            for rule in decision_rules:
                if _rule_matches(rule, tool_name, payload, candidate):
                    return PermissionResult(
                        decision=decision,
                        reason=f"matched permission rule: {rule}",
                        source=source,
                    )
        return None

    def _active_tool_rules(self) -> tuple[PermissionRules | None, str]:
        if self.mode == "full":
            return None, "full"
        if self.mode == "custom":
            return self.config.custom_permissions, str(self.config.source_path or "custom")
        return _default_permission_rules(), "default"

    def _check_shell_tool(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> PermissionResult:
        shell_rule = _match_shell_exec_rules(self.config.shell_exec_rules, tool_name, tool_input)
        if shell_rule is not None:
            return shell_rule

        if self.mode == "default":
            analysis = _analyze_shell_command(tool_name, tool_input)
            if analysis.read_only:
                return PermissionResult.allow(reason=analysis.reason, source=analysis.parser)
            if self.settings.shell_requires_approval:
                return PermissionResult.ask(reason="shell tool requires approval", source="default")
            return PermissionResult.allow(reason="shell tool allowed by settings", source="default")

        if self.mode == "full":
            return PermissionResult.allow(reason="shell command allowed by full permission mode", source="full")

        return PermissionResult(
            decision=_coerce_permission_decision(self.config.shell_exec_rules.unmatched),
            reason="shell command did not match any exec rule",
            source=str(self.config.source_path or "custom"),
        )

    def _default_decision(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> PermissionResult:
        if self.mode == "full":
            return PermissionResult.allow(reason="tool allowed by full permission mode", source="full")
        if self.mode == "custom":
            return PermissionResult.ask(reason="tool is not covered by custom permissions", source="custom")

        if _is_default_read_tool(tool_name, tool_input):
            return PermissionResult.allow(reason="read-only tool", source="default")
        return PermissionResult.ask(reason="tool can modify local state", source="default")

    def _check_sandbox_policy(self, tool_name: str, tool_input: dict[str, Any]) -> PermissionResult:
        if not self._is_shell_tool(tool_name, tool_input):
            return PermissionResult.allow(source="sandbox")
        policy = self.config.sandbox
        if not policy.enabled:
            return PermissionResult.allow(source="sandbox")
        if policy.allow_unsandboxed_commands:
            return PermissionResult.allow(
                reason="sandbox is configured but unsandboxed commands are allowed",
                source="sandbox",
            )
        if _native_windows():
            return PermissionResult.deny(
                reason=(
                    "sandbox is required, but native Windows PowerShell/Bash commands "
                    "cannot be isolated by SiYi v1"
                ),
                source="sandbox",
            )
        if policy.fail_if_unavailable:
            return PermissionResult.deny(
                reason="sandbox is required, but no sandbox backend is available",
                source="sandbox",
            )
        return PermissionResult.ask(
            reason="sandbox is unavailable; approve unsandboxed execution for this call",
            source="sandbox",
        )

    @staticmethod
    def _is_shell_tool(tool_name: str, tool_input: dict[str, Any] | None = None) -> bool:
        del tool_input
        return tool_name in {"shell", "Bash", "PowerShell", "ProcessStart"}


def load_permission_config(cwd: Path | None) -> PermissionConfig:
    del cwd
    global_path = get_global_permissions_path()
    if global_path.exists():
        return _read_permission_config(global_path)

    legacy_path = get_siyi_config_home() / "permissions.json"
    if legacy_path.exists():
        return _read_permission_config(legacy_path)

    return PermissionConfig()


def _read_permission_config(path: Path) -> PermissionConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raw = {}
    permissions_raw = raw.get("custom_permissions")
    if permissions_raw is None:
        permissions_raw = raw.get("permissions")
    if permissions_raw is None:
        permissions_raw = {
            "allow": raw.get("allow", []),
            "ask": raw.get("ask", []),
            "deny": raw.get("deny", []),
        }
    sandbox_raw = raw.get("sandbox") or {}
    config = PermissionConfig(
        custom_permissions=PermissionRules.model_validate(permissions_raw or {}),
        shell_exec_rules=ShellExecRules.model_validate(raw.get("shell_exec_rules") or {}),
        sandbox=SandboxPolicy.model_validate(sandbox_raw),
        source_path=path,
    )
    return config


def _default_permission_rules() -> PermissionRules:
    return PermissionRules(
        allow=[
            "Read(*)",
            "Glob(*)",
            "Grep(*)",
            "WebSearch(*)",
            "WebFetch(*)",
            "ToolSearch(*)",
            "ProcessRead(*)",
            "TaskGet(*)",
            "TaskList(*)",
            "Skill(*)",
            "ListMcpResourcesTool(*)",
            "ReadMcpResourceTool(*)",
            "AskUserQuestion(*)",
            "EnterPlanMode(*)",
            "ExitPlanMode(*)",
            "SendUserMessage(*)",
            "Config(get)",
            "Config(list)",
            "workflow(get)",
            "workflow(list)",
        ],
        ask=[
            "Bash(*)",
            "PowerShell(*)",
            "ProcessStart(*)",
            "ProcessWrite(*)",
            "ProcessStop(*)",
            "Write(*)",
            "Edit(*)",
            "NotebookEdit(*)",
            "Agent(*)",
            "SendMessage(*)",
            "TeamCreate(*)",
            "TeamDelete(*)",
            "TaskCreate(*)",
            "TaskUpdate(*)",
            "Config(set)",
            "Config(delete)",
            "workflow(run)",
        ],
    )


def _match_shell_exec_rules(
    rules: ShellExecRules,
    tool_name: str,
    tool_input: dict[str, Any],
) -> PermissionResult | None:
    command = str(tool_input.get("command") or "")
    lowered_command = command.lower()
    for decision, patterns in (
        ("deny", rules.deny_glob),
        ("ask", rules.ask_glob),
    ):
        for pattern in patterns:
            if fnmatch.fnmatchcase(lowered_command, pattern.lower()):
                return PermissionResult(
                    decision=decision,
                    reason=f"matched shell exec glob: {pattern}",
                    source="shell_exec_rules",
                )

    tokens = _shell_command_tokens(command)
    for decision, prefixes in (
        ("deny", rules.deny_prefix),
        ("ask", rules.ask_prefix),
        ("allow", rules.allow_prefix),
    ):
        for prefix in prefixes:
            prefix_tokens = _prefix_tokens(prefix)
            if prefix_tokens and _tokens_start_with(tokens, prefix_tokens):
                return PermissionResult(
                    decision=decision,
                    reason=f"matched shell exec prefix: {_format_prefix(prefix_tokens)}",
                    source="shell_exec_rules",
                )
    return None


def _shell_command_tokens(command: str) -> list[str]:
    try:
        tokens = _split_command(command, posix=True)
    except ValueError:
        try:
            tokens = _split_command(command, posix=False)
        except ValueError:
            return [part for part in command.strip().split() if part]
    return [_normalize_token(token) for token in tokens if token.strip()]


def _split_command(command: str, *, posix: bool) -> list[str]:
    import shlex

    lexer = shlex.shlex(command, posix=posix)
    lexer.whitespace_split = True
    lexer.commenters = ""
    return list(lexer)


def _prefix_tokens(prefix: list[str] | str) -> list[str]:
    if isinstance(prefix, str):
        return [_normalize_token(token) for token in prefix.split() if token.strip()]
    return [_normalize_token(token) for token in prefix if str(token).strip()]


def _tokens_start_with(tokens: list[str], prefix: list[str]) -> bool:
    if len(tokens) < len(prefix):
        return False
    return tokens[: len(prefix)] == prefix


def _normalize_token(token: object) -> str:
    return str(token).strip().strip("\"'").lower()


def _format_prefix(prefix: list[str]) -> str:
    return " ".join(prefix)


def _analyze_shell_command(tool_name: str, tool_input: dict[str, Any]):
    command = str(tool_input.get("command") or "")
    shell = str(tool_input.get("shell") or tool_name)
    if tool_name == "Bash" or shell == "Bash":
        return analyze_bash(command)
    return analyze_powershell(command)


def _coerce_permission_decision(value: object) -> PermissionDecision:
    if value in {"allow", "ask", "deny"}:
        return value  # type: ignore[return-value]
    return "ask"


def _combine_permissions(*results: PermissionResult) -> PermissionResult:
    denies = [result for result in results if result.decision == "deny"]
    if denies:
        return denies[0]
    asks = [result for result in results if result.decision == "ask"]
    if asks:
        return asks[0]
    return PermissionResult.allow(source="combined")


def _permission_payload(tool_name: str, tool_input: dict[str, Any]) -> str:
    if tool_name in {"Bash", "PowerShell", "ProcessStart", "shell"}:
        return str(tool_input.get("command") or "*")
    for key in ("action", "file_path", "path", "notebook_path", "uri", "query", "title", "task_id"):
        if key in tool_input and tool_input[key] is not None:
            return str(tool_input[key])
    return "*"


def _tool_input_summary(tool_name: str, tool_input: dict[str, Any]) -> str:
    payload = _permission_payload(tool_name, tool_input)
    if len(payload) > 240:
        payload = f"{payload[:240]}..."
    return f"{tool_name}({payload})"


def _rule_matches(rule: str, tool_name: str, payload: str, candidate: str) -> bool:
    rule = rule.strip()
    if not rule:
        return False
    if "(" not in rule or not rule.endswith(")"):
        return fnmatch.fnmatchcase(tool_name, rule) or fnmatch.fnmatchcase(candidate, rule)
    rule_tool, pattern = rule[:-1].split("(", 1)
    return fnmatch.fnmatchcase(tool_name, rule_tool) and fnmatch.fnmatchcase(payload, pattern)


def _is_default_read_tool(tool_name: str, tool_input: dict[str, Any]) -> bool:
    if tool_name in {
        "Read",
        "Glob",
        "Grep",
        "WebSearch",
        "WebFetch",
        "ToolSearch",
        "ProcessRead",
        "TaskGet",
        "TaskList",
        "Skill",
        "ListMcpResourcesTool",
        "ReadMcpResourceTool",
        "AskUserQuestion",
        "EnterPlanMode",
        "ExitPlanMode",
        "SendUserMessage",
    }:
        return True
    if tool_name == "Config":
        return str(tool_input.get("action") or "list") in {"get", "list"}
    if tool_name == "workflow":
        return str(tool_input.get("action") or "list") in {"get", "list"}
    return False


def _native_windows() -> bool:
    return sys.platform.startswith("win")
