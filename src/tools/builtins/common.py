from __future__ import annotations

import asyncio
import fnmatch
import html
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import httpx

from config.paths import get_global_skills_dir, get_project_state_dir, get_skill_paths_path
from runtime.ids import new_id
from tools.base import BaseTool, ToolContext, ToolResult, emit_tool_output_delta, truncate_text
from tools.governance import PermissionResult, ValidationResult
from tools.shell_analysis import analyze_bash, analyze_powershell


JsonObject = dict[str, Any]

DEFAULT_OUTPUT_IDLE_TIMEOUT_MS = 500 
# 交互式进程在没有输出时的空闲超时时间, 超过这个时间会直接将当前进程的输出结果返回给模型, 但不杀进程

DEFAULT_TTY_TIMEOUT_MS = 300_000
# 交互式进程的最大等待时间, 超过这个时间会强制杀死进程并返回结果.

DEFAULT_NON_TTY_TIMEOUT_MS = 10_000
# 非交互式进程的最大等待时间, 超过这个时间会强制杀死进程并返回结果.

MAX_TOOL_WAIT_MS = 30_000
# 工具的最大等待时间, 超过这个时间会强制返回结果, 但不杀进程.

def _schema(properties: JsonObject, required: list[str] | None = None) -> JsonObject:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }

def _ok(content: str, context: ToolContext, data: JsonObject | None = None) -> ToolResult:
    return ToolResult(
        success=True,
        content=truncate_text(content, context.max_result_chars),
        data=data,
    )

def _error(message: str, context: ToolContext, data: JsonObject | None = None) -> ToolResult:
    return ToolResult(
        success=False,
        content=truncate_text(message, context.max_result_chars),
        data=data,
        error=message,
    )

def _project_root(context: ToolContext) -> Path:
    return Path(context.project_root).expanduser().resolve()

def _resolve_path(context: ToolContext, value: str | None, *, default: str = ".") -> Path:
    raw = Path(value or default).expanduser()
    if not raw.is_absolute():
        raw = _project_root(context) / raw
    return raw.resolve()

def _state_dir(context: ToolContext) -> Path:
    path = get_project_state_dir(_project_root(context))
    path.mkdir(parents=True, exist_ok=True)
    return path

def _read_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def _write_json_file(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

def _coerce_source(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return str(value).splitlines(keepends=True)

def _timeout_seconds(raw_input: JsonObject, *, default: float = 10.0) -> float:
    if raw_input.get("timeout_ms") is None:
        return default
    try:
        value = float(raw_input["timeout_ms"]) / 1000.0
    except (TypeError, ValueError):
        return default
    return max(0.001, value)

__all__ = [
    "Any",
    "Callable",
    "HTMLParser",
    "Path",
    "asyncio",
    "dataclass",
    "field",
    "fnmatch",
    "html",
    "httpx",
    "json",
    "os",
    "parse_qs",
    "quote_plus",
    "re",
    "signal",
    "shutil",
    "subprocess",
    "suppress",
    "sys",
    "time",
    "unquote",
    "urlparse",
    "get_global_skills_dir",
    "get_project_state_dir",
    "get_skill_paths_path",
    "new_id",
    "BaseTool",
    "ToolContext",
    "ToolResult",
    "emit_tool_output_delta",
    "truncate_text",
    "PermissionResult",
    "ValidationResult",
    "analyze_bash",
    "analyze_powershell",
    "JsonObject",
    "DEFAULT_OUTPUT_IDLE_TIMEOUT_MS",
    "DEFAULT_TTY_TIMEOUT_MS",
    "DEFAULT_NON_TTY_TIMEOUT_MS",
    "MAX_TOOL_WAIT_MS",
    "_schema",
    "_ok",
    "_error",
    "_project_root",
    "_resolve_path",
    "_state_dir",
    "_read_json_file",
    "_write_json_file",
    "_coerce_source",
    "_timeout_seconds",
]
