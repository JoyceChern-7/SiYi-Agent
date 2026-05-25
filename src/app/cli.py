"""
数据结构:
argv: Sequence[str], 代表命令行参数列表. 例如 ["--model", "gpt-4", "--system-prompt", "You are a helpful assistant"]

namespace: 一个 argparse.ArgumentParser 用于解析命令行参数并将它们转换为 CLIOptions 对象.
解析命令行中的以下参数:
- model: string(可选, 表示主代理循环使用的模型名称)
- model-tier: string(可选, 表示使用 swift/balanced/depth 中的模型挡位)
- api-key: string(可选, 覆盖本次运行的 API key)
- base-url: string(可选, 覆盖本次运行的 OpenAI-compatible API base URL)
- system-prompt: string(可选, 覆盖此会话的默认系统提示)
- resume: string or bool(可选, 如果提供了 id, 则表示通过 id 恢复先前的会话. 如果没有提供 id, 则恢复最新的会话, 如果没有提供参数, 则不恢复任何会话)
- debug: bool(可选, 表示是否启用调试日志记录)
- print-thinking: bool(可选, 表示是否打印 model 的 CoT)
- cwd: Path(可选, 兼容参数, 作为 project_root 使用)
- version: 显示版本信息并退出

CLIOptions: 一个 dataclass, 包含从命令行参数解析的选项字段.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Sequence

from app.main import run
from app.options import CLIOptions

VERSION = "0.1.0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="siyi",
        description="Run the SiYi command line interface.",
    )
    parser.add_argument("--model", help="Model name for the main agent loop.")
    parser.add_argument(
        "--model-tier",
        choices=("swift", "balanced", "depth"),
        help="Model tier to use when --model is not supplied.",
    )
    parser.add_argument(
        "--api-key",
        help="API key for the current run. Prefer /login or environment variables for daily use.",
    )
    parser.add_argument(
        "--base-url",
        help="OpenAI-compatible API base URL for the current run.",
    )
    parser.add_argument(
        "--system-prompt",
        help="Override the default system prompt for this session.",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=None,
        help="Resume a previous session by id. If no id is supplied, resume the latest session.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "--print-thinking",
        action="store_true",
        help="Print thinking events when the model provider emits them.",
    )
    parser.add_argument(
        "--cwd",
        type=Path,
        help="Project root for tools and session context.",
    )
    parser.add_argument(
        "--internal-worker-prompt",
        dest="internal_worker_prompt",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> CLIOptions:
    namespace = build_parser().parse_args(argv)
    cwd = namespace.cwd.expanduser().resolve() if namespace.cwd else None
    return CLIOptions(
        prompt=namespace.internal_worker_prompt,
        internal_worker=namespace.internal_worker_prompt is not None,
        model=namespace.model,
        model_tier=namespace.model_tier,
        api_key=namespace.api_key,
        base_url=namespace.base_url,
        system_prompt=namespace.system_prompt,
        resume=namespace.resume,
        debug=namespace.debug,
        print_thinking=namespace.print_thinking,
        cwd=cwd,
        non_interactive=namespace.internal_worker_prompt is not None,
    )


def main(argv: Sequence[str] | None = None) -> int:
    if os.name == "nt":
        os.environ.setdefault("NoDefaultCurrentDirectoryInExePath", "1")

    options = parse_args(argv)
    return asyncio.run(run(options))


if __name__ == "__main__":
    raise SystemExit(main())
