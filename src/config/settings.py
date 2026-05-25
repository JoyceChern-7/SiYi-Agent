from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from config.paths import get_session_dir
from config.user_settings import load_user_settings, resolve_model_from_tier

DEFAULT_SYSTEM_PROMPT = (
    "You are SiYi, a careful general-purpose agent. "
    "Use tools when fresh data, files, or calculations are required. State "
    "uncertainty clearly and avoid pretending to have verified information "
    "when you have not."
)


class ModelPricing(BaseModel):
    input_per_million: float = 0.0
    output_per_million: float = 0.0


class ModelSettings(BaseModel):
    provider: str = "openai-compatible"
    model: str = "gpt-4o-mini"
    api_key: SecretStr | None = None
    base_url: str | None = None
    timeout_seconds: float = Field(default=600.0, gt=0)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    print_thinking: bool = False
    temperature: float = Field(default=1, ge=0.0, le=2.0)
    pricing: ModelPricing = Field(default_factory=ModelPricing)


class ToolSettings(BaseModel):
    web_search_enabled: bool = True
    read_file_enabled: bool = True
    shell_enabled: bool = True
    shell_requires_approval: bool = True
    sandbox_enabled: bool = False
    sandbox_fail_if_unavailable: bool = False
    allow_unsandboxed_commands: bool = True
    max_tool_result_chars: int = Field(default=20_000, ge=1_000)


class RuntimeSettings(BaseModel):
    project_root: Path
    debug: bool = False
    non_interactive: bool = False
    initial_prompt: str | None = None
    resume: str | bool | None = None
    session_dir: Path
    max_context_tokens: int = Field(default=1_000_000)
    max_output_tokens: int = Field(default=1_000_000)
    compaction_enabled: bool = True
    auto_compact_enabled: bool = True

    @field_validator("project_root", "session_dir")
    @classmethod
    def expand_paths(cls, value: Path) -> Path:
        return value.expanduser().resolve()


class AppSettings(BaseModel):
    model: ModelSettings
    tools: ToolSettings
    runtime: RuntimeSettings


class EnvSettings(BaseSettings):
    debug: bool = False
    max_context_tokens: int = 200_000
    max_output_tokens: int = 8_000

    model_config = SettingsConfigDict(env_prefix="SIYI_", extra="ignore")


def _first_value(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _env_value(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _env_int(*names: str) -> int | None:
    raw = _env_value(*names)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable must be an integer: {names[0]}") from exc


def _env_truthy(*names: str) -> bool:
    raw = _env_value(*names)
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_settings(options, project_root: Path) -> AppSettings:
    user_settings = load_user_settings()
    env = EnvSettings()
    default_model_settings = ModelSettings()
    session_dir = get_session_dir()
    compact_disabled = _env_truthy("SIYI_DISABLE_COMPACT")
    auto_compact_disabled = compact_disabled or _env_truthy("SIYI_DISABLE_AUTO_COMPACT")

    raw_api_key = _first_value(
        options.api_key,
        _env_value("SIYI_API_KEY", "OPENAI_API_KEY"),
        user_settings.api_key,
    )
    api_key = (
        SecretStr(raw_api_key)
        if raw_api_key
        else default_model_settings.api_key
    )

    base_url = _first_value(
        options.base_url,
        _env_value("SIYI_BASE_URL", "OPENAI_BASE_URL"),
        user_settings.base_url,
        default_model_settings.base_url,
    )
    api_timeout_ms = _first_value(
        _env_int("SIYI_API_TIMEOUT_MS", "API_TIMEOUT_MS"),
        600_000,
    )
    model_tier = _first_value(
        options.model_tier,
        _env_value("SIYI_MODEL_TIER"),
        user_settings.default_tier,
    )
    tier_model = resolve_model_from_tier(user_settings, model_tier)

    runtime = RuntimeSettings(
        project_root=project_root,
        debug=bool(options.debug or env.debug),
        non_interactive=bool(options.internal_worker),
        initial_prompt=options.prompt,
        resume=options.resume,
        session_dir=session_dir,
        max_context_tokens=env.max_context_tokens,
        max_output_tokens=env.max_output_tokens,
        compaction_enabled=not compact_disabled,
        auto_compact_enabled=not auto_compact_disabled,
    )
    model = ModelSettings(
        provider=_first_value(
            _env_value("SIYI_PROVIDER"),
            user_settings.provider,
            default_model_settings.provider,
        ),
        model=_first_value(
            options.model,
            _env_value("SIYI_MODEL"),
            tier_model,
            default_model_settings.model,
        ),
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=float(api_timeout_ms) / 1000.0,
        system_prompt=(
            options.system_prompt
            or _env_value("SIYI_SYSTEM_PROMPT")
            or DEFAULT_SYSTEM_PROMPT
        ),
        print_thinking=options.print_thinking,
    )
    tools = ToolSettings()
    return AppSettings(
        model=model,
        tools=tools,
        runtime=runtime,
    )
