from __future__ import annotations

import asyncio
import re

from engine.message_schema import user_message
from llm.base import LLMAssistantDone, LLMTextDelta, LLMToolUse, LLMAdapter


TITLE_SYSTEM_PROMPT = (
    "You generate concise Chinese UI titles for agent sessions. "
    "Return only one short title, without quotes, punctuation wrappers, or explanation. "
    "Use at most 10 Chinese characters or 10 English words."
)

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_TRAILING_PUNCTUATION = "。.!！?？:：,，;；"


class SessionTitleAgent:
    def __init__(self, llm: LLMAdapter, *, timeout_seconds: float = 12.0) -> None:
        self.llm = llm
        self.timeout_seconds = timeout_seconds

    async def generate(self, first_user_input: str) -> str:
        return await asyncio.wait_for(
            self._generate_inner(first_user_input),
            timeout=self.timeout_seconds,
        )

    async def _generate_inner(self, first_user_input: str) -> str:
        deltas: list[str] = []
        async for event in self.llm.stream_chat(
            messages=[user_message(first_user_input)],
            system_prompt=TITLE_SYSTEM_PROMPT,
            tools=[],
            temperature=0.2,
        ):
            if isinstance(event, LLMTextDelta):
                deltas.append(event.delta)
            elif isinstance(event, LLMToolUse | LLMAssistantDone):
                continue
        title = normalize_session_title("".join(deltas))
        if not title:
            raise ValueError("empty generated title")
        return title


def fallback_session_title(first_user_input: str) -> str:
    title = normalize_session_title(first_user_input)
    return title or "新会话"


def normalize_session_title(text: str) -> str:
    title = text.strip().strip("\"'`“”‘’")
    title = re.sub(r"\s+", " ", title)
    title = title.splitlines()[0].strip() if title else ""
    title = title.rstrip(_TRAILING_PUNCTUATION)
    return _clamp_session_title(title)


def _clamp_session_title(title: str) -> str:
    if not title:
        return ""
    if _CJK_RE.search(title):
        return title[:10].rstrip(f"{_TRAILING_PUNCTUATION} ")
    words = title.split()
    if len(words) <= 10:
        return title
    return " ".join(words[:10]).rstrip(_TRAILING_PUNCTUATION)
