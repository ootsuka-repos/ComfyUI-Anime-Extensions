"""Shared result and error types for the local OpenAI-compatible LLM."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AgentTextResult:
    text: str
    duration_ms: int | None = None
    total_cost_usd: float | None = None
    usage: dict[str, Any] | None = None


WROTE_RE = re.compile(r"^`*WROTE:\s*(\S.*?)\s*`*$", re.MULTILINE)


class AgentLimitError(RuntimeError):
    """The configured local model server rejected a request for a limit."""


class AgentRefusalError(RuntimeError):
    """The model responded with a refusal instead of the requested output."""


class AgentResultError(RuntimeError):
    """The local OpenAI-compatible API returned an invalid result."""


class AgentTransientError(AgentResultError):
    """A request failed in a way that an existing caller may retry."""


_REFUSAL_RE = re.compile(
    r"(?:"
    r"申し訳(?:ありません|ない)(?:が|です)?|"
    r"(?:対応|お手伝い|生成|作成|提供)(?:でき|いたしかね)(?:ません|ない)|"
    r"I\s*(?:cannot|can't|won't)\s+(?:assist|help|generate|create|provide)|"
    r"I(?:'m| am)\s+(?:not able|unable)\s+to\s+(?:assist|help|generate|create|provide)|"
    r"against\s+(?:my|the)\s+(?:usage\s+)?(?:guidelines|policies)|"
    r"(?:usage|content)\s+polic(?:y|ies)"
    r")",
    re.IGNORECASE,
)


def _looks_like_structured_payload(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith(("{", "[")):
        return True
    head = stripped[:120]
    if "```json" in head.lower():
        return True
    return "```" in head and ("{" in head or "[" in head)


def looks_like_refusal(text: str) -> bool:
    """Avoid treating structured output that quotes a refusal as a refusal."""

    return bool(
        text
        and text.strip()
        and not _looks_like_structured_payload(text)
        and _REFUSAL_RE.search(text)
    )


def raise_if_refusal_text(text: str) -> None:
    if not looks_like_refusal(text):
        return
    snippet = " ".join(text.strip().split())
    if len(snippet) > 200:
        snippet = snippet[:200] + "…"
    raise AgentRefusalError(f"Local model refused the request: {snippet}")
