"""Shared parsing and result-envelope helpers for rollout tools."""

from __future__ import annotations

import re
from typing import Any

RESULT_MAX_CHARS = 2000
TRUNCATION_SUFFIX = "\n...[output truncated]"


def _neutralize_result_tags(content: str) -> str:
    """Prevent tool-controlled text from closing the rollout result envelope."""

    return re.sub(
        r"<(/?)result>",
        lambda match: f"&lt;{match.group(1)}result&gt;",
        content,
        flags=re.IGNORECASE,
    )


def wrap_result(content: Any, *, max_chars: int = RESULT_MAX_CHARS) -> str:
    """Return a bounded, protocol-safe ``<result>`` observation."""

    if max_chars <= 0:
        max_chars = RESULT_MAX_CHARS
    try:
        normalized = content if isinstance(content, str) else str(content)
    except BaseException:
        normalized = "Error: tool result could not be converted to text."
    normalized = _neutralize_result_tags(normalized.strip())
    if len(normalized) > max_chars:
        if max_chars <= len(TRUNCATION_SUFFIX):
            normalized = TRUNCATION_SUFFIX[-max_chars:]
        else:
            keep = max_chars - len(TRUNCATION_SUFFIX)
            normalized = normalized[:keep] + TRUNCATION_SUFFIX
    return f"<result>\n{normalized}\n</result>"


def extract_tagged_content(raw_text: Any, tag: str) -> str | None:
    """Extract the final complete occurrence of ``tag`` from generated text."""

    if not isinstance(raw_text, str) or not isinstance(tag, str) or not tag:
        return None
    pattern = rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>"
    matches = re.findall(pattern, raw_text, flags=re.DOTALL)
    if not matches:
        return None
    return matches[-1].strip()
