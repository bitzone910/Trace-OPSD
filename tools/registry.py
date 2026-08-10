"""Compatibility registry for rollout loops that dispatch by tool name."""

from __future__ import annotations

from collections.abc import Callable

from .runtime import (
    call_tool,
    python_executor_tool,
    search_knowledge_tool,
    search_theory_tool,
)


TOOL_REGISTRY: dict[str, Callable[[object], str]] = {
    "python": python_executor_tool,
    "search_knowledge": search_knowledge_tool,
    "search_theory": search_theory_tool,
}

__all__ = ["TOOL_REGISTRY", "call_tool"]
