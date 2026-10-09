"""Public Trace-OPSD rollout-tool API."""

from __future__ import annotations

from .config import ToolConfig
from .runtime import (
    ToolRuntime,
    call_tool,
    initialize_tools,
    python_executor_tool,
    search_tool,
)


def create_app(runtime: ToolRuntime):
    """Lazily import FastAPI so direct function use stays dependency-light."""

    from .service import create_app as _create_app

    return _create_app(runtime)


__all__ = [
    "ToolConfig",
    "ToolRuntime",
    "call_tool",
    "create_app",
    "initialize_tools",
    "python_executor_tool",
    "search_tool",
]
