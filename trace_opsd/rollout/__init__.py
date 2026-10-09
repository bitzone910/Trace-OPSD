"""Shared tool trajectory protocol used by TRACE and evaluation."""

from .clients import ToolHTTPClient, VLLMClient
from .config import RolloutConfig, load_config
from .models import RolloutMode, RolloutSample, TrajectoryRecord
from .parser import (
    extract_boxed_answer,
    extract_last_complete_answer,
    parse_first_complete_action,
)
from .prompts import build_direct_prompt, build_tool_prompt
from .verifier import AnswerVerifier

__all__ = [
    "AnswerVerifier",
    "RolloutConfig",
    "RolloutMode",
    "RolloutSample",
    "ToolHTTPClient",
    "TrajectoryRecord",
    "VLLMClient",
    "build_direct_prompt",
    "build_tool_prompt",
    "extract_boxed_answer",
    "extract_last_complete_answer",
    "load_config",
    "parse_first_complete_action",
]
