"""Public vLLM adapter surface."""

from .client import VLLMClient, VLLMOpenAICompletionsClient
from .health import VLLMHealthResult, check_vllm_health
from .server import VLLMServerSpec, build_server_spec

__all__ = [
    "VLLMClient",
    "VLLMHealthResult",
    "VLLMOpenAICompletionsClient",
    "VLLMServerSpec",
    "build_server_spec",
    "check_vllm_health",
]
