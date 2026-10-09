"""HTTP Completion client for the separately running vLLM server.

The implementation remains owned by ``rollout.clients`` for backward
compatibility with existing callers.  New code should import it from this
module; the adapter deliberately does not vendor or import vLLM internals.
"""

from trace_opsd.rollout.clients import VLLMOpenAICompletionsClient

VLLMClient = VLLMOpenAICompletionsClient

__all__ = ["VLLMClient", "VLLMOpenAICompletionsClient"]
