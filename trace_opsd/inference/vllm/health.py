"""Read-only health checks for a local vLLM OpenAI-compatible server."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import httpx


@dataclass(frozen=True)
class VLLMHealthResult:
    healthy: bool
    base_url: str
    served_models: tuple[str, ...] = ()
    error: str | None = None


async def check_vllm_health(
    base_url: str,
    *,
    expected_model: str | None = None,
    timeout_seconds: float = 3.0,
) -> VLLMHealthResult:
    """Check both vLLM's ``/health`` and OpenAI ``/v1/models`` routes."""

    normalized = base_url.rstrip("/")
    root = normalized[:-3] if normalized.endswith("/v1") else normalized
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            health = await client.get(f"{root}/health")
            health.raise_for_status()
            models_response = await client.get(f"{normalized}/models")
            models_response.raise_for_status()
            payload: Any = models_response.json()
        data = payload.get("data", []) if isinstance(payload, Mapping) else []
        models = tuple(
            str(item["id"])
            for item in data
            if isinstance(item, Mapping) and item.get("id") is not None
        )
        if expected_model is not None and expected_model not in models:
            return VLLMHealthResult(
                False,
                normalized,
                models,
                f"expected served model {expected_model!r} was not advertised",
            )
        return VLLMHealthResult(True, normalized, models)
    except Exception as exc:
        return VLLMHealthResult(False, normalized, error=f"{type(exc).__name__}: {exc}")


__all__ = ["VLLMHealthResult", "check_vllm_health"]
