"""Construct the pinned vLLM OpenAI-server launch command."""

from __future__ import annotations

from dataclasses import dataclass

from trace_opsd.rollout.config import RolloutConfig


@dataclass(frozen=True)
class VLLMServerSpec:
    """Resolved process settings consumed by the one-click launcher."""

    command: tuple[str, ...]
    host: str
    port: int
    gpu: str
    conda_env: str
    health_timeout_seconds: float


def build_server_spec(config: RolloutConfig) -> VLLMServerSpec:
    """Build a deterministic vLLM 0.11 OpenAI server command from YAML."""

    service = config.services.vllm
    args = service.model_args
    command = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        config.model.name_or_path,
        "--served-model-name",
        config.model.served_model_name,
        "--host",
        service.host,
        "--port",
        str(service.port),
        "--dtype",
        args.dtype,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--generation-config",
        args.generation_config,
    ]
    if args.enable_prefix_caching:
        command.append("--enable-prefix-caching")
    return VLLMServerSpec(
        command=tuple(command),
        host=service.host,
        port=service.port,
        gpu=str(service.gpu),
        conda_env=service.conda_env,
        health_timeout_seconds=service.health_timeout_seconds,
    )


__all__ = ["VLLMServerSpec", "build_server_spec"]
