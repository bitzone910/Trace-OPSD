"""Validated configuration for Trace-OPSD rollout synthesis."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from trace_opsd.assets import asset_path, resolve_work_root

_ENV_PATTERN = re.compile(
    r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}"
)


def _expand_environment(value: Any) -> Any:
    """Recursively expand ``${VAR}`` and ``${VAR:-default}`` strings."""

    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            name = match.group("name")
            configured = os.environ.get(name)
            if configured is not None and configured != "":
                return configured
            default = match.group("default")
            if default is not None:
                return default
            raise ValueError(f"environment variable {name!r} is not set")

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_expand_environment(item) for item in value)
    if isinstance(value, Mapping):
        return {str(key): _expand_environment(item) for key, item in value.items()}
    return value


class _ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class _ExtensionConfig(BaseModel):
    """Validated mapping that deliberately permits launcher-owned fields."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)


class FutureTrainingConfig(_ExtensionConfig):
    framework: str = "verl"
    version: str | None = None
    install_now: bool = False


class ProjectConfig(_ExtensionConfig):
    name: str = "Trace-OPSD"
    stage: str = "rollout_synthesis"
    description: str = "Raw direct and tool-integrated rollout synthesis"
    prompt_profile: str = "search_r1_v0"
    raw_rollouts_only: bool = True
    future_training: FutureTrainingConfig = Field(default_factory=FutureTrainingConfig)


class VLLMModelArgsConfig(_ExtensionConfig):
    dtype: str = "bfloat16"
    tensor_parallel_size: int = Field(default=1, gt=0)
    max_model_len: int = Field(default=32768, gt=0)
    gpu_memory_utilization: float = Field(default=0.6, gt=0.0, le=1.0)
    enable_prefix_caching: bool = True
    generation_config: str = "vllm"


class VLLMServiceConfig(_ExtensionConfig):
    conda_env: str = "trace_opsd"
    url: str = "http://127.0.0.1:8000/v1"
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    gpu: str | int = 0
    start: bool = True
    health_timeout_seconds: float = Field(default=600.0, gt=0.0)
    model_args: VLLMModelArgsConfig = Field(default_factory=VLLMModelArgsConfig)


class ToolServiceConfig(_ExtensionConfig):
    conda_env: str = "trace_opsd"
    url: str = "http://127.0.0.1:8002"
    host: str = "127.0.0.1"
    port: int = Field(default=8002, ge=1, le=65535)
    gpu: str | int = 1
    start: bool = True
    health_timeout_seconds: float = Field(default=600.0, gt=0.0)
    general_index_path: str | None = None
    general_corpus_path: str | None = None
    retriever_model: str = Field(
        default_factory=lambda: str(asset_path("models", "e5-base-v2"))
    )
    device: str = "cpu"
    max_length: int = Field(default=256, gt=0)
    use_fp16: bool = False
    faiss_gpu: bool = False
    retrieval_batch_size: int = Field(default=128, gt=0)
    corpus_num_proc: int = Field(default=4, gt=0)
    search_topk: int = Field(default=3, gt=0)
    result_max_chars: int = Field(default=100_000, gt=0)
    raw_text_max_chars: int = Field(default=100_000, gt=0)
    python_code_max_chars: int = Field(default=20_000, gt=0)
    search_query_max_chars: int = Field(default=2000, gt=0)
    python_timeout_seconds: float = Field(default=10.0, gt=0.0)
    python_cpu_seconds: int = Field(default=10, gt=0)
    python_memory_mb: int = Field(default=512, gt=0)
    python_output_bytes: int = Field(default=65_536, gt=0)
    python_nofile: int = Field(default=64, gt=0)
    python_max_concurrency: int = Field(default=8, gt=0)
    python_queue_timeout_seconds: float = Field(default=30.0, gt=0.0)
    executor_uid: int | None = Field(default=None, ge=0)
    executor_gid: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _executor_ids_are_paired(self) -> "ToolServiceConfig":
        if (self.executor_uid is None) != (self.executor_gid is None):
            raise ValueError("executor_uid and executor_gid must be set together")
        return self


class ServicesConfig(_ExtensionConfig):
    """Optional service-launch metadata consumed by shell/CLI wrappers."""

    vllm: VLLMServiceConfig = Field(default_factory=VLLMServiceConfig)
    tools: ToolServiceConfig = Field(default_factory=ToolServiceConfig)


class InputConfig(_ConfigModel):
    path: str | None = None
    id_column: str = "id"
    problem_column: str = "problem"
    ground_truth_column: str = "ground_truth"

    @field_validator("id_column", "problem_column", "ground_truth_column")
    @classmethod
    def _column_is_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("column names must not be empty")
        return value


class OutputConfig(_ConfigModel):
    root_dir: str = Field(default_factory=lambda: str(resolve_work_root() / "rollouts"))
    run_name: str = "tool-trajectories"
    shard_size: int = Field(default=5000, gt=0)
    max_resume_cycles: int = Field(default=3, gt=0)

    @field_validator("root_dir", "run_name")
    @classmethod
    def _output_name_is_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("output paths and names must not be empty")
        return value


class ModelConfig(_ConfigModel):
    name_or_path: str = Field(
        default_factory=lambda: str(asset_path("models", "Qwen2.5-3B-Instruct"))
    )
    served_model_name: str = "qwen2.5-3b-instruct"
    base_url: str = "http://127.0.0.1:8000/v1"
    replica_base_urls: tuple[str, ...] = ()
    api_key: str = "EMPTY"

    @field_validator("name_or_path", "served_model_name")
    @classmethod
    def _model_name_is_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model names must not be empty")
        return value

    @field_validator("base_url")
    @classmethod
    def _base_url_is_http(cls, value: str) -> str:
        normalized = value.rstrip("/")
        if not normalized.startswith(("http://", "https://")):
            raise ValueError("base_url must use http:// or https://")
        return normalized

    @field_validator("replica_base_urls")
    @classmethod
    def _replica_urls_are_valid(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.rstrip("/") for value in values)
        if any(not value.startswith(("http://", "https://")) for value in normalized):
            raise ValueError("replica_base_urls must use http:// or https://")
        if len(set(normalized)) != len(normalized):
            raise ValueError("replica_base_urls must be unique")
        return normalized


class SamplingConfig(_ConfigModel):
    direct_samples: int = Field(default=1, ge=0)
    tool_samples: int = Field(default=1, ge=0)
    temperature: float = Field(default=0.7, ge=0.0)
    top_p: float = Field(default=0.8, gt=0.0, le=1.0)
    top_k: int = Field(default=20, ge=-1)
    repetition_penalty: float = Field(default=1.05, gt=0.0)
    max_tokens: int = Field(default=8192, gt=0)
    ignore_eos: bool = False
    base_seed: int = Field(default=42, ge=0)

    @model_validator(mode="after")
    def _has_samples(self) -> "SamplingConfig":
        if self.direct_samples + self.tool_samples == 0:
            raise ValueError("at least one sampling mode must be enabled")
        return self


class RolloutPolicyConfig(_ConfigModel):
    max_start_length: int = Field(default=8192, gt=0)
    max_prompt_length: int = Field(default=24576, gt=0)
    max_response_length: int | None = Field(default=None, gt=0)
    max_observation_length: int = Field(default=2048, gt=0)
    max_tool_calls: int = Field(default=5, gt=0)
    max_consecutive_invalid_actions: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _lengths_are_consistent(self) -> "RolloutPolicyConfig":
        if self.max_start_length > self.max_prompt_length:
            raise ValueError("max_start_length must not exceed max_prompt_length")
        return self


class ToolsConfig(_ConfigModel):
    base_url: str = "http://127.0.0.1:8002"
    batch_endpoint: str = "/tools/batch"
    endpoints: dict[str, str] = Field(
        default_factory=lambda: {
            "python": "/tools/python",
            "search": "/tools/search",
        }
    )
    enabled: tuple[str, ...] = ("python", "search")
    request_timeout_seconds: float = Field(default=30.0, gt=0.0)
    max_retries: int = Field(default=2, ge=0)
    retry_backoff_seconds: float = Field(default=1.0, ge=0.0)

    @field_validator("base_url")
    @classmethod
    def _tool_url_is_http(cls, value: str) -> str:
        normalized = value.rstrip("/")
        if not normalized.startswith(("http://", "https://")):
            raise ValueError("tools.base_url must use http:// or https://")
        return normalized

    @model_validator(mode="after")
    def _enabled_tools_have_endpoints(self) -> "ToolsConfig":
        missing = [name for name in self.enabled if name not in self.endpoints]
        if missing:
            raise ValueError(
                "enabled tools are missing endpoints: " + ", ".join(missing)
            )
        for name, endpoint in self.endpoints.items():
            if not name or not endpoint.startswith("/"):
                raise ValueError(
                    "tool endpoint names must be nonempty and paths must start with '/'"
                )
        if not self.batch_endpoint.startswith("/"):
            raise ValueError("tools.batch_endpoint must start with '/'")
        return self


class PilotConfig(_ConfigModel):
    enabled: bool = True
    size: int = Field(default=300, gt=0)
    auto_continue: bool = True
    transport_failure_rate_max: float = Field(default=0.01, ge=0.0, le=1.0)
    tool_service_availability_rate_min: float = Field(default=0.95, ge=0.0, le=1.0)
    answer_extraction_rate_min: float = Field(default=0.80, ge=0.0, le=1.0)
    verifier_completion_rate_min: float = Field(default=0.995, ge=0.0, le=1.0)


class RuntimeConfig(_ConfigModel):
    source_batch_size: int = Field(default=64, gt=0)
    request_concurrency: int = Field(default=16, gt=0)
    fallback_concurrency: int = Field(default=8, gt=0)
    generation_timeout_seconds: float = Field(default=300.0, gt=0.0)
    generation_max_retries: int = Field(default=2, ge=0)
    generation_retry_backoff_seconds: float = Field(default=1.0, ge=0.0)

    @model_validator(mode="after")
    def _fallback_does_not_exceed_primary(self) -> "RuntimeConfig":
        if self.fallback_concurrency > self.request_concurrency:
            raise ValueError("fallback_concurrency must not exceed request_concurrency")
        return self


class RolloutConfig(_ConfigModel):
    """Resolved, immutable-in-shape configuration used by all rollout code."""

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    services: ServicesConfig = Field(default_factory=ServicesConfig)
    input: InputConfig = Field(default_factory=InputConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    rollout: RolloutPolicyConfig = Field(default_factory=RolloutPolicyConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    pilot: PilotConfig = Field(default_factory=PilotConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @model_validator(mode="after")
    def _generation_fits_model_context(self) -> "RolloutConfig":
        required = self.rollout.max_prompt_length + self.sampling.max_tokens
        available = self.services.vllm.model_args.max_model_len
        if required > available:
            raise ValueError(
                "rollout.max_prompt_length + sampling.max_tokens must not "
                "exceed services.vllm.model_args.max_model_len"
            )
        if (
            self.rollout.max_response_length is not None
            and self.rollout.max_start_length + self.rollout.max_response_length
            > available
        ):
            raise ValueError(
                "rollout.max_start_length + rollout.max_response_length must "
                "not exceed services.vllm.model_args.max_model_len"
            )
        return self

    @model_validator(mode="after")
    def _inherit_service_urls(self) -> "RolloutConfig":
        if "base_url" not in self.model.model_fields_set:
            self.model.base_url = self.services.vllm.url.rstrip("/")
        if "base_url" not in self.tools.model_fields_set:
            self.tools.base_url = self.services.tools.url.rstrip("/")
        if (
            self.model.replica_base_urls
            and self.model.base_url != self.model.replica_base_urls[0]
        ):
            raise ValueError(
                "model.base_url must equal the first replica_base_urls entry"
            )
        return self

    @classmethod
    def load_yaml(cls, path: str | Path) -> "RolloutConfig":
        """Load, expand environment references, and validate a YAML file."""

        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - deployment dependency
            raise RuntimeError("PyYAML is required to load rollout config") from exc

        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        if payload is None:
            payload = {}
        if not isinstance(payload, Mapping):
            raise ValueError("rollout YAML must contain a mapping at the root")
        config = cls.model_validate(_expand_environment(payload))
        if config.input.path is None or not config.input.path.strip():
            raise ValueError("input.path must be configured in rollout YAML")
        return config

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def load_config(path: str | Path) -> RolloutConfig:
    """Compatibility function for callers that prefer a module-level loader."""

    return RolloutConfig.load_yaml(path)


__all__ = [
    "InputConfig",
    "FutureTrainingConfig",
    "ModelConfig",
    "OutputConfig",
    "PilotConfig",
    "ProjectConfig",
    "RolloutConfig",
    "RolloutPolicyConfig",
    "RuntimeConfig",
    "SamplingConfig",
    "ServicesConfig",
    "ToolServiceConfig",
    "ToolsConfig",
    "VLLMModelArgsConfig",
    "VLLMServiceConfig",
    "load_config",
]
