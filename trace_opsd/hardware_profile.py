"""Load and validate Trace-OPSD GPU hardware profiles without bootstrap deps."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from trace_opsd.compat import read_json

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_ROOT = REPO_ROOT / "configs" / "hardware"
SUPPORTED_PROFILES = ("a800",)


class HardwareProfileError(ValueError):
    """Raised when a hardware profile or the active GPU is incompatible."""


@dataclass(frozen=True)
class FlashAttentionArtifact:
    version: str
    filename: str
    url: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class HardwareProfile:
    path: Path
    name: str
    display_name: str
    workload: str
    vendored_frameworks: tuple[str, ...]
    name_patterns: tuple[str, ...]
    compute_capability: tuple[int, int]
    cache_arch: str
    torch_cuda_arch_list: str
    minimum_driver: str
    final_gpu_count: int
    distributed_gpu_counts: tuple[int, ...]
    runtime_config: Path
    dependency_input: Path
    dependency_lock: Path
    flash_attention: FlashAttentionArtifact
    inference_gpu_memory_utilization: float
    vllm_attention_backend: str
    xformers_kernel_policy: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["path"] = str(self.path)
        value["runtime_config"] = str(self.runtime_config)
        value["dependency_input"] = str(self.dependency_input)
        value["dependency_lock"] = str(self.dependency_lock)
        return value


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise HardwareProfileError(f"{name} must be a mapping")
    return value


def _repo_path(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise HardwareProfileError(f"{name} must be a nonempty path")
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise HardwareProfileError(f"{name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise HardwareProfileError(f"{name} must be a positive integer") from exc
    if parsed < 1:
        raise HardwareProfileError(f"{name} must be a positive integer")
    return parsed


def _load_document(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HardwareProfileError(f"cannot read hardware profile: {path}") from exc
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise HardwareProfileError(
                f"{path} is not JSON-compatible YAML and PyYAML is unavailable"
            ) from exc
        value = yaml.safe_load(text)
    return _mapping(value, "hardware profile")


def load_hardware_profile(name: str) -> HardwareProfile:
    """Load one of the repository-owned hardware profiles."""
    if name not in SUPPORTED_PROFILES:
        raise HardwareProfileError(
            f"unknown hardware profile {name!r}; expected one of: "
            + ", ".join(SUPPORTED_PROFILES)
        )
    path = (PROFILE_ROOT / f"{name}.yaml").resolve()
    root = _load_document(path)
    if root.get("schema_version") != 1:
        raise HardwareProfileError(
            f"unsupported profile schema: {root.get('schema_version')!r}"
        )
    if root.get("name") != name:
        raise HardwareProfileError(
            f"profile name mismatch: requested {name!r}, found {root.get('name')!r}"
        )
    gpu = _mapping(root.get("gpu"), "gpu")
    flash = _mapping(root.get("flash_attention"), "flash_attention")
    inference = _mapping(root.get("inference"), "inference")
    capability = gpu.get("compute_capability")
    if not (
        isinstance(capability, list)
        and len(capability) == 2
        and all(isinstance(part, int) and part >= 0 for part in capability)
    ):
        raise HardwareProfileError("gpu.compute_capability must be [major, minor]")
    patterns = gpu.get("name_patterns")
    if (
        not isinstance(patterns, list)
        or not patterns
        or not all(isinstance(item, str) and item for item in patterns)
    ):
        raise HardwareProfileError("gpu.name_patterns must be a nonempty string list")
    distributed = gpu.get("distributed_gpu_counts")
    if not isinstance(distributed, list) or not distributed:
        raise HardwareProfileError("gpu.distributed_gpu_counts must be nonempty")
    distributed_counts = tuple(
        _positive_int(item, "gpu.distributed_gpu_counts") for item in distributed
    )
    cache_arch = str(gpu.get("cache_arch", ""))
    expected_arch = f"sm{capability[0]}{capability[1]}"
    if cache_arch != expected_arch:
        raise HardwareProfileError(
            f"gpu.cache_arch must be {expected_arch}, found {cache_arch!r}"
        )
    sha256 = str(flash.get("sha256", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise HardwareProfileError(
            "flash_attention.sha256 must be 64 lowercase hex chars"
        )
    utilization = float(inference.get("gpu_memory_utilization"))
    if not 0.0 < utilization < 1.0:
        raise HardwareProfileError(
            "inference.gpu_memory_utilization must be between zero and one"
        )
    workload = str(root.get("workload", ""))
    if workload != "full":
        raise HardwareProfileError("the A800 environment requires workload='full'")
    vendored = root.get("vendored_frameworks")
    if (
        not isinstance(vendored, list)
        or not vendored
        or not all(item in {"verl", "llamafactory"} for item in vendored)
    ):
        raise HardwareProfileError(
            "vendored_frameworks must contain supported framework names"
        )
    if "verl" not in vendored:
        raise HardwareProfileError("vendored_frameworks must include verl")
    attention_backend = str(inference.get("vllm_attention_backend", ""))
    if attention_backend != "FLASH_ATTN":
        raise HardwareProfileError(
            "inference.vllm_attention_backend must be 'FLASH_ATTN'"
        )
    xformers_policy = str(inference.get("xformers_kernel_policy", ""))
    if xformers_policy not in {"enabled", "disabled"}:
        raise HardwareProfileError(
            "inference.xformers_kernel_policy must be 'enabled' or 'disabled'"
        )
    return HardwareProfile(
        path=path,
        name=name,
        display_name=str(root.get("display_name", "")),
        workload=workload,
        vendored_frameworks=tuple(vendored),
        name_patterns=tuple(patterns),
        compute_capability=(capability[0], capability[1]),
        cache_arch=cache_arch,
        torch_cuda_arch_list=str(gpu.get("torch_cuda_arch_list", "")),
        minimum_driver=str(gpu.get("minimum_driver", "")),
        final_gpu_count=_positive_int(
            gpu.get("final_gpu_count"), "gpu.final_gpu_count"
        ),
        distributed_gpu_counts=distributed_counts,
        runtime_config=_repo_path(root.get("runtime_config"), "runtime_config"),
        dependency_input=_repo_path(root.get("dependency_input"), "dependency_input"),
        dependency_lock=_repo_path(root.get("dependency_lock"), "dependency_lock"),
        flash_attention=FlashAttentionArtifact(
            version=str(flash.get("version", "")),
            filename=str(flash.get("filename", "")),
            url=str(flash.get("url", "")),
            sha256=sha256,
            size_bytes=_positive_int(
                flash.get("size_bytes"), "flash_attention.size_bytes"
            ),
        ),
        inference_gpu_memory_utilization=utilization,
        vllm_attention_backend=attention_backend,
        xformers_kernel_policy=xformers_policy,
    )


def validate_torch_devices(
    profile: HardwareProfile, expected_count: int | None
) -> list[dict[str, Any]]:
    """Validate every visible CUDA device against a profile."""
    import torch

    if not torch.cuda.is_available():
        raise HardwareProfileError("PyTorch cannot initialize CUDA")
    count = torch.cuda.device_count()
    if expected_count is not None and count != expected_count:
        raise HardwareProfileError(
            f"{profile.name} expects {expected_count} visible GPUs, found {count}"
        )
    devices: list[dict[str, Any]] = []
    for index in range(count):
        gpu_name = torch.cuda.get_device_name(index)
        capability = tuple(torch.cuda.get_device_capability(index))
        if capability != profile.compute_capability:
            raise HardwareProfileError(
                f"GPU {index} capability {capability} does not match "
                f"{profile.name} {profile.compute_capability}"
            )
        if not any(
            re.search(pattern, gpu_name, re.IGNORECASE)
            for pattern in profile.name_patterns
        ):
            raise HardwareProfileError(
                f"GPU {index} name {gpu_name!r} does not match {profile.name}"
            )
        devices.append(
            {"index": index, "name": gpu_name, "compute_capability": list(capability)}
        )
    return devices


def validate_install_state(
    profile: HardwareProfile, state_path: Path
) -> dict[str, Any]:
    """Reject an environment installed for a different hardware profile."""
    try:
        state = read_json(state_path)
    except OSError as exc:
        raise HardwareProfileError(f"missing installation state: {state_path}") from exc
    except json.JSONDecodeError as exc:
        raise HardwareProfileError(f"invalid installation state: {state_path}") from exc
    expected = {
        "hardware_profile": profile.name,
        "workload": profile.workload,
        "gpu_cache_arch": profile.cache_arch,
        "dependency_input": str(profile.dependency_input),
        "dependency_lock": str(profile.dependency_lock),
        "flash_attention_version": profile.flash_attention.version,
        "flash_attention_sha256": profile.flash_attention.sha256,
        "vllm_attention_backend": profile.vllm_attention_backend,
        "xformers_kernel_policy": profile.xformers_kernel_policy,
    }
    actual = {
        "hardware_profile": state.get("hardware_profile"),
        "workload": state.get("workload"),
        "gpu_cache_arch": state.get("gpu_cache_arch"),
        "dependency_input": state.get("dependency_input"),
        "dependency_lock": state.get("dependency_lock"),
        "flash_attention_version": (state.get("flash_attn") or {}).get("version"),
        "flash_attention_sha256": (state.get("flash_attn") or {}).get("sha256"),
        "vllm_attention_backend": (state.get("attention") or {}).get("vllm_backend"),
        "xformers_kernel_policy": (state.get("attention") or {}).get(
            "xformers_kernel_policy"
        ),
    }
    for key, wanted in expected.items():
        if actual[key] != wanted:
            raise HardwareProfileError(
                f"installed environment mismatch for {key}: "
                f"{actual[key]!r} != {wanted!r}"
            )
    return state


def _lookup(profile: HardwareProfile, key: str) -> object:
    values: dict[str, object] = {
        "name": profile.name,
        "display_name": profile.display_name,
        "workload": profile.workload,
        "vendored_frameworks": ",".join(profile.vendored_frameworks),
        "profile.path": profile.path,
        "gpu.compute_capability": ".".join(map(str, profile.compute_capability)),
        "gpu.compute_capability_major": profile.compute_capability[0],
        "gpu.compute_capability_minor": profile.compute_capability[1],
        "gpu.cache_arch": profile.cache_arch,
        "gpu.torch_cuda_arch_list": profile.torch_cuda_arch_list,
        "gpu.minimum_driver": profile.minimum_driver,
        "gpu.final_gpu_count": profile.final_gpu_count,
        "gpu.distributed_gpu_counts": ",".join(
            map(str, profile.distributed_gpu_counts)
        ),
        "runtime_config": profile.runtime_config,
        "dependency_input": profile.dependency_input,
        "dependency_lock": profile.dependency_lock,
        "flash_attention.version": profile.flash_attention.version,
        "flash_attention.filename": profile.flash_attention.filename,
        "flash_attention.url": profile.flash_attention.url,
        "flash_attention.sha256": profile.flash_attention.sha256,
        "flash_attention.size_bytes": profile.flash_attention.size_bytes,
        "inference.gpu_memory_utilization": profile.inference_gpu_memory_utilization,
        "inference.vllm_attention_backend": profile.vllm_attention_backend,
        "inference.xformers_kernel_policy": profile.xformers_kernel_policy,
    }
    try:
        return values[key]
    except KeyError as exc:
        raise HardwareProfileError(f"unknown hardware profile key: {key}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m trace_opsd.hardware_profile")
    parser.add_argument("--profile", choices=SUPPORTED_PROFILES, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    get = commands.add_parser("get")
    get.add_argument("key")
    commands.add_parser("show")
    devices = commands.add_parser("validate-devices")
    devices.add_argument("--expected-count", type=int)
    state = commands.add_parser("validate-state")
    state.add_argument("--state", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile = load_hardware_profile(args.profile)
    if args.command == "get":
        print(_lookup(profile, args.key))
    elif args.command == "show":
        print(json.dumps(profile.to_dict(), indent=2, sort_keys=True))
    elif args.command == "validate-devices":
        print(
            json.dumps(
                validate_torch_devices(profile, args.expected_count),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        validate_install_state(profile, args.state)
        print(f"installed environment matches hardware profile {profile.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
