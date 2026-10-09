"""Fail-fast validation for the AutoDL SFT launch."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from trace_opsd.compat import read_json

from .config import (
    effective_global_batch,
    load_training_config,
    training_fingerprint,
    validate_training_config,
)
from .dataset import (
    EXPECTED_SAMPLES,
    DatasetSummary,
    audit_token_lengths,
    load_rows,
    validate_dataset_file,
)


class PreflightError(RuntimeError):
    """Raised when production SFT prerequisites are not satisfied."""


@dataclass(frozen=True)
class PreflightResult:
    config_path: str
    config_sha256: str
    accelerate_config_path: str
    accelerate_config_sha256: str
    runtime_config_path: str
    runtime_config_sha256: str
    checkpoint_root: str
    gpu_count: int
    model_path: str
    model_digest: str
    dataset_file: str
    dataset_digest: str
    code_digest: str
    vendor_digest: str
    training_fingerprint: str
    effective_global_batch: int
    samples: int
    token_report: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def digest_directory(path: Path) -> str:
    if not path.is_dir():
        raise PreflightError(f"model path is not a directory: {path}")
    digest = hashlib.sha256()
    files = [item for item in path.rglob("*") if item.is_file()]
    if not files:
        raise PreflightError(f"model directory is empty: {path}")
    for item in sorted(files, key=lambda value: value.as_posix()):
        relative = item.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def digest_training_code(repo_root: Path) -> str:
    """Hash the Trace-OPSD-owned code that defines the SFT execution path."""

    sft_root = repo_root / "trace_opsd" / "sft"
    files = sorted(sft_root.rglob("*.py"), key=lambda item: item.as_posix())
    files.append(repo_root / "scripts" / "run_sft.sh")
    digest = hashlib.sha256()
    for item in files:
        if not item.is_file():
            raise PreflightError(f"missing SFT code file: {item}")
        relative = item.relative_to(repo_root).as_posix().encode()
        content = item.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _accelerate_training_digest(path: Path) -> str:
    """Hash FSDP behavior while deliberately excluding launch topology."""

    config = load_training_config(path)
    for key in ("num_processes", "machine_rank", "num_machines"):
        config.pop(key, None)
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_accelerate_config(path: Path) -> None:
    config = load_training_config(path)
    expected = {
        "distributed_type": "FSDP",
        "mixed_precision": "bf16",
        "num_machines": 1,
        "use_cpu": False,
        "num_processes": 4,
    }
    errors = [
        f"{key}={config.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if config.get(key) != value
    ]
    fsdp = config.get("fsdp_config")
    if not isinstance(fsdp, dict):
        errors.append("fsdp_config is missing")
    else:
        expected_fsdp = {
            "fsdp_version": 1,
            "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "fsdp_transformer_layer_cls_to_wrap": "Qwen2DecoderLayer",
            "fsdp_sharding_strategy": "FULL_SHARD",
            "fsdp_state_dict_type": "SHARDED_STATE_DICT",
            "fsdp_cpu_ram_efficient_loading": True,
            "fsdp_sync_module_states": True,
            "fsdp_use_orig_params": True,
            "fsdp_offload_params": False,
            "fsdp_activation_checkpointing": False,
            "fsdp_backward_prefetch": "BACKWARD_PRE",
            "fsdp_forward_prefetch": False,
        }
        errors.extend(
            f"fsdp_config.{key}={fsdp.get(key)!r}, expected {value!r}"
            for key, value in expected_fsdp.items()
            if fsdp.get(key) != value
        )
    if errors:
        raise PreflightError(
            "invalid Accelerate configuration:\n- " + "\n- ".join(errors)
        )


def verify_locked_runtime() -> None:
    expected = {
        "torch": "2.8.0",
        "transformers": "4.55.2",
        "accelerate": "1.11.0",
    }
    errors = []
    for package, version in expected.items():
        try:
            actual = importlib.metadata.version(package).split("+", 1)[0]
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"{package} is not installed")
            continue
        if actual != version:
            errors.append(f"{package}={actual}, expected {version}")
    if errors:
        raise PreflightError("runtime version mismatch:\n- " + "\n- ".join(errors))


def verify_gpu_runtime(gpu_count: int) -> None:
    try:
        import torch
    except ImportError as exc:
        raise PreflightError("PyTorch is not installed") from exc
    if torch.version.cuda != "12.8":
        raise PreflightError(
            f"torch.version.cuda={torch.version.cuda!r}, expected '12.8'"
        )
    if torch.cuda.device_count() != gpu_count:
        raise PreflightError(
            f"expected {gpu_count} visible GPUs, found {torch.cuda.device_count()}"
        )
    names = [torch.cuda.get_device_name(index) for index in range(gpu_count)]
    if any("A800" not in name for name in names):
        raise PreflightError(f"all GPUs must be A800; found {names}")
    if not torch.cuda.is_bf16_supported():
        raise PreflightError("CUDA runtime does not support BF16")


def verify_vendor(repo_root: Path) -> str:
    vendor_root = repo_root / "frameworks" / "LlamaFactory"
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "verify_vendored_framework.py"),
            str(vendor_root),
            "--expected-name",
            "LlamaFactory",
            "--print-digest",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise PreflightError(f"vendored LlamaFactory verification failed: {detail}")
    digest = result.stdout.strip()
    if len(digest) != 64:
        raise PreflightError(f"vendor verifier returned an invalid digest: {digest!r}")
    return digest


def _load_qwen2_5_encoding(model_path: Path, cutoff_len: int) -> tuple[Any, Any, Any]:
    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.data.processor.processor_utils import infer_seqlen
    from llamafactory.hparams import DataArguments, ModelArguments
    from llamafactory.model import load_tokenizer

    model_args = ModelArguments(
        model_name_or_path=str(model_path), trust_remote_code=False
    )
    data_args = DataArguments(template="qwen", cutoff_len=cutoff_len)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    return tokenizer, template, infer_seqlen


def validate_tokenized_dataset(
    dataset_file: Path, model_path: Path, cutoff_len: int
) -> dict[str, Any]:
    tokenizer, template, infer_seqlen = _load_qwen2_5_encoding(model_path, cutoff_len)
    return audit_token_lengths(
        load_rows(dataset_file),
        tokenizer=tokenizer,
        template=template,
        infer_seqlen=infer_seqlen,
        cutoff_len=cutoff_len,
    )


def run_preflight(
    *,
    repo_root: Path,
    config_path: Path,
    accelerate_config_path: Path,
    runtime_config_path: Path,
    checkpoint_root: Path,
    gpu_count: int,
    model_path: Path,
    dataset_file: Path,
    check_runtime: bool = True,
    check_tokens: bool = True,
    expected_samples: int | None = EXPECTED_SAMPLES,
) -> PreflightResult:
    config = load_training_config(config_path)
    config["model_name_or_path"] = str(model_path)
    validate_training_config(config)
    if gpu_count != 4:
        raise PreflightError("SFT requires the supported 4 x A800 topology")
    validate_accelerate_config(accelerate_config_path)
    vendor_digest = verify_vendor(repo_root)
    code_digest = digest_training_code(repo_root)
    if check_runtime:
        verify_locked_runtime()
        verify_gpu_runtime(gpu_count)
    dataset: DatasetSummary = validate_dataset_file(
        dataset_file, expected_samples=expected_samples
    )
    token_report = (
        validate_tokenized_dataset(dataset_file, model_path, int(config["cutoff_len"]))
        if check_tokens
        else None
    )
    model_digest = digest_directory(model_path)
    fingerprint = training_fingerprint(
        config,
        model_digest=model_digest,
        dataset_digest=dataset.digest,
        code_digest=code_digest,
        vendor_digest=vendor_digest,
        accelerate_config_digest=_accelerate_training_digest(accelerate_config_path),
    )
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    return PreflightResult(
        config_path=str(config_path.resolve()),
        config_sha256=_sha256_file(config_path),
        accelerate_config_path=str(accelerate_config_path.resolve()),
        accelerate_config_sha256=_sha256_file(accelerate_config_path),
        runtime_config_path=str(runtime_config_path.resolve()),
        runtime_config_sha256=_sha256_file(runtime_config_path),
        checkpoint_root=str(checkpoint_root.resolve()),
        gpu_count=gpu_count,
        model_path=str(model_path.resolve()),
        model_digest=model_digest,
        dataset_file=str(dataset_file.resolve()),
        dataset_digest=dataset.digest,
        code_digest=code_digest,
        vendor_digest=vendor_digest,
        training_fingerprint=fingerprint,
        effective_global_batch=effective_global_batch(config, world_size=gpu_count),
        samples=dataset.samples,
        token_report=token_report,
    )


def write_preflight_cache(result: PreflightResult, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(result.to_dict(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def read_preflight_cache(path: Path) -> PreflightResult:
    try:
        value = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"missing or invalid preflight cache: {path}") from exc
    return PreflightResult(**value)
