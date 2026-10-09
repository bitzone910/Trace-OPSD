"""Configuration loading and invariants for Trace-OPSD SFT."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


class SFTConfigurationError(ValueError):
    """Raised when an SFT configuration violates a fixed project invariant."""


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - locked runtime contains PyYAML
        raise RuntimeError(
            "PyYAML is required to read SFT configuration files"
        ) from exc

    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise SFTConfigurationError(f"configuration must be a mapping: {path}")
    return value


def load_training_config(path: str | Path) -> dict[str, Any]:
    """Load a LlamaFactory-compatible YAML mapping."""

    return _load_yaml(Path(path))


_EXPECTED_VALUES: dict[str, object] = {
    "do_train": True,
    "stage": "sft",
    "finetuning_type": "full",
    "template": "qwen",
    "cutoff_len": 15000,
    "max_samples": 1000000,
    "overwrite_cache": True,
    "preprocessing_num_workers": 16,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 7.0e-6,
    "num_train_epochs": 3.0,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "bf16": True,
    "logging_steps": 10,
    "save_steps": 2000,
    "ddp_timeout": 180000000,
    "train_on_prompt": False,
    "mask_history": False,
    "packing": False,
    "neat_packing": False,
    "trust_remote_code": False,
    "disable_gradient_checkpointing": False,
    "enable_liger_kernel": False,
    "overwrite_output_dir": False,
    "save_only_model": False,
    "restore_callback_states_from_checkpoint": True,
    "do_eval": False,
    "eval_strategy": "no",
    "load_best_model_at_end": False,
    "ignore_data_skip": False,
    "save_strategy": "steps",
    "plot_loss": True,
    "dataset": "sft_math_tir",
}

_FORBIDDEN_KEYS = {
    "deepspeed",
    "quantization_bit",
    "quantization_method",
    "adapter_name_or_path",
    "lora_rank",
    "use_liger_kernel",
    "weight_decay",
    "optim",
    "adam_beta1",
    "adam_beta2",
    "eval_dataset",
    "val_size",
    "metric_for_best_model",
    "max_steps",
}


def validate_training_config(config: Mapping[str, Any]) -> None:
    """Validate the immutable Tool-Star-derived optimization baseline."""

    expected = dict(_EXPECTED_VALUES)
    errors: list[str] = []
    for key, expected_value in expected.items():
        actual = config.get(key)
        if actual != expected_value:
            errors.append(f"{key}={actual!r}, expected {expected_value!r}")

    for key in sorted(_FORBIDDEN_KEYS):
        if key in config and config[key] not in (None, False, ""):
            errors.append(f"forbidden training option is set: {key}")

    model_name = str(config.get("model_name_or_path", ""))
    if not (
        model_name == "Qwen/Qwen2.5-3B-Instruct"
        or Path(model_name).name == "Qwen2.5-3B-Instruct"
    ):
        errors.append("model_name_or_path must identify Qwen2.5-3B-Instruct")

    if config.get("flash_attn") != "fa2":
        errors.append("flash_attn must be fa2")
    if config.get("use_reentrant_gc") is not True:
        errors.append("use_reentrant_gc must be true")
    if config.get("save_strategy") != "steps":
        errors.append("save_strategy must be steps")

    if errors:
        raise SFTConfigurationError(
            "invalid SFT configuration:\n- " + "\n- ".join(errors)
        )


def effective_global_batch(config: Mapping[str, Any], world_size: int = 4) -> int:
    return (
        int(config["per_device_train_batch_size"])
        * int(config["gradient_accumulation_steps"])
        * world_size
    )


def training_fingerprint(
    config: Mapping[str, Any],
    *,
    model_digest: str,
    dataset_digest: str,
    code_digest: str,
    vendor_digest: str,
    accelerate_config_digest: str,
) -> str:
    payload = {
        "accelerate_config_digest": accelerate_config_digest,
        "code_digest": code_digest,
        "config": dict(config),
        "dataset_digest": dataset_digest,
        "model_digest": model_digest,
        "vendor_digest": vendor_digest,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
