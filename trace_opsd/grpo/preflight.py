"""Fail-fast input checks for the Trace-OPSD GRPO/verl launch."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .schema import DatasetSummary, validate_dataset_directory

EXPECTED_VERL_COMMIT = "9c38b8bb1876a81273d76de3e79328b2dd2b7b32"


class PreflightError(RuntimeError):
    """Raised when a production GRPO prerequisite is not satisfied."""


@dataclass(frozen=True)
class PreflightResult:
    model_path: str
    model_type: str
    model_architecture: str
    thinking_mode: str
    dataset_dir: str
    dataset_digest: str
    train_samples: int
    validation_samples: int
    verl_commit: str
    verl_source_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingModel:
    model_type: str
    architecture: str
    tokenizer: Any


def _load_model_components(model_path: Path) -> tuple[Any, Any]:
    try:
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as exc:
        raise PreflightError(
            "transformers is required for local model validation"
        ) from exc
    try:
        config = AutoConfig.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False
        )
    except Exception as exc:
        raise PreflightError(
            f"cannot load local model metadata from {model_path}: {exc}"
        ) from exc
    return config, tokenizer


def _inspect_training_model(model_path: Path) -> TrainingModel:
    if not model_path.is_dir():
        raise PreflightError(f"training model directory not found: {model_path}")
    if not (model_path / "config.json").is_file():
        raise PreflightError(f"model config.json not found: {model_path}")
    weight_files = [
        *model_path.glob("*.safetensors"),
        *model_path.glob("pytorch_model*.bin"),
    ]
    if not weight_files:
        raise PreflightError(f"model weights not found in: {model_path}")

    config, tokenizer = _load_model_components(model_path)
    model_type = str(getattr(config, "model_type", ""))
    architectures = tuple(
        str(value) for value in getattr(config, "architectures", ()) or ()
    )
    supported = {
        "qwen2": "Qwen2ForCausalLM",
        "qwen3": "Qwen3ForCausalLM",
    }
    expected_architecture = supported.get(model_type)
    if expected_architecture is None or expected_architecture not in architectures:
        raise PreflightError(
            "training model must be Qwen2/Qwen2.5 or Qwen3 causal LM; "
            f"found model_type={model_type!r}, architectures={architectures!r}"
        )
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template.strip():
        raise PreflightError(
            f"model tokenizer has no usable chat template: {model_path}"
        )
    return TrainingModel(
        model_type=model_type,
        architecture=expected_architecture,
        tokenizer=tokenizer,
    )


def validate_training_model(model_path: Path) -> TrainingModel:
    """Validate a local raw or previously exported Qwen training model."""

    return _inspect_training_model(model_path.resolve())


def resolve_thinking_mode(model_type: str, requested: str) -> str:
    if requested not in {"auto", "on", "off"}:
        raise PreflightError(f"invalid thinking mode: {requested!r}")
    if requested == "auto":
        return "on" if model_type == "qwen3" else "off"
    if requested == "on" and model_type != "qwen3":
        raise PreflightError("thinking mode can only be enabled for Qwen3 models")
    return requested


def verify_verl_snapshot(repo_root: Path) -> str:
    snapshot = repo_root / "frameworks" / "verl"
    verifier = repo_root / "scripts" / "verify_vendored_framework.py"
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(verifier),
                str(snapshot),
                "--expected-name",
                "verl",
                "--expected-revision",
                EXPECTED_VERL_COMMIT,
                "--print-digest",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise PreflightError(f"cannot verify vendored verl snapshot: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise PreflightError(f"vendored verl snapshot verification failed: {detail}")
    digest = result.stdout.strip()
    if len(digest) != 64:
        raise PreflightError(f"verl verifier returned an invalid digest: {digest!r}")
    return digest


def run_preflight(
    *,
    repo_root: Path,
    model_path: Path,
    dataset_dir: Path,
    thinking_mode: str = "auto",
) -> PreflightResult:
    model_path = model_path.resolve()
    dataset_dir = dataset_dir.resolve()
    model = _inspect_training_model(model_path)
    resolved_thinking_mode = resolve_thinking_mode(model.model_type, thinking_mode)
    verl_digest = verify_verl_snapshot(repo_root.resolve())
    template_kwargs = (
        {"enable_thinking": resolved_thinking_mode == "on"}
        if model.model_type == "qwen3"
        else None
    )
    dataset: DatasetSummary = validate_dataset_directory(
        dataset_dir,
        tokenizer=model.tokenizer,
        max_prompt_length=2048,
        apply_chat_template_kwargs=template_kwargs,
    )
    return PreflightResult(
        model_path=str(model_path),
        model_type=model.model_type,
        model_architecture=model.architecture,
        thinking_mode=resolved_thinking_mode,
        dataset_dir=str(dataset_dir),
        dataset_digest=dataset.digest,
        train_samples=dataset.train.samples,
        validation_samples=dataset.validation.samples,
        verl_commit=EXPECTED_VERL_COMMIT,
        verl_source_sha256=verl_digest,
    )


__all__ = [
    "EXPECTED_VERL_COMMIT",
    "PreflightError",
    "PreflightResult",
    "TrainingModel",
    "resolve_thinking_mode",
    "run_preflight",
    "validate_training_model",
    "verify_verl_snapshot",
]
