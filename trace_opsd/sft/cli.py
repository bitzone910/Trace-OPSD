"""Command line interface for Trace-OPSD supervised fine-tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from trace_opsd.assets import asset_path, resolve_assets_root, resolve_work_root
from trace_opsd.compat import namespace_matches, read_json
from trace_opsd.runtime_config import (
    DEFAULT_RUNTIME_CONFIG,
    REPO_ROOT,
    RuntimeConfig,
    load_runtime_config,
)

from .artifact import ArtifactStoreError, CheckpointRepository
from .config import load_training_config, validate_training_config
from .dataset import validate_dataset_file, write_dataset_registry
from .preflight import (
    PreflightError,
    read_preflight_cache,
    run_preflight,
    write_preflight_cache,
)
from .trainer import TrainingArtifactContext

DEFAULT_TRAIN_CONFIG = REPO_ROOT / "configs" / "sft" / "qwen2_5_3b_full_toolstar.yaml"
DEFAULT_ACCELERATE_CONFIG = REPO_ROOT / "configs" / "accelerate" / "fsdp1_4xa800.yaml"


def _model_path(args: argparse.Namespace) -> Path:
    return (
        (
            args.model_path
            or asset_path("models", "Qwen2.5-3B-Instruct", assets_root=args.assets_root)
        )
        .expanduser()
        .resolve()
    )


def _dataset_file(args: argparse.Namespace) -> Path:
    return (
        (
            args.dataset_file
            or asset_path(
                "datasets", "sft", "sft_math_tir.json", assets_root=args.assets_root
            )
        )
        .expanduser()
        .resolve()
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _settings(args: argparse.Namespace) -> RuntimeConfig:
    settings = load_runtime_config(args.runtime_config)
    work_root = resolve_work_root(args.work_root, assets_root=args.assets_root)
    settings = RuntimeConfig(
        **{
            **settings.__dict__,
            "checkpoint_root": (
                args.checkpoint_root or work_root / "checkpoints"
            ).resolve(),
            "sft_keep_last": 2,
            "sft_keep_best": False,
        }
    )
    return settings


def _repository(settings: RuntimeConfig, run_name: str) -> CheckpointRepository:
    return CheckpointRepository(
        settings.sft_checkpoint_root,
        run_name,
        keep_last=settings.sft_keep_last,
        keep_best=settings.sft_keep_best,
    )


def _cache_path(settings: RuntimeConfig, run_name: str) -> Path:
    return (
        settings.checkpoint_root / ".runtime" / "sft" / "preflight" / f"{run_name}.json"
    )


def _add_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG)
    parser.add_argument("--assets-root", type=Path)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        help="override the base checkpoint root (defaults to runtime configuration)",
    )


def _add_preflight_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument(
        "--accelerate-config", type=Path, default=DEFAULT_ACCELERATE_CONFIG
    )
    _add_runtime_options(parser)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--dataset-file", type=Path)
    parser.add_argument("--no-runtime-check", action="store_true")
    parser.add_argument("--no-token-check", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trace-opsd-sft")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate SFT prerequisites")
    _add_preflight_options(validate)

    train = subparsers.add_parser("train", help="launch one distributed SFT worker")
    _add_preflight_options(train)
    train.add_argument("--run-name", required=True)
    train.add_argument(
        "--resume",
        default="auto",
        help="auto, none, latest, or a checkpoint path",
    )
    train.add_argument("--preflight-only", action="store_true")

    inspect_checkpoint = subparsers.add_parser(
        "inspect-checkpoint", help="verify a committed local checkpoint manifest"
    )
    _add_runtime_options(inspect_checkpoint)
    inspect_checkpoint.add_argument(
        "--run-name", "--run-id", dest="run_name", required=True
    )
    inspect_checkpoint.add_argument("--checkpoint", default="latest")
    inspect_checkpoint.add_argument("--deep", action="store_true")

    export = subparsers.add_parser("export", help="export a committed local checkpoint")
    _add_runtime_options(export)
    export.add_argument("--run-name", "--run-id", dest="run_name", required=True)
    export.add_argument("--model-path", type=Path)
    export.add_argument("--checkpoint", default="latest")
    export.add_argument("--target-dir", type=Path)
    return parser


def _preflight(args: argparse.Namespace, *, cache: bool) -> int:
    settings = _settings(args)
    result = run_preflight(
        repo_root=REPO_ROOT,
        config_path=args.config.resolve(),
        accelerate_config_path=args.accelerate_config.resolve(),
        runtime_config_path=settings.path,
        checkpoint_root=settings.checkpoint_root,
        gpu_count=settings.sft_gpu_count,
        model_path=_model_path(args),
        dataset_file=_dataset_file(args),
        check_runtime=not args.no_runtime_check,
        check_tokens=not args.no_token_check,
    )
    if cache:
        repository = _repository(settings, args.run_name)
        _validate_existing_run(repository, result.training_fingerprint)
        write_preflight_cache(result, _cache_path(settings, args.run_name))
        registry = repository.run_dir / "data"
        write_dataset_registry(_dataset_file(args), registry)
        (registry / "truncation_report.json").write_text(
            json.dumps(result.token_report, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(result.to_dict(), sort_keys=True, indent=2))
    return 0


def _validate_existing_run(
    repository: CheckpointRepository, training_fingerprint: str
) -> None:
    """Reject incompatible history before refreshing any preflight artifacts."""
    if not repository.run_manifest_path.exists():
        return
    try:
        manifest = read_json(repository.run_manifest_path)
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError("existing SFT run manifest is invalid") from exc
    if not isinstance(manifest, dict) or not namespace_matches(
        manifest.get("schema_version"), "trace_opsd.sft.run.v3"
    ):
        raise PreflightError(
            "existing SFT run uses a different schema; use a new run name"
        )
    if manifest.get("training_fingerprint") != training_fingerprint:
        raise PreflightError("existing SFT run has a different training fingerprint")


def _select_resume(repository: CheckpointRepository, value: str) -> str | None:
    if value == "best":
        raise RuntimeError(
            "SFT trains all rows without a validation/best-loss checkpoint"
        )
    if value == "none":
        if repository.run_manifest_path.exists():
            raise RuntimeError(
                f"run {repository.run_id!r} already exists; use --resume auto "
                "or a new run name"
            )
        return None
    if value == "auto":
        return "latest" if repository.try_resolve("latest") is not None else None
    repository.resolve(value)
    return value


def _run_training(args: argparse.Namespace) -> int:
    settings = _settings(args)
    if args.preflight_only:
        return _preflight(args, cache=True)
    if args.no_runtime_check or args.no_token_check:
        raise PreflightError(
            "training requires runtime and truncation preflight checks"
        )
    cached = read_preflight_cache(_cache_path(settings, args.run_name))
    if cached.config_sha256 != _sha256_file(args.config.resolve()):
        raise PreflightError("training config changed after preflight")
    if cached.accelerate_config_sha256 != _sha256_file(
        args.accelerate_config.resolve()
    ):
        raise PreflightError("Accelerate config changed after preflight")
    if cached.runtime_config_sha256 != _sha256_file(settings.path):
        raise PreflightError("runtime config changed after preflight")
    if cached.gpu_count != settings.sft_gpu_count:
        raise PreflightError("SFT GPU count changed after preflight")
    if Path(cached.checkpoint_root) != settings.checkpoint_root:
        raise PreflightError("checkpoint root changed after preflight")
    if Path(cached.model_path) != _model_path(args):
        raise PreflightError("model path changed after preflight")
    if Path(cached.dataset_file) != _dataset_file(args):
        raise PreflightError("dataset path changed after preflight")
    if validate_dataset_file(_dataset_file(args)).digest != cached.dataset_digest:
        raise PreflightError("SFT data changed after preflight")
    if cached.token_report is None:
        raise PreflightError("training requires the native truncation report")

    repository = _repository(settings, args.run_name)
    _validate_existing_run(repository, cached.training_fingerprint)
    resume = _select_resume(repository, args.resume)
    config = load_training_config(args.config)
    config["model_name_or_path"] = str(_model_path(args))
    config["dataset_dir"] = str(repository.run_dir / "data")
    config["output_dir"] = str(repository.run_dir / "trainer-output")
    validate_training_config(config)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank == 0:
        if repository.run_manifest_path.is_file():
            try:
                run_manifest = read_json(repository.run_manifest_path)
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError("existing run manifest is invalid") from exc
            if run_manifest.get("training_fingerprint") != cached.training_fingerprint:
                raise RuntimeError("existing run has a different training fingerprint")
        else:
            repository.write_run_manifest(
                {
                    "schema_version": "trace_opsd.sft.run.v3",
                    "run_id": args.run_name,
                    "training_fingerprint": cached.training_fingerprint,
                    "model_digest": cached.model_digest,
                    "dataset_digest": cached.dataset_digest,
                    "code_digest": cached.code_digest,
                    "vendor_digest": cached.vendor_digest,
                    "accelerate_config_sha256": cached.accelerate_config_sha256,
                    "initial_world_size": settings.sft_gpu_count,
                    "config": config,
                    "dataset_file": cached.dataset_file,
                    "samples": cached.samples,
                    "truncation": cached.token_report,
                    "supervision": "alpaca.full_output",
                }
            )

    from .workflow import run_sft

    metrics = run_sft(
        config,
        artifact_context=TrainingArtifactContext(
            repository=repository,
            training_fingerprint=cached.training_fingerprint,
        ),
        resume=resume,
    )
    if local_rank == 0:
        print(json.dumps(metrics, sort_keys=True, indent=2))
    return 0


def _inspect_checkpoint(args: argparse.Namespace) -> int:
    settings = _settings(args)
    repository = _repository(settings, args.run_name)
    path, manifest = repository.resolve(args.checkpoint)
    repository.verify(manifest, deep=args.deep)
    print(
        json.dumps({"path": str(path), **manifest.to_dict()}, sort_keys=True, indent=2)
    )
    return 0


def _export(args: argparse.Namespace) -> int:
    from .export import export_hf_model

    settings = _settings(args)
    repository = _repository(settings, args.run_name)
    target = args.target_dir or (
        resolve_work_root(args.work_root, assets_root=args.assets_root)
        / "exports"
        / "sft"
        / args.run_name
        / "final"
    )
    path = export_hf_model(
        repository=repository,
        model_path=_model_path(args),
        selector=args.checkpoint,
        target_dir=target,
    )
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(path)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.assets_root = resolve_assets_root(args.assets_root)
        if args.command == "validate":
            return _preflight(args, cache=False)
        if args.command == "train":
            return _run_training(args)
        if args.command == "inspect-checkpoint":
            return _inspect_checkpoint(args)
        if args.command == "export":
            return _export(args)
        raise AssertionError(f"unknown command: {args.command}")
    except (ArtifactStoreError, PreflightError, RuntimeError, ValueError) as exc:
        print(f"SFT error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
