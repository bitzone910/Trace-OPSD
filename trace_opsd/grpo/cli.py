"""Command-line interface for GRPO validation, run metadata, and export."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from trace_opsd.assets import (
    asset_path,
    configure_asset_environment,
    resolve_work_root,
)
from trace_opsd.compat import metadata_path, namespace_matches, read_json
from trace_opsd.runtime_config import DEFAULT_RUNTIME_CONFIG, load_runtime_config

from .checkpoint_retention import read_latest_step
from .credit_revision import (
    assert_trainable_trace_manifest,
    normalize_trace_credit_config,
    synchronize_trace_credit_manifest,
    validate_trace_credit_manifest,
)
from .export import export_checkpoint
from .preflight import PreflightError, run_preflight
from .sampling import TraceSamplingConfig
from .trace_opsd import TRACE_CREDIT_AGGREGATION, TRACE_TEACHER_MODE, TraceOPSDConfig

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = asset_path("datasets", "grpo")
DEFAULT_CHECKPOINT_ROOT = resolve_work_root() / "checkpoints" / "grpo"
TRACE_CONFIG_PATH = ROOT / "configs" / "grpo" / "trace_opsd.yaml"
RUN_SCHEMA_VERSION = "trace_opsd.grpo.run.v7"


def _positive_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def resolve_dynamic_sampling_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Only the approved binary-accuracy DAPO filter is supported."""

    if set(config) != {"enable", "metric", "max_num_gen_batches", "gen_batch_size"}:
        raise PreflightError("invalid TRACE dynamic sampling configuration fields")
    if config.get("enable") is not True or config.get("metric") != "acc":
        raise PreflightError(
            "TRACE dynamic sampling requires enable=true and metric=acc"
        )
    for key in ("max_num_gen_batches", "gen_batch_size"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise PreflightError(
                f"TRACE dynamic sampling {key} must be a positive integer"
            )
    return dict(config)


def _synchronize_dynamic_fields(
    value: dict[str, Any], requested: Mapping[str, Any], *, run_dir: Path
) -> None:
    resolved = resolve_dynamic_sampling_config(requested)
    old = value.get("trace_dynamic_sampling")
    if old is not None:
        old = resolve_dynamic_sampling_config(old)
        for key in ("enable", "metric", "gen_batch_size"):
            if old[key] != resolved[key]:
                raise PreflightError(
                    f"TRACE dynamic sampling {key} changed; use a new run name"
                )
        previous, current = old["max_num_gen_batches"], resolved["max_num_gen_batches"]
        if current < previous:
            raise PreflightError(
                "TRACE max_num_gen_batches may only increase on resume"
            )
        if current > previous:
            history = value.setdefault("trace_dynamic_sampling_budget_history", [])
            if not isinstance(history, list):
                raise PreflightError("invalid TRACE dynamic sampling budget history")
            history.append(
                {
                    "from": previous,
                    "to": current,
                    "completed_step": read_latest_step(run_dir, required=False) or 0,
                }
            )
    value["trace_dynamic_sampling"] = resolved
    value.setdefault("trace_dynamic_sampling_budget_history", [])


def _model_default() -> str | None:
    return os.getenv("TRACE_OPSD_GRPO_MODEL") or str(
        asset_path("models", "SFT-Qwen2.5-3B")
    )


def _safe_run_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise argparse.ArgumentTypeError(
            "run name must contain only letters, digits, '.', '_' or '-'"
        )
    return value


def _add_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--algorithm", choices=("trace_opsd",), default="trace_opsd")
    parser.add_argument(
        "--thinking-mode", choices=("auto", "on", "off"), default="auto"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trace-opsd-train")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser(
        "validate", help="validate local model, data, and vendored verl"
    )
    _add_inputs(validate)

    record = commands.add_parser(
        "record-run", help="commit preflight metadata for a run"
    )
    _add_inputs(record)
    record.add_argument("--run-name", type=_safe_run_name, required=True)
    record.add_argument("--checkpoint-root", type=Path, default=None)
    record.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG)
    record.add_argument("--training-gpus", required=True)
    record.add_argument("--retrieval-gpu-uuid", required=True)
    record.add_argument(
        "--retrieval-mode",
        choices=("local_dedicated", "local_cpu"),
        default="local_cpu",
    )
    record.add_argument("--profile", choices=("train",), required=True)
    record.add_argument("--max-num-gen-batches", type=_positive_integer, default=10)

    export = commands.add_parser("export", help="merge a local FSDP actor checkpoint")
    export.add_argument("--run-name", type=_safe_run_name, required=True)
    export.add_argument("--checkpoint", default="latest")
    export.add_argument("--checkpoint-root", type=Path, default=None)
    export.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG)
    export.add_argument("--target-dir", type=Path)
    for command in (validate, record, export):
        command.add_argument("--assets-root", type=Path)
        command.add_argument("--work-root", type=Path)
    return parser


def _preflight(args: argparse.Namespace):
    return run_preflight(
        repo_root=ROOT,
        model_path=Path(args.model_path),
        dataset_dir=args.dataset_dir,
        thinking_mode=args.thinking_mode,
    )


def _write_run_manifest(args: argparse.Namespace, result) -> Path:
    if args.algorithm == "trace_opsd":
        # Exact TRACE pool/batch validation happens after Hydra overrides are
        # composed and the fixed training pool has been loaded.
        required_train = 1
        required_validation = 1 if args.profile == "smoke" else 256
    else:
        required_train = 8 if args.profile == "smoke" else 512
        required_validation = 8 if args.profile == "smoke" else 256
    if result.train_samples < required_train:
        raise PreflightError(
            f"{args.profile} profile needs at least {required_train} train rows; "
            f"found {result.train_samples}"
        )
    if result.validation_samples < required_validation:
        raise PreflightError(
            f"{args.profile} profile needs at least "
            f"{required_validation} validation rows; "
            f"found {result.validation_samples}"
        )
    run_dir = (args.checkpoint_root / args.run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "trace_opsd_grpo_run.json"
    trace_config = None
    if args.algorithm == "trace_opsd":
        try:
            raw_trace = yaml.safe_load(TRACE_CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise PreflightError(
                f"cannot load TRACE-OPSD config {TRACE_CONFIG_PATH}: {exc}"
            ) from exc
        trace_config = TraceOPSDConfig.from_mapping(raw_trace).__dict__
    value = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_name": args.run_name,
        "profile": args.profile,
        "algorithm": args.algorithm,
        "distillation_enabled": False,
        "trace_opsd_enabled": args.algorithm == "trace_opsd",
        "trace_opsd": trace_config,
        "trace_credit_aggregation": (
            TRACE_CREDIT_AGGREGATION if args.algorithm == "trace_opsd" else None
        ),
        "trace_teacher_mode": (
            TRACE_TEACHER_MODE if args.algorithm == "trace_opsd" else None
        ),
        "trace_opsd_config_resolved": False,
        "trace_sampling": None,
        "trace_sampling_config_resolved": False,
        "trace_validation": None,
        "trace_validation_config_resolved": False,
        "trace_dynamic_sampling": (
            {
                "enable": True,
                "metric": "acc",
                "max_num_gen_batches": getattr(args, "max_num_gen_batches", 10),
                "gen_batch_size": 4 if args.profile == "smoke" else 32,
            }
            if args.algorithm == "trace_opsd"
            else None
        ),
        "trace_dynamic_sampling_config_resolved": False,
        "trace_dynamic_sampling_budget_history": [],
        "initial_training_gpus": args.training_gpus,
        "initial_retrieval_gpu_uuid": args.retrieval_gpu_uuid,
        "retrieval_mode": getattr(args, "retrieval_mode", "local_dedicated"),
        **result.to_dict(),
    }
    if args.algorithm == "trace_opsd":
        value["trace_dynamic_sampling"] = resolve_dynamic_sampling_config(
            value["trace_dynamic_sampling"]
        )
    serialized = json.dumps(value, sort_keys=True, indent=2) + "\n"
    if metadata_path(path).is_file():
        try:
            existing = read_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            raise PreflightError(f"invalid existing run metadata: {path}") from exc
        if args.algorithm == "trace_opsd" and (
            not namespace_matches(existing.get('schema_version'), RUN_SCHEMA_VERSION)
            or existing.get("trace_teacher_mode") != TRACE_TEACHER_MODE
        ):
            raise PreflightError(
                "existing TRACE run uses incompatible hindsight semantics; "
                "use a new run name"
            )
        if args.algorithm == "trace_opsd":
            assert_trainable_trace_manifest(existing)
        for key in (
            "profile",
            "model_path",
            "dataset_digest",
            "verl_commit",
            "algorithm",
            "model_type",
            "model_architecture",
            "thinking_mode",
            "retrieval_mode",
        ):
            if key not in existing:
                existing[key] = value[key]
                continue
            if existing.get(key) != value.get(key):
                raise PreflightError(
                    f"run metadata field {key!r} changed for {args.run_name!r}; "
                    "use a new run name"
                )
        if args.algorithm == "trace_opsd":
            validate_trace_credit_manifest(existing)
            if existing.get("trace_opsd") is not None:
                existing["trace_opsd"] = normalize_trace_credit_config(
                    existing["trace_opsd"]
                )
            _synchronize_dynamic_fields(
                existing, value["trace_dynamic_sampling"], run_dir=run_dir
            )
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(existing, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    else:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(path)
    return path


def synchronize_trace_run_manifest(
    run_dir: Path,
    trace_config: Mapping[str, Any],
    sampling_config: Mapping[str, Any],
    validation_config: Mapping[str, Any] | None = None,
    *,
    dynamic_sampling_config: Mapping[str, Any] | None = None,
    credit_transition_after_step: int | None = None,
    curriculum_start_step: int | None = None,
    curriculum_end_step: int | None = None,
    revalidate: bool = False,
) -> Path:
    """Record immutable core configs and explicit, validated credit revisions."""

    path = Path(run_dir).resolve() / "trace_opsd_grpo_run.json"
    try:
        value = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"missing or invalid run metadata: {path}") from exc
    if not isinstance(value, dict) or value.get("algorithm") != "trace_opsd":
        raise PreflightError(f"TRACE training requires a TRACE run manifest: {path}")
    if (
        not namespace_matches(value.get('schema_version'), RUN_SCHEMA_VERSION)
        or value.get("trace_teacher_mode") != TRACE_TEACHER_MODE
    ):
        raise PreflightError(
            "TRACE run manifest uses incompatible hindsight semantics; "
            "use a new run name"
        )
    assert_trainable_trace_manifest(value)
    resolved = TraceOPSDConfig.from_mapping(trace_config).__dict__
    resolved_sampling = TraceSamplingConfig.from_mapping(sampling_config).to_dict()
    resolved_validation = dict(validation_config or {})
    synchronize_trace_credit_manifest(
        value,
        path.parent,
        resolved,
        credit_transition_after_step=credit_transition_after_step,
        curriculum_start_step=curriculum_start_step,
        curriculum_end_step=curriculum_end_step,
        revalidate=revalidate,
    )
    if (
        value.get("trace_sampling_config_resolved")
        and value.get("trace_sampling") != resolved_sampling
    ):
        raise PreflightError(
            "resolved TRACE sampling config changed for an existing run; "
            "use a new run name"
        )
    if (
        value.get("trace_validation_config_resolved")
        and value.get("trace_validation") != resolved_validation
    ):
        raise PreflightError(
            "resolved TRACE validation set changed for an existing run; "
            "use a new run name"
        )
    value["trace_opsd_enabled"] = True
    value["trace_opsd_config_resolved"] = True
    value["trace_sampling"] = resolved_sampling
    value["trace_sampling_config_resolved"] = True
    value["trace_validation"] = resolved_validation
    value["trace_validation_config_resolved"] = True
    requested_dynamic = dynamic_sampling_config
    if requested_dynamic is None:
        requested_dynamic = value.get("trace_dynamic_sampling") or {
            "enable": True,
            "metric": "acc",
            "max_num_gen_batches": 10,
            "gen_batch_size": resolved_sampling["problems_per_step"],
        }
    _synchronize_dynamic_fields(value, requested_dynamic, run_dir=path.parent)
    value["trace_dynamic_sampling_config_resolved"] = True
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    assets, work = configure_asset_environment(args.assets_root, args.work_root)
    if args.command != "export":
        args.model_path = args.model_path or _model_default()
        args.dataset_dir = args.dataset_dir or Path(
            os.getenv("TRACE_OPSD_GRPO_DATA_DIR", str(assets / "datasets/grpo"))
        )
    if args.command in {"record-run", "export"}:
        args.checkpoint_root = args.checkpoint_root or (
            load_runtime_config(args.runtime_config).grpo_checkpoint_root
        )
    try:
        if args.command == "validate":
            result = _preflight(args)
            print(json.dumps(result.to_dict(), sort_keys=True, indent=2))
        elif args.command == "record-run":
            result = _preflight(args)
            path = _write_run_manifest(args, result)
            print(path)
        elif args.command == "export":
            settings = load_runtime_config(args.runtime_config)
            run_dir = args.checkpoint_root / args.run_name
            target = args.target_dir or (
                settings.export_root / args.run_name / str(args.checkpoint)
            )
            print(
                export_checkpoint(
                    run_dir=run_dir, checkpoint=args.checkpoint, target_dir=target
                )
            )
        else:  # pragma: no cover
            parser.error(f"unknown command: {args.command}")
    except (PreflightError, ValueError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
