"""Deterministic, checkpoint-bound difficulty curriculum for TRACE-OPSD."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq

from trace_opsd.compat import namespace_matches, read_json

from .checkpoint_retention import (
    read_best_checkpoint,
    reconcile_trace_checkpoints,
    verify_committed_checkpoint,
)
from .credit_revision import assert_trainable_trace_manifest
from .preflight import PreflightError
from .sampling import DifficultyPoolSampler, TraceSamplingConfig
from .schema import validate_dataset_directory

LEDGER_NAME = "trace_curriculum.json"
SCHEMA_VERSION = "trace_opsd.trace.curriculum.v3"


class CurriculumError(ValueError):
    """A requested curriculum stage cannot safely resume this run."""


def _digest(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise CurriculumError(f"missing or invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise CurriculumError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_difficulties(
    dataset_dir: Path, source_path: Path
) -> tuple[list[str], list[str], str]:
    """Align source difficulty with the immutable training parquet row order."""

    source_path = source_path.resolve()
    if not source_path.is_file():
        raise CurriculumError(f"difficulty source parquet is missing: {source_path}")
    dataset_dir = dataset_dir.resolve()
    manifest = _read_json(dataset_dir / "manifest.json")
    source_hash = _sha256(source_path)
    if manifest.get("source", {}).get("sha256") != source_hash:
        raise CurriculumError("difficulty source hash differs from dataset manifest")
    source = pq.read_table(source_path, columns=["id", "difficulty"]).to_pylist()
    by_id: dict[str, str] = {}
    for row in source:
        raw_id = row["id"]
        raw_difficulty = row["difficulty"]
        source_id = f"numinamath-{str(raw_id).strip()}"
        difficulty = (
            raw_difficulty.strip().lower() if isinstance(raw_difficulty, str) else None
        )
        if (
            raw_id is None
            or isinstance(raw_id, bool)
            or not str(raw_id).strip()
            or source_id in by_id
            or difficulty not in {"mid", "high"}
        ):
            raise CurriculumError(
                f"invalid source difficulty or duplicate ID: {source_id}"
            )
        by_id[source_id] = difficulty
    train = pq.read_table(
        dataset_dir / "train.parquet", columns=["extra_info"]
    ).to_pylist()
    validation = pq.read_table(
        dataset_dir / "validation.parquet", columns=["extra_info"]
    ).to_pylist()
    train_ids = [row["extra_info"]["source_id"] for row in train]
    validation_ids = [row["extra_info"]["source_id"] for row in validation]
    if (
        len(train_ids) != len(set(train_ids))
        or len(validation_ids) != len(set(validation_ids))
        or set(train_ids) & set(validation_ids)
        or set(train_ids) | set(validation_ids) != set(by_id)
    ):
        raise CurriculumError("source IDs do not match the train/validation split")
    return train_ids, [by_id[source_id] for source_id in train_ids], source_hash


def stage_plan_hash(stage: Mapping[str, Any]) -> str:
    """Lock a stage descriptor, not a reward-dependent candidate sequence."""

    return _digest(
        {
            key: stage.get(key)
            for key in (
                "start_step",
                "end_step",
                "mid_per_step",
                "high_per_step",
                "seed",
            )
        }
    )


def validate_curriculum_plan(
    config: TraceSamplingConfig,
    difficulties: Sequence[str],
    ledger: Mapping[str, Any],
    *,
    verify_last: bool = True,
) -> tuple[dict[str, Any], ...]:
    """Validate registered quotas; dynamic candidates are chosen at runtime."""

    config.validate()
    if len(difficulties) != config.pool_size or any(
        label not in {"mid", "high"} for label in difficulties
    ):
        raise CurriculumError("difficulty map size differs from training pool")
    if (
        not namespace_matches(ledger.get("schema_version"), SCHEMA_VERSION)
        or "legacy_prefix_steps" in ledger
    ):
        raise CurriculumError("unsupported curriculum ledger; old runs cannot resume")
    counts = {
        label: sum(value == label for value in difficulties)
        for label in ("mid", "high")
    }

    cursor = 0
    stages = ledger.get("stages", [])
    if not isinstance(stages, list):
        raise CurriculumError("invalid curriculum stages")
    for stage_number, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise CurriculumError("invalid curriculum stage")
        start, end = stage.get("start_step"), stage.get("end_step")
        mid, high = stage.get("mid_per_step"), stage.get("high_per_step")
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or start != cursor
            or not start < end <= config.total_steps
            or not isinstance(mid, int)
            or isinstance(mid, bool)
            or not isinstance(high, int)
            or isinstance(high, bool)
            or min(mid, high) < 0
            or mid + high != config.problems_per_step
            or isinstance(stage.get("seed"), bool)
            or stage.get("seed") != config.seed
        ):
            raise CurriculumError("invalid or non-contiguous curriculum stage")
        for difficulty, count in (("mid", mid), ("high", high)):
            if count > counts[difficulty]:
                raise CurriculumError(
                    f"{difficulty} pool cannot fill one actor update at stage "
                    f"{start}-{end}: required={count}, distinct={counts[difficulty]}"
                )
        if (verify_last or stage_number != len(stages) - 1) and stage.get(
            "stage_sha256"
        ) != stage_plan_hash(stage):
            raise CurriculumError(f"curriculum stage hash mismatch at step {start}")
        cursor = end
    return tuple(dict(stage) for stage in stages)


def _validate_ledger(
    ledger: Mapping[str, Any],
    *,
    dataset_digest: str,
    source_hash: str,
    config: TraceSamplingConfig,
) -> None:
    if (
        not namespace_matches(ledger.get("schema_version"), SCHEMA_VERSION)
        or "legacy_prefix_steps" in ledger
        or ledger.get("dataset_digest") != dataset_digest
        or ledger.get("source_sha256") != source_hash
        or ledger.get("sampling") != config.to_dict()
        or not isinstance(ledger.get("stages"), list)
    ):
        raise CurriculumError("curriculum ledger, dataset, or sampling config changed")


def stage_receipt_path(run_dir: Path, step: int) -> Path:
    return run_dir / f"trace_validation_step_{step}.json"


def stage_end_step(start_step: int, *, interval: int, total_steps: int) -> int:
    if (
        interval <= 0
        or start_step < 0
        or start_step >= total_steps
        or start_step % interval
    ):
        raise CurriculumError("invalid curriculum stage boundary")
    return min(start_step + interval, total_steps)


def _read_stage_receipt(path: Path, *, step: int, validation: Any) -> dict[str, Any]:
    value = _read_json(path)
    score = value.get("pass_at_1")
    if (
        not namespace_matches(
            value.get("schema_version"), "trace_opsd.trace.validation-receipt.v1"
        )
        or value.get("step") != step
        or value.get("validation") != validation
        or isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not 0 <= score <= 1
    ):
        raise CurriculumError(f"validation receipt is invalid: {path}")
    return value


def write_stage_receipt(
    run_dir: Path, *, step: int, score: float, validation: Mapping[str, Any]
) -> None:
    path = stage_receipt_path(run_dir, step)
    value = {
        "schema_version": "trace_opsd.trace.validation-receipt.v1",
        "step": step,
        "pass_at_1": score,
        "validation": dict(validation),
    }
    if path.exists():
        if _read_json(path) != value:
            raise CurriculumError(f"validation receipt changed: {path}")
        return
    _atomic_json(path, value)


def prepare_stage(
    *,
    run_dir: Path,
    dataset_dir: Path,
    source_path: Path,
    start_step: int,
    mid_per_step: int,
    high_per_step: int,
    interval: int = 50,
    export_dir: Path | None = None,
) -> dict[str, Any]:
    run_dir, dataset_dir = run_dir.resolve(), dataset_dir.resolve()
    manifest = _read_json(run_dir / "trace_opsd_grpo_run.json")
    profile = manifest.get("profile")
    if manifest.get("algorithm") != "trace_opsd" or profile not in {"train", "smoke"}:
        raise CurriculumError("curriculum requires a TRACE training run")
    if not namespace_matches(manifest.get("schema_version"), "trace_opsd.grpo.run.v7"):
        raise CurriculumError("unsupported TRACE run; old runs cannot resume")
    try:
        assert_trainable_trace_manifest(manifest)
    except PreflightError as exc:
        raise CurriculumError(str(exc)) from exc
    horizon, batch_size, stage_interval = (
        (622, 32, 50) if profile == "train" else (2, 4, 1)
    )
    # Smoke is one two-step stage; checkpoints/validation remain every step.
    if profile == "smoke":
        interval = 2
    elif interval != stage_interval:
        raise CurriculumError("formal curriculum requires 50-step stages")
    summary = validate_dataset_directory(dataset_dir)
    if manifest.get("dataset_digest") != summary.digest:
        raise CurriculumError("run dataset digest differs from current dataset")
    sampling = manifest.get("trace_sampling")
    if sampling:
        config = TraceSamplingConfig.from_mapping(sampling)
    else:
        config = TraceSamplingConfig(
            "curriculum_dynamic_cycles",
            42,
            summary.train.samples,
            horizon,
            batch_size,
            8,
        )
    config.validate()
    if (
        config.pool_size != summary.train.samples
        or config.total_steps != horizon
        or config.problems_per_step != batch_size
        or config.rollouts_per_problem != 8
    ):
        raise CurriculumError(
            f"curriculum requires the {horizon}-step, {batch_size}-problem, "
            "8-rollout TRACE plan"
        )
    train_ids, difficulties, source_hash = load_difficulties(dataset_dir, source_path)
    if (
        isinstance(start_step, bool)
        or not isinstance(start_step, int)
        or start_step < 0
        or start_step >= config.total_steps
        or interval <= 0
        or start_step % interval
    ):
        raise CurriculumError(
            f"stage start must be a {interval}-step boundary before the final step"
        )
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (mid_per_step, high_per_step)
        )
        or mid_per_step + high_per_step != config.problems_per_step
    ):
        raise CurriculumError(
            f"mid/high counts must be non-negative and sum to {batch_size}"
        )
    ledger_path = run_dir / LEDGER_NAME
    if ledger_path.exists():
        ledger = _read_json(ledger_path)
        _validate_ledger(
            ledger,
            dataset_digest=summary.digest,
            source_hash=source_hash,
            config=config,
        )
    else:
        if (
            (run_dir / "latest_checkpointed_iteration.txt").exists()
            or start_step
            or any(run_dir.glob("global_step_*"))
        ):
            raise CurriculumError(
                "new curriculum must start at step 0; old runs cannot resume"
            )
        ledger = {
            "schema_version": SCHEMA_VERSION,
            "dataset_digest": summary.digest,
            "source_sha256": source_hash,
            "sampling": config.to_dict(),
            "train_source_ids_sha256": _digest(train_ids),
            "stages": [],
        }
    if ledger.get("train_source_ids_sha256") != _digest(train_ids):
        raise CurriculumError("training source IDs changed")
    validate_curriculum_plan(config, difficulties, ledger)
    stages = ledger["stages"]
    sampler = DifficultyPoolSampler(
        train_ids, difficulties, seed=config.seed, dataset_digest=summary.digest
    )
    latest = (
        reconcile_trace_checkpoints(run_dir, sampler=sampler, sampling_config=config)
        or 0
    )
    end_step = stage_end_step(
        start_step, interval=interval, total_steps=config.total_steps
    )
    matching = next(
        (stage for stage in stages if stage["start_step"] == start_step), None
    )
    if matching is not None:
        if stages[-1] != matching or matching["end_step"] != end_step:
            raise CurriculumError("requested stage is not the active curriculum stage")
        if (matching["mid_per_step"], matching["high_per_step"]) != (
            mid_per_step,
            high_per_step,
        ):
            raise CurriculumError("a recorded stage cannot change its difficulty ratio")
        if latest < start_step or latest > end_step:
            raise CurriculumError("checkpoint is outside the requested stage")
        if latest == end_step:
            receipt = stage_receipt_path(run_dir, end_step)
            if receipt.exists():
                _read_stage_receipt(
                    receipt,
                    step=end_step,
                    validation=manifest.get("trace_validation"),
                )
                if export_dir is not None:
                    # A completed validation receipt can outlive a physically
                    # damaged best pointer. Load the committed latest weights
                    # so runtime retention can recover that pointer as well.
                    try:
                        best = read_best_checkpoint(run_dir)
                        verify_committed_checkpoint(
                            run_dir / f"global_step_{int(best['step'])}"
                        )
                    except (RuntimeError, OSError, ValueError):
                        return {
                            "action": "revalidate",
                            "start_step": start_step,
                            "end_step": end_step,
                        }
                    from .auto_export import recover_trace_exports

                    recover_trace_exports(
                        run_dir=run_dir,
                        export_dir=export_dir,
                        total_steps=config.total_steps,
                    )
                return {
                    "action": "complete",
                    "start_step": start_step,
                    "end_step": end_step,
                }
            return {
                "action": "revalidate",
                "start_step": start_step,
                "end_step": end_step,
            }
    else:
        if latest != start_step or (stages and stages[-1]["end_step"] != start_step):
            raise CurriculumError(
                "new difficulty ratio requires the latest boundary checkpoint"
            )
        if stages:
            receipt = stage_receipt_path(run_dir, start_step)
            if not receipt.exists():
                previous = stages[-1]
                return {
                    "action": "revalidate_previous",
                    "start_step": previous["start_step"],
                    "end_step": previous["end_step"],
                    "mid_per_step": previous["mid_per_step"],
                    "high_per_step": previous["high_per_step"],
                }
            _read_stage_receipt(
                receipt, step=start_step, validation=manifest.get("trace_validation")
            )
            if export_dir is not None:
                from .auto_export import recover_trace_exports

                recover_trace_exports(
                    run_dir=run_dir,
                    export_dir=export_dir,
                    total_steps=config.total_steps,
                )
        stage = {
            "start_step": start_step,
            "end_step": end_step,
            "mid_per_step": mid_per_step,
            "high_per_step": high_per_step,
            "seed": config.seed,
            "stage_sha256": "",
        }
        proposed = dict(ledger)
        proposed["stages"] = [*stages, stage]
        validate_curriculum_plan(config, difficulties, proposed, verify_last=False)
        stage["stage_sha256"] = stage_plan_hash(stage)
        validate_curriculum_plan(config, difficulties, proposed)
        _atomic_json(ledger_path, proposed)
    return {"action": "train", "start_step": start_step, "end_step": end_step}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "status"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--difficulty-source", type=Path)
    parser.add_argument("--after-step", type=int)
    parser.add_argument("--mid-per-step", type=int)
    parser.add_argument("--high-per-step", type=int)
    parser.add_argument("--export-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "status":
            print(
                json.dumps(
                    _read_json(args.run_dir / LEDGER_NAME), indent=2, sort_keys=True
                )
            )
            return 0
        if any(
            value is None
            for value in (
                args.dataset_dir,
                args.difficulty_source,
                args.after_step,
                args.mid_per_step,
                args.high_per_step,
            )
        ):
            parser.error(
                "prepare requires dataset, source, step, mid and high arguments"
            )
        result = prepare_stage(
            run_dir=args.run_dir,
            dataset_dir=args.dataset_dir,
            source_path=args.difficulty_source,
            start_step=args.after_step,
            mid_per_step=args.mid_per_step,
            high_per_step=args.high_per_step,
            export_dir=args.export_dir,
        )
    except (CurriculumError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
