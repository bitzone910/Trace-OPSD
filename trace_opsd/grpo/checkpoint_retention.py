"""Crash-safe retention of the newest two and best TRACE checkpoints."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping

from trace_opsd.compat import namespace_matches, read_json

from .credit_revision import (
    validate_trace_credit_manifest,
    verify_checkpoint_credit_state,
)
from .preflight import PreflightError
from .sampling import DifficultyPoolSampler, TraceSamplingConfig

BEST_CHECKPOINT_FILENAME = "best_checkpoint.json"
INITIAL_VALIDATION_FILENAME = "initial_model_validation.json"
LATEST_CHECKPOINT_FILENAME = "latest_checkpointed_iteration.txt"


def _checkpoint_path(run_dir: Path, step: int) -> Path:
    return run_dir / f"global_step_{step}"


def _is_committed(path: Path) -> bool:
    return path.is_dir() and (path / "manifest.json").is_file()


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"checkpoint manifest is missing or invalid: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"checkpoint manifest must be an object: {path}")
    return value


def _verify_objects(root: Path, manifest: Mapping[str, Any]) -> set[str]:
    objects = manifest.get("objects")
    if not isinstance(objects, list) or not objects:
        raise RuntimeError(f"checkpoint manifest has no integrity objects: {root}")
    names: set[str] = set()
    for item in objects:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise RuntimeError(f"invalid checkpoint integrity object: {root}")
        relative = Path(item["path"])
        object_path = (root / relative).resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or not object_path.is_relative_to(root)
            or relative.as_posix() in names
        ):
            raise RuntimeError(
                f"unsafe or duplicate checkpoint manifest path: {relative}"
            )
        size, expected_hash = item.get("size"), item.get("sha256")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(expected_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_hash)
        ):
            raise RuntimeError(
                f"invalid checkpoint object integrity metadata: {relative}"
            )
        if not object_path.is_file() or object_path.stat().st_size != size:
            raise RuntimeError(
                f"checkpoint object is missing or truncated: {object_path}"
            )
        digest = hashlib.sha256()
        with object_path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != expected_hash:
            raise RuntimeError(f"checkpoint object hash mismatch: {object_path}")
        names.add(relative.as_posix())
    return names


def verify_committed_checkpoint(path: Path, *, allow_quarantined: bool = False) -> None:
    """Re-read a committed Trace-OPSD checkpoint and verify its integrity manifest."""

    path = Path(path).resolve()
    manifest_path = path / "manifest.json"
    manifest = _read_manifest(manifest_path)
    match = re.fullmatch(r"global_step_(\d+)", path.name)
    if match is None and allow_quarantined:
        match = re.fullmatch(r"\.corrupt_global_step_(\d+)\.[0-9a-f]{32}", path.name)
    if match is None or int(match.group(1)) <= 0:
        raise RuntimeError(f"invalid committed checkpoint directory: {path}")
    expected_step = int(match.group(1))
    if (
        not namespace_matches(
            manifest.get("schema_version"), "trace_opsd.grpo.global-checkpoint.v1"
        )
        or isinstance(manifest.get("global_step"), bool)
        or not isinstance(manifest.get("global_step"), int)
        or manifest.get("global_step") != expected_step
    ):
        raise RuntimeError(f"checkpoint manifest metadata mismatch: {manifest_path}")
    names = _verify_objects(path, manifest)
    required = {"actor/manifest.json", "data.pt", "trainer_state.json"}
    if not required.issubset(names):
        raise RuntimeError(f"checkpoint manifest omits required components: {path}")
    for name in sorted(names):
        if not name.endswith("/manifest.json"):
            continue
        component = (path / name).parent.resolve()
        nested = _read_manifest(component / "manifest.json")
        world_size = nested.get("world_size")
        if (
            not namespace_matches(
                nested.get("schema_version"), "trace_opsd.grpo.fsdp-dcp.v1"
            )
            or nested.get("format") != "torch.distributed.checkpoint"
            or isinstance(nested.get("global_step"), bool)
            or not isinstance(nested.get("global_step"), int)
            or nested.get("global_step") != expected_step
            or isinstance(world_size, bool)
            or not isinstance(world_size, int)
            or world_size <= 0
        ):
            raise RuntimeError(
                f"checkpoint component manifest metadata mismatch: {component}"
            )
        component_names = _verify_objects(component, nested)
        if "dcp/.metadata" not in component_names or not any(
            item.startswith("dcp/") and item != "dcp/.metadata"
            for item in component_names
        ):
            raise RuntimeError(f"checkpoint component omits DCP shards: {component}")
        if not {
            f"sidecars/extra_state_rank_{rank}.pt" for rank in range(world_size)
        }.issubset(component_names):
            raise RuntimeError(
                f"checkpoint component omits scheduler/RNG sidecars: {component}"
            )


def verify_trace_checkpoint(
    path: Path,
    *,
    sampler: DifficultyPoolSampler,
    sampling_config: TraceSamplingConfig,
    max_step: int | None = None,
    allow_quarantined: bool = False,
) -> dict[str, Any]:
    """Verify weights and the exact dataset-bound, completed-update cursor."""
    path = Path(path).resolve()
    verify_committed_checkpoint(path, allow_quarantined=allow_quarantined)
    step = int(_read_manifest(path / "manifest.json")["global_step"])
    run = _read_manifest(path.parent / "trace_opsd_grpo_run.json")
    validate_trace_credit_manifest(run)
    if (
        not namespace_matches(run.get("schema_version"), "trace_opsd.grpo.run.v7")
        or run.get("algorithm") != "trace_opsd"
        or run.get("dataset_digest") != sampler.dataset_digest
        or (
            run.get("trace_sampling") is not None
            and run["trace_sampling"] != sampling_config.to_dict()
        )
    ):
        raise RuntimeError(f"TRACE checkpoint run or dataset identity changed: {path}")
    ledger = _read_manifest(path.parent / "trace_curriculum.json")
    if (
        not namespace_matches(
            ledger.get("schema_version"), "trace_opsd.trace.curriculum.v3"
        )
        or ledger.get("dataset_digest") != sampler.dataset_digest
        or ledger.get("sampling") != sampling_config.to_dict()
        or ledger.get("train_source_ids_sha256")
        != hashlib.sha256(
            json.dumps(
                list(sampler.source_ids), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
    ):
        raise RuntimeError(f"TRACE checkpoint curriculum identity changed: {path}")
    stages = ledger.get("stages")
    if not isinstance(stages, list) or not stages:
        raise RuntimeError(
            f"TRACE checkpoint has no registered curriculum stage: {path}"
        )
    cursor = 0
    for stage in stages:
        if not isinstance(stage, dict):
            raise RuntimeError(f"invalid TRACE checkpoint curriculum stage: {path}")
        keys = ("start_step", "end_step", "mid_per_step", "high_per_step", "seed")
        if (
            any(
                isinstance(stage.get(key), bool) or not isinstance(stage.get(key), int)
                for key in keys
            )
            or stage["start_step"] != cursor
            or not cursor < stage["end_step"] <= sampling_config.total_steps
            or min(stage["mid_per_step"], stage["high_per_step"]) < 0
            or stage["mid_per_step"] + stage["high_per_step"]
            != sampling_config.problems_per_step
            or stage["seed"] != sampling_config.seed
            or stage.get("stage_sha256")
            != hashlib.sha256(
                json.dumps(
                    {key: stage[key] for key in keys},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        ):
            raise RuntimeError(f"invalid TRACE checkpoint curriculum stage: {path}")
        cursor = stage["end_step"]
    ceiling = int(stages[-1]["end_step"])
    if not 0 < step <= min(ceiling, max_step if max_step is not None else ceiling):
        raise RuntimeError(f"TRACE checkpoint is outside registered curriculum: {path}")
    state = _read_manifest(path / "trainer_state.json")
    verify_checkpoint_credit_state(run, state, step=step)
    if (
        not namespace_matches(
            state.get("schema_version"), "trace_opsd.grpo.trainer-state.v1"
        )
        or isinstance(state.get("global_step"), bool)
        or not isinstance(state.get("global_step"), int)
        or state.get("global_step") != step
        or isinstance(state.get("consumed_samples"), bool)
        or not isinstance(state.get("consumed_samples"), int)
        or state.get("consumed_samples") != step * sampling_config.problems_per_step
        or state.get("kl_controller_value") is not None
    ):
        raise RuntimeError(f"TRACE checkpoint trainer state mismatch: {path}")
    restored = copy.deepcopy(sampler)
    restored.load_state_dict(
        state.get("trace_dynamic_sampling"),
        expected_global_step=step,
        problems_per_step=sampling_config.problems_per_step,
    )
    expected_counts = dict.fromkeys(("mid", "high"), 0)
    for stage in stages:
        updates = max(0, min(step, int(stage["end_step"])) - int(stage["start_step"]))
        for kind in expected_counts:
            expected_counts[kind] += updates * int(stage[f"{kind}_per_step"])
    if any(
        state["trace_dynamic_sampling"]["pools"][kind]["accepted_count"] != count
        for kind, count in expected_counts.items()
    ):
        raise RuntimeError(
            f"TRACE checkpoint accepted difficulty quotas mismatch: {path}"
        )
    return state


def reconcile_trace_checkpoints(
    run_dir: Path,
    *,
    sampler: DifficultyPoolSampler,
    sampling_config: TraceSamplingConfig,
    max_step: int | None = None,
) -> int | None:
    """Finish interrupted commits; quarantine corrupt newer steps without pruning."""
    run_dir = Path(run_dir).resolve()
    run = _read_manifest(run_dir / "trace_opsd_grpo_run.json")
    validate_trace_credit_manifest(run)
    # Credit identity errors are metadata/configuration errors. Fail before any
    # recovery mutation, rather than quarantining intact historical weights.
    for candidate in run_dir.glob("global_step_*"):
        match = re.fullmatch(r"global_step_(\d+)", candidate.name)
        if match is None or not candidate.is_dir():
            continue
        state_path = candidate / "trainer_state.json"
        if not state_path.is_file():
            continue
        try:
            state = _read_manifest(state_path)
        except RuntimeError:
            continue  # Broken/truncated state is handled by integrity recovery.
        verify_checkpoint_credit_state(run, state, step=int(match.group(1)))
    marker = run_dir / LATEST_CHECKPOINT_FILENAME
    previous = marker.read_text(encoding="utf-8").strip() if marker.is_file() else None
    valid: list[int] = []
    rejected: list[dict[str, Any]] = []
    for candidate in sorted(run_dir.glob("global_step_*")):
        match = re.fullmatch(r"global_step_(\d+)", candidate.name)
        if match is None or not candidate.is_dir():
            continue
        if candidate.resolve().parent != run_dir:
            raise RuntimeError(f"unsafe checkpoint directory: {candidate}")
        step = int(match.group(1))
        try:
            verify_trace_checkpoint(
                candidate,
                sampler=sampler,
                sampling_config=sampling_config,
                max_step=max_step,
            )
        except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
            rejected.append({"step": step, "reason": str(exc)})
        else:
            valid.append(step)
    if not valid:
        if rejected or previous is not None:
            raise RuntimeError(
                "no valid committed TRACE checkpoint; refusing step-zero restart: "
                f"{rejected}"
            )
        return None
    latest = max(valid)
    if previous != str(latest) or rejected:
        report = {
            "previous_latest": previous,
            "selected_step": latest,
            "rejected": rejected,
        }
        # Keep corrupt directories for diagnosis, but clear their canonical names
        # so replay from a valid checkpoint can commit those steps again.
        for item in rejected:
            candidate = _checkpoint_path(run_dir, item["step"])
            target = run_dir / f".corrupt_global_step_{item['step']}.{uuid.uuid4().hex}"
            if candidate.resolve().parent != run_dir or target.parent != run_dir:
                raise RuntimeError(f"unsafe checkpoint quarantine path: {candidate}")
            os.replace(candidate, target)
            item["quarantined_path"] = target.name
        publish_latest_step(run_dir, latest)
        _atomic_write(
            run_dir / "checkpoint_recovery.json", json.dumps(report, indent=2) + "\n"
        )
        print(
            f"TRACE checkpoint reconciliation: {json.dumps(report, sort_keys=True)}",
            file=sys.stderr,
        )
    return latest


def _fsync_directory(path: Path) -> None:
    """Best-effort directory fsync; unsupported platforms still get os.replace."""

    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, contents: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_latest_step(run_dir: Path, *, required: bool = True) -> int | None:
    marker = run_dir / LATEST_CHECKPOINT_FILENAME
    if not marker.is_file():
        if required:
            raise PreflightError(f"latest checkpoint pointer is missing: {marker}")
        return None
    value = marker.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"\d+", value):
        raise PreflightError(f"invalid latest checkpoint pointer: {marker}")
    step = int(value)
    if not _is_committed(_checkpoint_path(run_dir, step)):
        raise PreflightError(
            f"latest checkpoint pointer references an incomplete checkpoint: {step}"
        )
    return step


def publish_latest_step(run_dir: Path, step: int) -> None:
    """Atomically publish a latest pointer only for a committed checkpoint."""

    run_dir = Path(run_dir).resolve()
    if not _is_committed(_checkpoint_path(run_dir, int(step))):
        raise RuntimeError(f"cannot publish incomplete latest checkpoint: {step}")
    _atomic_write(run_dir / LATEST_CHECKPOINT_FILENAME, str(int(step)))
    if read_latest_step(run_dir) != int(step):
        raise RuntimeError("latest checkpoint pointer verification failed")


def restore_latest_step(run_dir: Path, step: int | None) -> None:
    """Restore the prior latest pointer after a failed checkpoint verification."""

    run_dir = Path(run_dir).resolve()
    marker = run_dir / LATEST_CHECKPOINT_FILENAME
    if step is None:
        if marker.exists():
            marker.unlink()
            _fsync_directory(run_dir)
        return
    publish_latest_step(run_dir, step)


def read_best_checkpoint(
    run_dir: Path, *, required: bool = True
) -> dict[str, Any] | None:
    path = run_dir / BEST_CHECKPOINT_FILENAME
    if not path.is_file():
        if required:
            raise PreflightError(f"best checkpoint pointer is missing: {path}")
        return None
    try:
        value = read_json(path)
        step = int(value["step"])
        score = float(value["pass_at_1"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise PreflightError(f"invalid best checkpoint pointer: {path}") from exc
    if not 0.0 <= score <= 1.0:
        raise PreflightError(f"best checkpoint score is outside [0, 1]: {path}")
    if not _is_committed(_checkpoint_path(run_dir, step)):
        raise PreflightError(
            f"best checkpoint pointer references an incomplete checkpoint: {step}"
        )
    return dict(value)


class TraceCheckpointRetention:
    """Publish best metadata, then retain the newest two plus the best checkpoint."""

    def __init__(
        self,
        *,
        run_dir: Path,
        metric_key: str,
        margin: float,
        validation: Mapping[str, Any],
        export_dir: Path | None = None,
        total_steps: int | None = None,
    ) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.metric_key = str(metric_key)
        self.margin = float(margin)
        self.validation = dict(validation)
        self.export_dir = Path(export_dir) if export_dir is not None else None
        self.total_steps = total_steps
        if (self.export_dir is None) != (self.total_steps is None):
            raise ValueError(
                "TRACE retention export directory and horizon must be provided together"
            )
        if not self.metric_key:
            raise ValueError("TRACE best-checkpoint metric key must be nonempty")
        if not 0.0 <= self.margin < 1.0:
            raise ValueError("TRACE best-checkpoint margin must be in [0, 1)")

    def recover(self) -> None:
        """Validate pointers without pruning before validation/export recovery."""

        latest = read_latest_step(self.run_dir, required=False)
        if latest is not None:
            verify_committed_checkpoint(_checkpoint_path(self.run_dir, latest))
        try:
            best = read_best_checkpoint(self.run_dir, required=False)
            if best is not None:
                verify_committed_checkpoint(
                    _checkpoint_path(self.run_dir, int(best["step"]))
                )
        except (PreflightError, RuntimeError) as exc:
            self._recover_best(str(exc))
            best = read_best_checkpoint(self.run_dir, required=False)
        if (
            best is None
            and latest is not None
            and any(self.run_dir.glob("trace_validation_step_*.json"))
        ):
            self._recover_best("best checkpoint pointer is missing")
            best = read_best_checkpoint(self.run_dir, required=False)
        if best is not None:
            if (
                not namespace_matches(
                    best.get("schema_version"), "trace_opsd.grpo.best-checkpoint.v1"
                )
                or not namespace_matches(best.get("metric_key"), self.metric_key)
                or best.get("validation") != self.validation
                or best.get("margin") != self.margin
            ):
                raise RuntimeError("TRACE best checkpoint validation identity changed")

    def _recover_best(self, reason: str) -> None:
        """Rebuild a damaged best pointer from surviving validated weights."""
        pointer = self.run_dir / BEST_CHECKPOINT_FILENAME
        candidates = []
        for checkpoint in self.run_dir.glob("global_step_*"):
            match = re.fullmatch(r"global_step_(\d+)", checkpoint.name)
            if match is None or not checkpoint.is_dir():
                continue
            step = int(match.group(1))
            receipt_path = self.run_dir / f"trace_validation_step_{step}.json"
            if not receipt_path.is_file():
                continue
            try:
                verify_committed_checkpoint(checkpoint)
                receipt = _read_manifest(receipt_path)
            except (OSError, RuntimeError):
                continue
            score = receipt.get("pass_at_1")
            if (
                namespace_matches(
                    receipt.get("schema_version"),
                    "trace_opsd.trace.validation-receipt.v1",
                )
                and receipt.get("step") == step
                and receipt.get("validation") == self.validation
                and not isinstance(score, bool)
                and isinstance(score, (int, float))
                and 0 <= score <= 1
            ):
                candidates.append((step, float(score)))
        selected = None
        for step, score in sorted(candidates):
            if selected is None or score > selected[1] + self.margin:
                selected = (step, score)
        quarantine = self.run_dir / f".corrupt_best_checkpoint.{uuid.uuid4().hex}.json"
        had_pointer = pointer.is_file()
        if had_pointer:
            os.replace(pointer, quarantine)
        if selected is not None:
            _atomic_write(
                pointer,
                json.dumps(
                    {
                        "schema_version": "trace_opsd.grpo.best-checkpoint.v1",
                        "step": selected[0],
                        "pass_at_1": selected[1],
                        "margin": self.margin,
                        "metric_key": self.metric_key,
                        "validation": self.validation,
                    },
                    sort_keys=True,
                    indent=2,
                )
                + "\n",
            )
        report = {
            "reason": reason,
            "selected_step": selected[0] if selected else None,
            "quarantined_pointer": quarantine.name if had_pointer else None,
        }
        _atomic_write(
            self.run_dir / "best_checkpoint_recovery.json",
            json.dumps(report, indent=2) + "\n",
        )
        print(
            f"TRACE best checkpoint recovery: {json.dumps(report, sort_keys=True)}",
            file=sys.stderr,
        )

    def record_initial_validation(
        self,
        *,
        score: float,
        model_path: str,
        generation: Mapping[str, Any],
    ) -> Path:
        """Persist the pre-update model score without treating it as a checkpoint."""

        score = float(score)
        if not 0.0 <= score <= 1.0:
            raise ValueError("pass@1 must be in [0, 1]")
        value = {
            "schema_version": "trace_opsd.grpo.initial-validation.v1",
            "step": 0,
            "model_path": str(model_path),
            "pass_at_1": score,
            "metric_key": self.metric_key,
            "validation": self.validation,
            "generation": dict(generation),
        }
        path = self.run_dir / INITIAL_VALIDATION_FILENAME
        _atomic_write(path, json.dumps(value, sort_keys=True, indent=2) + "\n")
        try:
            confirmed = read_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"initial validation receipt verification failed: {path}"
            ) from exc
        if confirmed != value:
            raise RuntimeError(
                f"initial validation receipt verification failed: {path}"
            )
        return path

    def record_validation(self, *, step: int, score: float) -> bool:
        """Update best when the strict margin is exceeded, then enforce retention."""

        score = float(score)
        if not 0.0 <= score <= 1.0:
            raise ValueError("pass@1 must be in [0, 1]")
        latest = read_latest_step(self.run_dir)
        if latest != int(step):
            raise RuntimeError(
                f"validation step {step} does not match committed latest step {latest}"
            )
        verify_committed_checkpoint(_checkpoint_path(self.run_dir, latest))
        if self.export_dir is not None:
            from .auto_export import recover_trace_exports

            recover_trace_exports(
                run_dir=self.run_dir,
                export_dir=self.export_dir,
                total_steps=int(self.total_steps),
            )

        previous = read_best_checkpoint(self.run_dir, required=False)
        promote = previous is None or score > float(previous["pass_at_1"]) + self.margin
        if promote:
            value = {
                "schema_version": "trace_opsd.grpo.best-checkpoint.v1",
                "step": int(step),
                "pass_at_1": score,
                "margin": self.margin,
                "metric_key": self.metric_key,
                "validation": self.validation,
            }
            path = self.run_dir / BEST_CHECKPOINT_FILENAME
            _atomic_write(path, json.dumps(value, sort_keys=True, indent=2) + "\n")
            confirmed = read_best_checkpoint(self.run_dir)
            if confirmed != value:
                raise RuntimeError(
                    f"best checkpoint pointer verification failed: {path}"
                )
            best_step = int(step)
        else:
            best_step = int(previous["step"])

        confirmed_latest = read_latest_step(self.run_dir)
        confirmed_best = read_best_checkpoint(self.run_dir)
        if confirmed_latest != latest or int(confirmed_best["step"]) != best_step:
            raise RuntimeError("TRACE checkpoint pointers changed during retention")
        committed = sorted(
            int(match.group(1))
            for path in self.run_dir.glob("global_step_*")
            if _is_committed(path)
            and (match := re.fullmatch(r"global_step_(\d+)", path.name))
        )
        if committed and committed[-1] > latest:
            raise RuntimeError("newer committed checkpoint is not the published latest")
        self._prune(set(committed[-2:]) | {latest, best_step})
        return promote

    def _prune(self, keep_steps: set[int]) -> None:
        for step in keep_steps:
            verify_committed_checkpoint(_checkpoint_path(self.run_dir, step))
        targets = []
        for candidate in self.run_dir.glob("global_step_*"):
            if not _is_committed(candidate):
                continue
            match = re.fullmatch(r"global_step_(\d+)", candidate.name)
            if not match or int(match.group(1)) in keep_steps:
                continue
            resolved = candidate.resolve()
            if resolved.parent != self.run_dir or not resolved.name.startswith(
                "global_step_"
            ):
                raise RuntimeError(
                    f"refusing to prune unsafe checkpoint path: {resolved}"
                )
            verify_committed_checkpoint(resolved)
            targets.append(resolved)
        for target in targets:
            shutil.rmtree(target)


__all__ = [
    "BEST_CHECKPOINT_FILENAME",
    "INITIAL_VALIDATION_FILENAME",
    "LATEST_CHECKPOINT_FILENAME",
    "TraceCheckpointRetention",
    "read_best_checkpoint",
    "read_latest_step",
    "publish_latest_step",
    "restore_latest_step",
    "verify_committed_checkpoint",
    "verify_trace_checkpoint",
    "reconcile_trace_checkpoints",
]
