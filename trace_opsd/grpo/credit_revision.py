"""Stage-bound TRACE credit revisions without rewriting historical checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

from trace_opsd.compat import (
    identity_for_source,
    is_historical_identity,
    metadata_equal,
    namespace_matches,
    read_json,
)

from .preflight import PreflightError
from .trace_opsd import (
    LEGACY_TOKEN_TEACHER_MODE,
    TRACE_CREDIT_AGGREGATION,
    TRACE_TEACHER_MODE,
    TraceOPSDConfig,
)

HISTORY_FIELD = "trace_credit_history"
LEGACY_CREDIT_VERSION = "trace_opsd.trace.credit.v1"
PYTHON_ERROR_CREDIT_VERSION = "trace_opsd.trace.credit.v2"
TURN_MEAN_CREDIT_VERSION = "trace_opsd.trace.credit.v3"
LEGACY_TRACE_TEACHER_MODE = "current_policy_solution_hindsight.v2"
PENALTY_FIELDS = frozenset(
    {"python_error_penalty_enabled", "python_error_penalty_coef"}
)


def normalize_trace_credit_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Missing error-penalty fields mean the historical, disabled mechanism."""
    try:
        return dict(TraceOPSDConfig.from_mapping(config).__dict__)
    except (TypeError, ValueError) as exc:
        raise PreflightError(f"invalid TRACE credit config: {exc}") from exc


def trace_credit_config_hash(
    config: Mapping[str, Any], *, aggregation: str | None = None
) -> str:
    normalized = normalize_trace_credit_config(config)
    if aggregation is not None:
        if aggregation != TRACE_CREDIT_AGGREGATION:
            raise PreflightError("unsupported TRACE credit aggregation")
        normalized = {
            "trace_credit_aggregation": aggregation,
            "trace_opsd": normalized,
        }
    return hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def trace_credit_aggregation(value: Mapping[str, Any]) -> str:
    """Resolve history without interpreting old token credit as turn credit."""
    mode = value.get("trace_teacher_mode")
    aggregation = value.get("trace_credit_aggregation")
    if mode == TRACE_TEACHER_MODE:
        if aggregation != TRACE_CREDIT_AGGREGATION:
            raise PreflightError(
                "TRACE turn-mean run is missing its credit aggregation"
            )
        return TRACE_CREDIT_AGGREGATION
    if mode in {LEGACY_TOKEN_TEACHER_MODE, LEGACY_TRACE_TEACHER_MODE, None}:
        if aggregation not in {None, "token"}:
            raise PreflightError(
                "historical TRACE run has incompatible credit aggregation"
            )
        return "token"
    raise PreflightError("TRACE run uses incompatible hindsight semantics")


def assert_trainable_trace_manifest(value: Mapping[str, Any]) -> None:
    """Reject historical training before changing run metadata or checkpoints."""
    if is_historical_identity(value.get("schema_version")):
        raise PreflightError(
            "historical run metadata is read-only; export its checkpoint to "
            "Hugging Face and start a new Trace-OPSD run"
        )
    if value.get("trace_teacher_mode") != TRACE_TEACHER_MODE:
        raise PreflightError(
            "existing TRACE run uses incompatible hindsight semantics; "
            "export its checkpoint to Hugging Face and use a new turn-mean run name"
        )
    trace_credit_aggregation(value)


def _entry(
    config: Mapping[str, Any], after_step: int, *, aggregation: str = "token"
) -> dict[str, Any]:
    resolved = normalize_trace_credit_config(config)
    return {
        "effective_after_step": after_step,
        "version": (
            TURN_MEAN_CREDIT_VERSION
            if aggregation == TRACE_CREDIT_AGGREGATION
            else (
                PYTHON_ERROR_CREDIT_VERSION
                if resolved["python_error_penalty_enabled"]
                else LEGACY_CREDIT_VERSION
            )
        ),
        "trace_opsd": resolved,
        "config_sha256": trace_credit_config_hash(
            resolved,
            aggregation=(
                aggregation if aggregation == TRACE_CREDIT_AGGREGATION else None
            ),
        ),
    }


def _core(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key not in PENALTY_FIELDS}


def validate_trace_credit_manifest(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate metadata before reconciliation can quarantine any weights."""
    current = normalize_trace_credit_config(value.get("trace_opsd"))
    aggregation = trace_credit_aggregation(value)
    raw_history = value.get(HISTORY_FIELD)
    if raw_history is None:
        initial = _entry(current, 0, aggregation=aggregation)
        initial["version"] = identity_for_source(
            initial["version"], value.get("schema_version")
        )
        return [initial]
    if not isinstance(raw_history, list) or not raw_history:
        raise PreflightError("invalid TRACE credit revision history")
    history: list[dict[str, Any]] = []
    previous = -1
    for raw in raw_history:
        if not isinstance(raw, dict) or set(raw) != {
            "effective_after_step",
            "version",
            "trace_opsd",
            "config_sha256",
        }:
            raise PreflightError("invalid TRACE credit revision entry")
        after = raw["effective_after_step"]
        if (
            isinstance(after, bool)
            or not isinstance(after, int)
            or after < 0
            or after <= previous
            or (not history and after != 0)
        ):
            raise PreflightError("invalid TRACE credit revision boundaries")
        expected = _entry(raw["trace_opsd"], after, aggregation=aggregation)
        if not metadata_equal(raw, expected):
            raise PreflightError(
                "TRACE credit revision version or config hash mismatch"
            )
        if history and _core(expected["trace_opsd"]) != _core(history[0]["trace_opsd"]):
            raise PreflightError("TRACE credit revision cannot change core OPSD config")
        history.append(dict(raw))
        previous = after
    if current != history[-1]["trace_opsd"]:
        raise PreflightError(
            "TRACE current config disagrees with credit revision history"
        )
    return history


def _read_run(run_dir: Path) -> dict[str, Any]:
    path = Path(run_dir) / "trace_opsd_grpo_run.json"
    try:
        value = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"missing or invalid TRACE run metadata: {path}") from exc
    if not isinstance(value, dict):
        raise PreflightError(f"invalid TRACE run metadata: {path}")
    return value


def _revision(value: Mapping[str, Any], step: int) -> dict[str, Any]:
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise PreflightError("TRACE credit step must be a nonnegative integer")
    history = validate_trace_credit_manifest(value)
    selected = history[0]
    for item in history[1:]:
        if step > item["effective_after_step"]:
            selected = item
    return selected


def resolve_trace_credit_config(run_dir: Path, step: int) -> dict[str, Any]:
    """Resolve the config that produced a checkpoint, including old best models."""
    return dict(_revision(_read_run(run_dir), step)["trace_opsd"])


def checkpoint_credit_state(run_dir: Path, step: int) -> dict[str, str]:
    revision = _revision(_read_run(run_dir), step)
    return {
        "trace_credit_version": revision["version"],
        "trace_credit_config_sha256": revision["config_sha256"],
    }


def verify_checkpoint_credit_state(
    value: Mapping[str, Any], state: Mapping[str, Any], *, step: int
) -> None:
    """Old digest-free checkpoints are legal only in the disabled initial era."""
    revision = _revision(value, step)
    keys = ("trace_credit_version", "trace_credit_config_sha256")
    if not any(key in state for key in keys):
        if (
            revision["effective_after_step"] == 0
            and trace_credit_aggregation(value) == "token"
            and not revision["trace_opsd"]["python_error_penalty_enabled"]
        ):
            return
        raise PreflightError(
            f"TRACE checkpoint {step} is missing credit revision state"
        )
    expected = (revision["version"], revision["config_sha256"])
    if (
        not namespace_matches(state.get(keys[0]), expected[0])
        or state.get(keys[1]) != expected[1]
    ):
        raise PreflightError(f"TRACE checkpoint {step} credit revision mismatch")


def _validate_transition_boundary(
    value: Mapping[str, Any],
    run_dir: Path,
    *,
    after_step: int,
    curriculum_start_step: int | None,
    curriculum_end_step: int | None,
) -> None:
    from .checkpoint_retention import read_latest_step, verify_committed_checkpoint

    if (
        isinstance(after_step, bool)
        or not isinstance(after_step, int)
        or after_step <= 0
        or curriculum_start_step != after_step
        or isinstance(curriculum_end_step, bool)
        or not isinstance(curriculum_end_step, int)
        or curriculum_end_step <= after_step
    ):
        raise PreflightError(
            "TRACE credit transition requires the next curriculum stage boundary"
        )
    latest = read_latest_step(run_dir)
    committed = [
        int(match.group(1))
        for path in run_dir.glob("global_step_*")
        if path.is_dir()
        and (path / "manifest.json").is_file()
        and (match := re.fullmatch(r"global_step_(\d+)", path.name))
    ]
    if latest != after_step or max(committed, default=0) != after_step:
        raise PreflightError(
            "TRACE credit transition requires the latest completed checkpoint"
        )
    verify_committed_checkpoint(run_dir / f"global_step_{after_step}")
    receipt_path = run_dir / f"trace_validation_step_{after_step}.json"
    ledger_path = run_dir / "trace_curriculum.json"
    try:
        receipt = read_json(receipt_path)
        ledger = read_json(ledger_path)
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(
            "TRACE credit transition requires a completed validation receipt "
            "and curriculum"
        ) from exc
    score = receipt.get("pass_at_1") if isinstance(receipt, dict) else None
    if (
        not isinstance(receipt, dict)
        or not namespace_matches(
            receipt.get("schema_version"), "trace_opsd.trace.validation-receipt.v1"
        )
        or receipt.get("step") != after_step
        or receipt.get("validation") != value.get("trace_validation")
        or isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
        or not 0 <= score <= 1
    ):
        raise PreflightError(
            "TRACE credit transition validation receipt is missing or incompatible"
        )
    stages = ledger.get("stages", []) if isinstance(ledger, dict) else []
    if not any(
        isinstance(stage, dict) and stage.get("end_step") == after_step
        for stage in stages
    ) or not any(
        isinstance(stage, dict)
        and stage.get("start_step") == after_step
        and stage.get("end_step") == curriculum_end_step
        for stage in stages
    ):
        raise PreflightError(
            "TRACE credit transition is not a registered curriculum boundary"
        )


def synchronize_trace_credit_manifest(
    value: dict[str, Any],
    run_dir: Path,
    requested: Mapping[str, Any],
    *,
    credit_transition_after_step: int | None = None,
    curriculum_start_step: int | None = None,
    curriculum_end_step: int | None = None,
    revalidate: bool = False,
) -> None:
    """Modify an in-memory manifest only after all transition checks pass."""
    assert_trainable_trace_manifest(value)
    resolved = normalize_trace_credit_config(requested)
    history = validate_trace_credit_manifest(value)
    if not value.get("trace_opsd_config_resolved"):
        if credit_transition_after_step is not None:
            raise PreflightError(
                "TRACE credit transition requires an existing resolved run"
            )
        value["trace_opsd"] = resolved
        value[HISTORY_FIELD] = [
            _entry(resolved, 0, aggregation=TRACE_CREDIT_AGGREGATION)
        ]
        return
    current = history[-1]["trace_opsd"]
    if _core(current) != _core(resolved):
        raise PreflightError(
            "resolved TRACE-OPSD config changed for an existing run; use a new run name"
        )
    if revalidate:
        if credit_transition_after_step is not None:
            raise PreflightError(
                "TRACE credit transitions cannot occur during revalidation"
            )
        if (
            curriculum_end_step is None
            or resolved != _revision(value, curriculum_end_step)["trace_opsd"]
        ):
            raise PreflightError(
                "TRACE revalidation must use the checkpoint's historical credit config"
            )
        # Keep the current run config and its latest history intact.
        value[HISTORY_FIELD] = history
        return
    if resolved == current:
        if credit_transition_after_step is not None:
            existing = next(
                (
                    item
                    for item in history
                    if item["effective_after_step"] == credit_transition_after_step
                ),
                None,
            )
            if existing is None or existing["trace_opsd"] != resolved:
                raise PreflightError(
                    "TRACE credit transition declaration has no matching revision"
                )
            if curriculum_start_step != credit_transition_after_step:
                raise PreflightError(
                    "TRACE credit transition retry must identify its original stage"
                )
        value["trace_opsd"] = resolved
        value[HISTORY_FIELD] = history
        return
    if credit_transition_after_step is None:
        raise PreflightError(
            "TRACE credit config changed; declare "
            "trainer.trace_credit_transition_after_step at a validated stage boundary"
        )
    if credit_transition_after_step <= history[-1]["effective_after_step"]:
        raise PreflightError(
            "TRACE credit transition would rewrite historical credit config"
        )
    _validate_transition_boundary(
        value,
        Path(run_dir),
        after_step=credit_transition_after_step,
        curriculum_start_step=curriculum_start_step,
        curriculum_end_step=curriculum_end_step,
    )
    value["trace_opsd"] = resolved
    value[HISTORY_FIELD] = [
        *history,
        _entry(
            resolved, credit_transition_after_step, aggregation=TRACE_CREDIT_AGGREGATION
        ),
    ]


__all__ = [
    "checkpoint_credit_state",
    "normalize_trace_credit_config",
    "assert_trainable_trace_manifest",
    "resolve_trace_credit_config",
    "synchronize_trace_credit_manifest",
    "trace_credit_config_hash",
    "trace_credit_aggregation",
    "validate_trace_credit_manifest",
    "verify_checkpoint_credit_state",
]
