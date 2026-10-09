"""Durable scalar metrics and read-only recovery from offline W&B records."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Mapping

from trace_opsd.compat import namespace_matches

REQUIRED_METRICS = (
    "trace/reference_kl/mean",
    "actor/kl_loss",
    "actor/ppo_kl",
    "actor/kl_coef",
    "actor/entropy",
    "actor/grad_norm",
    "actor/lr",
    "actor/optimizer_update_success",
    "trace/dynamic/candidate_acc_mean",
    "response_length/mean",
    "response_length/clip_ratio",
    "trace/token_credit_coverage",
    "trace/termination/invalid_action_limit_ratio",
    "training/epoch",
    "timing_s/step",
)

PYTHON_ERROR_PENALTY_METRICS = tuple(
    "trace/python_error_penalty/" + key
    for key in (
        "enabled",
        "coef",
        "observed_turns",
        "eligible_turns",
        "penalized_turns",
        "penalized_tokens",
        "token_ratio",
        "positive_to_negative_tokens",
        "positive_to_negative_ratio",
        "penalty_abs_mean",
        "penalty_max",
        "final_min",
        "final_max",
        "code_success_turns",
        "code_success_ratio",
        "error_turns",
        "origin_user_code_turns",
        "origin_execution_limit_turns",
        "origin_queue_turns",
        "origin_transport_turns",
        "origin_bootstrap_turns",
        "origin_sandbox_turns",
        "origin_unknown_turns",
        "origin_none_turns",
    )
)


def scalar_metrics(data: Mapping) -> dict[str, float]:
    result = {}
    for key, value in data.items():
        if not isinstance(value, Real) and getattr(value, "ndim", None) == 0:
            value = value.item()
        if isinstance(value, Real):
            result[str(key)] = float(value)
        elif isinstance(value, str) and value in {"NaN", "Infinity", "-Infinity"}:
            result[str(key)] = float(value)
    return result


def append_metrics(
    path: Path,
    *,
    step: int,
    data: Mapping,
    stage_start: int,
    stage_end: int,
    phase: str = "complete",
) -> None:
    """Single trainer writer; append+fsync preserves earlier steps on resume."""
    scalars = scalar_metrics(data)
    encoded = {
        k: v
        if math.isfinite(v)
        else str(v).replace("nan", "NaN").replace("inf", "Infinity")
        for k, v in scalars.items()
    }
    record = {
        "schema_version": "trace_opsd.trace.metrics.v1",
        "step": int(step),
        "stage_start": int(stage_start),
        "stage_end": int(stage_end),
        "phase": phase,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "data": encoded,
    }
    payload = (json.dumps(record, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    # A killed write may leave a partial last line. Preserve it as an invalid
    # line, separated from the next record; the reader reports and skips it.
    with path.open("a+b") as handle:
        if handle.tell():
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                handle.write(b"\n")
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def merge_metrics(target: dict, records: dict) -> list[int]:
    conflicts = []
    for step, values in records.items():
        previous = target.setdefault(step, {})
        if any(k in previous and previous[k] != v for k, v in values.items()):
            conflicts.append(step)
        previous.update(values)
    return conflicts


def read_metrics(path: Path) -> tuple[dict, list[str]]:
    records, warnings = {}, []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
                step = record["step"]
                if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                    raise ValueError("invalid global step")
                values = scalar_metrics(record["data"])
                if (
                    "training/global_step" in values
                    and values["training/global_step"] != step
                ):
                    raise ValueError("training/global_step differs from record step")
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                warnings.append(f"Invalid metrics record {path}:{line_number}: {exc}")
                continue
            if merge_metrics(records, {step: values}):
                warnings.append(f"Conflicting metrics at step {step}; last values used")
    return records, warnings


def metric_coverage(records: dict, after_step: int, step: int) -> dict:
    expected = range(after_step + 1, step + 1)
    missing_steps = [
        s for s in expected if "training/global_step" not in records.get(s, {})
    ]
    missing = {
        key: [s for s in expected if key not in records.get(s, {})]
        for key in REQUIRED_METRICS
    }
    nonfinite = {
        key: [
            s
            for s in expected
            if key in records.get(s, {}) and not math.isfinite(records[s][key])
        ]
        for key in REQUIRED_METRICS
    }
    enabled_steps = [
        s
        for s in expected
        if records.get(s, {}).get("trace/python_error_penalty/enabled") == 1
    ]
    for key in PYTHON_ERROR_PENALTY_METRICS:
        missing[key] = [s for s in enabled_steps if key not in records[s]]
        nonfinite[key] = [
            s
            for s in enabled_steps
            if key in records[s] and not math.isfinite(records[s][key])
        ]
    return {
        "complete": not missing_steps
        and not any(missing.values())
        and not any(nonfinite[key] for key in PYTHON_ERROR_PENALTY_METRICS),
        "expected_steps": step - after_step,
        "missing_steps": missing_steps,
        "missing_by_metric": {k: v for k, v in missing.items() if v},
        "nonfinite_by_metric": {k: v for k, v in nonfinite.items() if v},
        "python_error_penalty": {
            "availability": "available"
            if any(
                "trace/python_error_penalty/enabled" in records.get(s, {})
                for s in expected
            )
            else "unavailable",
            "enabled_steps": enabled_steps,
        },
    }


def decode_history(history) -> tuple[int, dict] | None:
    """W&B HistoryRecord retains global step, including slash/nested keys."""
    data = {}
    for item in history.item:
        key = item.key or "/".join(item.nested_key)
        try:
            data[key] = json.loads(item.value_json)
        except (ValueError, TypeError):
            continue
    values = scalar_metrics(data)
    number = values.get("training/global_step", values.get("_step"))
    if number is None and history.HasField("step"):
        number = history.step.num
    if (
        number is None
        or not math.isfinite(number)
        or number < 0
        or number != int(number)
    ):
        return None
    return int(number), values


def recover_wandb(
    wandb_root: Path, run_name: str, after_step: int, step: int
) -> tuple[dict, list[str], list[str]]:
    """No login/sync/network; only this project's exact run name is accepted."""
    paths = sorted(wandb_root.glob("*-run-*/run-*.wandb"))
    if not paths:
        return {}, [], []
    try:
        from wandb.proto import wandb_internal_pb2
        from wandb.sdk.internal.datastore import DataStore
    except ImportError:
        return {}, [], ["W&B offline recovery unavailable: wandb SDK not installed"]
    records, sources, warnings = {}, [], []
    for path in paths:
        store, matched, pending = DataStore(), False, {}
        identity = {"project": "", "display_name": ""}
        try:
            store.open_for_scan(str(path))
            while (payload := store.scan_data()) is not None:
                record = wandb_internal_pb2.Record()
                record.ParseFromString(payload)
                if record.HasField("run"):
                    # RunRecord is also used for partial updates; an omitted
                    # name/project must not erase the previously checked ID.
                    for key in identity:
                        value = getattr(record.run, key, "")
                        if value:
                            identity[key] = value
                    if (
                        identity["project"]
                        and not namespace_matches(identity["project"], "Trace-OPSD")
                    ) or (
                        identity["display_name"]
                        and identity["display_name"] != run_name
                    ):
                        break
                    matched = (
                        namespace_matches(identity["project"], "Trace-OPSD")
                        and identity["display_name"] == run_name
                    )
                if matched and record.HasField("history"):
                    decoded = decode_history(record.history)
                    if decoded is not None and after_step < decoded[0] <= step:
                        merge_metrics(pending, {decoded[0]: decoded[1]})
        except Exception as exc:
            # A running file may end in a not-yet-complete record. Already
            # decoded history is still useful; the watcher retries later.
            if matched:
                warnings.append(f"W&B scan ended early: {path}: {exc}")
        finally:
            try:
                store.close()
            except (OSError, AttributeError):
                pass
        if matched and pending:
            conflicts = merge_metrics(records, pending)
            sources.append(str(path))
            if conflicts:
                warnings.append(f"Conflicting W&B attempts at steps {conflicts}")
    return records, sources, warnings
