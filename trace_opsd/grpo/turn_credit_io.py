"""Durable per-turn credit diagnostics, written before a checkpoint commit."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.request import urlopen

from trace_opsd.compat import namespace_matches, read_json

PYTHON_EXECUTION_RESULT_VERSION = "trace_opsd.python-execution-result.v1"
TURN_CREDIT_SCHEMA = "trace_opsd.trace.turn-credit.v1"


def check_python_execution_protocol(base_url: str) -> None:
    """Fail before enabling credit changes against an incompatible service."""
    try:
        with urlopen(base_url.rstrip("/") + "/health", timeout=5) as response:
            health = json.load(response)
    except Exception as exc:
        raise ValueError(
            "Python error credit requires a reachable tool service"
        ) from exc
    if health.get("python_execution_result_version") != PYTHON_EXECUTION_RESULT_VERSION:
        raise ValueError(
            "Python error credit requires tool protocol "
            + PYTHON_EXECUTION_RESULT_VERSION
            + "; restart the tool service with the updated code"
        )


def write_turn_credit(
    run_dir: Path,
    *,
    step: int,
    credit_state: Mapping[str, Any],
    trajectory_count: int,
    turns: Sequence[Mapping[str, Any]],
) -> Path:
    directory = run_dir / "trace_turn_credit"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{step}.json"
    record = {
        "schema_version": TURN_CREDIT_SCHEMA,
        "step": step,
        "credit_state": dict(credit_state),
        "trajectory_count": trajectory_count,
        "turns": list(turns),
    }
    payload = json.dumps(record, ensure_ascii=False, allow_nan=False).encode("utf-8")
    descriptor, name = tempfile.mkstemp(prefix=f".{step}.", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, target)
    finally:
        if Path(name).exists():
            Path(name).unlink()
    return target


def read_turn_credit(path: Path, *, step: int) -> dict:
    value = read_json(path)
    if (
        not namespace_matches(value.get('schema_version'), TURN_CREDIT_SCHEMA)
        or value.get("step") != step
        or isinstance(value.get("step"), bool)
        or not isinstance(value.get("turns"), list)
        or not isinstance(value.get("trajectory_count"), int)
        or isinstance(value.get("trajectory_count"), bool)
        or value.get("trajectory_count", -1) < 0
        or not isinstance(value.get("credit_state"), dict)
    ):
        raise ValueError(f"Invalid turn credit diagnostic: {path}")
    return value
