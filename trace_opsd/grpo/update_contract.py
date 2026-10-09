"""Runtime guards for TRACE's one-snapshot, one-optimizer-update contract."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"TRACE requires a positive integer {name}")
    return int(value)


def validate_trace_update_config(config: Any, *, use_critic: bool) -> None:
    """Reject configurations that reuse one set of TRACE evidence across updates."""

    algorithm = _field(config, "algorithm")
    estimator = _field(algorithm, "adv_estimator")
    estimator = getattr(estimator, "value", estimator)
    if estimator != "grpo":
        raise ValueError("TRACE requires algorithm.adv_estimator=grpo")
    if use_critic:
        raise ValueError("TRACE does not support a critic")
    warmup = _field(_field(config, "trainer"), "critic_warmup", 0)
    if isinstance(warmup, bool) or warmup != 0:
        raise ValueError("TRACE requires trainer.critic_warmup=0")
    actor = _field(_field(config, "actor_rollout_ref"), "actor")
    epochs = _positive_integer(_field(actor, "ppo_epochs"), "actor.ppo_epochs")
    if epochs != 1:
        raise ValueError("TRACE requires actor.ppo_epochs=1")
    _positive_integer(_field(actor, "ppo_mini_batch_size"), "actor.ppo_mini_batch_size")


def validate_trace_actor_batch(config: Any, batch_rows: int) -> None:
    """Require one mini-batch after complete-group padding for actual actor DP."""

    rows = _positive_integer(batch_rows, "actor batch rows")
    actor_rollout_ref = _field(config, "actor_rollout_ref")
    actor = _field(actor_rollout_ref, "actor")
    mini_batch = _positive_integer(
        _field(actor, "ppo_mini_batch_size"), "actor.ppo_mini_batch_size"
    )
    rollout_n = _positive_integer(
        _field(_field(actor_rollout_ref, "rollout"), "n"), "rollout.n"
    )
    if mini_batch * rollout_n != rows:
        raise ValueError(
            "TRACE requires exactly one complete actor mini-batch after DP padding: "
            f"ppo_mini_batch_size * rollout.n={mini_batch * rollout_n}, "
            f"actor batch rows={rows}"
        )


def _metric_values(value: Any) -> list[float]:
    if isinstance(value, Real):
        return [float(value)]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [number for item in value for number in _metric_values(item)]
    raise RuntimeError(
        f"TRACE update metric has unsupported value type: {type(value).__name__}"
    )


def validate_actor_update_metrics(
    metrics: Mapping[str, Any], *, prefix: str = "actor/"
) -> None:
    """Check each raw rank's receipt; a mean can hide inconsistent step counts."""

    expected = {
        "optimizer_step_count": 1.0,
        "optimizer_steps_applied": 1.0,
        "optimizer_update_success": 1.0,
    }
    for name in (*expected, "grad_norm", "loss"):
        key = f"{prefix}{name}"
        if key not in metrics:
            raise RuntimeError(f"TRACE actor update is missing success metric {key}")
        values = _metric_values(metrics[key])
        if not values or not all(math.isfinite(value) for value in values):
            raise RuntimeError(
                f"TRACE actor update returned a non-finite or empty {key}"
            )
        if name in expected and any(value != expected[name] for value in values):
            raise RuntimeError(
                "TRACE actor update did not apply exactly one successful optimizer "
                f"step: {key}={values}"
            )


__all__ = [
    "validate_actor_update_metrics",
    "validate_trace_actor_batch",
    "validate_trace_update_config",
]
