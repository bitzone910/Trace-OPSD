"""Adjust a restored actor's LambdaLR plan without resetting optimizer progress."""

from __future__ import annotations

import math
from typing import Any

from torch.optim.lr_scheduler import LambdaLR

SCALE_STATE_KEY = "trace_opsd_trace_lr_scale"
CONSTANT_LR_STATE_KEY = "trace_opsd_trace_constant_lr"


def _scale(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("TRACE learning-rate scale must be a number in (0, 1]")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "TRACE learning-rate scale must be a number in (0, 1]"
        ) from exc
    if not math.isfinite(result) or not 0 < result <= 1:
        raise ValueError("TRACE learning-rate scale must be a number in (0, 1]")
    return result


def scale_restored_learning_rate(
    optimizer: Any, scheduler: Any, target_scale: float
) -> dict:
    """Set an absolute multiplier of the original plan, idempotently across resume.

    LambdaLR serializes the custom attribute with its normal scheduler state.
    Its lambda/epoch and all optimizer moments/step counters remain untouched.
    """
    target = _scale(target_scale)
    if not isinstance(scheduler, LambdaLR) or scheduler.optimizer is not optimizer:
        raise ValueError(
            "TRACE learning-rate scaling requires the actor's LambdaLR scheduler"
        )
    if getattr(scheduler, CONSTANT_LR_STATE_KEY, None) is not None:
        raise ValueError(
            "TRACE constant learning rate cannot be changed with trace_lr_scale"
        )
    previous = _scale(getattr(scheduler, SCALE_STATE_KEY, 1.0))
    ratio = target / previous
    groups = optimizer.param_groups
    if (
        not groups
        or len(scheduler.base_lrs) != len(groups)
        or len(scheduler.get_last_lr()) != len(groups)
    ):
        raise ValueError("TRACE learning-rate scheduler parameter groups disagree")
    # Validate everything before changing any parameter group.
    values = [*scheduler.base_lrs, *scheduler.get_last_lr()]
    values += [group[key] for group in groups for key in ("lr", "initial_lr")]
    if any(not math.isfinite(float(value)) or float(value) < 0 for value in values):
        raise ValueError("TRACE learning-rate state contains an invalid rate")
    before = [float(group["lr"]) for group in groups]
    scheduler.base_lrs = [float(value) * ratio for value in scheduler.base_lrs]
    scheduler._last_lr = [float(value) * ratio for value in scheduler.get_last_lr()]
    for group in groups:
        group["lr"] = float(group["lr"]) * ratio
        group["initial_lr"] = float(group["initial_lr"]) * ratio
    setattr(scheduler, SCALE_STATE_KEY, target)
    return {
        "previous_scale": previous,
        "scale": target,
        "last_epoch": scheduler.last_epoch,
        "before_lrs": before,
        "after_lrs": [float(group["lr"]) for group in groups],
        "base_lrs": list(scheduler.base_lrs),
    }


def constant_restored_learning_rate(
    optimizer: Any, scheduler: Any, learning_rate: float
) -> dict:
    """Hold a restored actor's LR constant while preserving optimizer progress.

    Keep relative rates between parameter groups and the global scheduler epoch.
    The rate is saved in LambdaLR's state and trainer_state; on resume the trainer
    reinstalls the constant lambda after checkpoint loading. Loading LambdaLR's
    state alone does not restore its callable, so that reinstall is essential.
    """
    if isinstance(learning_rate, bool):
        raise ValueError("TRACE constant learning rate must be finite and positive")
    try:
        target = float(learning_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "TRACE constant learning rate must be finite and positive"
        ) from exc
    if not math.isfinite(target) or target <= 0:
        raise ValueError("TRACE constant learning rate must be finite and positive")
    if not isinstance(scheduler, LambdaLR) or scheduler.optimizer is not optimizer:
        raise ValueError(
            "TRACE constant learning rate requires the actor's LambdaLR scheduler"
        )
    groups = optimizer.param_groups
    if (
        not groups
        or len(scheduler.base_lrs) != len(groups)
        or len(scheduler.get_last_lr()) != len(groups)
    ):
        raise ValueError("TRACE learning-rate scheduler parameter groups disagree")
    values = [*scheduler.base_lrs, *scheduler.get_last_lr()]
    values += [group[key] for group in groups for key in ("lr", "initial_lr")]
    if any(not math.isfinite(float(value)) or float(value) < 0 for value in values):
        raise ValueError("TRACE learning-rate state contains an invalid rate")
    anchor = max(float(value) for value in scheduler.base_lrs)
    if anchor <= 0:
        raise ValueError("TRACE constant learning rate requires a positive base rate")
    rates = [target * (float(value) / anchor) for value in scheduler.base_lrs]
    before = [float(group["lr"]) for group in groups]
    scheduler.base_lrs = list(rates)
    scheduler._last_lr = list(rates)
    scheduler.lr_lambdas = [lambda _step: 1.0 for _ in groups]
    for group, rate in zip(groups, rates, strict=True):
        group["lr"] = rate
        group["initial_lr"] = rate
    setattr(scheduler, CONSTANT_LR_STATE_KEY, target)
    return {
        "mode": "constant",
        "learning_rate": target,
        "last_epoch": scheduler.last_epoch,
        "before_lrs": before,
        "after_lrs": list(rates),
        "base_lrs": list(rates),
    }
