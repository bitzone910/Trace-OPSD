"""Small, per-step TRACE diagnostics computed before the actor update."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence


def summarize_trajectory_stability(
    *,
    uids: Sequence[Any],
    scores: Sequence[float],
    ended_with_eos: Sequence[bool],
    termination_reasons: Sequence[str],
) -> dict[str, float]:
    """Count informative binary-reward groups without padded trajectories."""

    size = len(uids)
    if not (size == len(scores) == len(ended_with_eos) == len(termination_reasons)):
        raise ValueError("TRACE stability metric columns have different lengths")

    groups: dict[str, list[float]] = defaultdict(list)
    for uid, score in zip(uids, scores, strict=True):
        value = float(score)
        if value not in (0.0, 1.0):
            raise ValueError(
                f"TRACE stability metrics require binary scores, got {value}"
            )
        groups[str(uid)].append(value)

    all_correct = sum(all(score == 1.0 for score in group) for group in groups.values())
    all_incorrect = sum(
        all(score == 0.0 for score in group) for group in groups.values()
    )
    mixed = len(groups) - all_correct - all_incorrect
    return {
        "trace/groups/count": float(len(groups)),
        "trace/groups/all_correct": float(all_correct),
        "trace/groups/all_incorrect": float(all_incorrect),
        "trace/groups/mixed": float(mixed),
        "trace/groups/mixed_ratio": mixed / len(groups) if groups else 0.0,
        "trace/termination/eos_ratio": (
            sum(bool(value) for value in ended_with_eos) / size if size else 0.0
        ),
        "trace/termination/invalid_action_limit_ratio": (
            sum(reason == "invalid_action_limit" for reason in termination_reasons)
            / size
            if size
            else 0.0
        ),
    }


__all__ = ["summarize_trajectory_stability"]
