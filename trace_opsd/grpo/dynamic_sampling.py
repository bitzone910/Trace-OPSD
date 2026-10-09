# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DAPO UID-group filtering, adapted for exact Trace-OPSD difficulty quotas.

The group-variance filtering is adapted from verl-recipe's
dapo/dapo_ray_trainer.py at commit
e477deff97b15d067f8b4e71f75b80ae58ad64c4:
https://github.com/verl-project/verl-recipe/blob/e477deff97b15d067f8b4e71f75b80ae58ad64c4/dapo/dapo_ray_trainer.py

Unlike the upstream singleton exception, TRACE requires every real attempt
group to contain exactly eight trajectories. Only binary mixed-accuracy groups
can contribute to a training update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .sampling import DIFFICULTIES


class DynamicSamplingError(ValueError):
    """An invalid group or exhausted refill budget cannot produce an update."""


class DynamicSamplingBudgetError(DynamicSamplingError):
    """The bounded refill budget was exhausted before exact quotas were met."""


@dataclass(frozen=True)
class RewardGroup:
    uid: str
    source_id: str
    difficulty: str
    row_indices: tuple[int, ...]
    correct_count: int

    @property
    def mixed(self) -> bool:
        return 0 < self.correct_count < len(self.row_indices)


@dataclass(frozen=True)
class BatchSelection:
    selected_groups: tuple[RewardGroup, ...]
    row_indices: tuple[int, ...]
    metrics: dict[str, float]


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DynamicSamplingError(f"{name} must be a positive integer")
    return value


def filter_mixed_reward_groups(
    uids: Sequence[str],
    source_ids: Sequence[str],
    difficulties: Sequence[str],
    accuracies: Sequence[float],
    *,
    rollouts_per_problem: int = 8,
    is_padding: Sequence[bool] | None = None,
    candidate_order: Sequence[str] | None = None,
) -> tuple[RewardGroup, ...]:
    """Validate real UID groups and preserve original candidate prompt order.

    Returns all validated groups, including homogeneous groups for diagnostics.
    Callers use RewardGroup.mixed (the upstream std > 0 predicate) to accept.
    Padding is excluded before validating metadata, accuracy, or group size.
    """

    repeats = _positive_integer(rollouts_per_problem, "rollouts_per_problem")
    if repeats <= 1:
        raise DynamicSamplingError("dynamic sampling requires multiple rollouts")
    row_count = len(uids)
    if any(
        len(values) != row_count for values in (source_ids, difficulties, accuracies)
    ):
        raise DynamicSamplingError("group metadata and accuracy lengths differ")
    if is_padding is not None and len(is_padding) != row_count:
        raise DynamicSamplingError("padding length differs from group metadata")
    rows_by_uid: dict[str, list[int]] = {}
    uid_sources: dict[str, str] = {}
    uid_difficulties: dict[str, str] = {}
    accuracy_by_uid: dict[str, list[float]] = {}
    seen_sources: dict[str, str] = {}
    for index in range(row_count):
        if is_padding is not None:
            padding = is_padding[index]
            if not isinstance(padding, (bool, np.bool_)):
                raise DynamicSamplingError("padding markers must be booleans")
            if padding:
                continue
        uid = uids[index]
        source_id = source_ids[index]
        difficulty = difficulties[index]
        if (
            not isinstance(uid, str)
            or not uid
            or not isinstance(source_id, str)
            or not source_id
            or difficulty not in DIFFICULTIES
        ):
            raise DynamicSamplingError("invalid real UID, source ID, or difficulty")
        try:
            accuracy = float(accuracies[index])
        except (TypeError, ValueError, OverflowError) as exc:
            raise DynamicSamplingError("accuracy must be finite binary 0/1") from exc
        if not np.isfinite(accuracy) or accuracy not in (0.0, 1.0):
            raise DynamicSamplingError("accuracy must be finite binary 0/1")
        if uid in rows_by_uid and (
            uid_sources[uid] != source_id or uid_difficulties[uid] != difficulty
        ):
            raise DynamicSamplingError("UID group source ID or difficulty changed")
        if source_id in seen_sources and seen_sources[source_id] != uid:
            raise DynamicSamplingError("duplicate candidate source ID in one batch")
        seen_sources[source_id] = uid
        uid_sources[uid] = source_id
        uid_difficulties[uid] = difficulty
        rows_by_uid.setdefault(uid, []).append(index)
        accuracy_by_uid.setdefault(uid, []).append(accuracy)
    if candidate_order is None:
        order = list(rows_by_uid)
    else:
        order = list(candidate_order)
        if (
            any(not isinstance(uid, str) or not uid for uid in order)
            or len(order) != len(set(order))
            or set(order) != set(rows_by_uid)
        ):
            raise DynamicSamplingError("candidate order does not match real UID groups")
    groups: list[RewardGroup] = []
    for uid in order:
        rows = rows_by_uid[uid]
        if len(rows) != repeats:
            raise DynamicSamplingError(
                f"UID {uid!r} must have exactly {repeats} real trajectories; "
                f"received {len(rows)}"
            )
        # Preserve DAPO's group-UID std > 0 filtering after stricter validation.
        mixed = bool(np.std(accuracy_by_uid[uid]) > 0)
        correct_count = int(sum(accuracy_by_uid[uid]))
        if mixed != (0 < correct_count < repeats):
            raise DynamicSamplingError("binary accuracy variance is inconsistent")
        groups.append(
            RewardGroup(
                uid=uid,
                source_id=uid_sources[uid],
                difficulty=uid_difficulties[uid],
                row_indices=tuple(rows),
                correct_count=correct_count,
            )
        )
    return tuple(groups)


class DynamicBatchAccumulator:
    """Keep exactly each difficulty's quota within one frozen actor snapshot."""

    def __init__(
        self,
        quotas: Mapping[str, int],
        *,
        rollouts_per_problem: int = 8,
        max_num_gen_batches: int = 10,
    ) -> None:
        if not isinstance(quotas, Mapping) or set(quotas) != set(DIFFICULTIES):
            raise DynamicSamplingError("quotas must contain mid and high")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in quotas.values()
        ) or not sum(quotas.values()):
            raise DynamicSamplingError("quotas must be nonnegative and nonempty")
        self.quotas = dict(quotas)
        self.rollouts_per_problem = _positive_integer(
            rollouts_per_problem, "rollouts_per_problem"
        )
        if self.rollouts_per_problem <= 1:
            raise DynamicSamplingError("dynamic sampling requires multiple rollouts")
        self.max_num_gen_batches = _positive_integer(
            max_num_gen_batches, "max_num_gen_batches"
        )
        self.batches_generated = 0
        self.accepted_counts = dict.fromkeys(DIFFICULTIES, 0)
        self.accepted_source_ids: set[str] = set()
        self._seen_uids: set[str] = set()
        self._totals = {
            "candidate_groups": 0,
            "all_wrong_groups": 0,
            "all_correct_groups": 0,
            "mixed_groups": 0,
            "accepted_groups": 0,
            "overflow_groups": 0,
            "candidate_correct_rollouts": 0,
            "accepted_correct_rollouts": 0,
        }

    @property
    def deficits(self) -> dict[str, int]:
        return {
            kind: self.quotas[kind] - self.accepted_counts[kind]
            for kind in DIFFICULTIES
        }

    @property
    def complete(self) -> bool:
        return all(value == 0 for value in self.deficits.values())

    def ensure_can_generate(self) -> None:
        if self.complete:
            raise DynamicSamplingError("dynamic batch is already complete")
        if self.batches_generated >= self.max_num_gen_batches:
            raise DynamicSamplingBudgetError(
                f"dynamic sampling exhausted {self.max_num_gen_batches} "
                f"candidate batches; unmet difficulty quotas={self.deficits}"
            )

    def ingest(self, groups: Sequence[RewardGroup]) -> BatchSelection:
        self.ensure_can_generate()
        if not groups:
            raise DynamicSamplingError("candidate generation contained no real groups")
        seen_uids: set[str] = set()
        seen_sources: set[str] = set()
        seen_rows: set[int] = set()
        for group in groups:
            if (
                not isinstance(group, RewardGroup)
                or group.difficulty not in DIFFICULTIES
                or not isinstance(group.uid, str)
                or not group.uid
                or not isinstance(group.source_id, str)
                or not group.source_id
                or group.uid in self._seen_uids
                or group.uid in seen_uids
                or group.source_id in seen_sources
                or group.source_id in self.accepted_source_ids
                or len(group.row_indices) != self.rollouts_per_problem
                or isinstance(group.correct_count, bool)
                or not isinstance(group.correct_count, int)
                or not 0 <= group.correct_count <= self.rollouts_per_problem
                or any(
                    isinstance(row, bool) or not isinstance(row, int) or row < 0
                    for row in group.row_indices
                )
                or len(set(group.row_indices)) != len(group.row_indices)
                or any(row in seen_rows for row in group.row_indices)
            ):
                raise DynamicSamplingError(
                    "invalid, duplicate, or overlapping reward group"
                )
            seen_uids.add(group.uid)
            seen_sources.add(group.source_id)
            seen_rows.update(group.row_indices)
        selected: list[RewardGroup] = []
        batch_counts = dict.fromkeys(self._totals, 0)
        for group in groups:
            batch_counts["candidate_groups"] += 1
            batch_counts["candidate_correct_rollouts"] += group.correct_count
            if not group.correct_count:
                batch_counts["all_wrong_groups"] += 1
            elif group.correct_count == self.rollouts_per_problem:
                batch_counts["all_correct_groups"] += 1
            else:
                batch_counts["mixed_groups"] += 1
                kind = group.difficulty
                if self.accepted_counts[kind] < self.quotas[kind]:
                    self.accepted_counts[kind] += 1
                    self.accepted_source_ids.add(group.source_id)
                    selected.append(group)
                    batch_counts["accepted_groups"] += 1
                    batch_counts["accepted_correct_rollouts"] += group.correct_count
                else:
                    batch_counts["overflow_groups"] += 1
        self._seen_uids.update(seen_uids)
        self.batches_generated += 1
        for key, count in batch_counts.items():
            self._totals[key] += count
        return BatchSelection(
            selected_groups=tuple(selected),
            row_indices=tuple(row for group in selected for row in group.row_indices),
            metrics={key: float(count) for key, count in batch_counts.items()},
        )

    def metrics(self) -> dict[str, float]:
        metrics = {key: float(value) for key, value in self._totals.items()}
        metrics["candidate_batches"] = float(self.batches_generated)
        candidates = self._totals["candidate_groups"]
        candidate_rollouts = candidates * self.rollouts_per_problem
        accepted_rollouts = (
            self._totals["accepted_groups"] * self.rollouts_per_problem
        )
        metrics["candidate_rollouts"] = float(candidate_rollouts)
        metrics["accepted_rollouts"] = float(accepted_rollouts)
        metrics["candidate_acc_mean"] = (
            self._totals["candidate_correct_rollouts"] / candidate_rollouts
            if candidate_rollouts
            else 0.0
        )
        metrics["accepted_acc_mean"] = (
            self._totals["accepted_correct_rollouts"] / accepted_rollouts
            if accepted_rollouts
            else 0.0
        )
        metrics["acceptance_ratio"] = (
            self._totals["accepted_groups"] / candidates if candidates else 0.0
        )
        for key in (
            "all_wrong_groups",
            "all_correct_groups",
            "mixed_groups",
            "overflow_groups",
        ):
            metrics[f"{key}_ratio"] = (
                self._totals[key] / candidates if candidates else 0.0
            )
        for kind in DIFFICULTIES:
            metrics[f"accepted_{kind}"] = float(self.accepted_counts[kind])
            metrics[f"deficit_{kind}"] = float(self.deficits[kind])
        return metrics


__all__ = [
    "BatchSelection",
    "DynamicBatchAccumulator",
    "DynamicSamplingError",
    "DynamicSamplingBudgetError",
    "RewardGroup",
    "filter_mixed_reward_groups",
]
