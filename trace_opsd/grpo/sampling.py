"""Resume-stable TRACE sampling and independently cycling difficulty pools."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from typing import Any, Iterator, Mapping, Sequence


class TraceSamplingConfigurationError(ValueError):
    """Raised when a TRACE fixed-pool schedule cannot be constructed safely."""


@dataclass(frozen=True)
class TraceSamplingConfig:
    """Resolved, resume-stable TRACE sampling plan."""

    mode: str
    seed: int
    pool_size: int
    total_steps: int
    problems_per_step: int
    rollouts_per_problem: int

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, allow_legacy: bool = False
    ) -> TraceSamplingConfig:
        raw = dict(value)
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise TraceSamplingConfigurationError(
                f"unknown TRACE sampling fields: {unknown}"
            )
        try:
            config = cls(**raw)
        except TypeError as exc:
            raise TraceSamplingConfigurationError(
                "TRACE sampling config is missing required fields"
            ) from exc
        config.validate(allow_legacy=allow_legacy)
        return config

    def validate(self, *, allow_legacy: bool = False) -> None:
        if self.mode != "curriculum_dynamic_cycles" and not (
            allow_legacy and self.mode == "curriculum_without_replacement"
        ):
            raise TraceSamplingConfigurationError(
                "TRACE sampling mode must be 'curriculum_dynamic_cycles'; "
                "old runs cannot resume"
            )
        for name in (
            "seed",
            "pool_size",
            "total_steps",
            "problems_per_step",
            "rollouts_per_problem",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TraceSamplingConfigurationError(f"{name} must be an integer")
        if self.seed < 0:
            raise TraceSamplingConfigurationError("seed must be non-negative")
        if self.pool_size <= 0:
            raise TraceSamplingConfigurationError("pool_size must be positive")
        if self.total_steps <= 0:
            raise TraceSamplingConfigurationError("total_steps must be positive")
        if self.problems_per_step <= 0:
            raise TraceSamplingConfigurationError("problems_per_step must be positive")
        if self.problems_per_step > self.pool_size:
            raise TraceSamplingConfigurationError(
                "fixed training pool must contain at least problems_per_step rows"
            )
        required = self.total_steps * self.problems_per_step
        if self.mode == "curriculum_without_replacement" and required > self.pool_size:
            raise TraceSamplingConfigurationError(
                f"training pool exhausted: required={required}, "
                f"remaining={self.pool_size}"
            )
        if self.rollouts_per_problem <= 0:
            raise TraceSamplingConfigurationError(
                "rollouts_per_problem must be positive"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DIFFICULTIES = ("mid", "high")
DYNAMIC_SAMPLER_SCHEMA = "trace_opsd.trace.dynamic-sampler.v1"
SHUFFLE_VERSION = "sha256_seed_difficulty_cycle.python_random.v1"


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TraceSamplingConfigurationError(
            f"{name} must be an integer greater than or equal to {minimum}"
        )
    return value


def _difficulty_counts(value: Mapping[str, int], name: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(DIFFICULTIES):
        raise TraceSamplingConfigurationError(f"{name} must contain mid and high")
    return {
        difficulty: _integer(value[difficulty], f"{name}.{difficulty}")
        for difficulty in DIFFICULTIES
    }


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class DifficultyPoolSampler:
    """Consume difficulty-specific shuffled cycles without speculative prefetch.

    Drawing advances durable candidate cursors, including excluded positions.
    The driver separately records generation batches and completed actor updates.
    """

    def __init__(
        self,
        source_ids: Sequence[str],
        difficulties: Sequence[str],
        *,
        seed: int,
        dataset_digest: str,
    ) -> None:
        self.seed = _integer(seed, "seed")
        if not isinstance(dataset_digest, str) or not dataset_digest:
            raise TraceSamplingConfigurationError("dataset_digest must be nonempty")
        self.dataset_digest = dataset_digest
        self.source_ids = tuple(source_ids)
        self.difficulties = tuple(difficulties)
        if (
            not self.source_ids
            or len(self.source_ids) != len(self.difficulties)
            or any(not isinstance(value, str) or not value for value in self.source_ids)
            or len(set(self.source_ids)) != len(self.source_ids)
            or any(value not in DIFFICULTIES for value in self.difficulties)
        ):
            raise TraceSamplingConfigurationError(
                "source IDs must be unique and aligned with mid/high difficulties"
            )
        self.source_ids_digest = _digest(list(self.source_ids))
        self.difficulties_digest = _digest(list(self.difficulties))
        self._pools = {
            difficulty: tuple(
                index
                for index, label in enumerate(self.difficulties)
                if label == difficulty
            )
            for difficulty in DIFFICULTIES
        }
        self._states = {
            difficulty: {
                "cycle_index": 0,
                "next_index": 0,
                "candidate_count": 0,
                "skipped_duplicate_slots": 0,
                "accepted_count": 0,
            }
            for difficulty in DIFFICULTIES
        }
        self._orders = {
            difficulty: self._order(difficulty, 0) for difficulty in DIFFICULTIES
        }
        self.candidate_batches_total = 0
        self.accepted_updates = 0
        self.accepted_problem_exposures = 0

    @property
    def candidate_problems_total(self) -> int:
        return sum(state["candidate_count"] for state in self._states.values())

    def _order(self, difficulty: str, cycle: int) -> tuple[int, ...]:
        indices = list(self._pools[difficulty])
        random.Random(_digest([self.seed, difficulty, cycle])).shuffle(indices)
        return tuple(indices)

    def available_counts(
        self, *, exclude_source_ids: Sequence[str] | set[str] = ()
    ) -> dict[str, int]:
        excluded = set(exclude_source_ids)
        return {
            difficulty: sum(
                self.source_ids[index] not in excluded
                for index in self._pools[difficulty]
            )
            for difficulty in DIFFICULTIES
        }

    def allocate_counts(
        self,
        deficits: Mapping[str, int],
        *,
        candidate_batch_size: int = 32,
        exclude_source_ids: Sequence[str] | set[str] = (),
    ) -> dict[str, int]:
        """Largest-remainder allocation, mid winning ties, with capacity refill."""

        remaining = _difficulty_counts(deficits, "deficits")
        batch_size = _integer(candidate_batch_size, "candidate_batch_size", minimum=1)
        available = self.available_counts(exclude_source_ids=exclude_source_ids)
        capacity = {
            difficulty: available[difficulty] if remaining[difficulty] else 0
            for difficulty in DIFFICULTIES
        }
        for difficulty in DIFFICULTIES:
            if remaining[difficulty] > available[difficulty]:
                raise TraceSamplingConfigurationError(
                    f"{difficulty} quota exceeds available distinct problems: "
                    f"required={remaining[difficulty]}, "
                    f"available={available[difficulty]}"
                )
        allocation = dict.fromkeys(DIFFICULTIES, 0)
        left = min(batch_size, sum(capacity.values()))
        while left:
            active = [
                difficulty
                for difficulty in DIFFICULTIES
                if allocation[difficulty] < capacity[difficulty]
            ]
            denominator = sum(remaining[difficulty] for difficulty in active)
            proposals = {
                difficulty: left * remaining[difficulty] // denominator
                for difficulty in active
            }
            remainder = left - sum(proposals.values())
            ranked = sorted(
                active,
                key=lambda difficulty: (
                    -(left * remaining[difficulty] % denominator),
                    DIFFICULTIES.index(difficulty),
                ),
            )
            for difficulty in ranked[:remainder]:
                proposals[difficulty] += 1
            used = 0
            for difficulty in active:
                count = min(
                    proposals[difficulty],
                    capacity[difficulty] - allocation[difficulty],
                )
                allocation[difficulty] += count
                used += count
            if not used:
                raise TraceSamplingConfigurationError("candidate allocation stalled")
            left -= used
        return allocation

    def draw(
        self,
        counts: Mapping[str, int],
        exclude_source_ids: Sequence[str] | set[str] = (),
    ) -> tuple[int, ...]:
        requested = _difficulty_counts(counts, "counts")
        excluded = set(exclude_source_ids)
        available = self.available_counts(exclude_source_ids=excluded)
        if any(requested[kind] > available[kind] for kind in DIFFICULTIES):
            raise TraceSamplingConfigurationError(
                "candidate draw exceeds available distinct source IDs"
            )
        selected: list[int] = []
        for difficulty in DIFFICULTIES:
            state = self._states[difficulty]
            count = requested[difficulty]
            taken = 0
            while taken < count:
                order = self._orders[difficulty]
                if state["next_index"] == len(order):
                    state["cycle_index"] += 1
                    state["next_index"] = 0
                    order = self._order(difficulty, state["cycle_index"])
                    self._orders[difficulty] = order
                index = order[state["next_index"]]
                state["next_index"] += 1
                source_id = self.source_ids[index]
                if source_id in excluded:
                    state["skipped_duplicate_slots"] += 1
                    continue
                excluded.add(source_id)
                selected.append(index)
                state["candidate_count"] += 1
                taken += 1
        return tuple(selected)

    def record_candidate_batch(self) -> None:
        if self.candidate_batches_total >= self.candidate_problems_total:
            raise TraceSamplingConfigurationError(
                "cannot record a candidate batch without drawn candidates"
            )
        self.candidate_batches_total += 1

    def record_accepted_update(self, counts: Mapping[str, int]) -> None:
        accepted = _difficulty_counts(counts, "accepted counts")
        if not sum(accepted.values()):
            raise TraceSamplingConfigurationError("accepted update must be nonempty")
        if any(
            self._states[kind]["accepted_count"] + accepted[kind]
            > self._states[kind]["candidate_count"]
            for kind in DIFFICULTIES
        ):
            raise TraceSamplingConfigurationError("accepted count exceeds candidates")
        for kind in DIFFICULTIES:
            self._states[kind]["accepted_count"] += accepted[kind]
        self.accepted_updates += 1
        self.accepted_problem_exposures += sum(accepted.values())

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DYNAMIC_SAMPLER_SCHEMA,
            "shuffle_version": SHUFFLE_VERSION,
            "dataset_digest": self.dataset_digest,
            "source_ids_digest": self.source_ids_digest,
            "difficulties_digest": self.difficulties_digest,
            "seed": self.seed,
            "candidate_batches_total": self.candidate_batches_total,
            "candidate_problems_total": self.candidate_problems_total,
            "accepted_updates": self.accepted_updates,
            "accepted_problem_exposures": self.accepted_problem_exposures,
            "pools": {kind: dict(state) for kind, state in self._states.items()},
        }

    def load_state_dict(
        self,
        value: Mapping[str, Any],
        *,
        expected_global_step: int | None = None,
        problems_per_step: int | None = None,
    ) -> None:
        """Validate the entire checkpoint before changing any live state."""

        expected = self.state_dict()
        if not isinstance(value, Mapping) or set(value) != set(expected):
            raise TraceSamplingConfigurationError(
                "invalid dynamic sampler state fields"
            )
        for key in (
            "schema_version",
            "shuffle_version",
            "dataset_digest",
            "source_ids_digest",
            "difficulties_digest",
            "seed",
        ):
            if value[key] != expected[key] or (
                key == "seed" and isinstance(value[key], bool)
            ):
                raise TraceSamplingConfigurationError(
                    f"dynamic sampler checkpoint {key} changed"
                )
        counters = {
            key: _integer(value[key], key)
            for key in (
                "candidate_batches_total",
                "candidate_problems_total",
                "accepted_updates",
                "accepted_problem_exposures",
            )
        }
        pools = value["pools"]
        if not isinstance(pools, Mapping) or set(pools) != set(DIFFICULTIES):
            raise TraceSamplingConfigurationError("invalid dynamic sampler pools")
        validated: dict[str, dict[str, int]] = {}
        for kind in DIFFICULTIES:
            state = pools[kind]
            if not isinstance(state, Mapping) or set(state) != set(self._states[kind]):
                raise TraceSamplingConfigurationError("invalid difficulty state fields")
            state = {
                key: _integer(item, f"{kind}.{key}") for key, item in state.items()
            }
            size = len(self._pools[kind])
            consumed_slots = state["cycle_index"] * size + state["next_index"]
            if (
                state["next_index"] > size
                or (not size and state["cycle_index"])
                or consumed_slots
                != state["candidate_count"] + state["skipped_duplicate_slots"]
                or state["accepted_count"] > state["candidate_count"]
            ):
                raise TraceSamplingConfigurationError(
                    f"invalid {kind} cycle, cursor, or candidate counters"
                )
            validated[kind] = state
        if (
            sum(state["candidate_count"] for state in validated.values())
            != counters["candidate_problems_total"]
            or sum(state["accepted_count"] for state in validated.values())
            != counters["accepted_problem_exposures"]
            or counters["candidate_batches_total"]
            > counters["candidate_problems_total"]
            or counters["accepted_updates"] > counters["candidate_batches_total"]
        ):
            raise TraceSamplingConfigurationError(
                "inconsistent dynamic sampler counters"
            )
        if (expected_global_step is None) != (problems_per_step is None):
            raise TraceSamplingConfigurationError(
                "expected_global_step and problems_per_step must be provided together"
            )
        if expected_global_step is not None:
            step = _integer(expected_global_step, "expected_global_step")
            batch_size = _integer(problems_per_step, "problems_per_step", minimum=1)
            if (
                counters["accepted_updates"] != step
                or counters["accepted_problem_exposures"] != step * batch_size
            ):
                raise TraceSamplingConfigurationError(
                    "dynamic sampler checkpoint accepted exposures mismatch"
                )
        self._states = validated
        self._orders = {
            kind: self._order(kind, state["cycle_index"])
            for kind, state in validated.items()
        }
        self.candidate_batches_total = counters["candidate_batches_total"]
        self.accepted_updates = counters["accepted_updates"]
        self.accepted_problem_exposures = counters["accepted_problem_exposures"]

    def metrics(self) -> dict[str, float]:
        metrics = {
            "candidate_batches_total": float(self.candidate_batches_total),
            "candidate_problems_total": float(self.candidate_problems_total),
            "accepted_updates": float(self.accepted_updates),
            "accepted_problem_exposures": float(self.accepted_problem_exposures),
        }
        for kind, state in self._states.items():
            for key, value in state.items():
                metrics[f"{kind}/{key}"] = float(value)
            metrics[f"{kind}/remaining"] = float(
                len(self._pools[kind]) - state["next_index"]
            )
        return metrics


class CurriculumDataset(Sequence[Any]):
    """Expose only the recorded curriculum prefix, never a fallback schedule."""

    def __init__(
        self,
        dataset: Sequence[Any],
        config: TraceSamplingConfig,
        *,
        schedule: Sequence[int],
        end_step: int,
    ) -> None:
        config.validate(allow_legacy=True)
        if config.mode != "curriculum_without_replacement":
            raise TraceSamplingConfigurationError(
                "explicit curriculum schedules only support legacy sampling"
            )
        if len(dataset) != config.pool_size:
            raise TraceSamplingConfigurationError(
                "sampling pool_size does not match the training dataset"
            )
        self.dataset = dataset
        self.config = config
        self.schedule = tuple(schedule)
        if (
            isinstance(end_step, bool)
            or not isinstance(end_step, int)
            or not 0 < end_step <= config.total_steps
            or len(self.schedule) != end_step * config.problems_per_step
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= config.pool_size
                for index in self.schedule
            )
            or len(set(self.schedule)) != len(self.schedule)
        ):
            raise TraceSamplingConfigurationError(
                "invalid or repeating curriculum schedule"
            )

    def __len__(self) -> int:
        return len(self.schedule)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[self.schedule[index]]

    def pool_index(self, index: int) -> int:
        """Return the underlying fixed-pool index for tests and diagnostics."""

        return self.schedule[index]

    def on_batch_end(self, *, batch: Any) -> Any:
        """Forward optional dataset lifecycle hooks used by the veRL trainer."""

        callback = getattr(self.dataset, "on_batch_end", None)
        return callback(batch=batch) if callback is not None else None


class OffsetSequentialSampler:
    """Resume a deterministic virtual schedule at an absolute sample offset."""

    def __init__(self, dataset: Sequence[Any], *, planned_samples: int) -> None:
        self.dataset = dataset
        if planned_samples < len(dataset):
            raise TraceSamplingConfigurationError(
                "planned horizon is shorter than schedule"
            )
        self.planned_samples = planned_samples
        self.start_index = 0

    def __iter__(self) -> Iterator[int]:
        if not 0 <= self.start_index <= len(self.dataset):
            raise TraceSamplingConfigurationError(
                "resume offset is outside curriculum schedule"
            )
        return iter(range(self.start_index, len(self.dataset)))

    def __len__(self) -> int:
        # Keep the full optimizer horizon when the dataloader is constructed.
        return self.planned_samples


__all__ = [
    "CurriculumDataset",
    "OffsetSequentialSampler",
    "TraceSamplingConfig",
    "TraceSamplingConfigurationError",
    "DIFFICULTIES",
    "DYNAMIC_SAMPLER_SCHEMA",
    "DifficultyPoolSampler",
    "SHUFFLE_VERSION",
]
